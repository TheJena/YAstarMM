#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2021 Federico Motta <191685@studenti.unimore.it>
#
# This file is part of YAstarMM
#
# YAstarMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# YAstarMM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YAstarMM.  If not, see <https://www.gnu.org/licenses/>.
"""
   Parallelise heavy Jupyter Notebook cells with multiprocessing.

   Usage:
            from  YAstarMM.parallel  import  (
                cast_date_time_columns_to_timestamps,
                rename_excel_sheets,
            )

   ( or from within the YAstarMM package )

            from          .parallel  import  (
                cast_date_time_columns_to_timestamps,
                rename_excel_sheets,
            )
"""

from .column_rules import (
    DAYFIRST_REGEXP,
    matches_date_time_rule,
    NORMALIZED_TIMESTAMP_COLUMNS,
)
from .utility import (
    black_magic,
)
from collections import Counter, namedtuple, OrderedDict
from datetime import datetime, timedelta
from gc import collect as run_garbage_collector
from hashlib import blake2b
from logging import debug, info, warning
from multiprocessing import cpu_count, Process, Queue
from numexpr import set_num_threads, set_vml_num_threads
from os.path import abspath, basename, isfile
from queue import Empty as EmptyQueue
from re import compile, IGNORECASE
from shutil import move
from string import ascii_letters, punctuation, whitespace
from sys import version_info
from time import time
import pandas as pd

SHEET_RENAMING_RULES = OrderedDict(
    {
        new_sheet_name: compile(case_insensitive_regexp, IGNORECASE)
        for new_sheet_name, case_insensitive_regexp in sorted(
            dict(
                anagraphic=r"",
                DRG=r"",
                emogas=r"",
                exams=r"",
                heparine_and_remdesivir=r"",
                ICD__9=r"",
                patient_journey=r"",
                sofa=r"",
                sofa_history=r"",
                symptoms=r"",
                unit_transfer=r"",
                anamnesis=str(
                    r""
                    r""  # logic or
                    r""
                    r""  # logic or
                    r""
                ),
                diary=r"",
                steroids=r"",
                swabs=r"",
                infections=r"",
                involved_units=r"",
                #
                # post-COVID sheets
                # v v v v v v v v v
                cog=r"",
                therapies=r"",
                hospitalization=r"",
                fibroscan=r"",
                frailty=r"",
                pneumological_exam=r"",
                radiology_report=r"",
                six_min_walk=r"",
                well_being=r"",
                nutrition=r"",
            ).items()
        )
    }
)
SORTING_PREFIX = "\t"
TITLE_COLUMN_REGEXP = compile(r"", IGNORECASE)

FileWorkerError = namedtuple("FileWorkerError", ["filename", "exception"])
FileWorkerInput = namedtuple("FileWorkerInput", ["filename", "rename_mapping"])
FileWorkerOutput = namedtuple("FileWorkerOutput", ["filename", "sheets_dict"])

GroupWorkerInput = namedtuple("GroupWorkerInput", ["group_name", "dataframe"])
GroupWorkerError = namedtuple("GroupWorkerError", ["group_name", "exception"])

InputOutputErrorQueues = namedtuple(
    "InputOutputErrorQueues", ["input_queue", "output_queue", "error_queue"]
)

SheetWorkerError = namedtuple(
    "SheetWorkerError", ["filename", "sheet_name", "exception"]
)
SheetWorkerInput = namedtuple(
    "SheetWorkerInput",
    ["filename", "old_sheet_name", "df"],
)
SheetWorkerOutput = namedtuple(
    "SheetWorkerOutput", ["filename", "new_sheet_name", "new_df"]
)

TimestampWorkerError = namedtuple(
    "TimestampWorkerError", ["sheet_name", "column_name", "exception"]
)
TimestampWorkerInputOutput = namedtuple(
    "TimestampWorkerInputOutput", ["sheet_name", "column_name", "series"]
)


_EIGHTEEN_CENTURIES_IN_MINUTES = 18 * 100 * 365 * 24 * 60


def _concat_same_name_sheets(all_sheets_dict):
    ret = dict()
    for filename, sheets_dict in all_sheets_dict.items():
        for new_sheet_name, new_df in sheets_dict.items():
            new_sheet_name = new_sheet_name.lstrip(SORTING_PREFIX).replace(
                "_",
                "-",
            )
            if new_df.empty or sorted(new_df.columns) == []:
                debug(f"Dropped empty or unneeded sheet '{new_sheet_name}'")
                continue
            if new_sheet_name not in ret:
                ret[new_sheet_name] = new_df
                debug(f"Added '{new_sheet_name}' dataframe to final df_dict")
            else:  # a df with the same new name already exist
                debug(f"Dataframe '{new_sheet_name}' already in final df_dict")
                old_df = ret[new_sheet_name]
                ret[new_sheet_name] = pd.concat(
                    [old_df, new_df], join="outer", sort=True
                )
                debug(
                    "Previous dataframe and current daframe correctly "
                    f"concatenated into '{new_sheet_name}' in final df_dict"
                )
    return OrderedDict(
        {k: v for k, v in sorted(ret.items(), key=lambda tup: tup[0].lower())}
    )


def _convert_single_cell_timestamp(cell, column_name, sheet_name):
    global _EIGHTEEN_CENTURIES_IN_MINUTES
    if pd.isna(cell):
        timestamp = pd.NaT
    else:
        cell_value = (
            str(cell)
            .strip(ascii_letters + punctuation + whitespace)
            .replace(
                ",",
                ".",
            )
        )
        if "." not in cell_value:
            try:
                cell_value = int(cell_value)
            except ValueError:
                pass
            else:
                if (
                    column_name == ""
                    and cell_value < _EIGHTEEN_CENTURIES_IN_MINUTES
                ):
                    # This is a useless incremental timestamp,
                    # let us drop it
                    return pd.NaT
                else:
                    debug(
                        f"Could not convert integer {cell_value} from column "
                        f"'{column_name}' of sheet '{sheet_name}' to timestamp"
                    )
                    return cell_value  # integer
        elif "." in cell_value and column_name == "":
            try:
                # Please take your seat before reading further ...
                #
                # From trial and error I discovered that the floating
                # point number in column '' is the number of
                # minutes passed since three days before 1 January 1AD
                cell_value = (
                    datetime.strptime("01/01/0001", "%d/%m/%Y")
                    + timedelta(minutes=float(cell_value))
                    - timedelta(days=3)
                )

            except Exception:
                debug(
                    f"Could not convert float {cell_value} from column "
                    f"'{column_name}' of sheet '{sheet_name}' to timestamp"
                )
                return cell_value  # floating point
            else:
                debug(
                    f"Converted {repr(cell).ljust(17)}"
                    " (minutes after three days before 1 Jan 1 Anno Domini)"
                    f" into {str(cell_value).ljust(36)} "
                    f"(column: {column_name}; sheet: {sheet_name})"
                )
                return cell_value
        try:
            fmt = None
            if DAYFIRST_REGEXP.match(cell) is not None:
                if cell[2] == cell[5] and cell[2] in ("/", "-"):
                    sep = cell[2]
                    fmt = f"%d{sep}%m{sep}%Y"
                    if ":" in cell:
                        fmt += " %H:%M:%S"
            timestamp = pd.to_datetime(
                cell_value,
                dayfirst=DAYFIRST_REGEXP.match(cell) is not None,
                format=fmt,
            )
        except Exception as e:
            debug(
                f"Could not convert {repr(cell)} to timestamp in column "
                f"'{column_name}' of sheet '{sheet_name}': {str(e)}"
            )
            return cell  # original value
        else:
            if (
                DAYFIRST_REGEXP.match(cell) is not None
                and str(DAYFIRST_REGEXP.match(cell).group("day"))
                != f"{timestamp.day:0>2d}"
            ):
                debug(
                    f"Poorly converted {repr(cell)} to timestamp in column "
                    f"'{column_name}' of sheet '{sheet_name}': "
                    "mismatch between the 'day' group of the regexp "
                    "and the actual 'day' parsed by pandas.to_datetime()"
                )
                return cell  # original value
    debug(
        f"Converted {repr(cell).ljust(71)} into {repr(timestamp).ljust(36)} "
        f"(column: {column_name}; sheet: {sheet_name})"
    )
    return timestamp


def _file_worker_body(input_queue, output_queue, error_queue):
    filename, sheet_rename_mapping = input_queue.get()
    debug(f"File worker in charge of '{abspath(filename)}' started")

    sheets_dict = dict()
    try:
        with pd.ExcelFile(filename) as xlsx:
            debug(f"Correctly parsed excel file '{abspath(filename)}'")
            for old_sheet_name, new_sheet_name in sheet_rename_mapping.items():
                if old_sheet_name is None:
                    sheets_dict = pd.read_excel(xlsx, sheet_name=None)
                    debug(
                        f"All sheets in '{abspath(filename)}' correctly read"
                    )  # no more sheets to read
                    break

                if isinstance(old_sheet_name, int) and old_sheet_name in range(
                    len(xlsx.sheet_names)
                ):
                    sheet_index = old_sheet_name
                    old_sheet_name = xlsx.sheet_names[old_sheet_name]
                    debug(
                        f"Substituted sheet identifier {sheet_index} of "
                        f"'{abspath(filename)}' with label '{old_sheet_name}'"
                    )

                assert new_sheet_name not in sheets_dict, str(
                    f"Could not rename sheet {repr(old_sheet_name).ljust(11)}"
                    f" of file '{filename}' to '{new_sheet_name}'"
                    " because a sheet with the latter name already"
                    " exists and it should not be overwritten."
                )
                sheets_dict[new_sheet_name] = pd.read_excel(
                    xlsx, sheet_name=old_sheet_name
                )
                debug(
                    f"Sheet '{old_sheet_name}' of file '{abspath(filename)}'"
                    " correctly read"
                )
    except Exception as e:
        error_queue.put(FileWorkerError(filename, e))
        output_queue.put(FileWorkerOutput(filename, None))
    else:
        error_queue.put(None)
        output_queue.put(FileWorkerOutput(filename, sheets_dict))
    finally:
        debug(f"File worker in charge of '{abspath(filename)}' ended")


def _file_writer_body(filename, sheet_dict, bkp_ext=".bkp"):
    debug(f"Excel writer in charge of '{abspath(filename)}' started")
    sheets_to_rename = [
        sheet_name
        for sheet_name in sorted(sheet_dict.keys())
        if sheet_name.startswith(SORTING_PREFIX)
    ]

    if len(sheets_to_rename) < 1:
        debug(f"No sheet to rename in '{abspath(filename)}'")
        return
    debug(
        f"Sheets to rename in '{abspath(filename)}' are: "
        + repr(sheets_to_rename).strip("[]")
        + "."
    )

    bkp_file = abspath(filename).split(bkp_ext)[0] + bkp_ext
    if not isfile(bkp_file):
        move(abspath(filename), bkp_file)
        info(
            f"A backup of the original excel '{basename(filename)}' "
            f"has been saved here: '{basename(bkp_file)}'"
        )
    else:
        debug(
            f"A previous backup of '{abspath(filename)}' already exists; "
            "let us do not overwrite it!"
        )

    with pd.ExcelWriter(abspath(filename)) as xlsx:
        for sheet_name, df in sorted(
            sheet_dict.items(), key=lambda tup: tup[0].lower()
        ):
            df.to_excel(
                xlsx,
                sheet_name=sheet_name.lstrip(SORTING_PREFIX),
                columns=sorted(df.columns, key=str.lower),
                index=False,
            )
            debug(
                f"Sheet '{sheet_name}' of file '{abspath(filename)}'"
                " correctly written"
            )
    info(
        f"{len(sheets_to_rename)} sheets in '{basename(filename)}'"
        " have been successfully renamed"
    )


def _get_new_sheet_name(filename, old_sheet_name, df):
    global LONGEST_FILENAME_LENGTH
    if df.empty:
        debug(
            f"Could not rename empty sheet '{old_sheet_name}' of "
            f"{repr(basename(filename))} because no column could ever "
            f"match regexp '{TITLE_COLUMN_REGEXP.pattern}')"
        )
        raise Warning(
            f"Could not rename empty sheet "
            f"{repr(old_sheet_name).ljust(11)} of "
            f"{repr(basename(filename)).ljust(LONGEST_FILENAME_LENGTH+2)}"
        )
    title_col = next(
        (c for c in df.columns if TITLE_COLUMN_REGEXP.match(c) is not None),
        None,
    )
    if title_col is None:
        debug(
            f"Could not rename sheet '{old_sheet_name}' of "
            f"{repr(basename(filename))} because no column matched "
            f"regexp '{TITLE_COLUMN_REGEXP.pattern}')"
        )
        raise Warning(
            f"Could not rename {' ' * len('empty')} sheet "
            f"{repr(old_sheet_name).ljust(11)} of "
            f"{repr(basename(filename)).ljust(LONGEST_FILENAME_LENGTH+2)}"
        )
    guessed_sheet_title = Counter(df[title_col].to_list()).most_common(1)
    if not guessed_sheet_title:
        raise Warning(
            f"Could not rename {' ' * len('empty')} sheet "
            f"{repr(old_sheet_name).ljust(11)} of "
            f"{repr(basename(filename)).ljust(LONGEST_FILENAME_LENGTH+2)} "
            f"(empty title column '{title_col}')"
        )
    guessed_sheet_title = guessed_sheet_title[0][0]
    for new_sheet_name, rule in SHEET_RENAMING_RULES.items():
        if rule.match(guessed_sheet_title) is None:
            continue
        if new_sheet_name.replace("__", "-") == old_sheet_name:
            debug(
                "Skipping already renamed sheet in "
                f"'{basename(filename)}' ({old_sheet_name})"
            )
            return old_sheet_name
        return f"{SORTING_PREFIX}{new_sheet_name.replace('__', '-')}"
    raise Warning(
        "No renaming rule was found for sheet "
        f"{repr(old_sheet_name).ljust(11)} (titled '{guessed_sheet_title}') "
        f"of '{basename(filename)}'"
    )


def _join_all(process_to_wait, label="spawned process"):
    debug(f"Waiting for all {label} to join ({len(process_to_wait)})")
    for p in process_to_wait:
        p.join()
    debug(f"All ({len(process_to_wait)}) {label} joined")


def _sheet_worker_body(input_queue, output_queue, error_queue):
    filename, old_sheet_name, df = input_queue.get()
    debug(
        f"Sheet worker in charge of '{old_sheet_name}' from "
        f"'{abspath(filename)}' started"
    )
    try:
        new_sheet_name = _get_new_sheet_name(filename, old_sheet_name, df)
        debug(
            f"Sheet '{old_sheet_name}' from '{abspath(filename)}' "
            f"will be renamed '{new_sheet_name}'"
        )
    except Exception as e:
        error_queue.put(SheetWorkerError(filename, old_sheet_name, e))
        output_queue.put(SheetWorkerOutput(filename, old_sheet_name, df))
    else:
        error_queue.put(None)
        output_queue.put(SheetWorkerOutput(filename, new_sheet_name, df))
    finally:
        debug(
            f"Sheet worker in charge of '{old_sheet_name}' from "
            f"'{abspath(filename)}' ended"
        )


def _show_sorted_warnings(sheet_worker_errors):
    warnings, exceptions = list(), dict()
    for swe in sheet_worker_errors:
        if swe is not None:
            filename, sheet_name, exception = swe
            filename = basename(filename)
            if isinstance(exception, Warning):
                warnings.append(str(exception))
            else:
                exceptions[str(exception)] = exceptions.get(
                    str(exception), dict(filename=list())
                )
                exceptions[str(exception)][filename].append(sheet_name)
    warning("\n    " + str("\n    ".join(sorted(warnings))))
    for exception, dictionary in sorted(exceptions.items()):
        for filename, sheet_name in dictionary.items():
            warning(
                f"'{exception}' occurred while renaming "
                f"'{sheet_name}' in '{filename}'"
            )


def _spwan_parallel_excel_writers(all_sheets_dict):
    parallel_writers = list()
    for filename, sheet_dict in all_sheets_dict.items():
        parallel_writer = Process(
            target=_file_writer_body,
            args=(filename, sheet_dict),
        )
        parallel_writers.append(parallel_writer)
        parallel_writer.start()
    return parallel_writers


def _timestamp_worker_body(input_queue, output_queue, error_queue):
    sheet_name, column_name, old_series = input_queue.get()
    debug(
        f"Timestamp worker in charge of column '{column_name}' "
        f"in '{sheet_name}' started"
    )
    try:
        assert isinstance(old_series, pd.Series), str(
            f"Expected a Series; not a '{type(old_series)}'"
        )
        new_series = old_series.apply(
            _convert_single_cell_timestamp, args=(column_name, sheet_name)
        ).convert_dtypes(
            infer_objects=True,  # applied function usually returns Timestamps
            convert_floating=True,  # applied function can return floats
            convert_boolean=False,
            convert_integer=False,
            convert_string=False,
        )
    except Exception as e:
        error_queue.put(TimestampWorkerError(sheet_name, column_name, e))
        output_queue.put(
            TimestampWorkerInputOutput(sheet_name, column_name, old_series)
        )
    else:
        debug(
            f"Timestamp worker in charge of column '{column_name}' "
            f"in '{sheet_name}' returned a series of type '{new_series.dtype}'"
        )
        error_queue.put(None)
        output_queue.put(
            TimestampWorkerInputOutput(sheet_name, column_name, new_series)
        )
    finally:
        debug(
            f"Timestamp worker in charge of column '{column_name}' "
            f"in '{sheet_name}' ended"
        )


@black_magic
def cast_date_time_columns_to_timestamps(df_dict, **kwargs):
    timestamp_workers = list()
    tw_input_queue, tw_output_queue, tw_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )

    for sheet_name, df in df_dict.items():
        for column_name in set(df.columns):
            if matches_date_time_rule(column_name):
                tw_input_queue.put(
                    TimestampWorkerInputOutput(
                        sheet_name, column_name, df.loc[:, column_name]
                    )
                )
                timestamp_worker = Process(
                    target=_timestamp_worker_body,
                    args=(tw_input_queue, tw_output_queue, tw_error_queue),
                )
                timestamp_workers.append(timestamp_worker)
                timestamp_worker.start()

    new_df_columns = dict()
    for _ in timestamp_workers:
        timestamp_worker_error = tw_error_queue.get()
        if timestamp_worker_error is not None:
            sheet_name, column_name, exception = timestamp_worker_error
            warning(
                "While converting dates/times of "
                f"column '{column_name}' in sheet '{sheet_name}' "
                f"got the following exception: {str(exception)}"
            )
    for _ in timestamp_workers:
        sheet_name, column_name, new_series = tw_output_queue.get()
        new_df_columns[sheet_name] = new_df_columns.get(sheet_name, dict())
        if column_name in NORMALIZED_TIMESTAMP_COLUMNS:
            new_df_columns[sheet_name][column_name] = new_series.astype(
                "datetime64[ns]"
            ).dt.normalize()
        else:
            new_df_columns[sheet_name][column_name] = new_series
        debug(
            f"Added new Timestamp series '{column_name}' to "
            f"the other ones for '{sheet_name}'"
        )
    _join_all(timestamp_workers, "timestamp workers")
    sheet_name_pad = 2 + max((len(sn) for sn in new_df_columns.keys()))
    for sheet_name, new_columns in sorted(
        new_df_columns.items(), key=lambda tup: tup[0].lower()
    ):
        df_dict[sheet_name] = df_dict[sheet_name].assign(**new_columns)
        info(
            f"Successfully converted {len(new_columns): >2d} columns of "
            f"sheet {repr(sheet_name).ljust(sheet_name_pad)} to pd.Timestamp"
        )
    return df_dict


@black_magic
def rename_excel_sheets(*args, **kwargs):
    """
    *args = list(filename_1, filename_2, ..)          # all sheets will be read
    **kwargs = dict(new_sheet_name=filename, ..)  # only 1st sheet will be read
    """
    tasks = dict()  # excel_file: [ dict(old_sheet_name=new_sheet_name), ... ]
    for excel_file in args:
        assert isfile(excel_file), f"Could not find '{abspath(excel_file)}'"
        tasks[excel_file] = {
            None: None,  # all sheets available in the excel file
        }
    for new_sheet_name, excel_file in kwargs.items():
        assert isfile(excel_file), f"Could not find '{abspath(excel_file)}'"
        tasks[excel_file] = {0: new_sheet_name}  # just 1st sheet

    file_workers = list()
    fw_input_queue, fw_output_queue, fw_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    global LONGEST_FILENAME_LENGTH
    LONGEST_FILENAME_LENGTH = -1
    for excel_file, sheet_renaming_mapping in tasks.items():
        LONGEST_FILENAME_LENGTH = max(
            LONGEST_FILENAME_LENGTH, len(basename(excel_file))
        )
        fw_input_queue.put(FileWorkerInput(excel_file, sheet_renaming_mapping))
        file_worker = Process(
            target=_file_worker_body,
            args=(fw_input_queue, fw_output_queue, fw_error_queue),
        )
        file_workers.append(file_worker)
        file_worker.start()  # spawn a process for each excel to read

    sheet_workers = list()
    sw_input_queue, sw_output_queue, sw_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    for _ in file_workers:
        filename, sheets_dict = fw_output_queue.get()
        if sheets_dict is not None:
            for old_sheet_name, current_df in sheets_dict.items():
                sw_input_queue.put(
                    SheetWorkerInput(filename, old_sheet_name, current_df)
                )
                sheet_worker = Process(
                    target=_sheet_worker_body,
                    args=(sw_input_queue, sw_output_queue, sw_error_queue),
                )
                sheet_workers.append(sheet_worker)
                sheet_worker.start()  # spawn a process for each sheet read
    for _ in file_workers:
        file_worker_error = fw_error_queue.get()
        if file_worker_error is not None:
            filename, exception = file_worker_error
            warning(
                f"Excel file worker in charge of '{filename}' failed "
                f"with the following exception: {str(exception)}."
            )
    _join_all(file_workers, "file workers")

    all_sheets = dict()
    for _ in sheet_workers:
        filename, new_sheet_name, new_df = sw_output_queue.get()
        hashed_filename = (
            blake2b(str.encode(filename), digest_size=16).hexdigest().upper()
        )  # hex-string
        new_df = new_df.assign(
            provenance=pd.Series(
                [hashed_filename for _ in range(new_df.shape[0])],
            )
        )
        all_sheets[filename] = all_sheets.get(filename, dict())
        if new_sheet_name not in all_sheets[filename]:
            all_sheets[filename][new_sheet_name] = new_df
            debug(
                f"Added '{new_sheet_name}' dataframe to final df_dict of "
                f"{abspath(filename)}"
            )
        else:
            debug(
                f"Dataframe '{new_sheet_name}' already in final df_dict of "
                f"{abspath(filename)}"
            )
            old_df = all_sheets[filename][new_sheet_name]
            all_sheets[filename][new_sheet_name] = pd.concat(
                [old_df, new_df], join="outer", sort=True
            )
            debug(
                "Previous dataframe and current daframe correctly "
                f"concatenated into '{new_sheet_name}' in final df_dict of"
                f" {abspath(filename)}"
            )
    _show_sorted_warnings([sw_error_queue.get() for _ in sheet_workers])
    _join_all(sheet_workers, "sheet workers")

    parallel_writers = _spwan_parallel_excel_writers(all_sheets)

    # the following concat is quite heavy; let us do it in parallel
    # with the excel writers
    ret = _concat_same_name_sheets(all_sheets)  # do not move after join !!!

    _join_all(parallel_writers, "excel writers")

    return ret



if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.parallel",
            "YAstarMM.parallel",
            "parallel",
        ),
        "cast_date_time_columns_to_timestamps" in globals(),
        "rename_excel_sheets" in globals(),
    )
), "Please update 'Usage' section of module docstring"
