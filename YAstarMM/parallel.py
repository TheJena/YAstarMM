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
    matches_date_time_rule,
    new_key_col_value,
)
from .constants import (
    DAYFIRST_REGEXP,
    EIGHTEEN_CENTURIES_IN_MINUTES,
    FileWorkerError,
    FileWorkerInput,
    FileWorkerOutput,
    GroupWorkerError,
    GroupWorkerInput,
    InputOutputErrorQueues,
    NORMALIZED_TIMESTAMP_COLUMNS,
    NewKeyError,
    NewKeyOutput,
    RowFillerError,
    RowFillerOutput,
    SHEET_RENAMING_RULES,
    SORTING_PREFIX,
    SheetMergerInput,
    SheetMergerOutput,
    SheetWorkerError,
    SheetWorkerInput,
    SheetWorkerOutput,
    TITLE_COLUMN_REGEXP,
    TimestampWorkerError,
    TimestampWorkerInputOutput,
)
from .utility import (
    black_magic,
    enough_ram,
    row_selector,
    swap_month_and_day,
)
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from gc import collect as run_garbage_collector
from hashlib import blake2b
from logging import debug, info, warning
from multiprocessing import cpu_count, Process, Queue
from numexpr import set_num_threads, set_vml_num_threads
from os.path import abspath, basename, isfile
from queue import Empty as EmptyQueue
from psutil import virtual_memory
from shutil import move
from string import ascii_letters, punctuation, whitespace
from sys import version_info
from time import time
import pandas as pd


def _concat_same_name_sheets(all_sheets_dict):
    ret = dict()
    for filename, sheets_dict in all_sheets_dict.items():
        for new_sheet_name, new_df in sheets_dict.items():
            new_sheet_name = new_sheet_name.lstrip(SORTING_PREFIX).replace(
                "_",
                "-",
            )
            if new_df.empty or sorted(new_df.columns) == ["Name", "Value"]:
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
                    and cell_value < EIGHTEEN_CENTURIES_IN_MINUTES
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


def _fill_rows_matching_truth_body(
    input_queue,
    output_queue,
    error_queue,
    read_only_truth,
    empty_columns,
    new_key,
    indirect_obj,
    added_date_cols,
    use_all_available_dates,
):
    debug("Row filler started")

    stats = Counter()
    while True:
        row = input_queue.get()
        if row is None:
            debug("Row filler got a termination signal")
            break

        try:
            selector = row_selector(index=read_only_truth.index)
            assert isinstance(row, dict)
            assert "Index" in row
            for col, cell_value in row.items():
                if (
                    pd.notna(cell_value)
                    and col not in ("date", "Index")
                    and col not in added_date_cols
                ):
                    selector = (selector) & (
                        read_only_truth[col] == cell_value
                    )
            for col in empty_columns:
                selector = (selector) & (read_only_truth[col].isna())
            possible_ids = set(
                read_only_truth.loc[selector, new_key].dropna().tolist(),
            )
            if len(possible_ids) == 0:
                stats.update([f"did not match any existing '{indirect_obj}'"])
                continue
            elif len(possible_ids) == 1:
                stats.update(
                    [f"successfully matched a single '{indirect_obj}'"]
                )
                chosen_id = possible_ids.pop()
            else:
                date_columns = ["date"]
                if use_all_available_dates:
                    date_columns += added_date_cols
                chosen_id = None
                for date_col in date_columns:
                    date = row.get(date_col, None)
                    if pd.isna(date):
                        continue
                    normalized_date = pd.to_datetime(
                        pd.to_datetime(date).date()
                    )
                    sub_df = read_only_truth.loc[
                        read_only_truth.loc[:, new_key].isin(possible_ids),
                        ["admission_date", "discharge_date", new_key],
                    ]
                    possible_ids = set()
                    possible_ids_considering_date_typos = set()
                    for sub_row in sub_df.itertuples():
                        start = getattr(sub_row, "admission_date")
                        if pd.isna(start):
                            continue
                        end = getattr(sub_row, "discharge_date")
                        if pd.isna(end):
                            end = pd.to_datetime(datetime.now().date())
                        if normalized_date in pd.date_range(
                            start, end, freq="D", normalize=True
                        ):
                            possible_ids.add(getattr(sub_row, new_key))
                        try:
                            mistyped_date = swap_month_and_day(normalized_date)
                        except ValueError:
                            pass
                        else:
                            if mistyped_date in pd.date_range(
                                start, end, freq="D", normalize=True
                            ):
                                possible_ids_considering_date_typos.add(
                                    getattr(sub_row, new_key)
                                )
                    if len(possible_ids) == 1:
                        stats.update(
                            [
                                "successfully matched several "
                                f"'{indirect_obj}' and\n"
                                "only one also\nmatched the date range"
                            ]
                        )
                        chosen_id = possible_ids.pop()
                        break
                    elif len(possible_ids_considering_date_typos) == 1:
                        stats.update(
                            [
                                "successfully matched several "
                                f"'{indirect_obj}' and\n"
                                "only one (having a typo in the 'date')"
                                " also\nmatched the date range"
                            ]
                        )
                        chosen_id = possible_ids_considering_date_typos.pop()
                        break
                if chosen_id is None:
                    stats.update(
                        [
                            f"matched several '{indirect_obj}' but "
                            "no date information\nhelped "
                            "disambiguating among them"
                        ]
                    )
                    continue
            output_queue.put(RowFillerOutput(row["Index"], chosen_id, None))
        except Exception as e:
            error_queue.put(RowFillerError(row, str(e)))
    output_queue.put(RowFillerOutput(None, None, stats))
    output_queue.put(None)
    error_queue.put(None)
    debug("Row filler acked twice to termination signal")
    debug("Row filler is ready to rest in peace")


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


def _group_worker_body(tail, input_queue, output_queue, error_queue):
    worker_results = list()
    while True:
        group_worker_input = input_queue.get()
        if group_worker_input is None:
            debug("Group worker got a termination signal")
            break

        group_name, old_df = group_worker_input
        debug(f"Group worker started  filling Nan of {repr(group_name)}")
        try:
            new_df = old_df.fillna(method="bfill").fillna(method="ffill")
        except Exception as e:
            error_queue.put(GroupWorkerError(group_name, e))
            worker_results.append(old_df.tail(tail))
        else:
            worker_results.append(new_df.tail(tail))
        finally:
            debug(f"Group worker finished filling Nan of {repr(group_name)}")

    output_queue.put(pd.concat(worker_results, join="outer", sort=True))
    error_queue.put(None)
    output_queue.put(None)
    debug("Group worker acked twice to termination signal")
    debug("Group worker is ready to rest in peace")


def _join_all(process_to_wait, label="spawned process"):
    debug(f"Waiting for all {label} to join ({len(process_to_wait)})")
    for p in process_to_wait:
        p.join()
    debug(f"All ({len(process_to_wait)}) {label} joined")


def _new_key_worker_body(
    input_queue, output_queue, error_queue, aux_df, selector, ids_columns
):
    debug("New key worker started")

    while True:
        ids_values = input_queue.get()
        if ids_values is None:
            debug("New key worker got a termination signal")
            break
        try:
            if len(ids_columns) == 1:
                ids_values = [ids_values]  # make single pd.Timestamp iterable
            selected_rows = row_selector(index=selector.index)
            for col, val in zip(ids_columns, ids_values):
                selected_rows = (selected_rows) & (aux_df[col] == val)
            new_key_value = new_key_col_value(
                **{col: val for col, val in zip(ids_columns, ids_values)}
            )
            output_queue.put(NewKeyOutput(selected_rows, new_key_value))
        except Exception as e:
            error_queue.put(NewKeyError(ids_values, str(e)))
    output_queue.put(None)
    error_queue.put(None)
    debug("New key worker acked twice to termination signal")
    debug("New key worker is ready to rest in peace")


def _sheet_merger_worker_body(input_queue, output_queue, error_queue, key_col):
    debug("Sheet merger worker started")

    while True:
        try:
            workload = input_queue.get(block=True, timeout=0.5)  # seconds
        except EmptyQueue:
            continue  # keep polling
        else:
            if workload is None:
                debug("Sheet merger worker got a termination signal")
                break  # received termination signal

        left_name, right_name, left_df, right_df, join_dict = workload
        worker_name = f"Merger of: [{left_name}] + [{right_name}]"
        debug(f"{worker_name} started")

        type_cast_mapping = {
            column: "object"
            for column in set(left_df.columns) & set(right_df.columns)
            if str(left_df.dtypes[column]) != str(right_df.dtypes[column])
        }
        if type_cast_mapping:
            debug(
                f"{worker_name} (type_cast_mapping: {repr(type_cast_mapping)}"
            )  # make black auto-formatting prettier
        type_cast_mapping.update(join_dict)
        debug(f"{worker_name} (join_dict: {repr(join_dict)})")
        try:
            left_df = left_df.astype(type_cast_mapping)
            right_df = right_df.astype(type_cast_mapping)
            merged_df = pd.merge(
                left=left_df,
                right=right_df,
                how="outer",
                on=list(
                    set(join_dict.keys())
                    | (set(left_df.columns) & set(right_df.columns))
                ),
                copy=False,
            )
            merged_df = merged_df.drop_duplicates(ignore_index=True)
            if all(
                (
                    key_col in merged_df.columns,
                    "date" in merged_df.columns,
                )
            ):
                debug(
                    f"{worker_name} started trimming multiple records"
                    f" about the same ({key_col}, date)"
                )
                t0 = time()
                merged_df = fill_nan_backward_forward(
                    worker_name,
                    merged_df,
                    sort_criteria=["date"],
                    group_criteria=[key_col, "date"],
                    dropna=False,
                    tail=1,
                )
                debug(
                    f"Trimming of multiple records {' ' * 61}"
                    f"took {time() - t0:10.6f} seconds.\t({worker_name})"
                )
        except Exception as e:
            debug(f"{worker_name} failed")
            error_queue.put(e)
            output_queue.put(SheetMergerOutput(left_name, left_df))
            output_queue.put(SheetMergerOutput(right_name, right_df))
        else:
            error_queue.put(None)
            output_queue.put(
                SheetMergerOutput(f"({left_name} + {right_name})", merged_df)
            )
            debug(f"{worker_name} succeeded")
        finally:
            debug(f"{worker_name} ended")
            run_garbage_collector()

    # ack to termination signal
    debug("Sheet merger worker is ready to rest in peace")
    output_queue.put(None)
    error_queue.put(None)


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


def _update_all_sheets(
    input_queue, output_queue, error_queue, lookup_table, receiver, giver
):
    if not isinstance(receiver, str) or not isinstance(giver, str):
        warning(
            "updating with giver/receiver tuples matches records "
            "only against the first giver/receiver"
        )
        assert len(receiver) == len(giver), str(
            "when using giver/receiver tuples, their length must be the same"
        )

    if isinstance(receiver, str):
        receiver_list = [receiver]
    else:
        receiver_list = list(receiver)
    if isinstance(giver, str):
        giver_list = [giver]
    else:
        giver_list = list(giver)

    for giver in giver_list:
        assert giver in lookup_table.columns, str(
            f"giver '{giver}' column must be in lookup_table.columns"
        )

    swi = input_queue.get()
    sheet_name, df = swi.old_sheet_name, swi.df.loc[:, :]
    debug(f"Sheet updater in charge of '{sheet_name}' started")
    try:
        for giver_value, sub_lookup_table in lookup_table.loc[
            lookup_table[giver_list[0]].notna(),
            list(
                set(
                    giver_list
                    + list(
                        set(lookup_table.columns).intersection(set(df.columns))
                    ),
                )
            ),
        ].groupby(giver_list[0]):
            selector = row_selector(index=df.index)
            for sub_col, sub_series in sub_lookup_table.items():
                if sub_col in giver_list:
                    continue
                selector = (selector) & df[sub_col].isin(sub_series.unique())
                if not selector.any():
                    # selector is not selecting any row, go on
                    break
            else:
                df.loc[selector, receiver_list[0]] = giver_value
                for receiver_col, giver_col in zip(
                    receiver_list[1:], giver_list[1:]
                ):
                    receiver_values = (
                        lookup_table.loc[
                            lookup_table[giver_list[0]] == giver_value,
                            receiver_col,
                        ]
                        .dropna()
                        .tolist()
                    )
                    if len(set(receiver_values)) == 1:
                        df.loc[
                            (selector) & (df[receiver_list[0]] == giver_value),
                            receiver_col,
                        ] = receiver_values.pop()
                    else:
                        warning(
                            "could not choose between so many receiver "
                            f"values ({repr(receiver_values)}) for "
                            f"'{receiver_col}' and "
                            f"'{receiver_list[0]}'=='{giver_value}'"
                        )
    except Exception as e:
        error_queue.put(SheetWorkerError("df_dict", sheet_name, str(e)))
        output_queue.put(SheetWorkerOutput("df_dict", sheet_name, swi.df))
    else:
        error_queue.put(None)
        output_queue.put(SheetWorkerOutput("df_dict", sheet_name, df))
    debug(
        f"Sheet updater in charge of '{sheet_name}' is ready to rest in peace"
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
def fill_nan_backward_forward(
    name, df, sort_criteria, group_criteria, tail, dropna=True, **kwargs
):
    if dropna:
        warning(
            f"calling 'fill_nan_backward_forward()' with "
            f"dropna=True may cause the loss of rows where any of "
            f"the [{repr(group_criteria).strip('[(,)]')},] is nan"
        )
    max_cpus = cpu_count() - 1

    gw_input_queue, gw_output_queue, gw_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    group_workers = [
        Process(
            target=_group_worker_body,
            args=(tail, gw_input_queue, gw_output_queue, gw_error_queue),
        )
        for _ in range(max_cpus)
    ]
    for gw in group_workers:
        gw.start()
    group_is_not_a_tuple = isinstance(group_criteria, str)
    for group_name, group_df in df.sort_values(sort_criteria).groupby(
        group_criteria, dropna=dropna
    ):
        if group_is_not_a_tuple:
            group_name = [group_name]
        if not dropna and any(pd.isna(value) for value in group_name):
            debug(
                "skipping fill-backward-forward+tail for group "
                f"{group_name}; since it contains nan "
            )
            gw_output_queue.put(group_df.loc[:, :])  # pass a copy of group
        else:
            gw_input_queue.put(GroupWorkerInput(group_name, group_df))
    for _ in group_workers:
        gw_input_queue.put(None)  # send termination signals

    all_groups = list()
    output_ack, error_ack = len(group_workers), len(group_workers)
    while output_ack > 0:
        group_df = gw_output_queue.get()
        if group_df is None:  # receive ack to termination signal
            output_ack -= 1
            continue
        all_groups.append(group_df)  # tail already done by workers
    while error_ack > 0:
        group_worker_error = gw_error_queue.get()
        if group_worker_error is None:  # receive ack to termination signal
            error_ack -= 1
            continue
        warning(
            "While filling backward/forward NaN values of sheet '{name}';"
            f" worker {repr(group_worker_error.group_name)} got the "
            f"follwing exception: {str(group_worker_error.exception)}"
        )
    _join_all(group_workers, "group workers")
    return pd.concat(all_groups, join="outer", sort=True).drop_duplicates(
        ignore_index=True
    )


# DO NOT CACHE THIS FUNCTION
def fill_rows_matching_truth(
    aux_df,
    read_only_truth,
    empty_columns,
    new_key,
    indirect_obj,
    added_date_cols=list(),
    stats=Counter(),
    use_all_available_dates=False,
):
    assert new_key in aux_df.columns, str(
        f"new_key ({new_key}) must be a column in aux_df"
    )
    assert all(
        (
            "admission_date" in read_only_truth.columns,
            "discharge_date" in read_only_truth.columns,
        )
    )

    rf_input_queue, rf_output_queue, rf_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    row_filler_workers = [
        Process(
            target=_fill_rows_matching_truth_body,
            args=(
                rf_input_queue,
                rf_output_queue,
                rf_error_queue,
                read_only_truth,
                empty_columns,
                new_key,
                indirect_obj,
                added_date_cols,
                use_all_available_dates,
            ),
        )
        for _ in range(cpu_count() - 1)
    ]

    for rf in row_filler_workers:
        rf.start()

    for row in aux_df.loc[aux_df[new_key].isna(), :].itertuples():
        # putting the row as it raises PicklingError
        rf_input_queue.put(
            {k: v for k, v in row._asdict().items() if pd.notna(v)}
        )
    for rf in row_filler_workers:
        rf_input_queue.put(None)  # sent termination signals

    output_ack, error_ack = len(row_filler_workers), len(row_filler_workers)
    while output_ack > 0:
        rfo = rf_output_queue.get()
        if rfo is None:  # receive ack to termination signal
            output_ack -= 1
            continue
        if rfo.stats is not None:
            stats.update(rfo.stats)
        else:
            aux_df.loc[rfo.row_index, new_key] = rfo.chosen_id
    while error_ack > 0:
        rfe = rf_error_queue.get()
        if rfe is None:  # receive ack to termination signal
            error_ack -= 1
            continue
        warning(
            "While filling rows matching truth; "
            f" worker in charge of row: {repr(rfe.row)} got the "
            f"follwing exception: {str(rfe.exception)}"
        )
    _join_all(row_filler_workers, "row filler workers")
    return (aux_df, stats)


# DO NOT CACHE THIS FUNCTION
def find_valid_keys(aux_df, selector, ids_columns, new_key, stats):
    nk_input_queue, nk_output_queue, nk_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    new_key_workers = [
        Process(
            target=_new_key_worker_body,
            args=(
                nk_input_queue,
                nk_output_queue,
                nk_error_queue,
                aux_df,
                selector,
                ids_columns,
            ),
        )
        for _ in range(cpu_count() - 1)
    ]
    for nk in new_key_workers:
        nk.start()

    for ids_values, _ in aux_df.loc[selector, ids_columns].groupby(
        ids_columns
    ):
        nk_input_queue.put(ids_values)
    for nk in new_key_workers:
        nk_input_queue.put(None)  # sent termination signals

    output_ack, error_ack = len(new_key_workers), len(new_key_workers)
    while output_ack > 0:
        nko = nk_output_queue.get()
        if nko is None:  # receive ack to termination signal
            output_ack -= 1
            continue
        aux_df.loc[nko.selected_rows, new_key] = nko.new_value
        stats.update(
            {
                str(
                    f"successfully identified by a '{new_key}' value",
                ): len(list(nko.selected_rows.index.array))
            }
        )
    while error_ack > 0:
        nke = nk_error_queue.get()
        if nke is None:  # receive ack to termination signal
            error_ack -= 1
            continue
        warning(
            "While finding {repr(tuple(ids_columns))} values, valid as key; "
            f"worker in charge of ids_values: {repr(nke.ids_values)} got the "
            f"follwing exception: {str(nke.exception)}"
        )
    _join_all(new_key_workers, "new_key workers")
    return (aux_df, stats)


@black_magic
def merge_sheets(
    df_dict, key_col, key_col_dtype, enable_automagic=True, **kwargs
):
    """The default amount of ram is an heuristically determined value

    Elapsed (wall clock) time (h:mm:ss or m:ss):     0:07:42
    Maximum resident set size (kbytes):              6828496
    """
    max_cpus = 1
    needed_gb_ram = 6828496 / 1024 ** 2
    max_wall_clock_time = timedelta(hours=0, minutes=7, seconds=42)
    if enable_automagic and enough_ram(needed_gb_ram):
        info("Automagic discovered enough memory to try multiprocessing")
        max_cpus = cpu_count()
        set_vml_num_threads(1)  # we are already using multiprocessing
        set_num_threads(1)  # we are already using multiprocessing
    else:
        debug("Not enough memory to use multiprocessing")
        set_vml_num_threads(max(1, cpu_count() // 2))  # use numexpr
        set_num_threads(max(1, cpu_count() // 2))  # multithreading instead

    info(
        "Congratulations! You just won some free time; "
        "please come back at:\n\t"
        f"{(datetime.now()+max_wall_clock_time).strftime('%A %d, %H:%M:%S')}\n"
        + str(
            "(since there is not enough RAM available,\n at least "
            + f"{needed_gb_ram - virtual_memory().available/1024**3:.3f}"
            + " GiB of swap will be used;\n"
            + " this could increase the required time)"
            if virtual_memory().available < needed_gb_ram * 1024 ** 3
            else ""
        )
    )

    sm_input_queue, sm_output_queue, sm_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    sheet_merger_workers = [
        Process(
            target=_sheet_merger_worker_body,
            args=(sm_input_queue, sm_output_queue, sm_error_queue, key_col),
        )
        for _ in range(min(round(len(df_dict) / 2), max_cpus))
    ]
    for sm_worker in sheet_merger_workers:
        sm_worker.start()

    preferred_join_order = (
        # http://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
        {"date": "datetime64[ns]", "": "Int64"},
        {"date": "datetime64[ns]", key_col: key_col_dtype},
        {"": "Int64"},
        {key_col: key_col_dtype},
        {"admission_code": "Int64"},
    )

    output_to_wait, last_debug_msg = 0, ""
    while len(df_dict) > 1 or output_to_wait > 0:
        for join_dict in preferred_join_order:
            sheet_list = [
                sheet_name
                for sheet_name, df in df_dict.items()
                if all(col in df.columns for col in join_dict)
            ]
            if sheet_list:
                msg = "sheet_list: " + str(", ".join(sheet_list))
                if msg != last_debug_msg:
                    last_debug_msg = msg
                    debug(msg)
            while len(sheet_list) >= 2:
                left_name, right_name = sheet_list.pop(0), sheet_list.pop(0)
                sm_input_queue.put(
                    SheetMergerInput(
                        left_name,
                        right_name,
                        df_dict.pop(left_name),
                        df_dict.pop(right_name),
                        join_dict,
                    )
                )
                output_to_wait += 1
            if output_to_wait > 0:
                try:
                    exception = sm_error_queue.get(
                        block=True,
                        timeout=0.5,  # seconds
                    )
                except EmptyQueue:
                    continue
                debug(f"output_to_wait: {output_to_wait}")
                output_to_wait -= 1
                if exception is None:
                    sheet_merger_output = sm_output_queue.get()
                    assert sheet_merger_output is not None
                    merged_name, merged_df = sheet_merger_output
                    debug(
                        f"added '{merged_name}' to df_dict "
                        f"{repr(merged_df.columns.tolist())}"
                    )
                    df_dict[merged_name] = merged_df
                else:
                    names, dataframes = dict(), dict()
                    for pos in ("left", "right"):
                        output = sm_output_queue.get()
                        assert output is not None
                        names[pos] = output.name
                        dataframes[pos] = output.df
                        df_dict[output.name] = output.df
                    debug(f"\n\n{'~' * 80}\n\n")
                    for pos in ("left", "right"):
                        sheet_name, df = names[pos], dataframes[pos]
                        debug(f"Sheet '{sheet_name}':")
                        for column in sorted(df.columns, key=str.lower):
                            debug(
                                " ".join(
                                    (
                                        str(df.dtypes[column]).ljust(16),
                                        repr(column),
                                    )
                                )
                            )
                    warning(
                        f"While merging '{names['left']}' and "
                        f"'{names['right']}' got the following "
                        f"exception: {str(exception)}"
                    )
            else:
                continue  # no couple of sheets found with join_dict criterion
            break  # at least a couple was found, retry all join_dict criteria
    assert output_to_wait == 0
    assert len(df_dict) == 1

    for sm_worker in sheet_merger_workers:
        sm_input_queue.put(None)  # send termination signal

    for i, _ in enumerate(sheet_merger_workers):
        assert (
            sm_output_queue.get() is None
        ), f"output-{i+1}-of-{len(sheet_merger_workers)}"
        assert (
            sm_error_queue.get() is None
        ), f"output-{i+1}-of-{len(sheet_merger_workers)}"

    for i, q in enumerate((sm_input_queue, sm_output_queue, sm_error_queue)):
        assert q.empty, f"Queue {i+1} is not empty"

    _join_all(sheet_merger_workers, "sheet merger workers")

    merged_order, merged_df = df_dict.popitem()

    for col in sorted(merged_df.columns, key=str.lower):
        debug(f"{str(merged_df.dtypes[col]).ljust(16)} {repr(col)}")

    info(f"Sheets were merged in the following order:\n    {merged_order}")
    return merged_df


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


# DO NOT CACHE THIS FUNCTION
def update_all_sheets(df_dict, lookup_table, receiver, giver):
    sw_input_queue, sw_output_queue, sw_error_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )

    sheet_workers = list()
    for sheet_name in sorted(df_dict.keys()):
        sw_input_queue.put(
            SheetWorkerInput("df_dict", sheet_name, df_dict.pop(sheet_name))
        )
        sheet_worker = Process(
            target=_update_all_sheets,
            args=(
                sw_input_queue,
                sw_output_queue,
                sw_error_queue,
                lookup_table,
                receiver,
                giver,
            ),
        )
        sheet_workers.append(sheet_worker)
        sheet_worker.start()  # spawn a process for each sheet in df_dict

    for _ in sheet_workers:
        swe = sw_error_queue.get()
        if swe is not None:
            warning(
                "while updating all sheets against lookup table "
                f"'{swe.sheet_name}' got {swe.exception}"
            )
        _, sheet_name, new_df = sw_output_queue.get()
        df_dict[sheet_name] = new_df
    _join_all(sheet_workers, "sheet updaters")
    return df_dict


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
