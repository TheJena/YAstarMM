#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020 Federico Motta <191685@studenti.unimore.it>
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
   Remove duplicated columns in DataFrame by merging their actual content.

   Usage:
            from  YAstarMM.deduplicate_columns  import   (
                columns_to_keep, fix_duplicated_columns,
            )

   ( or from within the YAstarMM package )

            from          .deduplicate_columns  import   (
                columns_to_keep, fix_duplicated_columns,
            )
"""

from .column_rules import rename_helper
from .constants import (  # without the dot notebook raises ModuleNotFoundError
    #
    # these are mostly tuples (or strings) of italian words
    #
    COLUMNS_AFTER_STATE_TRANSITION_COLUMNS,
    COLUMNS_BEFORE_STATE_TRANSITION_COLUMNS,
    COLUMNS_CONTAINING_EXAM_DATE,
    COLUMNS_NOT_SO_EASY_TO_MERGE,  # because of contraddicting data
    COLUMNS_TO_KEEP_DICTIONARY,  # dictionary
    COLUMNS_TO_MAXIMIZE,
    COLUMNS_TO_MAXIMIZE_DATE,
    COLUMNS_TO_MINIMIZE,
    COLUMNS_TO_MINIMIZE_DATE,
    COLUMN_CONTAINING_PERCENTAGES,
    COLUMN_HOSPITAL_UNIT,
    COLUMN_RECALCULATED_AFTERWARDS,
    COLUMN_WITH_EXECUTED_EXAM,
    COLUMN_WITH_REASON,
    DECEASED_VALUE,  # string
    EXECUTED_VALUE,  # string
    EXECUTING_IN_JUPYTER_KERNEL,  # bool
    FLOAT_EQUALITY_THRESHOLD,
    NASTY_SUFFIXES,
    ORDINARILY_HOME_DISCHARGED,  # string
    PREFIX_OF_COLUMNS_CONTAINING_EXAM_DATE,
    TRANSFERRED_VALUE,  # string
)
from .model import (  # without the dot notebook raises ModuleNotFoundError
    State,
    ordered_state_transition_columns,
)
from collections import Counter, OrderedDict
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count
from sys import stderr, version_info
import numpy as np
import pandas as pd


def all_equal_floats(iterable):
    """Return whether iterable contains all equal floats."""
    first_float = next((f for f in iterable if isinstance(f, float)))
    for f in iterable:
        if isinstance(f, str):
            f = f.replace(",", ".")  # italians sometimes use comma
        try:
            f = float(f)  # is f a string representing first_float ?
        except ValueError:
            return False
        else:
            if abs(first_float - f) > FLOAT_EQUALITY_THRESHOLD:
                return False
    return True


def all_timestamps(iterable):
    """Return if iterable contains only and all timestamps compatible data."""
    assert datetime.now().year <= 2029, str(
        "Please fix all tests using string '/202' to determine "
        "dayfirst/yearfirst boolean values to help Pandas parsing dates"
    )
    for t in iterable:
        try:
            pd.to_datetime(t, dayfirst="/202" in repr(t))
        except Exception:
            return False
    return True


def all_timestamps_and_equal(iterable):
    """Return whether iterable contains all equal timestamps."""
    if not any((isinstance(t, pd.Timestamp) for t in iterable)):
        return False
    return all(
        (
            all_timestamps(iterable),
            len(set(convert_all_to_timestamps(iterable))) == 1,
            True,  # make black auto-formatting prettier
        )
    )


def columns_to_keep(df):
    """Return df.columns except those added by merge operations gone wrong.

    And actually except also those not so easy to merge, but droppable
    because currently unused.
    """
    ret = COLUMNS_TO_KEEP_DICTIONARY
    ret["index"] = "int64"
    ret[rename_helper("ActualState")] = pd.CategoricalDtype(
        # make black auto-formatting prettier
        categories=State.names(),
        ordered=True,
    )
    ret[rename_helper("ActualState_val")] = pd.CategoricalDtype(
        categories=State.values(), ordered=True
    )
    for col in df.columns:
        if col.lower() not in (
            column_name.lower() for column_name in COLUMNS_NOT_SO_EASY_TO_MERGE
        ):
            for nasty_suffix in NASTY_SUFFIXES:
                if all(
                    (
                        col.endswith(nasty_suffix),
                        not col.endswith("O2"),  # oxygen
                        not col.endswith("02"),  # typo of oxygen
                    )
                ):
                    break  # drop column
            else:
                if col not in ret:
                    ret[col] = df[col].dtype  # no nasty suffix, keep column

    # Wait a moment, by doing only the above for loop we will drop all
    # those columns appearing only with a nasty suffix, thus without
    # at least a good column name; the following loop include them.
    for col in df.columns:
        for nasty_suffix in NASTY_SUFFIXES:
            if all(
                (
                    col.endswith(nasty_suffix),
                    not col.endswith("O2"),  # oxygen
                    not col.endswith("02"),  # typo of oxygen
                )
            ):
                good_col_name = nasty_suffix.join(col.split(nasty_suffix)[:-1])
                if all(
                    (
                        good_col_name not in ret,
                        good_col_name.lower()
                        not in [
                            column_name.lower()
                            for column_name in COLUMNS_NOT_SO_EASY_TO_MERGE
                        ],
                    )
                ):
                    ret[good_col_name] = None
    return ret


def convert_all_to_timestamps(iterable):
    assert all_timestamps(iterable)
    return (pd.to_datetime(t, dayfirst="/202" in repr(t)) for t in iterable)


def drop_duplicates_and_nan(list_of_lists):
    return list(
        set(
            (
                value
                for value_list in list_of_lists
                for value in value_list
                if all(
                    (
                        pd.notna(value),
                        #######################################################
                        isinstance(value, str)
                        and value
                        not in (  # involved column names
                        ),
                        #######################################################
                        isinstance(value, str)
                        and not value.startswith("")
                        and not value.startswith(""),
                    )
                )
            )
        )
    )


def find_values(good_column_name, old_row):
    """Return dictionary with good and nasty column names as keys.

    Dictionary values are lists containing found values for the
    respective key
    """
    ret = dict()
    for suffix in set(("",) + NASTY_SUFFIXES):
        nasty_column_name = f"{good_column_name}{suffix}"
        if nasty_column_name in old_row:
            if nasty_column_name not in ret:
                ret[nasty_column_name] = list()
            value = old_row[nasty_column_name]
            if hasattr(value, "tolist"):
                # value is actually a series because of multiple
                # columns named nasty_column_name in old_row :O
                ret[nasty_column_name].extend(value.tolist())
            else:
                ret[nasty_column_name].append(value)
    return ret


def print_updated_percentage(leading_string, iteration, tot_iterations):
    percentage = 100.0 * iteration / tot_iterations
    if EXECUTING_IN_JUPYTER_KERNEL:
        if (
            round(percentage) % 10 == 0
            and round(round(percentage) * tot_iterations / 100.0) == iteration
        ):
            # since jupyter notebooks:
            # 1) are not very resistant to aggressive print floods
            # 2) do not correctly show backspace character
            # show progress on new line every 10 %
            print(f"{leading_string}{percentage:6.2f} %")
    else:
        print("\b" * (8 + len(leading_string)), end="")
        print(
            f"{leading_string}{percentage:6.2f} %",
            end="",
            flush=True,
        )


def sort_dictionary_keys(ordered_dict):
    for col in ordered_state_transition_columns():
        assert col not in COLUMNS_BEFORE_STATE_TRANSITION_COLUMNS, str(
            f"Remove state transition column '{col}' from the above constant"
        )
        assert col not in COLUMNS_AFTER_STATE_TRANSITION_COLUMNS, str(
            f"Remove state transition column '{col}' from the above constant"
        )

    for col in reversed(
        COLUMNS_BEFORE_STATE_TRANSITION_COLUMNS
        + ordered_state_transition_columns()
        + COLUMNS_AFTER_STATE_TRANSITION_COLUMNS
    ):
        if col not in ordered_dict:
            print(f"WARNING: Column '{col}' not found in df_merged...")
            continue
        ordered_dict.move_to_end(col, last=False)  # move to beginning
    return


def worker_body(final_columns, input_queue, output_queue, warning_queue):
    """Deduplicates one row at a time, until there are no more rows."""
    _DEBUG = False
    final_columns = sorted(set(final_columns).difference({"index"}))
    while True:
        index, old_row = input_queue.get()
        if old_row is None:  # got termination signal
            output_queue.put(None)  # ack to termination signal
            break  # die
        new_row = dict()
        new_row["index"] = index
        for good_column_name in final_columns:
            found_values = find_values(good_column_name, old_row)
            possible_values = drop_duplicates_and_nan(found_values.values())
            if len(possible_values) == 0:
                new_row[good_column_name] = np.nan
                continue
            if len(possible_values) == 1:
                new_row[good_column_name] = possible_values.pop()
                continue
            if good_column_name.lower() in [
                column_name.lower() for column_name in COLUMNS_TO_MAXIMIZE
            ]:
                # By supposing incremental values in df[good_column_name]
                # we can pretty safely use the maximum one.
                #
                # Please also note that the decision to apply this
                # easy fix has been took considering that any
                # subsequent application taking this data in input
                # PROBABLY does not use this piece of information
                warning_queue.put(
                    f"column '{good_column_name}' has been "
                    f"determined\n{' ' * 8} by choosing the maximum "
                    f"value between {len(possible_values)} available."
                )
                new_row[good_column_name] = max(
                    tuple((float(f) for f in possible_values))
                )
                continue
            if good_column_name.lower() in [
                column_name.lower() for column_name in COLUMNS_TO_MINIMIZE
            ]:
                # By supposing incremental values in df[good_column_name]
                # we can pretty safely use the minimum one.
                #
                # Please also note that the decision to apply this
                # easy fix has been took considering that any
                # subsequent application taking this data in input
                # PROBABLY does not use this piece of information
                warning_queue.put(
                    f"column '{good_column_name}' has been "
                    f"determined\n{' ' * 8} by choosing the minimum "
                    f"value between {len(possible_values)} available."
                )
                new_row[good_column_name] = min(
                    tuple((float(f) for f in possible_values))
                )
                continue
            if any((isinstance(f, float) for f in possible_values)):
                # drop '%' simbols from possible values
                if good_column_name == COLUMN_CONTAINING_PERCENTAGES:
                    possible_values = [
                        float(v.replace("%", ""))
                        if all((isinstance(v, str), "%" in repr(v)))
                        else v
                        for v in possible_values
                    ]
                if all_equal_floats(possible_values):
                    new_row[good_column_name] = next(
                        (v for v in possible_values if isinstance(v, float))
                    )
                    continue
                else:
                    if (
                        good_column_name.lower()
                        == COLUMN_RECALCULATED_AFTERWARDS.lower()
                    ):
                        new_row[good_column_name] = np.nan
                        continue  # it will be computed again afterwards
                    warning_queue.put(
                        f"column '{good_column_name}' could not be "
                        f"determined\n{' ' * 8} because at least two"
                        " float values were find differing more than "
                        f"{FLOAT_EQUALITY_THRESHOLD:g};\n{' ' * 8}"
                        " None value was set by default."
                    )
                    new_row[good_column_name] = None
                    continue
            if all_timestamps_and_equal(possible_values):
                new_row[good_column_name] = next(
                    (v for v in possible_values if isinstance(v, pd.Timestamp))
                )
                continue
            if good_column_name.lower() == COLUMN_WITH_EXECUTED_EXAM.lower():
                if (
                    len(
                        set(
                            (
                                repr(v).lower().strip("'")
                                in (EXECUTED_VALUE.lower(), repr(True).lower())
                                for v in possible_values
                            )
                        )
                    )
                    == 1
                ):
                    new_row[good_column_name] = EXECUTED_VALUE.lower()
                    continue
            if good_column_name.lower().startswith(
                PREFIX_OF_COLUMNS_CONTAINING_EXAM_DATE.lower()
            ):
                possible_values = [
                    t for t in possible_values if not isinstance(t, bool)
                ]
            if all_timestamps(possible_values):
                if (
                    good_column_name
                    in COLUMNS_TO_MINIMIZE_DATE + COLUMNS_CONTAINING_EXAM_DATE
                ):
                    new_row[good_column_name] = min(
                        convert_all_to_timestamps(possible_values)
                    )
                    continue
                elif good_column_name in COLUMNS_TO_MAXIMIZE_DATE:
                    new_row[good_column_name] = max(
                        convert_all_to_timestamps(possible_values)
                    )
                    continue
            if good_column_name.lower() == COLUMN_WITH_REASON.lower():
                if rename_helper("ActualState") in old_row:
                    # Insomnia (version >= 3) already determined the last
                    # reason; let's use what it wrote in there
                    possible_value = find_values(
                        rename_helper("ActualState"), old_row
                    )[rename_helper("ActualState")].pop()
                    if any(
                        (
                            possible_value in State.non_final_states_names(),
                            pd.isna(possible_value),
                            False,  # make black auto-formatting prettier
                        )
                    ):
                        warning_queue.put(
                            f"column '{good_column_name}' could not be "
                            f"\n{' ' * 8} determined because a non-final"
                            " state was found in "
                            f"'{rename_helper('ActualState')}';\n"
                            f"{' ' * 8} NaN value was set by default."
                        )
                        # unfortunately we did not get the record
                        # where Insomnia (version >= 3) wrote :(
                        new_row[good_column_name] = np.nan
                        continue
                    elif str(State.Deceased) == possible_value:
                        new_row[good_column_name] = DECEASED_VALUE
                        continue
                    elif str(State.Discharged) == possible_value:
                        new_row[good_column_name] = ORDINARILY_HOME_DISCHARGED
                        continue
                    elif str(State.Transferred) == possible_value:
                        # let's try to get the correct transfer type
                        possible_values = list(
                            set(possible_values).difference(
                                {ORDINARILY_HOME_DISCHARGED}
                            )
                        )
                        if len(possible_values) == 1:
                            new_row[good_column_name] = possible_values.pop()
                            continue
                        else:
                            warning_queue.put(
                                f"column '{good_column_name}' could "
                                f"not be be\n{' ' * 8} determined "
                                f"because {len(possible_values)} values "
                                "meaning 'Transferred' were found;\n"
                                f"{' ' * 8} value "
                                f"'{TRANSFERRED_VALUE}' was set by default."
                            )
                            new_row[good_column_name] = TRANSFERRED_VALUE
                            continue
                else:  # Insomnia not yet run
                    possible_values = [
                        value
                        for value_list in found_values.values()
                        for value in value_list
                        if pd.notna(value)
                    ]
                    if str(State.Deceased) in possible_values:
                        new_row[good_column_name] = DECEASED_VALUE
                        continue
                    if str(State.Discharged) not in possible_values:
                        new_row[good_column_name] = TRANSFERRED_VALUE
                        continue
                    warning_queue.put(
                        f"column '{good_column_name}' has been "
                        f"determined\n{' ' * 8} by choosing the most frequent "
                        f"value between {len(possible_values)} available."
                    )
                    if (
                        str(State.Discharged)
                        == Counter(possible_values).most_common(1)[0]
                    ):
                        new_row[good_column_name] = ORDINARILY_HOME_DISCHARGED
                        continue
                    else:
                        new_row[good_column_name] = TRANSFERRED_VALUE
                        continue
            if good_column_name.lower() in COLUMN_HOSPITAL_UNIT.lower():
                possible_values = [
                    value
                    for value_list in found_values.values()
                    for value in value_list
                    if pd.notna(value)
                ]
                if len(possible_values) > 2:
                    new_row[good_column_name] = Counter(
                        possible_values  # make black auto-formatting prettier
                    ).most_common(1)[0]
                    warning_queue.put(
                        f"column '{good_column_name}' has been "
                        f"determined\n{' ' * 8} by choosing the most frequent "
                        f"value between {len(possible_values)} available."
                    )
                    continue
            if good_column_name.lower() in rename_helper(("free_text_notes",)):
                possible_values = [
                    note
                    for value_list in found_values.values()
                    for value in value_list
                    for note in value.split("\n")
                ]
                new_row[good_column_name] = "\n".join(set(possible_values))
                warning_queue.put(
                    f"column '{good_column_name}' has been "
                    f"determined\n{' ' * 8} by joining available values. "
                )
                continue
            assert good_column_name not in new_row, str(
                "Did you forget to place a 'continue' statement "
                f"after setting value {repr(new_row[good_column_name])} "
                f"in column '{good_column_name}'?"
            )
            warning_queue.put(
                f"column '{good_column_name}' could not be\n{' ' * 8} "
                f"determined because {len(possible_values)} values were"
                " found; None value was set by default.\t"
                + repr(Counter(possible_values))
            )
            if _DEBUG:
                print(f"{repr(found_values)}", file=stderr, flush=True)
            else:
                warning_queue.put(
                    f"\nHint:    Set _DEBUG=True in {worker_body.__name__}()"
                    " for an easier debug experience ;-)"
                )
            new_row[good_column_name] = None
        output_queue.put(new_row)


def fix_duplicated_columns(
    df_merged,
    final_columns=None,
    skip_sort_dictionary_keys=False,
    skip_slowest_type_checking=False,
):
    if final_columns is None:
        # Insomnia already run
        final_columns = columns_to_keep(df_merged)
    else:
        # we just need to deduplicate AdmissionCode_[x|y] columns
        skip_sort_dictionary_keys = True  # Insomnia not yet run
        skip_slowest_type_checking = True  # Insomnia will do it

    scatter_q, gather_q, warning_q = Queue(), Queue(), Queue()
    workers_list = [
        Process(
            target=worker_body,
            args=(
                final_columns.keys(),
                scatter_q,
                gather_q,
                warning_q,
            ),
        )
        for _ in range(cpu_count())
    ]

    for w in workers_list:
        w.start()

    print(
        "Scattering rows with duplicated columns over"
        f" {len(workers_list)} parallel workers"
    )
    for i, (index, row) in enumerate(df_merged.iterrows()):
        scatter_q.put((index, row))

    # append a termination signal for each worker
    for w in workers_list:
        scatter_q.put((None, None))

    i, tot = 0, i
    data = OrderedDict({col: list() for col in final_columns.keys()})

    for w in workers_list:
        while True:
            new_row = gather_q.get()
            if new_row is None:
                break  # receive termination of all workers
            print_updated_percentage("Gathering deduplicated rows: ", i, tot)
            i += 1
            for k, v in new_row.items():
                data[k].append(v)
    assert all((scatter_q.empty(), gather_q.empty()))
    print()

    warning_list = list()
    while not warning_q.empty():
        warning_list.append(warning_q.get())
    if warning_list:
        for warning_msg, count in Counter(warning_list).most_common():
            if "Hint" not in warning_msg:
                print(f"\nWARNING: In {count} ", end="")
                print(f"record{'s' if count > 1 else ''}, ", end="")
            print(warning_msg)
        print()

    for w in workers_list:
        w.join()

    print("Building deduplicated dataframe")
    if not skip_sort_dictionary_keys:
        sort_dictionary_keys(data)  # Sort future dataframe columns
    new_df = pd.DataFrame(data=data)

    print("Verifing index integrity")
    # uncomment if original index is not automagically detected and kept
    new_df = new_df.set_index("index", verify_integrity=True)

    print_after = list()
    tot = len(final_columns) - 1  # because i starts from zero
    for i, (col_name, col_type) in enumerate(final_columns.items()):
        if skip_slowest_type_checking:
            continue
        print_updated_percentage("Verifing columns data types: ", i, tot)
        if col_name == "index":
            continue  # it is not a column anymore, pandas got it was an Index
        elif col_name not in new_df.columns:
            print(f"WARNING: Column '{col_name}' not found in df_merged...")
        elif col_type is not None and any(
            (
                repr(col_type).startswith("CategoricalDtype"),
                str(new_df[col_name].dtype) != str(col_type),
            )
        ):
            print_after.append(
                f"Fixed column '{col_name}' dtype"
                f"({str(new_df[col_name].dtype).ljust(14)}"
                f" ~> {str(col_type).rjust(14)})"
            )
            new_df.astype({col_name: col_type})  # really slow :(
    print()
    if print_after:
        pad = max((len(line.split("dtype")[0]) for line in print_after))
        for line in sorted(print_after):
            print(line.split("dtype")[0].ljust(pad), end="dtype ")
            print("dtype".join(line.split("dtype")[1:]))
        print()

    print("Deduplicated dataframe is ready")
    return new_df


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.deduplicate_columns",
            "YAstarMM.deduplicate_columns",
            "deduplicate_columns",
        ),
        "columns_to_keep" in globals(),
        "fix_duplicated_columns" in globals(),
    )
), "Please update 'Usage' section of module docstring"
