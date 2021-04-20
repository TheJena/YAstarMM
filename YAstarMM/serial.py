#!/usr/in/env python3
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
   Functions for notebook 00 which do not use multiprocessing.

   Usage:
            from  YAstarMM.serial  import  (
                fix_bad_date_range,
                deduplicate_dataframe_columns
            )

   ( or from within the YAstarMM package )

            from          .serial  import  (
                fix_bad_date_range,
                deduplicate_dataframe_columns,
            )
"""

from .column_rules import (
    BOOLEANISATION_MAP,
    ENUM_GRAVITY_LIST,
    NORMALIZED_TIMESTAMP_COLUMNS,
    rename_helper,
)
from .utility import (
    AVERAGE_DAYS_PER_YEAR,
    black_magic,
    swap_month_and_day,
)
from collections import Counter
from datetime import timedelta
from logging import debug, info, warning
from os.path import expanduser, isdir, join as join_path
from string import ascii_letters
from sys import version_info
import numpy as np
import pandas as pd


# The following list of columns are only considered during the merge
# of multiple columns with the same name.
COLUMNS_TO_BOOLEANISE = [
]
COLUMNS_TO_JOIN = [
]
COLUMNS_TO_MAXIMISE = [
]
COLUMNS_TO_MINIMISE = list()
ENUM_TO_MAXIMISE = [
]


def duplicated_columns(df):
    return sorted(df.loc[:, df.columns.duplicated()].columns)


def _merge_multiple_columns(sheet_name, df, col):
    debug(
        "simple column deduplication was not enough for "
        f"'{sheet_name}'; column '{col}' was still replicated;"
        " row-by-row merging of its values started"
    )
    data = list()
    for _, row in df.loc[:, col].iterrows():
        if col not in NORMALIZED_TIMESTAMP_COLUMNS:
            row = set(row.dropna())
        else:
            row = set(row.dropna().astype("datetime64[ns]").dt.normalize())
        if not row:
            data.append(np.nan)
        elif len(row) == 1:
            data.append(row.pop())
        elif col in COLUMNS_TO_BOOLEANISE:
            row = [BOOLEANISATION_MAP.get(cell, cell) for cell in row]
            assert all((isinstance(cell, bool) for cell in row)), str(
                f"row of booleans for column '{col}' has spurious values: "
                + repr(sorted(row)).strip("[]")
            )
            if sum(row) > len(row) - sum(row):
                data.append(True)
            elif sum(row) < len(row) - sum(row):
                data.append(False)
            else:  # no majority was found
                data.append(np.nan)
        elif col in COLUMNS_TO_JOIN:
            debug(f"row of '{col}' before joining cells: {repr(row)}")
            row = " ".join(str(".¿@?".join(row)).split()).replace("¿@?", "\n")
            debug(f"row of '{col}' after joining cells: {repr(row)}")
            data.append(row)  # it is now a single string
        elif col in COLUMNS_TO_MAXIMISE:
            debug(f"row of '{col}' before using maximum value: {repr(row)}")
            data.append(max(row))
            debug(f"row of '{col}' after using maximum value: {max(row)}")
        elif col in COLUMNS_TO_MINIMISE:
            debug(f"row of '{col}' before using minimum value: {repr(row)}")
            data.append(min(row))
            debug(f"row of '{col}' after using minimum value: {min(row)}")
        elif col in ENUM_TO_MAXIMISE:
            debug(f"row of '{col}' before using maximum value: {repr(row)}")
            max_value = ENUM_GRAVITY_LIST[
                max(ENUM_GRAVITY_LIST.index(cell) for cell in row)
            ]
            data.append(max_value)
            debug(f"row of '{col}' after using maximum value: {max_value}")
        else:
            raise NotImplementedError(
                f"row of '{col}' was not covered by any case: {repr(row)}"
            )
    assert len(data) == df.shape[0], str(
        f"merged columns '{col}' resulted in a series with a different "
        f"length from the dataframe (len(new_col): {len(data)}; "
        f"df.shape[0]: {df.shape[0]}"
    )
    return df.drop(columns=col).assign(**{col: pd.Series(data)})


@black_magic
def deduplicate_dataframe_columns(
    sheet_name,
    df,
    skip_simple_attempt=False,
    **kwargs,
):
    duplicated_columns_qty = len(duplicated_columns(df))

    if not skip_simple_attempt:
        new_df = df.transpose().drop_duplicates().transpose()
    else:
        new_df = df
    if not new_df.columns.is_unique:
        for col in set(duplicated_columns(new_df)):
            try:
                new_df = _merge_multiple_columns(sheet_name, new_df, col)
            except Exception as e:
                warning(
                    f"While deduplicating column '{col}' of sheet "
                    f"'{sheet_name}' got the following exception: {str(e)}"
                )

    if new_df.columns.is_unique:
        info(
            f"Deduplicated {duplicated_columns_qty: >2d} columns "
            f"of sheet '{sheet_name}' ..."
        )
    else:
        warning(
            f"sheet '{sheet_name}' still has duplicated columns:\n    "
            + repr(
                sorted(
                    new_df.loc[:, new_df.columns.duplicated()].columns,
                    key=str.lower,
                )
            ).strip("[]")
            + "."
        )
    return new_df


@black_magic
def fix_bad_date_range(
    df_dict, start_col, end_col, admit_start_end_swap=False, **kwargs
):
    fixed_dates = Counter()
    for sheet_name in sorted(df_dict.keys()):
        df = df_dict.pop(sheet_name)
        if start_col in df.columns and end_col in df.columns:
            for (start_value, end_value), _ in df.loc[
                :,
                [start_col, end_col],
            ].groupby([start_col, end_col]):
                if start_value <= end_value:
                    continue  # good date range, nothing to fix
                debug(
                    f"negative date range ('{start_col}': '{start_value}'; "
                    f"'{end_col}': '{end_value}') will be replaced with ..."
                )
                selected_rows = (df[start_col] == start_value) & (
                    df[end_col] == end_value
                )
                swapped = dict()
                for col, old_col_value in {
                    start_col: start_value,
                    end_col: end_value,
                }.items():
                    try:
                        new_col_value = swap_month_and_day(old_col_value)
                    except ValueError:
                        swapped[col] = old_col_value
                    else:
                        swapped[col] = new_col_value
                previous_date_range_size = round(120 * AVERAGE_DAYS_PER_YEAR)
                for start, end, swap_num in (
                    (swapped[start_col], end_value, 1),
                    (start_value, swapped[end_col], 1),
                    (swapped[start_col], swapped[end_col], 2),
                    (
                        end_value if admit_start_end_swap else start_value,
                        start_value if admit_start_end_swap else end_value,
                        int(admit_start_end_swap),
                    ),
                ):
                    if start > end:
                        continue  # still a negative date range, skip it
                    current_date_range_size = pd.date_range(
                        start, end, freq="D", normalize=True
                    ).size
                    if current_date_range_size < previous_date_range_size:
                        debug(
                            f"{'.'*len('negative date range')} "
                            f"('{start_col}': '{start}'; "
                            f"'{end_col}': '{end}')"
                            + str(
                                " which is shorter ("
                                f"{current_date_range_size} < "
                                f"{previous_date_range_size})"
                                if previous_date_range_size
                                != round(120 * AVERAGE_DAYS_PER_YEAR)
                                else ""
                            )
                        )
                        previous_date_range_size = current_date_range_size
                        start_value, end_value = start, end
                        fixed_dates.update([sheet_name] * swap_num)
                df.loc[selected_rows, start_col] = start_value
                df.loc[selected_rows, end_col] = end_value
        df_dict[sheet_name] = df

    if fixed_dates:
        pad = len(str(fixed_dates.most_common(1)[0][1]))
        for sheet_name, count in fixed_dates.most_common():
            info(
                f"{str(count).rjust(pad)} dates causing "
                f"negative date ranges fixed in '{sheet_name}'"
            )
    return df_dict


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.serial",
            "YAstarMM.serial",
            "serial",
        ),
        "deduplicate_dataframe_columns" in globals(),
        "cast_columns_to_booleans" in globals(),
        "cast_columns_to_categorical" in globals(),
        "cast_columns_to_floating_point" in globals(),
        "fix_bad_date_range" in globals(),
    )
), "Please update 'Usage' section of module docstring"
