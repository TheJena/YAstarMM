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
                cast_columns_to_booleans,
                cast_columns_to_categorical,
                cast_columns_to_floating_point,
                fix_bad_date_range,
                deduplicate_dataframe_columns
            )

   ( or from within the YAstarMM package )

            from          .serial  import  (
                cast_columns_to_booleans,
                cast_columns_to_categorical,
                cast_columns_to_floating_point,
                fix_bad_date_range,
                deduplicate_dataframe_columns,
            )
"""

from .column_rules import (
    BOOLEANISATION_MAP,
    does_not_match_categorical_rule,
    does_not_match_float_rule,
    ENUM_GRAVITY_LIST,
    matched_enumerated_rule,
    matches_boolean_rule,
    matches_date_time_rule,
    matches_integer_rule,
    NORMALIZED_TIMESTAMP_COLUMNS,
    rename_helper,
)
from .parallel import (
    fill_rows_matching_truth,
    find_valid_keys,
    update_all_sheets,
)
from .utility import (
    AVERAGE_DAYS_PER_YEAR,
    black_magic,
    new_key_col_value,
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


def _auxiliary_dataframe(df_dict, aux_cols, new_empty_col, sortby=list()):
    ret = pd.DataFrame(columns=sorted(aux_cols, key=str.lower))
    for df in df_dict.values():
        sub_df = df.loc[:, [c for c in aux_cols if c in df.columns]]
        ret = pd.concat(
            [
                ret,
                sub_df.assign(
                    **{
                        c: pd.Series([np.nan for _ in range(df.shape[0])])
                        for c in set(aux_cols).difference(set(sub_df.columns))
                    }
                ),
            ],
            join="outer",
            ignore_index=True,
        ).drop_duplicates()
    if new_empty_col is not None:
        ret = ret.assign(
            **{
                new_empty_col: pd.Series(
                    [np.nan for _ in range(ret.shape[0])]
                ),
            }
        )
    return (
        ret.astype(
            {
                col: pd.Int64Dtype()
                for col in ret.columns
                if all(
                    (
                        # pd.Timestamp
                        "date" not in col.lower(),
                        not matches_date_time_rule(col),
                        # string
                        "" not in col.lower(),
                        col != rename_helper(""),
                    )
                )
            }
        )
        .sort_values([c for c in sortby if c in ret.columns])
        .reset_index(drop=True)
    )


def _convert_single_cell_boolean(cell, column_name=None):
    if column_name is not None and column_name in (
    ):
        if pd.isna(cell):
            return np.nan
        return bool(" ".join(str(cell).lower().split()) != "")
    else:
        return BOOLEANISATION_MAP.get(
            cell.lower() if isinstance(cell, str) else cell,
            cell,
        )


def _convert_single_cell_float(value, column=None):
    if pd.isna(value):
        return np.nan
    try:
        ret = str(value).lower()
        if any(
            (
                "" in ret and "" in ret,
                "" in ret and "" in ret,
                "" in ret and "" in ret,
                "" in ret and "" in ret,
            )
        ):
            return np.nan
        ret = (
            ret.replace("", " 0 ")
            .replace("", " 0 ")
            .replace("", " 0 ")
        )
        ret = ret.strip(f"{ascii_letters} /%+=(<@>)")
        ret = " ".join(ret.split()).replace(",", ".").replace("|", "-")
        if matches_integer_rule(column) and ret.isnumeric():
            ret = int(round(float(ret)))  # treat integer numbers
        elif ret.startswith("-"):  # treat negative numbers
            ret = -1 * float(ret.strip("-"))
        else:  # treate positive numbers and range
            ret = np.mean([float(number) for number in ret.split("-")])
    except ValueError:
        debug(
            f"Could not convert {repr(value)}"
            f"{' of ' + repr(column) if column is not None else ''} to "
            f"{'integer' if matches_integer_rule(column) else 'float'} "
            f"\t(stripped value: {repr(ret)})"
        )
        return np.nan
    else:
        if repr(value) != repr(ret):
            debug(
                f"Converted {repr(value)}"
                f"{' of ' + repr(column) if column is not None else ''} "
                f"into floating-point {repr(ret)}"
            )
        return ret


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


def _read_only_copy(aux_df, key_col, excluded_cols=list()):
    assert key_col in aux_df.columns, str(
        f"key_col '{key_col}' must be in aux_df.columns"
    )
    return (
        aux_df.loc[
            aux_df[key_col].notna(),
            [col for col in aux_df.columns if col not in excluded_cols],
        ]
        .drop_duplicates()
        .sort_values(key_col)
        .reset_index(drop=True)
    )


def cast_columns_to_booleans(df_dict):
    new_df_columns = dict()
    for sheet_name, df in df_dict.items():
        for column in df.columns:
            assert isinstance(df.loc[:, column], pd.Series), str(
                f"Did you deduplicate columns of '{sheet_name}'? Because "
                f"column '{column}' seems to appear more than once.."
            )
            if matches_boolean_rule(column, df.loc[:, column].unique()):
                debug(
                    f"Column '{column}' in sheet '{sheet_name}' "
                    "probably contains booleans. "
                    + repr(
                        sorted(
                            set(df.loc[:, column].unique()),
                            key=lambda cell: str(cell).lower(),
                        )
                    )
                )
                new_df_columns[sheet_name] = new_df_columns.get(
                    sheet_name,
                    dict(),
                )
                old_col_repr = repr(df.loc[:, column].tolist())
                new_df_columns[sheet_name][column] = (
                    df.loc[:, column]
                    .apply(_convert_single_cell_boolean, args=(column,))
                    .convert_dtypes(
                        infer_objects=True,
                        convert_boolean=True,
                        convert_floating=False,
                        convert_integer=False,
                        convert_string=False,
                    )
                )
                new_col_repr = repr(
                    new_df_columns[sheet_name][column].tolist()
                )
                if old_col_repr != new_col_repr:
                    debug(
                        f"Column '{column}' of sheet '{sheet_name}' was:       "
                        + old_col_repr
                    )
                    debug(
                        f"Column '{column}' of sheet '{sheet_name}' now is:    "
                        + new_col_repr
                    )
    sheet_name_pad = 2 + max((len(sn) for sn in new_df_columns.keys()))
    for sheet_name, new_columns in sorted(
        new_df_columns.items(), key=lambda tup: tup[0].lower()
    ):
        df_dict[sheet_name] = df_dict[sheet_name].assign(**new_columns)
        info(
            f"Successfully converted {len(new_columns): >2d} columns of "
            f"sheet {repr(sheet_name).ljust(sheet_name_pad)} to boolean"
        )
    return df_dict


@black_magic
def cast_columns_to_categorical(df_dict, **kwargs):
    new_df_columns = dict()
    for sheet_name, df in df_dict.items():
        for column in df.columns:
            if does_not_match_categorical_rule(column, df):
                continue
            column_unique_values = set(df.loc[:, column].unique())
            debug(
                f"Column {repr(column).ljust(64)} in sheet "
                f"{repr(sheet_name).ljust(25)} probably contains enumerated "
                + repr(
                    sorted(
                        column_unique_values,
                        key=lambda cell: str(cell).lower(),
                    )
                )
            )
            dtype, conversion_map = matched_enumerated_rule(
                column, column_unique_values
            )
            if dtype is None or conversion_map is None:
                debug(
                    f"No known enum rule did match column '{column}' "
                    f"of sheet '{sheet_name}'"
                )
                continue
            old_col_repr = repr(df.loc[:, column].tolist())
            new_series = df.loc[:, column].apply(
                lambda cell: conversion_map.get(cell, cell)
            )
            new_col_repr = repr(new_series.tolist())
            if old_col_repr != new_col_repr:
                debug(
                    f"Column '{column}' of sheet '{sheet_name}' was:       "
                    + old_col_repr
                )
                debug(
                    f"Column '{column}' of sheet '{sheet_name}' now is:    "
                    + new_col_repr
                )
            new_df_columns[sheet_name] = new_df_columns.get(sheet_name, dict())
            try:
                new_df_columns[sheet_name][column] = new_series.astype(dtype)
            except Exception as e:
                debug(
                    f"Could not force {repr(dtype)} on column '{column}' "
                    f"of sheet '{sheet_name}' because {str(e)}"
                )
                new_df_columns[sheet_name][column] = new_series
            else:
                debug(
                    f"Successfully converted column '{column}' "
                    f"of sheet '{sheet_name}' into '"
                    + " ".join(repr(dtype).split())
                    + "'"
                )
    sheet_name_pad = 2 + max((len(sn) for sn in new_df_columns.keys()))
    for sheet_name, new_columns in sorted(
        new_df_columns.items(), key=lambda tup: tup[0].lower()
    ):
        df_dict[sheet_name] = df_dict[sheet_name].assign(**new_columns)
        info(
            f"Successfully converted {len(new_columns): >2d} columns of "
            f"sheet {repr(sheet_name).ljust(sheet_name_pad)} to Categorical"
        )
    return df_dict


@black_magic
def cast_columns_to_floating_point(df_dict, **kwargs):
    new_df_columns = dict()
    for sheet_name, df in df_dict.items():
        for column in df.columns:
            if does_not_match_float_rule(column, df.dtypes[column]):
                continue
            new_df_columns[sheet_name] = new_df_columns.get(sheet_name, dict())
            old_col_repr = repr(df.loc[:, column].tolist())
            new_df_columns[sheet_name][column] = (
                df.loc[:, column]
                .apply(_convert_single_cell_float, args=(column,))
                .convert_dtypes(
                    infer_objects=True,
                    convert_floating=True,
                    convert_boolean=False,
                    convert_integer=False,
                    convert_string=False,
                )
            )
            new_col_repr = repr(new_df_columns[sheet_name][column].tolist())
            if old_col_repr != new_col_repr:
                debug(
                    f"Column '{column}' of sheet '{sheet_name}' was:       "
                    + old_col_repr
                )
                debug(
                    f"Column '{column}' of sheet '{sheet_name}' now is:    "
                    + new_col_repr
                )
    sheet_name_pad = 2 + max((len(sn) for sn in new_df_columns.keys()))
    for sheet_name, new_columns in sorted(
        new_df_columns.items(), key=lambda tup: tup[0].lower()
    ):
        amount = sum(
            str(series.dtype).lower().startswith("float")
            for series in new_columns.values()
        )
        if amount > 0:
            info(
                f"Successfully converted {amount: >2d} columns of "
                f"sheet {repr(sheet_name).ljust(sheet_name_pad)} to "
                f"floating-point"
            )
    for sheet_name, new_columns in sorted(
        new_df_columns.items(), key=lambda tup: tup[0].lower()
    ):
        df_dict[sheet_name] = df_dict[sheet_name].assign(**new_columns)
    debug(
        "Please check how cells were casted to floating point numbers "
        "with this bash command:\n\t"
        "ls -v1 /tmp/*debugging* "
        "| tail -n1 "
        "| xargs grep -i 'converted.*'" + '"\'"' + "' of.*to' "
        "| grep -iv 'categorical' "
        "| sort -V "
        "| uniq "
        "| less -S"
    )
    debug(
        "Please check which cells failed floating point casting "
        "with this bash command:\n\t"
        "ls -v1 /tmp/*debugging* "
        "| tail -n1 "
        "| xargs grep -i 'could.*not.*convert.*float'"
        "| sort -V "
        "| uniq "
        "| less -S"
    )
    return df_dict


@black_magic
def create_new_unique_identifier(
    df_dict, aux_cols, new_key_col, autoselect_date_columns=False, **kwargs
):
    debug(f"original aux_cols: {repr(aux_cols)}")
    original_aux_cols, added_cols = list(aux_cols), list()
    if autoselect_date_columns:
        added_cols = sorted(
            set(
                col
                for df in df_dict.values()
                for col in df.columns
                if matches_date_time_rule(col)
            ).difference(set(aux_cols)),
            key=str.lower,
        )
    aux_cols = original_aux_cols + added_cols
    debug(
        f"building auxiliary dataframe with columns {repr(aux_cols)}"
        " from all sheets"  # make black auto-formatting prettier
    )
    aux_df = _auxiliary_dataframe(  # all aux_cols in df_dict.values()
        df_dict,
        aux_cols=aux_cols,
        new_empty_col=new_key_col,
        sortby=["admission_date", "discharge_date"],
    ).astype({new_key_col: "string"})

    stats = Counter()
    for ids_columns, empty_columns in (
        (["admission_date", "birth_date", "discharge_date"], []),
        (["admission_date", "birth_date"], ["discharge_date"]),
        (["admission_date", "discharge_date"], ["birth_date"]),
        (["admission_date"], ["birth_date", "discharge_date"]),
    ):
        selector = aux_df[new_key_col].isna()
        for col in ids_columns:
            assert col in aux_df, f"'{col}' must be in aux_df"
            selector = (selector) & (aux_df[col].notna())
        for col in empty_columns:
            assert col in aux_df, f"'{col}' has to be in aux_df"
            selector = (selector) & (aux_df[col].isna())

        debug(f"finding {repr(tuple(ids_columns))} values, valid as key")
        aux_df, stats = find_valid_keys(
            aux_df, selector, ids_columns, new_key=new_key_col, stats=stats
        )

        debug(f"building read only table with '{new_key_col}' valid values")
        read_only_truth = _read_only_copy(
            aux_df, key_col=new_key_col, excluded_cols=["date"]
        )

        debug(
            f"filling missing '{new_key_col}' values of records matching"
            " those in the read only table"
        )
        aux_df, stats = fill_rows_matching_truth(
            aux_df,
            read_only_truth,
            empty_columns,
            new_key=new_key_col,
            indirect_obj=new_key_col,  # en.wikipedia.org/wiki/Object_(grammar)
            added_date_cols=added_cols,
            stats=stats,
            use_all_available_dates=autoselect_date_columns,
        )
        debug(f"With identification columns: {repr(tuple(ids_columns))}")
        for msg, count in stats.most_common():
            debug(f"{count:9d} records {msg}")
        stats = Counter(
            {
                str(
                    f"{msg} built with\n{repr(tuple(ids_columns))}"
                    if "successfully" in msg and "built with" not in msg
                    else msg
                ): count
                for msg, count in stats.most_common()
                if "successfully" in msg or len(ids_columns) == 1
            }
        )

    aux_df = (
        aux_df.loc[:, original_aux_cols + [new_key_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    debug(f"updating all sheets with the new key column '{new_key_col}'")
    df_dict = update_all_sheets(
        df_dict,
        aux_df.drop(columns=["date"] if "date" in aux_df.columns else []),
        receiver=new_key_col,
        giver=new_key_col,
    )

    tot = sum(qty for _, qty in stats.most_common())
    pad1 = len(str(stats.most_common(1)[0][1]))
    pad2 = {msg: msg.lstrip().find(repr(new_key_col)) for msg in stats.keys()}
    for msg, count in stats.most_common():
        if count == tot:
            continue
        prefix = (
            f"{str(count).rjust(pad1)} ({100 * count / tot:6.3f}%) records "
        )
        info(
            prefix
            + msg.split("\n")[0].lstrip().rjust(max(pad2.values()) - pad2[msg])
        )
        for line in msg.split("\n")[1:]:
            info(f"{' ' * len(prefix.rstrip())} {line.lstrip()}")

    debug(
        "aux_df =\n"
        + aux_df.sort_values(
            [new_key_col, "admission_date", "birth_date", "discharge_date"]
        )
        .drop_duplicates()
        .reset_index(drop=True)
        .to_string()
    )
    if isdir(join_path(expanduser("~"), "RAMDISK")):
        aux_df.sort_values(
            [
                "admission_date",
                "discharge_date",
                "admission_code",
                "admission_id",
            ]
        ).drop(columns=["date"]).dropna(
            subset=[""]
        ).drop_duplicates().reset_index(
            drop=True
        ).to_pickle(
            join_path(expanduser("~"), "RAMDISK", "aux_df.pkl")
        )
    debug(f"amount of          in aux_df: {aux_df[].count():9d}")
    debug(f"amount of          in aux_df: {aux_df[].count():9d}")
    debug(f"size of               aux_df: {aux_df.shape[0]:9d}")
    return df_dict


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
def fill_guessable_nan_keys(df_dict, key_col, aux_cols, **kwargs):
    if key_col != "":
        warning(
            f"This function was designed to fill missing "
            "but guessable '' values.\nIts usage against "
            f"'{key_col}' was not tested, YOU HAVE BEEN WARNED!"
        )
    new_key_col = f"GUESSED_{key_col}".upper()

    debug(
        f"building auxiliary dataframe with columns {repr(aux_cols)}"
        " from all sheets"  # make black auto-formatting prettier
    )
    aux_df = _auxiliary_dataframe(
        df_dict,
        aux_cols=aux_cols,
        new_empty_col=new_key_col,
        sortby=["", "", "admission_date", "discharge_date"],
    )
    stats = Counter()
    debug(f"finding existing and valid '{key_col}' values")
    for key_value, df in (
        aux_df.drop(
            columns=set(("date", new_key_col)).intersection(
                set(aux_df.columns),
            )
        )
        .dropna()
        .groupby(key_col)
    ):
        aux_df.loc[df.index, new_key_col] = int(key_value)
        stats.update(
            {
                " ".join(
                    (
                        "successfully identified by an existing",
                        repr(key_col.upper()),
                    )
                ): len(list(df.index.array))
            }
        )

    debug("building read only table with found valid values")
    read_only_truth = _read_only_copy(
        aux_df, key_col=new_key_col, excluded_cols=["date"]
    )

    debug(
        f"fill missing '{new_key_col}' values of records matching"
        " those in the read only table"
    )
    aux_df, stats = fill_rows_matching_truth(
        aux_df,
        read_only_truth,
        empty_columns=list(),
        new_key=new_key_col,
        indirect_obj=key_col,
        stats=stats,
    )

    debug("updating all sheets with the successfully guessed key values")
    df_dict = update_all_sheets(
        df_dict,
        aux_df.drop(columns=["date"] if "date" in aux_df.columns else []),
        receiver=key_col,
        giver=new_key_col,
    )  # make black auto-formatting prettier

    tot = sum(qty for _, qty in stats.most_common())
    pad = len(str(stats.most_common(1)[0][1]))
    for msg, count in stats.most_common():
        info(
            f"{str(count).rjust(pad)} ({100 * count / tot:5.2f}%) records "
            + msg.split("\n")[0].lstrip()
        )
        for line in msg.split("\n")[1:]:
            info(f"{' ' * (pad + 10 + len('records'))} {line.lstrip()}")
    return df_dict


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


@black_magic
def identify_remaining_records(
    df_dict,
    aux_cols,
    old_key_tuple,
    new_key_col,
    guess_admission_date=None,
    guess_discharge_date=None,
    ignore_provenance=False,
    **kwargs,
):
    assert isinstance(old_key_tuple, tuple)
    if not ignore_provenance:
        assert "provenance" in old_key_tuple, str(
            "Please put 'provenance' also in old_key_tuple argument"
        )
    aux_cols = list(aux_cols)
    for new_col in (new_key_col, guess_admission_date, guess_discharge_date):
        if new_col is not None:
            aux_cols.append(new_col)
    debug(
        f"building auxiliary dataframe with columns {repr(aux_cols)}"
        " from all sheets"  # make black auto-formatting prettier
    )
    for col in ("admission_date", "discharge_date", "provenance"):
        assert col in aux_cols, f"'{col}' must be in aux_cols"
    aux_df = _auxiliary_dataframe(  # all aux_cols in df_dict.values()
        df_dict,
        aux_cols=aux_cols,
        new_empty_col=None,  # do not add anything
        sortby=list(old_key_tuple)
        + [
            "provenance",
            guess_admission_date
            if guess_admission_date is not None
            else "admission_date",
            guess_discharge_date
            if guess_discharge_date is not None
            else "discharge_date",
        ],
    ).astype({new_key_col: "string"})

    # add to aux_df two new columns for guessed admission/discharge dates
    for guess_col, new_col in {
        guess_admission_date: "NEW_admission_date",
        guess_discharge_date: "NEW_discharge_date",
    }.items():
        if guess_col is not None:
            aux_df = aux_df.assign(
                **{
                    new_col: pd.Series(
                        [np.nan for _ in range(aux_df.shape[0])],
                        dtype="datetime64[ns]",
                    )
                }
            )

    stats = Counter()

    # select only rows not covered by create_new_unique_identifier()
    selector = aux_df[new_key_col].isna()
    for col in aux_df.columns:
        if all(
            (
                col not in old_key_tuple,
                col != "provenance",
                (guess_admission_date is None)
                or (col != guess_admission_date),
                (guess_discharge_date is None)
                or (col != guess_discharge_date),
            )
        ):
            selector = (selector) & (aux_df[col].isna())

    old_identifiers = [
        dict(zip(old_key_tuple, old_key_values))
        for old_key_values, _ in aux_df.loc[selector, :].groupby(
            list(old_key_tuple)
        )
    ]
    while old_identifiers:
        old_id = old_identifiers.pop()
        patient_selector = selector
        for col, val in old_id.items():
            assert col in aux_df, f"'{col}' must be in aux_df"
            patient_selector = (patient_selector) & (aux_df[col] == val)

        new_admission_date = (
            aux_df.loc[
                patient_selector,
                guess_admission_date
                if guess_admission_date is not None
                else "admission_date",
            ]
            .dropna()
            .min()
        )
        if pd.isna(new_admission_date):
            continue

        new_discharge_date = (
            aux_df.loc[
                patient_selector,
                guess_discharge_date
                if guess_discharge_date is not None
                else "discharge_date",
            ]
            .dropna()
            .max()
        )
        if pd.isna(new_discharge_date):
            new_discharge_date = None

        debug(
            f"patient ({repr(old_id)}) has: ("
            + str(
                repr(guess_admission_date)
                if guess_admission_date is not None
                else repr("admission_date")
            )
            + f"=='{new_admission_date}', "
            + str(
                repr(guess_discharge_date)
                if guess_discharge_date is not None
                else repr("discharge_date")
            )
            + f"=='{new_discharge_date}') "
            + f"and {patient_selector.sum()} records in aux_df."
        )

        new_key_value = new_key_col_value(
            admission_date=new_admission_date,
            discharge_date=new_discharge_date,
        )
        if new_key_value in set(aux_df[new_key_col].dropna().tolist()):
            warning(f"patient '{new_key_value}' is already in aux_df")
            debug(
                "\n"
                + aux_df.loc[
                    (aux_df[new_key_col].isin([new_key_value]))
                    | patient_selector,
                    :,
                ].to_string()
            )
        aux_df.loc[patient_selector, new_key_col] = new_key_value

        aux_df.loc[patient_selector, "NEW_admission_date"] = new_admission_date
        aux_df.loc[patient_selector, "NEW_discharge_date"] = new_discharge_date

        stats.update(
            {
                str(
                    f"successfully identified by a '{new_key_col}'"
                    " value built with\n("
                    + str(
                        repr(guess_admission_date)
                        if guess_admission_date is not None
                        else "admission_date"
                    )
                    + ", "
                    + str(
                        repr(guess_discharge_date)
                        if guess_discharge_date is not None
                        else "discharge_date"
                    )
                    + ")"
                ): patient_selector.sum()
            }
        )

    # drop 'date' column if present since it is not needed anymore
    aux_df = (
        aux_df.drop(columns=["date"] if "date" in aux_df.columns else [])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    giver = tuple([new_key_col, "NEW_admission_date", "NEW_discharge_date"])
    receiver = tuple([new_key_col, "admission_date", "discharge_date"])
    info("")
    debug(
        f"updating all sheets columns {repr(receiver)}' with information "
        f"from aux_df[{repr(giver)}'].notna() (when rows match)"
    )
    df_dict = update_all_sheets(
        df_dict, aux_df, receiver=receiver, giver=giver
    )
    info("")
    pad1 = len(str(stats.most_common(1)[0][1]))
    for msg, count in stats.most_common():
        prefix = f"{str(count).rjust(pad1)} records "
        info(prefix + msg.split("\n")[0].lstrip())
        for line in msg.split("\n")[1:]:
            info(f"{' ' * len(prefix.rstrip())} {line.lstrip()}")

    debug(
        "aux_df =\n"
        + aux_df.sort_values(
            [
                guess_admission_date
                if guess_admission_date is not None
                else "admission_date",
                new_key_col,
                guess_discharge_date
                if guess_discharge_date is not None
                else "discharge_date",
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
        .to_string()
    )
    debug(
        f"amount of NEW_KEY            in aux_df: "
        f"{aux_df[new_key_col].count():9d}"
    )
    debug(
        f"amount of NEW_admission_date in aux_df: "
        f"{aux_df['NEW_admission_date'].count():9d}"
    )
    debug(
        f"amount of NEW_discharge_date in aux_df: "
        f"{aux_df['NEW_discharge_date'].count():9d}"
    )
    debug(f"size of                         aux_df: {aux_df.shape[0]:9d}")
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
