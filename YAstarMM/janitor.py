#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2018-2020 pyjanitor devs
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
   Clean and format Pandas DataFrame column names.

   This module is a free adaptation of:
        https://github.com/ericmjl/pyjanitor
   (originally licensed under the MIT or Expat license, which is
    compatible with the present GPL-3.0-or-later license)

   Usage:
            from  YAstarMM.janitor  import  clean_names

   ( or from within the YAstarMM package )

            from           janitor  import  clean_names
"""

import pandas as pd
import re
from sys import version_info
from typing import Union
import unicodedata

_underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
_underscorer2 = re.compile("([a-z0-9])([A-Z])")


def _camel2snake(col_name: str) -> str:
    """Convert camelcase names to snake case.

    Credits to: https://gist.github.com/jaytaylor/3660565
                https://stackoverflow.com/a/1176023
    """
    subbed = _underscorer1.sub(r"\1_\2", col_name)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


def _change_case(col: str, case_type: str) -> str:
    """Change case of a column name."""
    case_types = ("preserve", "lower", "upper", "snake")
    if case_type.lower() not in case_types:
        raise ValueError(f"case_type must be one of: {case_types}")

    if case_type.lower() != "preserve":
        if case_type.lower() == "lower":
            col = col.lower()
        elif case_type.lower() == "upper":
            col = col.upper()
        elif case_type.lower() == "snake":
            col = _camel2snake(col)
    return col


def _normalize_1(col_name: str) -> str:
    """Perform normalization of column name."""
    result = str(col_name)
    for search, replace in [(r"[ /:,?()\.-]", "_"), (r"['’]", "")]:
        result = re.sub(search, replace, result)
    return result


def _remove_special(col_name: str) -> str:
    """Remove special characters from column name."""
    return "".join(
        item
        for item in str(col_name)
        if any(
            (
                item.isalnum(),
                "_" in item,
            )
        )
    )


def _strip_accents(col_name: str) -> str:
    """Remove accents from a DataFrame column name.

    Credits to: https://stackoverflow.com/a/517974
    """
    return "".join(
        letter
        for letter in unicodedata.normalize("NFD", col_name)
        if not unicodedata.combining(letter)
    )


def _strip_underscores_func(
    col: str, strip_underscores: Union[str, bool, None] = None
) -> str:
    """Strip underscores from a string."""
    underscore_options = ("both", "l", "left", "r", "right", None, True)
    if strip_underscores not in underscore_options:
        raise ValueError(
            f"Unrecognized strip_underscores value ({strip_underscores}); "
            f"allowed values are: {underscore_options}"
        )

    if strip_underscores in ("left", "l"):
        col = col.lstrip("_")
    elif strip_underscores in ("right", "r"):
        col = col.rstrip("_")
    elif strip_underscores == "both" or strip_underscores is True:
        col = col.strip("_")
    return col


def _strip_underscores(
    df: pd.DataFrame, strip_underscores: Union[str, bool, None] = None
) -> pd.DataFrame:
    """Strip underscores from DataFrames column names.

    Underscores can be stripped from the beginning, end or both.

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer
         underscores from all column names. By default, None keeps
         outer underscores. Values can be either 'left', 'right' or
         'both' or the respective shorthand 'l', 'r' and True.
    :returns: A pandas DataFrame with underscores removed.
    """
    return df.rename(
        columns=lambda colum_name: _strip_underscores_func(
            colum_name, strip_underscores
        )
    )


def clean_names(
    df: pd.DataFrame,
    case_type: str = "lower",
    enforce_string: bool = True,
    preserve_original_columns: bool = True,
    remove_special: bool = False,
    strip_accents: bool = True,
    strip_underscores: Union[str, bool, None] = None,
) -> pd.DataFrame:
    """Clean column names.

    Takes all column names, converts them to lowercase, then
    replaces all spaces with underscores.

    By default, column names are converted to string types.  This
    can be switched off by passing in ``enforce_string=False``.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    :Example of transformation:

        Columns before: First Name, Last Name, Employee Status, Subject
        Columns after: first_name, last_name, employee_status, subject

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer
        underscores from all column names. Default None keeps outer
        underscores. Values can be either 'left', 'right' or 'both'
        or the respective shorthand 'l', 'r' and True.
    :param case_type: (optional) Whether to make columns lower or
        uppercase.  Current case may be preserved with 'preserve',
        while snake case conversion (from CamelCase or camelCase
        only) can be turned on using "snake".  Default 'lower'
        makes all characters lowercase.
    :param remove_special: (optional) Remove special characters
        from columns.  Only letters, numbers and underscores are
        preserved.
    :param preserve_original_columns: (optional) Preserve original names.
        This is later retrievable using `df.original_columns`.
    :param enforce_string: Whether or not to convert all column names
        to string type. Defaults to True, but can be turned off.
        Columns with >1 levels will not be converted by default.
    :returns: A pandas DataFrame.
    """
    original_column_names = list(df.columns)

    if enforce_string:
        df = df.rename(columns=lambda x: str(x))

    df = df.rename(columns=lambda x: _change_case(x, case_type))

    df = df.rename(columns=_normalize_1)

    if remove_special:
        df = df.rename(columns=_remove_special)

    if strip_accents:
        df = df.rename(columns=_strip_accents)

    df = df.rename(columns=lambda x: re.sub("_+", "_", x))
    df = _strip_underscores(df, strip_underscores)

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        df.__dict__["original_columns"] = original_column_names
    return df


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__ in ("YAstarMM.janitor", "janitor"),
        "clean_names" in globals(),
    )
), "Please update 'Usage' section of module docstring"
