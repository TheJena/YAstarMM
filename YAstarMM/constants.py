#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020-2021 Federico Motta <191685@studenti.unimore.it>
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
   Return constant values and translated values

   Usage:
            from  YAstarMM.constants  import  (
                ALLOWED_OUTPUT_FORMATS, LOGGING_LEVEL,
            )

   ( or from within the YAstarMM package )

            from          .constants  import  (
                ALLOWED_OUTPUT_FORMATS, LOGGING_LEVEL,
            )
"""

from .column_rules import rename_helper
from re import VERBOSE, compile
from typing import Dict, Tuple
import logging
import pandas as pd
import sys

try:
    from IPython import get_ipython
except (ImportError, ModuleNotFoundError):
    pass  # ipython not installed

ALLOWED_OUTPUT_FORMATS: Tuple[str, ...] = (
    "csv",
    "json",
    "pickle",
    "pkl",
    "xlsx",
)
"""Allowed DataFrame export formats."""

APPLE_GREEN = "#8DB600"
CAPRI_BLUE = "#00BFFF"


COLUMNS_AFTER_STATE_TRANSITION_COLUMNS = rename_helper(
    (
        #
        # Order does matter, do not change it please
        #
    )
)

COLUMNS_BEFORE_STATE_TRANSITION_COLUMNS = rename_helper(
    (
        #
        # Order does matter, do not change it please
        #
    )
)

COLUMNS_CONTAINING_EXAM_DATE = rename_helper(
    (
    )
)

COLUMNS_NOT_SO_EASY_TO_MERGE = rename_helper(
    (  # because of contraddicting data
    )
)

COLUMNS_TO_KEEP_DICTIONARY = {
    rename_helper(k): v
    for k, v in {
        # column_name: column_type,
    }.items()
}

COLUMNS_TO_MAXIMIZE = rename_helper(
    (
    )
)
COLUMNS_TO_MAXIMIZE_DATE = rename_helper(
    (
    )
)

COLUMNS_TO_MINIMIZE = rename_helper(
    (
    )
)
COLUMNS_TO_MINIMIZE_DATE = rename_helper(
    (
    )
)

COLUMN_CONTAINING_PERCENTAGES = rename_helper("")
COLUMN_HOSPITAL_UNIT = rename_helper("")
COLUMN_RECALCULATED_AFTERWARDS = rename_helper("")
COLUMN_WITH_EXECUTED_EXAM = rename_helper("")
COLUMN_WITH_REASON = rename_helper("")

DECEASED_VALUE = ""

DEFAULT_RENAME_DICT: Dict[str, str]
DEFAULT_RENAME_DICT = dict(
)

EPSILON = 1e-6

EXECUTED_VALUE = ""

EXECUTING_IN_JUPYTER_KERNEL = (
    False
    if "IPython" not in sys.modules
    else False
    if "get_ipython" not in globals()
    else False
    if get_ipython() is None
    else getattr(get_ipython(), "kernel", None) is not None
)

FLOAT_EQUALITY_THRESHOLD = 1e-9

LOGGING_FORMAT = "[{levelname:^8}][{filename:^16}]{message}"
# "[%(levelname)s][%(filename)s]%(message)s"
LOGGING_LEVEL = logging.WARNING
LOGGING_STREAM = None  # i.e. stderr
LOGGING_STYLE = "{"

NASTY_SUFFIXES = (
    "_x",
    "_y",
) + tuple(f"_z{'z' * i}" for i in range(16))

ORDINARILY_HOME_DISCHARGED = ""

PER_STATE_TRANSITION_OBSERVABLES = {
    ("No O2", "O2"): rename_helper(
        (
        )
    ),
    ("O2", "NIV"): rename_helper(
        (
        )
    ),
    ("NIV", "Intubated"): rename_helper(
        (
        )
    ),
}

PREFIX_OF_COLUMNS_CONTAINING_EXAM_DATE = rename_helper("")

REGEX_FLOAT_AND_DATE = compile(
    r"""
    ^'?\s*
    (\d+[.,]?\d*)?                        # 1st group: the float
    \s*
    (\(\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\))?  # 2nd group: the date
    \s*'?$
    """,
    VERBOSE,
)
"""1st group contains the float whilst 2nd group contains the date."""

REGEX_RANGE_OF_FLOATS = compile(
    r"""
    ^'?\s*
    (\d+[.,]?\d*)               # 1st group: range start
    \s*[-÷]\s*
    (\d+[.,]?\d*)               # 2nd group: range end
    \s*'?$
    """,
    VERBOSE,
)
"""1st group contains the range start whilst the 2nd the range end."""

REGEX_UNDER_THRESHOLD_FLOAT = compile(
    r"""
    ^'?\s*
    <\s*
    (\d+[.,]?\d*)               # 1st group: threshold
    \s*'?$
    """,
    VERBOSE,
)
"""1st group contains the float used as threshold."""

TRANSFERRED_VALUE = ""

TRUTH_DICT = {
}
