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


COLUMNS_AFTER_STATE_TRANSITION_COLUMNS = (
    #
    # Order does matter, do not change it please
    #
)

COLUMNS_BEFORE_STATE_TRANSITION_COLUMNS = (
    #
    # Order does matter, do not change it please
    #
)

COLUMNS_CONTAINING_EXAM_DATE = (
)

COLUMNS_NOT_SO_EASY_TO_MERGE = (  # because of contraddicting data
)

COLUMNS_TO_KEEP_DICTIONARY = {
}

COLUMNS_TO_MAXIMIZE = (
)

COLUMNS_TO_MAXIMIZE_DATE = ()
COLUMNS_TO_MINIMIZE_DATE = ()

COLUMN_CONTAINING_PERCENTAGES = ""
COLUMN_RECALCULATED_AFTERWARDS = ""
COLUMN_WITH_EXECUTED_EXAM = ""
COLUMN_WITH_REASON = ""

DECEASED_VALUE = ""

DEFAULT_RENAME_DICT: Dict[str, str]
DEFAULT_RENAME_DICT = dict(
)

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

NASTY_SUFFIXES = ("_x", "_y",) + tuple(f"_z{'z' * i}" for i in range(16))

ORDINARILY_HOME_DISCHARGED = ""

PREFIX_OF_COLUMNS_CONTAINING_EXAM_DATE = ""

TRANSFERRED_VALUE = ""
