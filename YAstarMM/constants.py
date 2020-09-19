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
                ALLOWED_OUTPUT_FORMATS, LOGGING_LEVEL, NON_FINAL_STATES,
            )

   ( or from within the YAstarMM package )

            from          .constants  import  (
                ALLOWED_OUTPUT_FORMATS, LOGGING_LEVEL, NON_FINAL_STATES,
            )
"""

import logging
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


EXECUTING_IN_JUPYTER_KERNEL = (
    False
    if "IPython" not in sys.modules
    else False
    if "get_ipython" not in globals()
    else False
    if get_ipython() is None
    else getattr(get_ipython(), "kernel", None) is not None
)

LOGGING_FORMAT = "[{levelname:^8}][{filename:^16}]{message}"
# "[%(levelname)s][%(filename)s]%(message)s"
LOGGING_LEVEL = logging.WARNING
LOGGING_STREAM = None  # i.e. stderr
LOGGING_STYLE = "{"

NASTY_SUFFIXES = ("_x", "_y",) + tuple(f"_z{'z' * i}" for i in range(16))
