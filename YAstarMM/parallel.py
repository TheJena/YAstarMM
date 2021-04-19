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
            )

   ( or from within the YAstarMM package )

            from          .parallel  import  (
            )
"""

from collections import namedtuple

GroupWorkerInput = namedtuple("GroupWorkerInput", ["group_name", "dataframe"])
GroupWorkerError = namedtuple("GroupWorkerError", ["group_name", "exception"])

InputOutputErrorQueues = namedtuple(
    "InputOutputErrorQueues", ["input_queue", "output_queue", "error_queue"]
)


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
    )
), "Please update 'Usage' section of module docstring"
