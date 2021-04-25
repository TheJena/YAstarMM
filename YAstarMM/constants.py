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
                BOOLEANIZATION_MAP,
                EXECUTING_IN_JUPYTER_KERNEL,
                EXTRACTION_REGEXP,
                InputOutputErrorQueues,
                LOGGING_LEVEL,
                MIN_PYTHON_VERSION,
                SHEET_RENAMING_RULES,
                TITLE_COLUMN_REGEXP,
            )

   ( or from within the YAstarMM package )

            from          .constants  import  (
                BOOLEANIZATION_MAP,
                EXECUTING_IN_JUPYTER_KERNEL,
                EXTRACTION_REGEXP,
                InputOutputErrorQueues,
                LOGGING_LEVEL,
                MIN_PYTHON_VERSION,
                SHEET_RENAMING_RULES,
                TITLE_COLUMN_REGEXP,
            )
"""

from collections import namedtuple, OrderedDict
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import re
import sys
import yaml


MIN_PYTHON_VERSION = (3, 8)
assert (
    sys.version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"


try:
    from IPython import get_ipython
except (ImportError, ModuleNotFoundError):
    pass  # ipython not installed


APPLE_GREEN = "#8DB600"

# https://en.wikipedia.org/w/index.php?title=Bissextile_year&redirect=yes
AVERAGE_DAYS_PER_YEAR = 365 + 1 / 4 - 1 / 100 + 1 / 400

BOOLEANIZATION_MAP = {
    "": np.nan,
    "nan": np.nan,
    False: False,
    True: True,
    float(0.0): False,
    float(1.0): True,
    int(0): False,
    int(1): True,
    np.nan: np.nan,
    pd.NA: np.nan,
    pd.NaT: np.nan,
    str(float(0.0)): False,
    str(float(1.0)): True,
    str(int(0)): False,
    str(int(1)): True,
}

CAPRI_BLUE = "#00BFFF"

# The following list of columns are only considered during the merge
# of multiple columns with the same name.
COLUMNS_TO_BOOLEANIZE = [
    "remdesivir",
]
COLUMNS_TO_JOIN = [
    "symptoms_list",
    "free_text_notes",
]
COLUMNS_TO_MAXIMIZE = [
]
COLUMNS_TO_MINIMIZE = [
]


#
# Some reminders about regexp:
#
# \d is equivalent to [0-9]
# \D is equivalent to [^0-9]
# \w is equivalent to [a-zA-Z0-9_]
DAYFIRST_REGEXP = re.compile(
    str(
        r"[^0-9]*"  # junk text
        r"(?P<day>[012][1-9]|30|31)"
        r"[^a-zA-Z0-9]"  # separator
        r"(?P<month>0[1-9]|1[012])"
        r"[^a-zA-Z0-9]"  # separator
        r"(?P<year>1[89]\d\d|2[01]\d\d)"  # years between 1800 and 2199
        r"\s*"  # white space
        r"("  # optional time start
        r"(?P<hour>[01]\d|2[0123])"
        r":"  # separator
        r"(?P<minute>[012345]\d)"
        r":"  # separator
        r"(?P<second>[012345]\d)"
        r")?"  # optional time end
        r"[^0-9]*"  # junk text
    )
)
assert datetime.today().year < 2200, "Please fix DAYFIRST regular expression"

DECEASED = ""

EIGHTEEN_CENTURIES_IN_MINUTES = 18 * 100 * AVERAGE_DAYS_PER_YEAR * 24 * 60

ENUM_GRAVITY_LIST = [  # from lower gravity
    "Absent",
    "With reservoir bag",
    "Venturi mask",
    "Venturi mask without reservoir bag",
    "Venturi mask with reservoir bag",
    "Nasal cannula",
    "HFNO",
    "NIV",
]  # to higher gravity

ENUM_TO_MAXIMIZE = [
]

EPSILON = 1e-6

EXECUTING_IN_JUPYTER_KERNEL = (
    False
    if "IPython" not in sys.modules
    else False
    if "get_ipython" not in globals()
    else False
    if get_ipython() is None
    else getattr(get_ipython(), "kernel", None) is not None
)

EXTRACTION_REGEXP = re.compile(
    r"(Extraction)"
    r"[ _-]*"  # separator
    r"(?P<year>2[01]\d\d)"  # valid until 2199
    r"[ _-]*"  # separator
    r"(?P<month>0[1-9]|1[012])"
    r"[ _-]*"  # separator
    r"(?P<day>[012][1-9]|30|31)"
    r".*",  # whatever
    re.IGNORECASE,
)
assert datetime.today().year < 2200, "Please fix the above regular expression"

FileWorkerError = namedtuple("FileWorkerError", ["filename", "exception"])
FileWorkerInput = namedtuple("FileWorkerInput", ["filename", "rename_mapping"])
FileWorkerOutput = namedtuple("FileWorkerOutput", ["filename", "sheets_dict"])

GroupWorkerError = namedtuple("GroupWorkerError", ["group_name", "exception"])
GroupWorkerInput = namedtuple("GroupWorkerInput", ["group_name", "dataframe"])
GroupWorkerOutput = namedtuple("GroupWorkerOutput", ["df", "stats"])

InputOutputErrorQueues = namedtuple(
    "InputOutputErrorQueues", ["input_queue", "output_queue", "error_queue"]
)

LOGGING_FORMAT = "[{levelname:^8}][{filename:^16}]{message}"
# "[%(levelname)s][%(filename)s]%(message)s"
LOGGING_LEVEL = logging.WARNING
LOGGING_STREAM = None  # i.e. stderr
LOGGING_STYLE = "{"

NASTY_SUFFIXES = (
    "_x",
    "_y",
) + tuple(f"_z{'z' * i}" for i in range(16))

NewKeyError = namedtuple("NewKeyError", ["ids_values", "exception"])
NewKeyOutput = namedtuple("NewKeyOutput", ["selected_rows", "new_value"])

NORMALIZED_TIMESTAMP_COLUMNS = [
    # These are the columns we are going to use in "db-like-join"
    # merge operations between sheets; for this reason it is important
    # that two record do not differ for a few seconds/minutes (since
    # we are interested in a daily time granularity). So we are going
    # to keep the date information and drop the time information.
    "date",
]

ORDINARILY_HOME_DISCHARGED = ""

RowFillerError = namedtuple("RowFillerError", ["row", "exception"])
RowFillerOutput = namedtuple(
    "RowFillerOutput", ["row_index", "chosen_id", "stats"]
)

YamlSafeLoader = getattr(
    yaml,
    "CSafeLoader",  # faster compiled (safe) Loader
    yaml.Loader,  # fallback, slower interpreted (safe) Loader
)

SheetMergerInput = namedtuple(
    "SheetMergerInput",
    ["left_name", "right_name", "left_df", "right_df", "join_dict"],
)
SheetMergerOutput = namedtuple("SheetMergerOutput", ["name", "df"])

SHEET_RENAMING_RULES = OrderedDict(
    {
        new_sheet_name: re.compile(case_insensitive_regexp, re.IGNORECASE)
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
                    r"|"  # logic or
                    r""
                    r"|"  # logic or
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
                fibroscan=r"",
                frailty=r"",
                hospitalization=r"",
                nutrition=r"",
                pneumological_exam=r"",
                radiology_report=r"",
                six_min_walk=r"",
                therapies=r"",
                well_being=r"",
            ).items()
        )
    }
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

SORTING_PREFIX = "\t"

SummarizeFeatureItem = namedtuple(
    "SummarizeFeatureItem", ["old_columns", "old_values_checker", "new_enum"]
)

SwitchToDateValue = namedtuple("SwitchToDateValue", ["true_val", "date_col"])

TimestampWorkerError = namedtuple(
    "TimestampWorkerError", ["sheet_name", "column_name", "exception"]
)
TimestampWorkerInputOutput = namedtuple(
    "TimestampWorkerInputOutput", ["sheet_name", "column_name", "series"]
)

TITLE_COLUMN_REGEXP = re.compile(
    r"^\s*(title|description)\s*$", re.IGNORECASE
)

VerticalizeFeatureItem = namedtuple(
    "VerticalizeFeatureItem", ["date_column", "column_name", "related_columns"]
)


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert __name__ in (
    "analisi.src.YAstarMM.constants",
    "YAstarMM.constants",
    "constants",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
