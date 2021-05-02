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

HARDCODED_COLUMN_NAMES = dict(
    ALCOHOLIC_LIVER_DISEASE="",
    ANOTHER_DIABETES_COLUMN="",
    AUTOIMMUNE_HEPATITIS="",
    ActualState="",
    ActualState_val="",
    BLOOD_NEOPLASMS="",
    CARDIAC_ISCHEMIA="",
    CEREBROVASCULAR_DISEASE="",
    CHARLSON_AGE="",
    CHARLSON_AIDS="",
    CHARLSON_BLOOD_DISEASE="",
    CHARLSON_CONNECTIVE_TISSUE_DISEASE="",
    CHARLSON_COPD="",
    CHARLSON_CVA_OR_TIA="",
    CHARLSON_DEMENTIA="",
    CHARLSON_DIABETES="",
    CHARLSON_HEART_FAILURE="",
    CHARLSON_HEMIPLEGIA="",
    CHARLSON_INDEX="",
    CHARLSON_KIDNEY_DISEASE="",
    CHARLSON_LIVER_DISEASE="",
    CHARLSON_LIVER_FAILURE="",
    CHARLSON_MYOCARDIAL_ISCHEMIA="",
    CHARLSON_PEPTIC_ULCER_DISEASE="",
    CHARLSON_SOLID_TUMOR="",
    CHARLSON_VASCULAR_DISEASE="",
    CIRRHOSIS="",
    CKD="",
    COPD="",
    CREATININE="",
    DEMENTIA="",
    DYSPNEA="",
    DYSPNEA_START="",
    D_DIMER="",
    DayCount="",
    day_count="",
    GPT_ALT="",
    HEART_FAILURE="",
    HFNO_STATE="",
    HFNO_STATE_END="",
    HFNO_STATE_START="",
    HOROWITZ_INDEX_UNDER_150="",
    HOROWITZ_INDEX_UNDER_250="",
    ICU_TRANSFER="",
    INFECTIOUS_DISEASES_UNIT_TRANSFER="",
    INTUBATION_STATE="",
    INTUBATION_STATE_END="",
    INTUBATION_STATE_START="",
    LACTATES="",
    LDH="",
    LYMPHOCYTE="",
    NIV_STATE="",
    NIV_STATE_END="",
    NIV_STATE_START="",
    NO_OXYGEN_THERAPY_STATE="",
    NO_OXYGEN_THERAPY_STATE_END="",
    NO_OXYGEN_THERAPY_STATE_START="",
    OTHER_LIVER_PATOLOGIES="",
    OXYGEN_THERAPY_STATE="",
    OXYGEN_THERAPY_STATE_END="",
    OXYGEN_THERAPY_STATE_START="",
    PHOSPHOCREATINE="",
    PORTAL_HYPERTENSION="",
    POST_HFNO_STATE_END="",
    POST_HFNO_STATE_START="",
    POST_NIV_STATE_END="",
    POST_NIV_STATE_START="",
    POST_OXYGEN_THERAPY_STATE_END="",
    POST_OXYGEN_THERAPY_STATE_START="",
    PROCALCITONIN="",
    RESPIRATORY_RATE="",
    SOLID_TUMOR="",
    SWAB="",
    SYMPTOMS_START="",
    State="",
    TEMPERATURE="",
    UREA="",
    USE_OXYGEN="",
)
"""Lazy renaming mapping for columns not covered by regular expressions"""

ICD9_CODES = {
    "aids_hiv": (42,),
    "cerebrovascular_disease": (
        362.34,
        430,
        431,
        432,
        432.1,
        432.9,
        433,
        433.01,
        433.1,
        433.11,
        433.2,
        433.21,
        433.3,
        433.31,
        433.8,
        433.81,
        433.9,
        433.91,
        434,
        434.01,
        434.1,
        434.11,
        434.9,
        434.91,
        435,
        435.1,
        435.2,
        435.3,
        435.8,
        435.9,
        436,
        437,
        437.1,
        437.2,
        437.3,
        437.4,
        437.5,
        437.6,
        437.7,
        437.8,
        437.9,
        438,
        438.1,
        438.11,
        438.12,
        438.13,
        438.14,
        438.19,
        438.2,
        438.21,
        438.22,
        438.3,
        438.31,
        438.32,
        438.4,
        438.41,
        438.42,
        438.5,
        438.51,
        438.52,
        438.53,
        438.6,
        438.7,
        438.8,
        438.81,
        438.82,
        438.83,
        438.84,
        438.85,
        438.89,
        438.9,
    ),
    "chronic_pulmonary_disease": (
        490,
        491,
        491.1,
        491.2,
        491.21,
        491.22,
        491.8,
        491.9,
        492,
        492.8,
        493,
        493.01,
        493.02,
        493.1,
        493.11,
        493.12,
        493.2,
        493.21,
        493.22,
        493.8,
        493.81,
        493.82,
        493.9,
        493.91,
        494,
        494.1,
        508.1,
        508.8,
    ),
    "congestive_heart_failure": (
        398.91,
        402.01,
        402.11,
        402.91,
        404.01,
        404.03,
        404.11,
        404.13,
        404.91,
        404.93,
        425.4,
        425.5,
        425.7,
        425.8,
        425.9,
        428,
        428.1,
        428.2,
        428.21,
        428.22,
        428.23,
        428.3,
        428.31,
        428.32,
        428.33,
        428.4,
        428.41,
        428.42,
        428.43,
        428.9,
    ),
    "dementia": (
        290,
        290.1,
        290.11,
        290.12,
        290.13,
        290.2,
        290.21,
        290.3,
        290.4,
        290.41,
        290.42,
        290.43,
        290.8,
        290.9,
        294.1,
        331.2,
    ),
    "diabetes_without_complication": (
        250,
        250.01,
        250.02,
        250.03,
        250.1,
        250.2,
        250.3,
        250.7,
        250.8,
        250.9,
    ),
    "diabetes_with_chronic_complication": (
        250.4,
        250.41,
        250.42,
        250.43,
        250.5,
        250.51,
        250.52,
        250.53,
        250.6,
        250.61,
        250.62,
        250.63,
        250.7,
        250.71,
        250.72,
        250.73,
    ),
    "hemiplegia": (
        342,
        342.01,
        342.02,
        342.1,
        342.11,
        342.12,
        342.8,
        342.81,
        342.82,
        342.9,
        342.91,
        342.92,
        343,
        343.1,
        343.2,
        343.3,
        343.4,
        343.8,
        343.9,
        344,
        344.01,
        344.02,
        344.03,
        344.04,
        344.1,
        344.2,
        344.3,
        344.4,
        344.5,
        344.6,
        344.9,
    ),
    "leukemia": (
        203.02,
        203.1,
        203.11,
        203.12,
        203.8,
        204,
        204.01,
        204.02,
        204.1,
        204.11,
        204.12,
        204.2,
        204.21,
        204.22,
        204.8,
        204.81,
        204.82,
        204.9,
        204.91,
        204.92,
        205,
        205.01,
        205.02,
        205.1,
        205.11,
        205.12,
        205.2,
        205.21,
        205.22,
        205.3,
        205.31,
        205.32,
        205.8,
        205.81,
        205.82,
        205.9,
        205.91,
        205.92,
        206,
        206.01,
        206.02,
        206.1,
        206.11,
        206.12,
        206.2,
        206.21,
        206.22,
        206.8,
        206.81,
        206.82,
        206.9,
        206.91,
        206.92,
        207,
        207.01,
        207.02,
        207.1,
        207.11,
        207.12,
        207.2,
        207.21,
        207.22,
        207.8,
        207.81,
        207.82,
        208,
        208.01,
        208.02,
        208.1,
        208.11,
        208.12,
        208.2,
        208.21,
        208.22,
        208.8,
        208.81,
        208.82,
        208.9,
        208.91,
        208.92,
    ),
    "lymphoma": (
        196,
        196.1,
        196.2,
        196.3,
        196.5,
        196.6,
        196.8,
        196.9,
        200,
        200.01,
        200.02,
        200.03,
        200.04,
        200.05,
        200.06,
        200.07,
        200.08,
        200.1,
        200.11,
        200.12,
        200.13,
        200.14,
        200.15,
        200.16,
        200.17,
        200.18,
        200.2,
        200.21,
        200.22,
        200.23,
        200.24,
        200.25,
        200.26,
        200.27,
        200.28,
        200.3,
        200.31,
        200.32,
        200.33,
        200.34,
        200.35,
        200.36,
        200.37,
        200.38,
        200.4,
        200.41,
        200.42,
        200.43,
        200.44,
        200.45,
        200.46,
        200.47,
        200.48,
        200.5,
        200.51,
        200.52,
        200.53,
        200.54,
        200.55,
        200.56,
        200.57,
        200.58,
        200.6,
        200.61,
        200.62,
        200.63,
        200.64,
        200.65,
        200.66,
        200.67,
        200.68,
        200.7,
        200.71,
        200.72,
        200.73,
        200.74,
        200.75,
        200.76,
        200.77,
        200.78,
        200.8,
        200.81,
        200.82,
        200.83,
        200.84,
        200.85,
        200.86,
        200.87,
        200.88,
        201,
        201.01,
        201.02,
        201.03,
        201.04,
        201.05,
        201.06,
        201.07,
        201.08,
        201.1,
        201.11,
        201.12,
        201.13,
        201.14,
        201.15,
        201.16,
        201.17,
        201.18,
        201.2,
        201.21,
        201.22,
        201.23,
        201.24,
        201.25,
        201.26,
        201.27,
        201.28,
        201.4,
        201.41,
        201.42,
        201.43,
        201.44,
        201.45,
        201.46,
        201.47,
        201.48,
        201.5,
        201.51,
        201.52,
        201.53,
        201.54,
        201.55,
        201.56,
        201.57,
        201.58,
        201.6,
        201.61,
        201.62,
        201.63,
        201.64,
        201.65,
        201.66,
        201.67,
        201.68,
        201.7,
        201.71,
        201.72,
        201.73,
        201.74,
        201.75,
        201.76,
        201.77,
        201.78,
        201.9,
        201.91,
        201.92,
        201.93,
        201.94,
        201.95,
        201.96,
        201.97,
        201.98,
        202,
        202.01,
        202.02,
        202.03,
        202.04,
        202.05,
        202.06,
        202.07,
        202.08,
        202.1,
        202.11,
        202.12,
        202.13,
        202.14,
        202.15,
        202.16,
        202.17,
        202.18,
        202.2,
        202.21,
        202.22,
        202.23,
        202.24,
        202.25,
        202.26,
        202.27,
        202.28,
        202.3,
        202.31,
        202.32,
        202.33,
        202.34,
        202.35,
        202.36,
        202.37,
        202.38,
        202.4,
        202.41,
        202.42,
        202.43,
        202.44,
        202.45,
        202.46,
        202.47,
        202.48,
        202.5,
        202.51,
        202.52,
        202.53,
        202.54,
        202.55,
        202.56,
        202.57,
        202.58,
        202.6,
        202.61,
        202.62,
        202.63,
        202.64,
        202.65,
        202.66,
        202.67,
        202.68,
        202.7,
        202.71,
        202.72,
        202.73,
        202.74,
        202.75,
        202.76,
        202.77,
        202.78,
        202.8,
        202.81,
        202.82,
        202.83,
        202.84,
        202.85,
        202.86,
        202.87,
        202.88,
        202.9,
        202.91,
        202.92,
        202.93,
        202.94,
        202.95,
        202.96,
        202.97,
        202.98,
    ),
    "metastatic_solid_tumor": (
        197,
        197.1,
        197.2,
        197.3,
        197.4,
        197.5,
        197.6,
        197.7,
        197.8,
        198,
        198.1,
        198.2,
        198.3,
        198.4,
        198.5,
        198.6,
        198.7,
        198.8,
        198.81,
        198.82,
        198.89,
        199,
        199.1,
        199.2,
    ),
    "mild_liver_disease": (
        70.22,
        70.23,
        70.32,
        70.33,
        70.44,
        70.54,
        70.6,
        70.9,
        570,
        571,
        571.1,
        571.2,
        571.3,
        571.4,
        571.41,
        571.42,
        571.49,
        571.5,
        571.6,
        571.8,
        571.9,
        573.3,
        573.4,
        573.8,
        573.9,
    ),
    "moderate_to_severe_liver_disease": (
        456,
        456.1,
        456.2,
        572,
        572.1,
        572.2,
        572.3,
        572.4,
        572.8,
    ),
    "myocardial_infarction": (
        410,
        410.01,
        410.02,
        410.1,
        410.11,
        410.12,
        410.2,
        410.21,
        410.22,
        410.3,
        410.31,
        410.32,
        410.4,
        410.41,
        410.42,
        410.5,
        410.51,
        410.52,
        410.6,
        410.61,
        410.62,
        410.7,
        410.71,
        410.72,
        410.8,
        410.81,
        410.82,
        410.9,
        410.91,
        410.92,
        412,
    ),
    "peptic_ulcer_disease": (
        531,
        531.01,
        531.1,
        531.11,
        531.2,
        531.21,
        531.3,
        531.31,
        531.4,
        531.41,
        531.5,
        531.51,
        531.6,
        531.61,
        531.7,
        531.71,
        531.9,
        531.91,
        532,
        532.01,
        532.1,
        532.11,
        532.2,
        532.21,
        532.3,
        532.31,
        532.4,
        532.41,
        532.5,
        532.51,
        532.6,
        532.61,
        532.7,
        532.71,
        532.9,
        532.91,
        533,
        533.01,
        533.1,
        533.11,
        533.2,
        533.21,
        533.3,
        533.31,
        533.4,
        533.41,
        533.5,
        533.51,
        533.6,
        533.61,
        533.7,
        533.71,
        533.9,
        533.91,
        534,
        534.01,
        534.1,
        534.11,
        534.2,
        534.21,
        534.3,
        534.31,
        534.4,
        534.41,
        534.5,
        534.51,
        534.6,
        534.61,
        534.7,
        534.71,
        534.9,
        534.91,
    ),
    "peripheral_vascular_disease": (
        47.1,
        93.0,
        437.3,
        440,
        440.1,
        440.2,
        440.21,
        440.22,
        440.23,
        440.24,
        440.29,
        440.3,
        440.31,
        440.32,
        440.4,
        440.8,
        440.9,
        441,
        441.01,
        441.02,
        441.03,
        441.1,
        441.2,
        441.3,
        441.4,
        441.5,
        441.6,
        441.7,
        441.9,
        443,
        443.1,
        443.2,
        443.21,
        443.22,
        443.23,
        443.24,
        443.29,
        443.8,
        443.81,
        443.82,
        443.89,
        443.9,
        557.1,
        557.9,
        785.4,
    ),
    "renal_disease": (
        403.01,
        403.11,
        403.91,
        404.02,
        404.03,
        404.12,
        404.13,
        404.92,
        404.93,
        582,
        582.1,
        582.2,
        582.4,
        582.8,
        582.81,
        582.89,
        582.9,
        583,
        583.1,
        583.2,
        583.4,
        583.6,
        583.7,
        585,
        585.1,
        585.2,
        585.3,
        585.4,
        585.5,
        585.6,
        585.9,
        586,
        588,
        588.1,
        588.8,
        588.81,
        588.89,
        588.9,
    ),
    "rheumatic_disease": (
        446.5,
        710,
        710.1,
        710.2,
        710.3,
        710.4,
        714.0,
        714.1,
        714.2,
        714.81,
        725,
    ),
}
r"""
Many thanks to Table 1 from https://www.jstor.org/stable/3768193 and
http://www.icd9data.com/2015/Volume1/default.htm
from which the above tuples were built with the following BASH function:
fun () {
    # e.g. "Acute myocardial infarction"
    # $1="http://www.icd9data.com/2015/Volume1/390-459/410-414/410/default.htm"
    clear;
    curl -sL $1 \
    | egrep -o '>[0-9]*\.?[0-9]*</a>' \
    | sed 's/^>//;s/<\/a>/,/' \
    | sort -V \
    | uniq \
    | grep -v '^20[012][0-9]' \
    | tr -s '\n' ' ' \
    | sed 's/,\s*,/,/g';
    echo;
}
"""

InputOutputErrorQueues = namedtuple(
    "InputOutputErrorQueues", ["input_queue", "output_queue", "error_queue"]
)

LOGGING_FORMAT = "[{levelname:^8}][{filename:^16}]{message}"
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
