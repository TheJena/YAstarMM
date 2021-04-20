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
   TODO HERE
"""

from .column_rules import rename_helper
from .constants import (
    EPSILON,
    LOGGING_LEVEL,
    PER_STATE_TRANSITION_OBSERVABLES,
    REGEX_RANGE_OF_FLOATS,
    REGEX_UNDER_THRESHOLD_FLOAT,
from .charlson_index import (
    compute_charlson_index,
    estimated_ten_year_survival,
    max_charlson_col_length,
    most_common_charlson,
    reset_charlson_counter,
)
from .flavoured_parser import parsed_args

from .model import State
from .preprocessing import clear_and_refill_state_transition_columns
from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from numpy.random import RandomState
from os import makedirs
from os.path import join as join_path, isdir
from sys import stdout, version_info
from threading import Lock
from yaml import dump, Dumper, SafeDumper
import logging
import numpy as np
import pandas as pd
import pickle

TestResult = namedtuple("TestResult", ("success", "value", "msg"))
BAD_TEST_RESULT = TestResult(success=False, value=None, msg="")
NAN_TEST_RESULT = TestResult(
    success=True, value=np.nan, msg=str(np.nan).center(len(f"{0.0:e}"))
)

MIN_MAX_DICT = {
    rename_helper(k): v
    for k, v in {  # reference ranges in comments
        "": dict(min=0, max=150),
        "": dict(min=0, max=37),
        "": dict(min=0, max=15),  # 0.7 - 1.3 mg/dL
        "": dict(min=50, max=40000),  # 0 - 1700 g/mL
        "": dict(min=12, max=40),
        "": dict(min=5, max=255),  # <= 45 IU/L
        "": dict(min=0, max=1),  # just a boolean
        "": dict(min=50, max=1550),  # 50 - 150 U/L
        "": dict(min=0, max=100),  # % on total
        "": dict(min=0, max=50),  # 0 - 0.7 ???
        "": dict(min=0, max=10),  # 0 - 0.5 ng/mL
        "": dict(min=5, max=155),  # 15 - 55 mg/dL
        "": dict(min=20, max=80),  # 35 - 45 mmHg
        "": dict(min=6, max=8),  # ???
        "": dict(min=50, max=450),  #  ???
    }.items()
}
"""Built considering 0.03 and 0.97 percentile of the respective columns."""

def test_float(f):
    try:
        f = float(str(f).replace(",", "."))  # italians sometimes use comma
    except ValueError:
        return BAD_TEST_RESULT
    else:
        return TestResult(success=True, value=f, msg="")


def test_float_and_date(f):
    match_obj = REGEX_FLOAT_AND_DATE.match(repr(f))
    if match_obj is not None:
        if match_obj.group(1) is None:
            # regexp matched but the "float" group is empty, let's use a NaN
            return NAN_TEST_RESULT
        else:
            value = float(match_obj.group(1).replace(",", "."))
            return TestResult(success=True, value=value, msg=f"{value:e}")
    return BAD_TEST_RESULT


def test_float_range(f):
    match_obj = REGEX_RANGE_OF_FLOATS.match(repr(f))
    if match_obj is not None:
        value = 0.5 * sum(
            float(match_obj.group(i + 1).replace(",", "."))
            for i in range(2)  # starts from zero
        )
        return TestResult(success=True, value=value, msg=f"{value:e}")
    return BAD_TEST_RESULT


def test_float_under_threshold(f):
    match_obj = REGEX_UNDER_THRESHOLD_FLOAT.match(repr(f))
    if match_obj is not None:
        value = float(match_obj.group(1)) - EPSILON
        return TestResult(success=True, value=value, msg=f"{value:e}")
    return BAD_TEST_RESULT


def test_missing_value(f):
    if "missing" in repr(f).lower():
        return NAN_TEST_RESULT
    return BAD_TEST_RESULT


def float_or_nan(value, log_prefix=""):
    for func in (  # order does matter, do not change it please
        test_float,
        test_missing_value,
        test_float_under_threshold,
        test_float_range,
        lambda _: NAN_TEST_RESULT,
    ):
        test_result = func(value)
        if test_result.success:
            if test_result.msg:
                logging.debug(
                    f"{log_prefix}"  # make black auto-formatting prettier
                    f" Using {test_result.msg} instead of {repr(value)}"
                )
            return test_result.value
    raise NotImplementedError(
        "You probably forgot to include a function "
        "returning a default to use as fallback value"
    )


def aggregate_constant_values(sequence):
    sequence = set(s for s in sequence if pd.notna(s))
    assert (
        len(sequence) <= 1
    ), f"sequence {repr(sequence)} does not contain a constant value"
    return sequence.pop() if sequence else np.nan


def function_returning_worst_value_for(column):
    if column in rename_helper(
        (
        )
    ):  # a higher value means a worst patient health
        return np.max
    elif column in rename_helper(
        (
        )
    ):  # a lower value means a worst patient health
        return np.min
    elif column in rename_helper(
        (
        )
    ):  # both a higher or a lower value mean a worst patient health
        return np.mean
    elif column in rename_helper(
        (
        )
    ):  # all values should be the same, check it
        return aggregate_constant_values
    raise NotImplementedError(
        f"No aggregation function was found for column {repr(column)};"
        " please add it in the above switch-case"
    )


def preprocess_single_patient_df(df, observed_variables):
    assert (
        len(df.loc[:, rename_helper("")].sort_values().unique()) == 1
    ), str("This function should process one patient at a time")
    patient_id = int(
        set(df.loc[:, rename_helper("")].to_list()).pop()
    )
    log_prefix = "".join(
        (
            f"[{preprocess_single_patient_df.__name__}]",
            f"[patient{patient_id}]",
        )
    ).strip()

    # let's compute the charlson-index before dropping unobserved columns
    cci = compute_charlson_index(df)
    if pd.isna(cci):
        logging.debug(f"{log_prefix} Charlson-Index is not computable.")
    else:
        logging.debug(f"{log_prefix} Charlson-Index is {cci:2.0f}")

    # drop unobserved columns
    df = df.loc[
        :,
        list(
            rename_helper(
                (
                )
            )
        )
        + observed_variables,
    ]

    # ensure all columns (except '') contain float (or nan) values
    max_col_length = max(len(col) for col in df.columns)
    for col in set(df.columns).difference({""}):
        df.loc[:, col] = df.loc[:, col].apply(
            float_or_nan,
            log_prefix=f"{log_prefix}[column {col.rjust(max_col_length)}]",
        )

        # dates of interest are those with a valid ActualState_val
        dates_of_interest = df[df[rename_helper("ActualState_val")].notna()][
            "date"
        ]

    if len(dates_of_interest) < 2:
        return

    # ensure dates of interest are a proper range of dates, i.e. without holes
    dates_of_interest = pd.date_range(
        start=min(dates_of_interest),
        end=max(dates_of_interest),
        freq="D",
        normalize=True,
    )

    # add an empty record for each date of interest
    nan_series = pd.Series([np.nan for _ in range(len(dates_of_interest))])
    charlson_series = pd.Series([cci for _ in range(len(dates_of_interest))])
    df = pd.concat(
        [
            df,
            pd.DataFrame.from_dict(
                defaultdict(
                    lambda _: nan_series,  # default_factory for missing keys
                    {
                        rename_helper(""): dates_of_interest,
                    },
                )
            ),
        ]
    ).sort_values("date")

    # ensure each date has exactly one record; if there are multiple
    # values they will be aggregated by choosing the one which denotes
    # the worst health state
    df = df.groupby("date", as_index=False).aggregate(
        {
            col: function_returning_worst_value_for(col)
            for col in df.columns
            if col != "date"  # otherwise pandas complains
        }
    )

    # fill forward/backward some columns
    fill_forward_columns = rename_helper(
        (
        )
    )
    fill_backward_columns = rename_helper(())
    for col in fill_forward_columns:
        if col in df.columns:
            df.loc[:, col] = df.loc[:, col].fillna(method="ffill")
    for col in fill_backward_columns:
        if col in df.columns:
            df.loc[:, col] = df.loc[:, col].fillna(method="bfill")

    # drop records not in the date range of interest
    df = df.loc[
        (df["date")] >= min(dates_of_interest))
        & (df["date"] <= max(dates_of_interest)),
    ]

    return df.sort_values("date")


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__ in ("analisi.src.YAstarMM.hmm", "YAstarMM.hmm", "hmm"),
        # TODO HERE in globals(),
    )
), "Please update 'Usage' section of module docstring"
