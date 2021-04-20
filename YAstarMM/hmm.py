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

GLOBAL_LOCK = Lock()
NEW_DF = None

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
    global NEW_DF, GLOBAL_LOCK
    GLOBAL_LOCK.acquire()
    if NEW_DF is None:
        NEW_DF = df.copy()
    else:
        NEW_DF = pd.concat([NEW_DF, df.copy()])
    GLOBAL_LOCK.release()

    return df.sort_values("date")


def dataframe_to_numpy_matrix(df, only_columns=None, normalize=False):
    """Many thanks to https://stackoverflow.com/a/41532180"""
    if only_columns is None:
        raise ValueError("only_columns argument should be an iterable")

    only_columns = [col for col in only_columns if col in df.columns]

    if normalize:
        return (
            df.loc[:, only_columns]
            .sub(
                [MIN_MAX_DICT[col]["min"] for col in only_columns],
                axis="columns",
            )
            .div(
                [
                    MIN_MAX_DICT[col]["max"] - MIN_MAX_DICT[col]["min"]
                    for col in only_columns
                ],
                axis="columns",
            )
            .to_numpy()
        )
    return df.loc[:, only_columns].to_numpy()


class MetaModel(object):
    @property
    def input_data(self):
        """Return { patient: { timestamp: State, ... }, ..., }"""
        if self._input_data_dict is None:
            self._input_data_dict = dict()
            for index, row in self._df.iterrows():
                patient, date = (
                    row[rename_helper("")],
                    row[rename_helper("")],
                )
                if pd.isna(date):
                    continue
                assert isinstance(date, pd.Timestamp), str(
                    f"Patient '{patient}' has '{repr(date)}' in column "
                    f"'{rename_helper('')}' instead of a pd.Timestamp"
                )

                date = pd.to_datetime(date.date())  # truncate HH:MM:SS
                state_val = row[rename_helper("ActualState_val")]
                assert isinstance(state_val, (int, float)), str(
                    f"Patient '{repr(patient)}' has {repr(state_val)} "
                    f"in columnn '{rename_helper('ActualState_val')}'"
                    " instead of a float or integer."
                )
                actual_state = State(state_val)
                if patient not in self._input_data_dict:
                    self._input_data_dict[patient] = dict()
                self._input_data_dict[patient][date] = max(
                    # worst state between the current one and any other
                    # already inserted for the same patient in the same
                    # date
                    self._input_data_dict[patient].get(
                        date,
                        actual_state,
                    ),
                    actual_state,
                )

            # ensure each day in between first and last patient timestamp
            # exist and has a state
            log_empty_line = False
            for patient in sorted(self._input_data_dict.keys()):
                if log_empty_line:
                    logging.info(
                        "[PREPROCESSING]"
                    )  # separate added timestamps of different users
                    log_empty_line = False

                records = self._input_data_dict[patient]
                for date in sorted(
                    pd.date_range(
                        start=min(records.keys()),
                        end=max(records.keys()),
                        freq="D",
                        normalize=True,
                    )
                    .to_series()
                    .tolist()
                ):
                    if date not in records.keys():
                        log_empty_line = True
                        prev_date = max(d for d in records.keys() if d < date)
                        assert (  # make black auto-formatting prettier
                            date - prev_date
                        ).total_seconds() == 24 * 60 ** 2, str(
                            f"Timedelta '{repr(date - prev_date)}' should "
                            "be a day"  # make black auto-formatting prettier
                        )
                        logging.info(
                            f"[PREPROCESSING]"
                            f"[patient{int(patient)}] added missing state "
                            f"({str(State(records[prev_date]))}) "
                            f"for {repr(date)}"
                        )
                        records[date] = records[prev_date]
        return self._input_data_dict

    @property
    def occurrences_matrix(self):
        rows = 1 + max(State)
        cols = rows
        ret = np.zeros(shape=(rows, cols), dtype=np.uint16)
        for patient, records in self.input_data.items():
            previous_date = None
            previous_state = State.No_O2  # i.e. the day before admission
            for current_date in sorted(  # enforce chronological order
                records.keys()
            ):  # make black auto-formatting prettier
                assert current_date != previous_date
                current_state = self.input_data[patient][current_date]

                if current_state not in self.oxygen_states:
                    continue

                if current_state != previous_state:
                    ret[previous_state][current_state] += 1
                else:
                    ret[current_state][current_state] += 1

                previous_date = current_date
                previous_state = current_state
        return ret

    @property
    def random_state(self):
        return self._random_state

    @property
    def start_prob(self):
        start_occurrences = np.zeros(
            shape=(1 + max(State),),
            dtype=np.uint16,  # 0 to 65535
        )
        for patient, records in self.input_data.items():
            for date in sorted(records.keys()):
                state = self.input_data[patient][date]

                if state not in self.oxygen_states:
                    continue

                start_occurrences[state] += 1
                break  # use just first state of each patient
        ret = np.array(start_occurrences, dtype=np.float64)
        return ret / ret.sum()

    @property
    def validation_matrix(self):
        if self._validation_df is None:
            self._split_dataset()
        return dataframe_to_numpy_matrix(
            self._validation_df,
            only_columns=list(rename_helper(("ActualState_val",)))
            + self.observed_variables,
        )

    @property
    def training_matrix(self):
        if self._training_df is None:
            self._split_dataset()
        return dataframe_to_numpy_matrix(
            self._training_df,
            only_columns=list(rename_helper(("ActualState_val",)))
            + self.observed_variables,
        )

    @property
    def transition_matrix(self):
        ret = np.array(self.occurrences_matrix, dtype=np.float64)
        for row, row_sum in enumerate(self.occurrences_matrix.sum(axis=1)):
            if row_sum > 0:
                ret[row] = ret[row] / float(row_sum)
            else:  # Avoid divisions by zero
                ret[row] = np.zeros(ret[row].shape, dtype=np.float64)
        return ret

    def __init__(
        self,
        df,
        oxygen_states=None,
        observed_variables=None,
        random_seed=None,
        save_to_dir=None,
    ):
        assert observed_variables is not None and isinstance(
            observed_variables,
            (list, tuple),
        ), str(
            "Expected list or tuple as 'observed_variables', "
            f"got '{repr(observed_variables)}' instead"
        )

        self._input_data_dict = None
        self._random_seed = random_seed
        self._random_state = RandomState(seed=self._random_seed)
        self._training_df = None
        self._validation_df = None
        self.observed_variables = rename_helper(tuple(observed_variables))
        if not oxygen_states:
            self.oxygen_states = State.values()
        else:
            self.oxygen_states = [
                getattr(State, state_name).value
                for state_name in oxygen_states
            ]

        reset_charlson_counter()
        self._df_old = df
        self._df_old = self._df_old.assign(
            **{
                rename_helper(""): self._df_old.loc[
                    :, rename_helper("")
                ].astype(
                    np.float64
                ),
                rename_helper(""): self._df_old.loc[
                    :, rename_helper("")
                ].apply(
                    lambda timestamp: pd.to_datetime(timestamp.date())
                ),  # truncate HH:MM:SS
            }
        )
        self._df_old.sort_values(rename_helper("")).groupby(
            "",
            as_index=False,
        ).apply(preprocess_single_patient_df, observed_variables)

        max_col_length = max_charlson_col_length()
        for charlson_col, count in most_common_charlson():
            logging.debug(
                f" {count:6d} patients had necessary data to choose "
                f"{charlson_col.rjust(max_col_length)} "
                "to compute Charlson-Index"
            )

        global GLOBAL_LOCK, NEW_DF
        GLOBAL_LOCK.acquire()
        self._df = NEW_DF[
            # cut away records about oxygen state not of interest
            NEW_DF[rename_helper("ActualState_val")].isin(self.oxygen_states)
        ].copy()
        NEW_DF = None
        GLOBAL_LOCK.release()

        show_final_hint = False
        for col, data in MIN_MAX_DICT.items():
            if col not in self._df.columns:
                continue
            logging.debug(
                f" Statistical description of column '{col}':\t"
                + repr(
                    self._df.loc[:, col]
                    .describe(percentiles=[0.03, 0.25, 0.50, 0.75, 0.97])
                    .to_dict()
                )
            )
            lower_outliers = self._df[  # make black auto-formatting prettier
                self._df[col] < MIN_MAX_DICT[col]["min"]
            ][col].count()
            upper_outliers = self._df[  # make black auto-formatting prettier
                self._df[col] > MIN_MAX_DICT[col]["max"]
            ][col].count()
            if lower_outliers > 0 or upper_outliers > 0:
                logging.debug(
                    f" Column '{col}' has {lower_outliers} values under "
                    f"the lower limit ({MIN_MAX_DICT[col]['min']}) and "
                    f"{upper_outliers} values above the upper limit "
                    f"({MIN_MAX_DICT[col]['max']}); these outliers will"
                    f" be clipped to the respective limits."
                )
                show_final_hint = True
        if show_final_hint:
            logging.debug(
                " To change the above lower/upper limits please "
                "consider the column percentiles in the debug log"
            )

        # force outlier values to lower/upper bounds of each column
        self._df.loc[
            :, [c for c in MIN_MAX_DICT.keys() if c in self._df.columns]
        ] = self._df.loc[
            :, [c for c in MIN_MAX_DICT.keys() if c in self._df.columns]
        ].clip(
            lower={
                col: data["min"]
                for col, data in MIN_MAX_DICT.items()
                if col in self._df.columns
            },
            upper={
                col: data["max"]
                for col, data in MIN_MAX_DICT.items()
                if col in self._df.columns
            },
            axis="columns",
        )

    def _split_dataset(self, ratio=None):
        """Split dataset into training set and validation set"""
        assert isinstance(ratio, float) and ratio > 0 and ratio < 1, str(
            "Validation set ratio (CLI argument --ratio) is not in (0, 1)"
        )
        logging.debug(
            " full dataset shape: "
            f"{self._df.shape[0]} rows, "
            f"{self._df.shape[1]} columns"
        )
        target_validation_rows = max(1, round(self._df.shape[0] * ratio))

        patients_left = [
            patient_id
            for patient_id, _ in Counter(
                self._df[rename_helper("")].to_list()
            ).most_common()  # make black auto-formatting prettier
        ]
        while (
            self._validation_df is None
            or self._validation_df.shape[0] < target_validation_rows
        ):
            assert len(patients_left) >= 1, "No patient left"
            patient_id = patients_left.pop(0)
            patient_df = self._df[
                self._df[rename_helper("")].isin([patient_id])
            ].copy()  # make black auto-formatting prettier
            if bool(self.random_state.randint(2)):  # toss a coin
                # try to add all the patient's records to the validation set
                if (
                    self._validation_df is None
                    or patient_df.shape[0] + self._validation_df.shape[0]
                    <= target_validation_rows
                ):
                    if self._validation_df is None:
                        self._validation_df = patient_df
                    else:
                        self._validation_df = pd.concat(
                            [self._validation_df, patient_df]
                        )
                    continue  # successfully added all patients records
            # try to add the last ratio of patient's records to the
            # validation set
            cut_row = round(patient_df.shape[0] * (1 - ratio))
            if self._validation_df is not None and (
                patient_df.shape[0]
                - cut_row  # validation records
                + self._validation_df.shape[0]
                > target_validation_rows
            ):
                cut_row = patient_df.shape[0]
                -(target_validation_rows - self._validation_df.shape[0]),
            if self._training_df is None:
                self._training_df = patient_df.iloc[:cut_row, :]
            else:
                self._training_df = pd.concat(
                    [self._training_df, patient_df.iloc[:cut_row, :]]
                )  # make black auto-formatting prettier
            if self._validation_df is None:
                self._validation_df = patient_df.iloc[cut_row:, :]
            else:
                self._validation_df = pd.concat(
                    [self._validation_df, patient_df.iloc[cut_row:, :]]
                )
        assert self._validation_df.shape[0] == target_validation_rows, str(
            f"validation matrix has {self._validation_df.shape[0]} "
            f"rows instead of {target_validation_rows}"
        )
        # add patients left to training set
        self._training_df = pd.concat(
            [
                self._training_df,
                self._df[
                    self._df[rename_helper("")].isin(patients_left)
                ].copy(),
            ]
        )
        assert (
            self._training_df.shape[0] + self._validation_df.shape[0]
            == self._df.shape[0]
        ), str(
            f"training matrix has {self._training_df.shape[0]} "
            "rows instead of "
            f"{self._df.shape[0] - self._validation_df.shape[0]}"
        )

        logging.debug(
            " training set shape: "
            f"{self._training_df.shape[0]} rows, "
            f"{self._training_df.shape[1]} columns"
        )
        logging.debug(
            " validation set shape: "
            f"{self._validation_df.shape[0]} rows, "
            f"{self._validation_df.shape[1]} columns"
        )

    def print_matrix(
        self,
        occurrences_matrix=False,
        transition_matrix=False,
        training_matrix=False,
        file_obj=stdout,
        #
        # style parameters
        #
        float_decimals=3,
        cell_pad=2,
        separators=True,
    ):
        assert (
            len(
                [
                    b
                    for b in (
                        occurrences_matrix,
                        transition_matrix,
                        training_matrix,
                    )
                    if bool(b)
                ]
            )
            == 1
        ), str(
            "Please set only one flag between "
            "{occurrences,transition,training}_matrix"
        )

        col_names = State.values()
        if occurrences_matrix:
            matrix = self.occurrences_matrix
            title = "Occurrences"
        elif transition_matrix:
            matrix = self.transition_matrix
            title = "Transition"
        elif training_matrix:
            matrix = self.training_matrix
            title = "Training"
        else:
            raise ValueError(
                "Please set only one flag between "
                "{occurrences,transition,training}_matrix"
            )
        title += " matrix (count_all)"

        legend_size = max(len(name) for name in State.names())
        cell_size = max(len(str(col)) for col in col_names)
        if isinstance(
            matrix[0][0],
            (
                int,
                np.uint16,
            ),
        ):
            cell_size = max(cell_size, len(str(np.max(matrix))))
        elif isinstance(
            matrix[0][0],
            (
                float,
                np.float64,
            ),
        ):
            cell_size = max(cell_size, float_decimals + 2)

        header = " " * (3 + 3) + "From / to".center(legend_size) + " " * 3
        header += "".join(  # make black auto-formatting prettier
            str(col).rjust(cell_size) + " " * cell_pad for col in col_names
        )
        if title:
            print(title, file=file_obj)
        print(header, file=file_obj)
        if separators:
            print("_" * len(header), file=file_obj)
        for row in State.values():
            print(
                f"{row:3d} = " + str(State(row)).center(legend_size),
                end=" | ",
                file=file_obj,
            )
            for col in range(len(col_names)):
                cell_value = matrix[row][col]
                if isinstance(
                    cell_value,
                    (
                        int,
                        np.uint16,
                    ),
                ):
                    cell_str = str(cell_value)
                elif isinstance(
                    cell_value,
                    (
                        float,
                        np.float64,
                    ),
                ):
                    if not np.isnan(cell_value):
                        assert cell_value <= 1 and cell_value >= 0, str(
                            "This function expects float values "
                            "to be in "
                            "[0, 1] like probabilities; "
                            f"got {cell_value:g} instead."
                        )
                    cell_str = "{:.16f}".format(  # right pad with many zeros
                        round(cell_value, float_decimals)
                    )
                    cell_str = cell_str[:cell_size]  # cut unneded "padding" 0
                else:
                    raise NotImplementedError(
                        f"Please add support to {type(cell_value)} cells"
                    )
                print(
                    cell_str.rjust(cell_size),
                    end=" " * cell_pad,
                    file=file_obj,
                )
            print("", file=file_obj)
        if separators:
            print("_" * len(header), file=file_obj)
        print(file=file_obj)

    def print_start_probability(
        self,
        float_decimals=3,
        cell_pad=2,
        file_obj=stdout,
    ):
        print("Start probability:  ", end="", file=file_obj)
        print(
            str(" " * cell_pad).join(
                str("{:.16f}".format(round(p, float_decimals)))[
                    : float_decimals + 2
                ]  # make black auto-formatting prettier
                for p in self.start_prob
            ),
            end="\n\n",
            file=file_obj,
        )

    def validation_matrix_labels(self, unordered_model_states):
        new_index_of_state = {
            str(state.name).replace(" ", "_"): i
            for i, state in enumerate(unordered_model_states)
        }

        for state_enum in self.oxygen_states:
            state_name = str(State(state_enum)).replace(" ", "_")
            assert state_name in new_index_of_state, str(
                f"Could not found any state named '{state_name}'."
                "\nWhen building the Hidden Markov Model, please "
                "pass to the 'state_names' argument what you "
                "passed to the 'oxygen_states' argument in MetaModel "
                "constructor; i.e. State.names() or a subset of it."
            )
            state = getattr(State, state_name)
            logging.info(
                "In the current model, state "
                + repr(str(state)).ljust(2 + max(len(s.name) for s in State))
                + f" has index {new_index_of_state[state.name]} "
                f"while its default enum value is {state.value}"
            )

        return np.array(
            [
                new_index_of_state[State(old_index).name]
                for old_index in self._validation_df.loc[
                    :, rename_helper("ActualState_val")
                ].to_list()  # make black auto-formatting prettier
            ]
        )

    def save_to(self, dir_name):
        dir_name = join_path(dir_name, "MetaModel_class")
        if not isdir(dir_name):
            makedirs(dir_name)

        with open(f"{dir_name}/input_data.pickle", "wb") as f:
            pickle.dump(self.input_data, f)

        for object_name in (
            "input_data",
            "_random_seed",
            "observed_variables",
            "oxygen_states",
        ):
            with open(f"{dir_name}/{object_name}.yaml", "w") as f:
                dump(
                    getattr(self, object_name),
                    f,
                    Dumper=SafeDumper if "data" not in object_name else Dumper,
                    default_flow_style=False,
                )
        with open(f"{dir_name}/oxygen_states.yaml", "a") as f:
            for state in self.oxygen_states:
                f.write(f"# State({state}).name == '{State(state).name}'\n")

        with open(f"{dir_name}/clip_out_outliers_dictionary.yaml", "w") as f:
            dump(MIN_MAX_DICT, f, Dumper=SafeDumper, default_flow_style=False)

        np.savetxt(
            f"{dir_name}/start_prob.txt",
            self.start_prob,
            fmt="%10.9f",
            header=repr([State(state).name for state in self.oxygen_states]),
        )
        np.save(
            f"{dir_name}/start_prob.npy",
            self.start_prob,
            allow_pickle=False,
        )

        np.save(
            f"{dir_name}/occurrences_matrix.npy",
            self.occurrences_matrix,
            allow_pickle=False,
        )
        with open(f"{dir_name}/occurrences_matrix.txt", "w") as f:
            self.print_matrix(
                occurrences_matrix=True,
                file_obj=f,
            )

        np.save(
            f"{dir_name}/transition_matrix.npy",
            self.transition_matrix,
            allow_pickle=False,
        )
        with open(f"{dir_name}/transition_matrix.txt", "w") as f:
            self.print_matrix(
                transition_matrix=True,
                file_obj=f,
                float_decimals=6,
            )

        for matrix_name in ("validation_matrix", "training_matrix"):
            np.savetxt(
                f"{dir_name}/{matrix_name}.txt",
                getattr(self, f"{matrix_name}"),
                fmt="%16.9e",
                header=str(
                    list(rename_helper(("ActualState_val",)))
                    + self.observed_variables
                ),
            )
            np.save(
                f"{dir_name}/{matrix_name}.npy",
                getattr(self, f"{matrix_name}"),
                allow_pickle=False,
            )

        for df_name in ("_df", "_validation_df", "_training_df"):
            getattr(self, df_name).to_csv(f"{dir_name}/{df_name}.csv")


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
