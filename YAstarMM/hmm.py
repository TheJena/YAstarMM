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
   Build model from input.

   Usage:
            from  YAstarMM.hmm  import  MetaModel, run as run_hmm_training
"""

from .charlson_index import (
    compute_charlson_index,
    max_charlson_col_length,
    most_common_charlson,
    reset_charlson_counter,
)
from .column_rules import (
    minimum_maximum_column_limits,
    rename_helper,
)
from .constants import LOGGING_LEVEL, MIN_PYTHON_VERSION
from .flavoured_parser import parsed_args
from .model import State
from .preprocessing import clear_and_refill_state_transition_columns
from .utility import initialize_logging
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Lock
from numpy.random import RandomState
from os import makedirs
from os.path import join as join_path, isdir
from sys import stdout, version_info
from yaml import dump, Dumper, SafeDumper
import logging
import numpy as np
import pandas as pd
import pickle

_NEW_DF_LOCK = Lock()
_NEW_DF = None


def aggregate_constant_values(sequence):
    sequence = set(s for s in sequence if pd.notna(s))
    assert (
        len(sequence) <= 1
    ), f"sequence {repr(sequence)} does not contain a constant value"
    return sequence.pop() if sequence else np.nan


def function_returning_worst_value_for(column):
    if column in rename_helper(
        (
            "ActualState_val",
            "AGE",
            "CHARLSON-INDEX",
            "CREATININE",
            "D_DIMER",
            "RESPIRATORY_RATE",
            "GPT_ALT",
            "DYSPNEA",
            "LDH",
            "LYMPHOCYTE",
            "PROCALCITONIN",
            "Urea",
        )
    ):  # a higher value means a worst patient health
        return np.max
    elif column in rename_helper(
        (
            "PHOSPHOCREATINE",
            "HOROWITZ_INDEX",
        )
    ):  # a lower value means a worst patient health
        return np.min
    elif column in rename_helper(
        (
            "CARBON_DIOXIDE_PARTIAL_PRESSURE",
            "PH",
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


def preprocess_single_patient_df(
    df, observed_variables, hexadecimal_patient_id=False
):
    assert (
        len(df.loc[:, rename_helper("")].sort_values().unique()) == 1
    ), str("This function should process one patient at a time")
    patient_id = int(
        set(df.loc[:, rename_helper("")].to_list()).pop(),
        base=16 if hexadecimal_patient_id else 10,
    )
    log_prefix = f"[patient {patient_id}]"

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
            set(
                rename_helper(("", "DataRef", "ActualState_val"))
            ).union(set(rename_helper(tuple(observed_variables))))
        ),
    ]

    # add an empty record for each date of interest
    nan_series = pd.Series([np.nan for _ in range(df.shape[0])])
    charlson_series = pd.Series([cci for _ in range(df.shape[0])])
    df = pd.concat(
        [
            df,
            pd.DataFrame.from_dict(
                defaultdict(
                    lambda _: nan_series,  # default_factory for missing keys
                    {
                        rename_helper("CHARLSON-INDEX"): charlson_series,
                    },
                )
            ),
        ]
    ).sort_values(rename_helper("DataRef"))

    # ensure each date has exactly one record; if there are multiple
    # values they will be aggregated by choosing the one which denotes
    # the worst health state
    df = df.groupby(rename_helper("DataRef"), as_index=False).aggregate(
        {
            col: function_returning_worst_value_for(col)
            for col in df.columns
            if col != rename_helper("DataRef")  # otherwise pandas complains
        }
    )

    global _NEW_DF, _NEW_DF_LOCK
    _NEW_DF_LOCK.acquire()
    if _NEW_DF is None:
        _NEW_DF = df.copy()
    else:
        _NEW_DF = pd.concat([_NEW_DF, df.copy()])
    _NEW_DF_LOCK.release()

    return df.sort_values(rename_helper("DataRef"))


def dataframe_to_numpy_matrix(df, only_columns=None, normalize=False):
    """Many thanks to https://stackoverflow.com/a/41532180"""
    if only_columns is None:
        raise ValueError("only_columns argument should be an iterable")

    only_columns = [col for col in only_columns if col in df.columns]

    if normalize:
        return (
            df.loc[:, only_columns]
            .sub(
                [
                    minimum_maximum_column_limits()[col]["min"]
                    for col in only_columns
                ],
                axis="columns",
            )
            .div(
                [
                    minimum_maximum_column_limits()[col]["max"]
                    - minimum_maximum_column_limits()[col]["min"]
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
                    row[rename_helper("DataRef")],
                )
                if pd.isna(date):
                    continue
                assert isinstance(date, pd.Timestamp), str(
                    f"Patient '{patient}' has '{repr(date)}' in column "
                    f"'{rename_helper('DataRef')}' instead of a pd.Timestamp"
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
            only_columns=list(
                set(rename_helper(("ActualState_val",))).union(
                    set(rename_helper(tuple(self.observed_variables)))
                )
            ),
        )

    @property
    def training_matrix(self):
        if self._training_df is None:
            self._split_dataset()
        return dataframe_to_numpy_matrix(
            self._training_df,
            only_columns=list(
                set(rename_helper(("ActualState_val",))).union(
                    set(rename_helper(tuple(self.observed_variables)))
                )
            ),
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
        hexadecimal_patient_id=False,
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
                ].astype("string" if hexadecimal_patient_id else "Int64"),
                rename_helper("DataRef"): self._df_old.loc[
                    :, rename_helper("DataRef")
                ].apply(
                    lambda timestamp: pd.to_datetime(timestamp.date())
                ),  # truncate HH:MM:SS
            }
        )
        if hexadecimal_patient_id:
            logging.debug(
                f"Assuming key column '{rename_helper('')}'"
                " contains strings representing hexadecimal values"
            )
        self._df_old.sort_values(rename_helper("DataRef")).groupby(
            "",
            as_index=False,
        ).apply(
            preprocess_single_patient_df,
            observed_variables,
            hexadecimal_patient_id=hexadecimal_patient_id,
        )

        max_col_length = max_charlson_col_length()
        for charlson_col, count in most_common_charlson():
            logging.debug(
                f" {count:6d} patients had necessary data to choose "
                f"{charlson_col.rjust(max_col_length)} "
                "to compute Charlson-Index"
            )

        global _NEW_DF_LOCK, _NEW_DF
        _NEW_DF_LOCK.acquire()
        self._df = _NEW_DF[
            # cut away records about oxygen state not of interest
            _NEW_DF[rename_helper("ActualState_val")].isin(self.oxygen_states)
        ].copy()
        _NEW_DF = None
        _NEW_DF_LOCK.release()

        show_final_hint = False
        for col, data in minimum_maximum_column_limits().items():
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
                self._df[col] < minimum_maximum_column_limits()[col]["min"]
            ][col].count()
            upper_outliers = self._df[  # make black auto-formatting prettier
                self._df[col] > minimum_maximum_column_limits()[col]["max"]
            ][col].count()
            if lower_outliers > 0 or upper_outliers > 0:
                logging.debug(
                    f" Column '{col}' has {lower_outliers} values under "
                    "the lower limit "
                    f"({minimum_maximum_column_limits()[col]['min']}) and "
                    f"{upper_outliers} values above the upper limit "
                    f"({minimum_maximum_column_limits()[col]['max']}); these "
                    "outliers will be clipped to the respective limits."
                )
                show_final_hint = True
        if show_final_hint:
            logging.debug(
                " To change the above lower/upper limits please "
                "consider the column percentiles in the debug log"
            )

        # force outlier values to lower/upper bounds of each column
        self._df.loc[
            :,
            [
                c
                for c in minimum_maximum_column_limits().keys()
                if c in self._df.columns
            ],
        ] = self._df.loc[
            :,
            [
                c
                for c in minimum_maximum_column_limits().keys()
                if c in self._df.columns
            ],
        ].clip(
            lower={
                col: data["min"]
                for col, data in minimum_maximum_column_limits().items()
                if col in self._df.columns
            },
            upper={
                col: data["max"]
                for col, data in minimum_maximum_column_limits().items()
                if col in self._df.columns
            },
            axis="columns",
        )

        if save_to_dir is not None:
            self.save_to(save_to_dir)

    def _split_dataset(self, ratio=None):
        """Split dataset into training set and validation set"""
        if ratio is None:
            ratio = getattr(parsed_args(), "validation_set_ratio_hmm", 0.1)
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
            dump(
                minimum_maximum_column_limits(),
                f,
                Dumper=SafeDumper,
                default_flow_style=False,
            )

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
                    list(
                        set(rename_helper(("ActualState_val",))).union(
                            set(rename_helper(tuple(self.observed_variables)))
                        )
                    )
                ),
            )
            np.save(
                f"{dir_name}/{matrix_name}.npy",
                getattr(self, f"{matrix_name}"),
                allow_pickle=False,
            )

        for df_name in ("_df", "_validation_df", "_training_df"):
            getattr(self, df_name).to_csv(f"{dir_name}/{df_name}.csv")


def run():
    initialize_logging(getattr(parsed_args(), "log_level", LOGGING_LEVEL))
    assert getattr(parsed_args(), "save_dir") is not None

    if not isdir(getattr(parsed_args(), "save_dir")):
        makedirs(getattr(parsed_args(), "save_dir"))

    df = clear_and_refill_state_transition_columns(
        pd.read_excel(excel_file=parsed_args().input),
        patient_key_col=rename_helper(""),
        log_level=logging.CRITICAL,
        show_statistics=getattr(
            parsed_args(), "show_preprocessing_statistics", False
        ),  # make black auto-formatting prettier
        use_dumbydog=getattr(parsed_args(), "use_dumbydog", False),
        use_insomnia=getattr(parsed_args(), "use_insomnia", False),
    )

    if getattr(parsed_args(), "random_seed", None) is not None:
        seed = getattr(parsed_args(), "random_seed")
        logging.warning(f"Using manually set seed {seed}")

    mm1 = MetaModel(
        df,
        oxygen_states=getattr(
            parsed_args(), "oxygen_states", [state.name for state in State]
        ),
        observed_variables=getattr(parsed_args(), "observed_variables"),
        random_seed=seed,
        save_to_dir=getattr(parsed_args(), "save_dir"),
        hexadecimal_patient_id=str(
            pd.api.types.infer_dtype(df.loc[:, rename_helper("")])
        )
        == "string",
    )

    print("\n")
    mm1.print_start_probability()
    mm1.print_matrix(occurrences_matrix=True)
    mm1.print_matrix(transition_matrix=True)

    print("Validation matrix:\n" + repr(mm1.validation_matrix))
    print("Training matrix:\n" + repr(mm1.training_matrix))


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.hmm",
    "YAstarMM.hmm",
    "hmm",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
