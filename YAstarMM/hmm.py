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
from .constants import LOGGING_LEVEL, HmmTrainerInput, MIN_PYTHON_VERSION
from .flavoured_parser import parsed_args
from .model import State
from .parallel import _join_all
from .preprocessing import clear_and_refill_state_transition_columns
from .utility import initialize_logging
from collections import Counter
from multiprocessing import cpu_count, Lock, Process, Queue
from numpy.random import RandomState
from os import makedirs
from os.path import abspath, join as join_path, isdir
from pomegranate import (
    HiddenMarkovModel,
    NormalDistribution,
)
from pomegranate.callbacks import CSVLogger
from sys import version_info
from yaml import dump, Dumper, SafeDumper
import logging
import numpy as np
import pandas as pd
import pickle
import queue
import signal

_NEW_DF = None
_NEW_DF_LOCK = Lock()
_SIGNAL_QUEUE = Queue()
_WORKERS_POOL = list()


def _hmm_trainer(hti, **kwargs):
    worker_id, num_workers = hti.worker_id, hti.num_workers
    assert worker_id >= 0, repr(worker_id)
    assert num_workers > worker_id, repr(num_workers, worker_id)

    seed = worker_id
    if getattr(parsed_args(), "random_seed", None) is not None:
        seed = getattr(parsed_args(), "random_seed")
        logging.warning(f"Using manually set seed {seed}")

    for i in range(300):  # more or less a day
        light_mm = LightMetaModel(hti.df, random_seed=seed, **kwargs)

        light_mm.print_start_probability()
        light_mm.print_matrix(occurrences_matrix=True)
        light_mm.print_matrix(transition_matrix=True)

        hmm_kwargs = light_mm.hidden_markov_model_kwargs
        hmm_save_dir = light_mm.hidden_markov_model_dir
        light_mm.save_to_disk()

        hmm = HiddenMarkovModel.from_samples(
            X=light_mm.training_matrix,
            callbacks=[CSVLogger(join_path(hmm_save_dir, "training_log.csv"))],
            distribution=NormalDistribution,
            n_jobs=1,  # already using multiprocessing
            random_state=light_mm.random_state,
            verbose=getattr(parsed_args(), "log_level", LOGGING_LEVEL)
            <= logging.INFO,
            **hmm_kwargs,
        )

        save_hidden_markov_model(
            dir_name=hmm_save_dir,
            hmm=hmm,
            hmm_kwargs=hmm_kwargs,
            validation_matrix=light_mm.validation_matrix,
            validation_labels=light_mm.validation_matrix_labels(hmm.states),
            logger=light_mm,
        )

        if getattr(parsed_args(), "random_seed", None) is None:
            try:
                if hti.signal_queue.get(timeout=0.5) is None:
                    logging.info(f"Worker {worker_id} acked to SIGTERM")
                    break
            except queue.Empty:
                seed += num_workers
        else:
            logging.warning(
                "Setting manually the seed forces exit "
                "after the first model training"
            )
            break
    logging.info(f"Worker {worker_id} is ready to rest in peace")


def _sig_handler(sig_num, stack_frame):
    global _SIGNAL_QUEUE, _WORKERS_POOL
    for w in _WORKERS_POOL:
        logging.info(
            f"Received signal {sig_num}; sending SIGTERM to worker {w.name}"
        )
        _SIGNAL_QUEUE.put(None)


def aggregate_constant_values(sequence):
    sequence = set(s for s in sequence if pd.notna(s))
    assert (
        len(sequence) <= 1
    ), f"sequence {repr(sequence)} does not contain a constant value"
    return sequence.pop() if sequence else np.nan


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


def function_returning_worst_value_for(column, patient_key_col):
    if column in rename_helper(
        (
            "ActualState_val",
            "CREATININE",
            "D_DIMER",
            "RESPIRATORY_RATE",
            "GPT_ALT",
            "DYSPNEA",
            "LDH",
            "LYMPHOCYTE",
            "PROCALCITONIN",
            "UREA",
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
    elif column in (
        patient_key_col,
        rename_helper("AGE"),
        rename_helper("CHARLSON_INDEX"),
    ):  # all values should be the same, check it
        return aggregate_constant_values
    raise NotImplementedError(
        f"No aggregation function was found for column {repr(column)};"
        " please add it in the above switch-case"
    )


def preprocess_single_patient_df(
    df,
    patient_key_col,
    observed_variables,
    hexadecimal_patient_id=False,
    logger=logging,
):
    assert len(df.loc[:, patient_key_col].sort_values().unique()) == 1, str(
        "This function should process one patient at a time"
    )
    patient_id = int(
        set(df.loc[:, patient_key_col].to_list()).pop(),
        base=16 if hexadecimal_patient_id else 10,
    )
    log_prefix = str(
        "[patient "
        + str(
            f"{patient_id:X}" if hexadecimal_patient_id else f"{patient_id:d}"
        )
        + "]"
    )

    if any(
        (
            rename_helper("CHARLSON_INDEX") in observed_variables,
            "CHARLSON_INDEX" in observed_variables,
        )
    ):
        # let's compute the charlson-index before dropping unobserved columns
        logger.debug(f"{log_prefix}")
        cci = compute_charlson_index(df, logger=logger, log_prefix=log_prefix)
        logger.debug(
            f"{log_prefix} Charlson-Index is "
            + str(f"= {cci:2.0f}" if pd.notna(cci) else "not computable")
            + "\n"
        )
        # assign updated Charlson-Index
        df.loc[:, rename_helper("CHARLSON_INDEX")] = (
            float(cci) if pd.notna(cci) else np.nan
        )

    # drop unobserved columns
    df = df.loc[
        :,
        [patient_key_col]
        + list(
            set(rename_helper(("DataRef", "ActualState_val")))
            .union(set(rename_helper(tuple(observed_variables))))
            .difference({patient_key_col})
        ),
    ].sort_values(rename_helper("DataRef"))

    # ensure each date has exactly one record; if there are multiple
    # values they will be aggregated by choosing the one which denotes
    # the worst health state
    df = df.groupby(rename_helper("DataRef"), as_index=False).aggregate(
        {
            col: function_returning_worst_value_for(col, patient_key_col)
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


def run():
    assert getattr(parsed_args(), "max_workers") > 0
    assert getattr(parsed_args(), "save_dir") is not None

    initialize_logging(getattr(parsed_args(), "log_level", LOGGING_LEVEL))

    patient_key_col = rename_helper(
        getattr(parsed_args(), "patient_key_col"), errors="warn"
    )
    df = clear_and_refill_state_transition_columns(
        parsed_args().input.name
        if parsed_args().input.name.endswith(".xlsx")
        else parsed_args().input,
        patient_key_col=patient_key_col,
        log_level=logging.CRITICAL,
        show_statistics=getattr(
            parsed_args(), "show_preprocessing_statistics", False
        ),
        use_dumbydog=getattr(parsed_args(), "use_dumbydog", False),
        use_insomnia=getattr(parsed_args(), "use_insomnia", False),
    )

    # finish dataframe preprocessing and split training/test set
    heavy_mm = MetaModel(
        df,
        patient_key_col=patient_key_col,
        observed_variables=getattr(parsed_args(), "observed_variables"),
        oxygen_states=getattr(
            parsed_args(), "oxygen_states", [state.name for state in State]
        ),
        save_to_dir=getattr(parsed_args(), "save_dir"),
        hexadecimal_patient_id=str(
            pd.api.types.infer_dtype(df.loc[:, patient_key_col])
        )
        == "string",
    )

    light_mm_df = heavy_mm.light_meta_model_df
    light_mm_kwargs = heavy_mm.light_meta_model_kwargs
    heavy_mm.save_to_disk()

    num_workers = min(
        getattr(parsed_args(), "max_workers"),
        1
        if getattr(parsed_args(), "random_seed", None) is not None
        else cpu_count(),
    )

    global _SIGNAL_QUEUE, _WORKERS_POOL
    for sig_num in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig_num, _sig_handler)
    for i in range(num_workers):
        w = Process(
            target=_hmm_trainer,
            name=str(i),
            args=(
                HmmTrainerInput(
                    signal_queue=_SIGNAL_QUEUE,
                    df=light_mm_df,
                    worker_id=i,
                    num_workers=num_workers,
                ),
            ),
            kwargs=light_mm_kwargs,
        )
        _WORKERS_POOL.append(w)
        w.start()
    while not _SIGNAL_QUEUE.empty():
        _SIGNAL_QUEUE.get()
    _join_all(_WORKERS_POOL, "Workers in charge of HMMs training")


def save_hidden_markov_model(
    dir_name,
    hmm,
    hmm_kwargs,
    validation_matrix,
    validation_labels,
    logger=logging,
    pad=32,
):
    logger.info(
        f"Saving {type(hmm).__name__.ljust(pad-1)} to {abspath(dir_name)}"
    )

    with open(
        join_path(dir_name, "hmm_constructor_parameters.yaml"), "w"
    ) as f:
        dump(hmm_kwargs, f, Dumper=SafeDumper, default_flow_style=False)
        logger.info(f"Saved {'hmm_kwargs'.rjust(pad)} to {f.name}")

    for k, v in dict(
        log_probability=hmm.log_probability(validation_matrix),
        predict=hmm.predict(validation_matrix, algorithm="map"),
        score=float(hmm.score(validation_matrix, validation_labels)),
    ).items():
        logger.info(f"{k}: {repr(v)}")
        with open(join_path(dir_name, f"{k}.yaml"), "w") as f:
            dump(v, f, Dumper=SafeDumper, default_flow_style=False)
            logger.info(f"Saved {k.rjust(pad)} to {f.name}")

    for k, v in dict(
        predict_proba=hmm.predict_proba(validation_matrix),
        predict_log_proba=hmm.predict_log_proba(validation_matrix),
        dense_transition_matrix=hmm.dense_transition_matrix(),
    ).items():
        np.savetxt(join_path(dir_name, f"{k}.txt"), v, fmt="%16.9e")
        np.save(join_path(dir_name, f"{k}.npy"), v, allow_pickle=False)
        logger.info(
            f"Saved {k.rjust(pad)} to {str(join_path(dir_name, k))}"
            + ".{txt,npy}"
        )

    with open(
        join_path(dir_name, "hmm_trained_and_serialized.json"), "w"
    ) as f:
        f.write(hmm.to_json())  # logged with the next one

    with open(
        join_path(dir_name, "hmm_trained_and_serialized.yaml"), "w"
    ) as f:
        f.write(hmm.to_yaml())
        logger.info(
            f"Saved {str('trained ' + HiddenMarkovModel.__name__).rjust(pad)}"
            f" to {'.'.join(f.name.split('.')[:-1])}"
            + ".{json,"
            + f.name.split(".")[-1]
            + "}"
        )


class MetaModel(object):
    @property
    def input_data(self):
        """Return { patient: { timestamp: State, ... }, ..., }"""
        if self._input_data_dict is None:
            self._input_data_dict = dict()
            for index, row in self._df.iterrows():
                patient, date = (
                    row[self.patient_id],
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
                    self.info(
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
                        assert (
                            date - prev_date
                        ).total_seconds() == 24 * 60 ** 2, str(
                            f"Timedelta '{repr(date - prev_date)}' should "
                            "be a day"  # make black auto-formatting prettier
                        )
                        self.info(
                            f"[PREPROCESSING]"
                            f"[patient{int(patient)}] added missing state "
                            f"({str(State(records[prev_date]))}) "
                            f"for {repr(date)}"
                        )
                        records[date] = records[prev_date]
        return self._input_data_dict

    @property
    def light_meta_model_df(self):
        if self._training_df is None:
            self._split_dataset()
        return self._training_df.copy(deep=True)

    @property
    def light_meta_model_kwargs(self):
        return dict(
            hexadecimal_patient_id=None,
            observed_variables=self.observed_variables,
            oxygen_states=[
                State(state_num).name for state_num in self.oxygen_states
            ],
            patient_key_col=self.patient_id,
            # random_seed will be taken from parsed_args()
            ratio=self.fixed_validation_set_ratio(),
            save_to_dir=self.worker_dir,
            # skip_preprocessing will be set by LightMetaModel.__init__()
        )

    @property
    def logger_prefix(self):
        assert self._random_seed >= 0 and self._random_seed <= 999999
        return f"[seed {str(self._random_seed).rjust(6, '0')}]"

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
            ):
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
    def patient_id(self):
        assert any(
            (
                hasattr(self, "_df_old")
                and self._patient_key_col in self._df_old,
                hasattr(self, "_df") and self._patient_key_col in self._df,
            )
        )
        return self._patient_key_col

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
    def test_matrix(self):
        if isinstance(self, LightMetaModel):
            raise AttributeError(
                f"Property available only in {MetaModel.__name__}"
            )
        if self._test_df is None:
            self._split_dataset()
        return dataframe_to_numpy_matrix(
            self._test_df,
            only_columns=list(
                set(rename_helper(("ActualState_val",))).union(
                    set((self.observed_variables))
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
                    set(self.observed_variables)
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

    @property
    def validation_matrix(self):
        if not isinstance(self, LightMetaModel):
            raise AttributeError(
                f"Property available only in {LightMetaModel.__name__}"
            )
        if self._validation_df is None:
            self._split_dataset()
        return dataframe_to_numpy_matrix(
            self._validation_df,
            only_columns=list(
                set(rename_helper(("ActualState_val",))).union(
                    set((self.observed_variables))
                )
            ),
        )

    @property
    def worker_dir(self):
        if self._save_to_dir is None:
            return None
        ret, suffix = self._save_to_dir, f"{MetaModel.__name__}_class"
        if not ret.endswith(suffix):
            ret = join_path(ret, f"seed_{self._random_seed:0>6d}", suffix)
        if not isdir(ret):
            makedirs(ret)
        return ret

    def __init__(
        self,
        df,
        patient_key_col,
        observed_variables=None,
        oxygen_states=None,
        random_seed=None,
        ratio=None,
        save_to_dir=None,
        skip_preprocessing=False,
        hexadecimal_patient_id=False,
    ):
        self._input_data_dict = None
        self._patient_key_col = patient_key_col
        if random_seed is None:
            self._random_seed = RandomState(seed=None).randint(0, 999999)
        else:
            self._random_seed = int(random_seed)
        self._random_state = RandomState(seed=self._random_seed)
        self._ratio = ratio
        self._save_to_dir = save_to_dir
        self._test_df = None
        self._training_df = None
        self._validation_df = None

        self.info(f"Creating {type(self).__name__}")

        assert observed_variables is not None and isinstance(
            observed_variables,
            (list, tuple),
        ), str(
            "Expected list or tuple as 'observed_variables', "
            f"got '{repr(observed_variables)}' instead"
        )
        self.observed_variables = sorted(
            rename_helper(tuple(observed_variables), errors="quiet"),
            key=str.lower,
        )
        self.debug(
            f"observed_variables: {repr(self.observed_variables)[1:-1]}."
        )

        if oxygen_states is None or not oxygen_states:
            self.oxygen_states = State.values()
        else:
            self.oxygen_states = [
                getattr(State, state_name).value
                for state_name in oxygen_states
            ]
        self.debug(f"oxygen_states: {repr(self.oxygen_states)[1:-1]}.")

        if skip_preprocessing:
            self.info("Skipping preprocessing")
            self._old_df = None
            self._df = df
        else:
            self.info(
                "Starting preprocessing ("
                "\n\taggregate multiple values with the worst one, "
                "\n\tclip columns to min/max limits, "
                "\n\tdrop unobserved columns, "
                "\n\tre-compute Charlson-Index, "
                "\n\tetc.\n)"
            )
            reset_charlson_counter()
            self._df_old = df
            if hexadecimal_patient_id:
                self.debug(
                    f"Assuming key column '{self.patient_id}'"
                    " contains strings representing hexadecimal values"
                )
            self._df_old = self._df_old.assign(
                **{
                    self.patient_id: self._df_old.loc[
                        :, self.patient_id
                    ].astype("string" if hexadecimal_patient_id else "Int64"),
                    rename_helper("DataRef"): self._df_old.loc[
                        :, rename_helper("DataRef")
                    ].dt.normalize(),  # truncate HH:MM:SS
                }
            )
            self._df_old.sort_values(rename_helper("DataRef")).groupby(
                self.patient_id, as_index=False
            ).apply(  # TODO parallelize this bottleneck
                preprocess_single_patient_df,
                patient_key_col=self.patient_id,
                observed_variables=observed_variables,
                hexadecimal_patient_id=hexadecimal_patient_id,
                logger=self,
            )

            max_col_length = max_charlson_col_length()
            for charlson_col, count in most_common_charlson():
                self.debug(
                    f"{count:6d} patients had necessary data to choose "
                    f"{charlson_col.rjust(max_col_length)}"
                    " to compute Charlson-Index"
                )

            global _NEW_DF_LOCK, _NEW_DF
            _NEW_DF_LOCK.acquire()
            self._df = _NEW_DF[
                # cut away records about oxygen state not of interest
                _NEW_DF[rename_helper("ActualState_val")].isin(
                    self.oxygen_states
                )
            ].copy()
            _NEW_DF = None
            _NEW_DF_LOCK.release()

            show_final_hint = False
            for col, data in minimum_maximum_column_limits().items():
                if col not in self._df.columns:
                    continue
                self.debug(
                    f"Statistical description of column '{col}':\n\t"
                    + repr(
                        self._df.loc[:, col]
                        .describe(percentiles=[0.03, 0.25, 0.50, 0.75, 0.97])
                        .to_dict()
                    ).replace(" 'min'", "\n\t 'min'")
                )
                lower_outliers = self._df[
                    self._df[col] < minimum_maximum_column_limits()[col]["min"]
                ][col].count()
                upper_outliers = self._df[
                    self._df[col] > minimum_maximum_column_limits()[col]["max"]
                ][col].count()
                if lower_outliers > 0 or upper_outliers > 0:
                    self.warning(
                        f"Column '{col}' has:"
                        + str(
                            f"\n\t{lower_outliers} values < "
                            f"{minimum_maximum_column_limits()[col]['min']}"
                            if lower_outliers > 0
                            else ""
                        )
                        + str(
                            f"\n\t{upper_outliers} values > "
                            f"{minimum_maximum_column_limits()[col]['max']}"
                            if upper_outliers > 0
                            else ""
                        )
                        + "\n\tthese outliers will be clipped "
                        "to the respective limits."
                    )
                    show_final_hint = True
            if show_final_hint:
                self.warning(
                    "To change the above lower/upper limits please "
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
            self.info("Ended preprocessing")

        self._split_dataset()

    def _log(self, level, msg):
        msg = str(
            f"{self.logger_prefix}"
            f"{' ' if not msg.startswith('[') else ''}"
            f"{msg}"
        )
        logging.log(level, msg)

    def _split_dataset(self):
        """Split dataset into training set and validation (or test) set"""
        if self._ratio is None:
            if isinstance(self, LightMetaModel):
                self._ratio = getattr(
                    parsed_args(), "validation_set_ratio_hmm", 0.1
                )
                self.debug(f"Validation set ratio is: {self._ratio:.3f}")
            else:
                self._ratio = getattr(
                    parsed_args(), "test_set_ratio_composer", 0.1
                )
                self.debug(f"Test set ratio is: {self._ratio:.3f}")
        assert (
            isinstance(self._ratio, float)
            and self._ratio > 0
            and self._ratio < 1
        ), str(
            "Ratio (CLI argument --ratio-{validation,test}-set) "
            "is not in (0, 1)"
        )
        self.debug(f"Splitting with ratio {self._ratio:.6f}")
        self.debug(
            "full dataset shape: "
            f"{self._df.shape[0]} rows, "
            f"{self._df.shape[1]} columns"
        )
        target_df = None
        target_rows = max(1, round(self._df.shape[0] * self._ratio))

        patients_left = [
            patient_id
            for patient_id, _ in Counter(
                self._df[self.patient_id].to_list()
            ).most_common()
        ]
        while target_df is None or target_df.shape[0] < target_rows:
            assert len(patients_left) >= 1, "No patient left"
            patient_id = patients_left.pop(0)
            patient_df = self._df[
                self._df[self.patient_id].isin([patient_id])
            ].copy()
            if bool(self.random_state.randint(2)):  # toss a coin
                # try to add all the patient's records to the
                # validation/test set
                if (
                    target_df is None
                    or patient_df.shape[0] + target_df.shape[0] <= target_rows
                ):
                    if target_df is None:
                        target_df = patient_df
                    else:
                        target_df = pd.concat([target_df, patient_df])
                    continue  # successfully added all patients records
            # try to add the last ratio of patient's records to the
            # validation/test set
            cut_row = round(patient_df.shape[0] * (1 - self._ratio))
            if target_df is not None and (
                patient_df.shape[0]
                - cut_row  # validation/test records
                + target_df.shape[0]
                > target_rows
            ):
                cut_row = patient_df.shape[0]
                -(target_rows - target_df.shape[0]),
            if self._training_df is None:
                self._training_df = patient_df.iloc[:cut_row, :]
            else:
                self._training_df = pd.concat(
                    [self._training_df, patient_df.iloc[:cut_row, :]]
                )
            if target_df is None:
                target_df = patient_df.iloc[cut_row:, :]
            else:
                target_df = pd.concat(
                    [target_df, patient_df.iloc[cut_row:, :]]
                )
        assert target_df.shape[0] == target_rows, str(
            f"{'validation' if isinstance(self, LightMetaModel) else 'test'} "
            f"matrix has {target_df.shape[0]} "
            f"rows instead of {target_rows}"
        )
        # add patients left to training set
        self._training_df = pd.concat(
            [
                self._training_df,
                self._df[self._df[self.patient_id].isin(patients_left)].copy(),
            ]
        )
        assert (
            self._training_df.shape[0] + target_df.shape[0]
            == self._df.shape[0]
        ), str(
            f"training matrix has {self._training_df.shape[0]} "
            "rows instead of "
            f"{self._df.shape[0] - target_df.shape[0]}"
        )

        self.debug(
            "training set shape: "
            f"{self._training_df.shape[0]} rows, "
            f"{self._training_df.shape[1]} columns"
        )
        self.debug(
            f"{'validation' if isinstance(self, LightMetaModel) else 'test'} "
            "set shape: "
            f"{target_df.shape[0]} rows, "
            f"{target_df.shape[1]} columns"
        )

        if isinstance(self, LightMetaModel):
            self._validation_df = target_df
        else:
            self._test_df = target_df

    def _state_name_to_index_mapping(self, unordered_model_states):
        ret = {
            str(state.name).replace(" ", "_"): i
            for i, state in enumerate(unordered_model_states)
        }

        for state_enum in self.oxygen_states:
            state_name = str(State(state_enum)).replace(" ", "_")
            assert state_name in ret, str(
                f"Could not found any state named '{state_name}'."
                "\nWhen building the Hidden Markov Model, please "
                "pass to the 'state_names' argument what you "
                "passed to the 'oxygen_states' argument in MetaModel "
                "constructor; i.e. State.names() or a subset of it."
            )
            state = getattr(State, state_name)
            self.info(
                "In the current model, state "
                + repr(str(state)).ljust(2 + max(len(s.name) for s in State))
                + f" has index {ret[state.name]} "
                f"while its default enum value is {state.value}"
            )
        return ret

    def debug(self, msg=""):
        """Log debug message."""
        self._log(logging.DEBUG, msg)

    def fixed_validation_set_ratio(self):
        """test_records : test_ratio = validation_records : validation_ratio"""
        test_records = self._test_df.shape[0]
        validation_records = (
            test_records
            * getattr(parsed_args(), "validation_set_ratio_hmm")
            / getattr(parsed_args(), "test_set_ratio_composer")
        )
        new_validation_ratio = validation_records / self._training_df.shape[0]
        self.debug(
            f"{validation_records} records will be put in validation set;"
            " changing ratio accordingly ("
            f"{getattr(parsed_args(), 'validation_set_ratio_hmm')}"
            f"~> {new_validation_ratio})"
        )
        return new_validation_ratio

    def info(self, msg=""):
        """Log info message."""
        self._log(logging.INFO, msg)

    def print_matrix(
        self,
        occurrences_matrix=False,
        transition_matrix=False,
        training_matrix=False,
        file_obj=None,
        #
        # style parameters
        #
        cell_pad=2,
        float_decimals=3,
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
        print_buffer = str()

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
        header += "".join(
            str(col).rjust(cell_size) + " " * cell_pad for col in col_names
        )
        if title:
            print_buffer += f"{title}\n"
        print_buffer += f"{header}\n"
        if separators:
            print_buffer += f"{'_' * len(header)}\n"
        for row in State.values():
            print_buffer += str(
                f"{row:3d} = " + str(State(row)).center(legend_size)
            )
            print_buffer += " | "
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
                print_buffer += f"{cell_str.rjust(cell_size)}{' ' * cell_pad}"
            print_buffer += "\n"
        if separators:
            print_buffer += f"{'_' * len(header)}\n"
        print_buffer += "\n"

        if file_obj is not None:
            print(print_buffer, file=file_obj)
        else:
            for line in print_buffer.split("\n"):
                self.info(line)

    def print_start_probability(
        self, float_decimals=3, cell_pad=2, file_obj=None
    ):
        print_buffer = "\n\nStart probability:  "
        print_buffer += str(" " * cell_pad).join(
            str("{:.16f}".format(round(p, float_decimals)))[
                : float_decimals + 2
            ]
            for p in self.start_prob
        )
        print_buffer += "\n"

        if file_obj is not None:
            print(print_buffer, file=file_obj)
        else:
            for line in print_buffer.split("\n"):
                self.info(line)

    def save_to_disk(self, dir_name=None, pad=32):
        if dir_name is None:
            dir_name = self.worker_dir
        if dir_name is None:
            self.warning(
                f"Could not save {type(self).__name__} (None destination)"
            )
            return
        if not isdir(dir_name):
            makedirs(dir_name)
        self.info(
            f"Saving {type(self).__name__.ljust(pad-1)} to "
            f"{abspath(dir_name)}"
        )

        with open(join_path(dir_name, "input_data.pickle"), "wb") as f:
            pickle.dump(self.input_data, f)
            self.info(
                f"Saved {'input_data'.rjust(pad)} to "
                f"{'.'.join(f.name.split('.')[:-1])}"
                + ".{yaml,"
                + f.name.split(".")[-1]
                + "}"
            )

        for object_name in (
            "input_data",
            "_random_seed",
            "observed_variables",
            "oxygen_states",
        ):
            with open(join_path(dir_name, f"{object_name}.yaml"), "w") as f:
                dump(
                    getattr(self, object_name),
                    f,
                    Dumper=SafeDumper if "data" not in object_name else Dumper,
                    default_flow_style=False,
                )
                if object_name != "input_data":  # logged with .pickle
                    self.info(f"Saved {object_name.rjust(pad)} to {f.name}")
        with open(join_path(dir_name, "oxygen_states.yaml"), "a") as f:
            for state in self.oxygen_states:
                f.write(f"# State({state}).name == '{State(state).name}'\n")

        with open(
            join_path(dir_name, "clip_out_outliers_dictionary.yaml"), "w"
        ) as f:
            dump(
                minimum_maximum_column_limits(),
                f,
                Dumper=SafeDumper,
                default_flow_style=False,
            )
            self.info(
                f"Saved {'min/max column limits dictionary'.rjust(pad)} "
                f"to {f.name}"
            )

        if isinstance(self, LightMetaModel):
            np.savetxt(
                join_path(dir_name, "start_prob.txt"),
                self.start_prob,
                fmt="%10.9f",
                header=repr(
                    [State(state).name for state in self.oxygen_states]
                ),
            )
            np.save(
                join_path(dir_name, "start_prob.npy"),
                self.start_prob,
                allow_pickle=False,
            )
            self.info(
                f"Saved {'start_prob'.rjust(pad)} to "
                f"{str(join_path(dir_name,'start_prob'))}" + ".{txt,npy}"
            )

            np.save(
                join_path(dir_name, "occurrences_matrix.npy"),
                self.occurrences_matrix,
                allow_pickle=False,
            )
            with open(join_path(dir_name, "occurrences_matrix.txt"), "w") as f:
                self.print_matrix(
                    occurrences_matrix=True,
                    file_obj=f,
                )
            self.info(
                f"Saved {'occurrences_matrix'.rjust(pad)} to "
                + f"{str(join_path(dir_name, 'occurrences_matrix'))}"
                + ".{txt,npy}"
            )

            np.save(
                join_path(dir_name, "transition_matrix.npy"),
                self.transition_matrix,
                allow_pickle=False,
            )
            with open(join_path(dir_name, "transition_matrix.txt"), "w") as f:
                self.print_matrix(
                    transition_matrix=True,
                    file_obj=f,
                    float_decimals=6,
                )
            self.info(
                f"Saved {'transition_matrix'.rjust(pad)} to "
                + f"{str(join_path(dir_name, 'transition_matrix'))}"
                + ".{txt,npy}"
            )

        for matrix_name in (
            "training_matrix",
            "validation_matrix",
            "test_matrix",
        ):
            if not hasattr(self, matrix_name):
                self.debug(f"No {type(self).__name__}.{matrix_name} was found")
                continue
            np.savetxt(
                join_path(dir_name, f"{matrix_name}.txt"),
                getattr(self, matrix_name),
                fmt="%16.9e",
                header=str(
                    list(
                        set(rename_helper(("ActualState_val",))).union(
                            set(self.observed_variables)
                        )
                    )
                ),
            )
            np.save(
                join_path(dir_name, f"{matrix_name}.npy"),
                getattr(self, f"{matrix_name}"),
                allow_pickle=False,
            )
            self.info(
                f"Saved {matrix_name.rjust(pad)} to "
                + f"{str(join_path(dir_name, matrix_name))}"
                + ".{txt,npy}"
            )

        for df_name in ("_df", "_training_df", "_validation_df", "_test_df"):
            if getattr(self, df_name, None) is None:
                self.debug(f"No {type(self).__name__}.{df_name} was found")
                continue
            getattr(self, df_name).to_csv(
                join_path(dir_name, f"{df_name}.csv")
            )
            self.info(
                f"Saved {df_name.rjust(pad)} to "
                + str(join_path(dir_name, f"{df_name}.csv"))
            )

        if not isinstance(self, LightMetaModel):
            self.warning(
                "The following properties are only saved for "
                f"{LightMetaModel.__name__} objects:\n\t"
                + str(
                    "\n\t".join(
                        (
                            "start_prob.npy",
                            "start_prob.txt",
                            "occurrences_matrix.npy",
                            "occurrences_matrix.txt",
                            "transition_matrix.npy",
                            "transition_matrix.txt",
                        )
                    )
                )
            )

    def test_matrix_labels(self, unordered_model_states):
        index_of = self._state_name_to_index_mapping(unordered_model_states)
        return np.array(
            [
                index_of[State(old_index).name]
                for old_index in self._test_df.loc[
                    :, rename_helper("ActualState_val")
                ].to_list()
            ]
        )

    def validation_matrix_labels(self, unordered_model_states):
        index_of = self._state_name_to_index_mapping(unordered_model_states)
        return np.array(
            [
                index_of[State(old_index).name]
                for old_index in self._validation_df.loc[
                    :, rename_helper("ActualState_val")
                ].to_list()
            ]
        )

    def warning(self, msg=""):
        """Log warning message."""
        self._log(logging.WARNING, msg)


class LightMetaModel(MetaModel):
    @property
    def hidden_markov_model_dir(self):
        if super().worker_dir is None:
            return None
        ret = join_path(
            super().worker_dir,
            f"seed_{self._random_seed:0>6d}",
            "HiddenMarkovModel_class",
        )
        if not isdir(ret):
            makedirs(ret)
        return ret

    @property
    def hidden_markov_model_kwargs(self):
        return dict(
            algorithm="baum-welch",
            max_iterations=getattr(parsed_args(), "max_iter", 1e8),
            n_components=len(self.oxygen_states),
            n_init=128,  # initialize kmeans n times before taking the best
            name=f"HMM__seed_{self._random_seed:0>6d}",
            state_names=[State(state).name for state in self.oxygen_states],
            stop_threshold=getattr(parsed_args(), "stop_threshold", 1e-9),
        )

    @property
    def worker_dir(self):
        if super().worker_dir is None:
            return None
        ret = join_path(
            super().worker_dir,
            f"seed_{self._random_seed:0>6d}",
            f"{LightMetaModel.__name__}_class",
        )
        if not isdir(ret):
            makedirs(ret)
        return ret

    def __init__(self, df, patient_key_col, **kwargs):
        super().__init__(
            df=df,
            patient_key_col=patient_key_col,
            skip_preprocessing=True,
            **kwargs,
        )


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
