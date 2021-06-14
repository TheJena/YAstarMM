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
   Build a model from the input data and train it.

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
    matches_static_rule,
    minimum_maximum_column_limits,
    rename_helper,
)
from .constants import LOGGING_LEVEL, HmmTrainerInput, MIN_PYTHON_VERSION
from .flavoured_parser import parsed_args
from .model import State
from .parallel import _join_all
from .plot import plot_histogram_distribution
from .preprocessing import clear_and_refill_state_transition_columns
from .utility import initialize_logging, random_string
from collections import Counter
from multiprocessing import cpu_count, Lock, parent_process, Process, Queue
from numpy.random import RandomState
from os import listdir, makedirs, remove, rmdir
from os.path import abspath, basename, join as join_path, isdir, isfile
from pomegranate import (
    HiddenMarkovModel,
    NormalDistribution,
)
from pomegranate.callbacks import Callback, CSVLogger
from random import randint
from sys import version_info
from time import time
from yaml import dump, Dumper, SafeDumper
from zipfile import ZipFile, ZIP_DEFLATED
import logging
import numpy as np
import pandas as pd
import pickle
import queue
import signal

_LOGGING_LOCK = Lock()
_NEW_DF = None
_NEW_DF_LOCK = Lock()
_SIGNAL_COUNTER = 0
_SIGNAL_QUEUE = Queue()
_STATS_QUEUE = Queue()
_WORKERS_POOL = list()


def _hmm_trainer(hti, **kwargs):
    worker_id, num_workers = hti.worker_id, hti.num_workers
    assert worker_id >= 0, repr(worker_id)
    assert num_workers > worker_id, repr(num_workers, worker_id)
    logging.info(f"Worker {worker_id} successfully started")

    seed_whitelist = {
        worker_id + (i * num_workers)
        for i in range(
            hti.skip_first_seeds, hti.skip_first_seeds + hti.num_iterations
        )
    }
    hmm_seeds = getattr(parsed_args(), "light_meta_model_random_seeds", None)
    if hmm_seeds is not None and isinstance(hmm_seeds, (list, tuple, set)):
        seed_whitelist.intersection_update(set(hmm_seeds))
        logging.warning(f"Using manually set seed list {repr(seed_whitelist)}")

    log_msg_queue = list()
    for seed in seed_whitelist:
        try:
            light_mm = LightMetaModel(
                hti.df,
                log_msg_queue=log_msg_queue,
                oxygen_states=[
                    State(state_value).name
                    for state_value in hti.workload_mapping[
                        (seed) % len(hti.workload_mapping)
                    ]  # round robin load balancing
                ],
                random_seed=seed,
                **kwargs,
            )

            light_mm.show_start_probability()
            light_mm.show(occurrences_matrix=True)
            light_mm.show(transition_matrix=True)
            if False:
                light_mm.show(training_matrix=True)
                light_mm.show(validation_matrix=True)

            hmm_kwargs = light_mm.hidden_markov_model_kwargs
            hmm_save_dir = light_mm.hidden_markov_model_dir
            light_mm.save_to_disk(compress=True)

            light_mm.info()
            light_mm.info("Building HMM from samples")
            hmm = HiddenMarkovModel.from_samples(
                X=light_mm.training_matrix,
                callbacks=[
                    CSVLogger(join_path(hmm_save_dir, "training_log.csv")),
                    StreamLogger(light_mm.info),
                ],
                distribution=NormalDistribution,
                n_jobs=1,  # already using multiprocessing
                random_state=light_mm.random_state,
                verbose=False,
                **hmm_kwargs,
            )

            save_hidden_markov_model(
                dir_name=hmm_save_dir,
                hmm=hmm,
                hmm_kwargs=hmm_kwargs,
                validation_matrix=light_mm.validation_matrix,
                validation_labels=light_mm.validation_matrix_labels(
                    hmm.states
                ),
                logger=light_mm,
                compress=True,
            )
        except Exception as e:
            if "light_mm" in locals():
                light_mm._log(logging.CRITICAL, str(e))
                light_mm.flush_logging_queue()  # blocking
            else:
                # something crashed before/during light_mm memory allocation
                logging.critical(str(e))
                raise e
        finally:
            if "light_mm" in locals():
                log_msg_queue = light_mm.flush_logging_queue(timeout=1.5)
            else:
                logging.warning(
                    "something crashed before/during light_mm allocation"
                )
                while log_msg_queue:
                    level, msg = log_msg_queue.pop(0)
                    logging.log(level, msg)

        try:
            if hti.signal_queue.get(timeout=0.5) is None:
                logging.info(f"Worker {worker_id} acked to SIGTERM")
                break
        except queue.Empty:
            pass

    if seed_whitelist:
        light_mm.flush_logging_queue()  # blocking

    global _STATS_QUEUE
    _STATS_QUEUE.put(None)  # no more stats from this worker

    logging.info(f"Worker {worker_id} is ready to rest in peace")


def _sig_handler(sig_num, stack_frame):
    if parent_process() is not None:
        return

    # only main process arrives here
    global _SIGNAL_COUNTER, _SIGNAL_QUEUE, _WORKERS_POOL
    for w in _WORKERS_POOL:
        logging.info(
            f"Received signal {sig_num}; sending SIGTERM to worker {w.name}"
        )
        _SIGNAL_QUEUE.put(None)

    logging.info("Please wait until all workers ACK and finish gracefully")

    _SIGNAL_COUNTER += 1
    if _SIGNAL_COUNTER >= 3:
        for w in _WORKERS_POOL:
            logging.warning(f"Killing worker {w.name}")
            w.kill()
        logging.critical("EXIT FORCED")
        raise SystemExit("\nEXIT FORCED\n")


def aggregate_constant_values(sequence):
    sequence = set(s for s in sequence if pd.notna(s))
    assert (
        len(sequence) <= 1
    ), f"sequence {repr(sequence)} does not contain a constant value"
    return sequence.pop() if sequence else np.nan


def compress_directory(path, logger=logging, max_seeds=100):
    """100 explored seeds produce about a GiB of uncompressed results"""
    if all(
        (
            getattr(parsed_args(), "debug_mode", False),
            getattr(parsed_args(), "seeds_to_explore", None) is None
            or getattr(parsed_args(), "seeds_to_explore") <= max_seeds,
            getattr(parsed_args(), "light_meta_model_random_seeds") is None
            or len(getattr(parsed_args(), "light_meta_model_random_seeds"))
            <= max_seeds,
        )
    ):
        logger.info("Compression disabled by -d/--debug-mode CLI argument")
        return
    if not isdir(path):
        logger.warning(
            f"Path '{path}' is not an existing directory; "
            "compression aborted."
        )
        return

    entries = [
        join_path(path, entry)
        for entry in sorted(listdir(path), key=str.lower)
    ]
    data = dict()
    for entry in entries:
        if not isfile(entry):
            logger.warning(
                f"Aborting compression because '{entry}' (in '{path}') "
                f"is not a file as expected.\t({compress_directory.__name__}"
                " does not recursively visit 'path' argument"
            )
            return
        data[join_path(basename(path), basename(entry))] = open(
            entry, "rb"
        ).read()

    zip_filename = f"{path}.zip"
    try:
        with ZipFile(
            zip_filename, mode="x", compression=ZIP_DEFLATED, compresslevel=-1
        ) as archive:
            for arcname, raw_data in data.items():
                archive.writestr(arcname, raw_data)
    except FileExistsError as e:
        logger.warning(str(e))
        return

    if isfile(zip_filename):
        logger.debug(f"Successfully compressed {path} to into .zip archive")
        for entry in entries:
            remove(entry)
            logger.debug(f"Removed '{entry}'")
        rmdir(path)
        logger.debug(f"Removed '{path}'")
        return
    logger.warning(f"Compression of '{path}' did not succeed")


def dataframe_to_numpy_matrix(df, only_columns=None, normalize=False):
    """Many thanks to https://stackoverflow.com/a/41532180"""
    if only_columns is None or not isinstance(
        only_columns, (list, set, tuple)
    ):
        raise ValueError("only_columns argument should be an iterable")

    only_columns = sort_cols_as_in_df(only_columns, df)

    if normalize:
        raise NotImplementedError(  # define {MINIMUM,MAXIMUM}_COLUMN_LIMIT
            """
            return (
                df.loc[:, only_columns]
                .sub(
                    [MINIMUM_COLUMN_LIMIT for col in only_columns],
                    axis="columns",
                )
                .div(
                    [
                        MAXIMUM_COLUMN_LIMIT - MINIMUM_COLUMN_LIMIT
                        for col in only_columns
                    ],
                    axis="columns",
                )
                .to_numpy()
            )
            """
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
            "URINE_PH",
        )
    ):  # both a higher or a lower value mean a worst patient health
        return np.mean
    elif column in (
        patient_key_col,
        rename_helper("AGE"),
        rename_helper("CHARLSON_INDEX"),
        rename_helper("DAYS_IN_STATE", errors="quiet"),
        rename_helper("UPDATED_CHARLSON_INDEX", errors="quiet"),
    ):  # all values should be the same, check it
        return aggregate_constant_values
    raise NotImplementedError(
        f"No aggregation function was found for column {repr(column)};"
        " please add it in the above switch-case"
    )


def observables_of_interest(state_sequence):
    assert isinstance(state_sequence, (list, tuple)) and all(
        isinstance(state_value, int) for state_value in state_sequence
    ), str("state_sequence must be a sequence of integer")
    ret = set([rename_helper("DAYS_IN_STATE", errors="quiet")])
    state_sequence = set(state_sequence).intersection(
        {
            getattr(State, name.replace(" ", "_"))
            for name in State.non_final_states_names()
        }
    )
    for state_a in state_sequence:
        for state_b in state_sequence:
            if state_a >= state_b:
                continue
            # half of the len(state_sequence)**2 combinations arrive here
            for state_pool, observables in {
                (State.No_O2.value, State.O2.value): (
                    "CREATININE",
                    "DYSPNEA",
                    "D_DIMER",
                    "GPT_ALT",
                    "HOROWITZ_INDEX",
                    "LYMPHOCYTE",
                    "PHOSPHOCREATINE",
                    "PROCALCITONIN",
                    "RESPIRATORY_RATE",
                ),
                (State.O2.value, State.HFNO.value, State.NIV.value): (
                    "CARBON_DIOXIDE_PARTIAL_PRESSURE",
                    "DYSPNEA",
                    "HOROWITZ_INDEX",
                    "LDH",
                    "PH",
                    "RESPIRATORY_RATE",
                    "URINE_PH",
                ),
                (State.NIV.value, State.Intubated.value): (
                    "AGE",
                    "CARBON_DIOXIDE_PARTIAL_PRESSURE",
                    "CHARLSON_INDEX",
                    "DYSPNEA",
                    "HOROWITZ_INDEX",
                    "LDH",
                    "LYMPHOCYTE",
                    "PH",
                    "UPDATED_CHARLSON_INDEX",
                    "UREA",
                    "URINE_PH",
                ),
            }.items():
                if state_a in state_pool and state_b in state_pool:
                    ret.update(set(rename_helper(observables, errors="quiet")))
                    break
    return tuple(sorted(ret, key=str.lower))


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
        set(df.loc[:, patient_key_col].astype("string").to_list()).pop(),
        base=16 if hexadecimal_patient_id else 10,
    )
    log_prefix = str(
        "[patient "
        + str(
            f"{patient_id:X}" if hexadecimal_patient_id else f"{patient_id:d}"
        )
        + "]"
    )

    if (
        rename_helper("UPDATED_CHARLSON_INDEX", errors="quiet")
        in observed_variables
    ):
        # let's compute updated charlson-index before dropping unobserved cols
        logger.debug(f"{log_prefix}")
        cci = compute_charlson_index(df, logger=logger, log_prefix=log_prefix)
        logger.debug(
            f"{log_prefix} Updated Charlson-Index is "
            + str(f"= {cci:2.0f}" if pd.notna(cci) else "not computable")
            + "\n"
        )
        df.loc[:, rename_helper("UPDATED_CHARLSON_INDEX", errors="quiet")] = (
            float(cci) if pd.notna(cci) else np.nan
        )
    else:
        logger.info(
            "Charlson-Index will be used as it is "
            "(because of --update-charlson-index False CLI argument)"
        )

    # drop unobserved columns
    df = df.loc[
        :,
        sort_cols_as_in_df(
            set(
                rename_helper((patient_key_col, "DataRef", "ActualState_val"))
            ).union(
                set(rename_helper(tuple(observed_variables), errors="quiet"))
            ),
            df,
        ),
    ].sort_values(rename_helper("DataRef"))

    logger.debug(f"{log_prefix} original patient_df was:\n" + df.to_string())

    # ensure each date has exactly one record; if there are multiple
    # values they will be aggregated by choosing the one which denotes
    # the worst health state; finally drop nan dates
    df = (
        df.groupby(rename_helper("DataRef"), as_index=False)
        .aggregate(
            {
                col: function_returning_worst_value_for(col, patient_key_col)
                for col in df.columns
                if col
                != rename_helper("DataRef")  # otherwise pandas complains
            }
        )
        .dropna(subset=rename_helper(["DataRef"]))
    )

    # now date column can be safely used as an index
    df = df.set_index(
        keys=rename_helper("DataRef"), drop=False, verify_integrity=True
    )

    # let's compute the days passed in a state
    days_in_state, count = list(), 1
    for today_state, tomorrow_state in zip(
        df[rename_helper("ActualState_val")].tolist(),
        df[rename_helper("ActualState_val")].shift(periods=1).tolist(),
    ):
        if today_state == tomorrow_state:
            count += 1
        else:
            count = 1
        days_in_state.append(count)

    df = df.assign(
        **{"DAYS_IN_STATE": pd.Series(days_in_state, dtype="int64")}
    ).reset_index(
        drop=True
    )  # drop dates in favour of 0 to N-1

    old_num_records = df.shape[0]
    if str(getattr(parsed_args(), "drop_duplicates", False)) == str(True):
        df = df.drop_duplicates(
            subset=[
                col
                for col in df.columns
                if col != rename_helper("DataRef")
                and not matches_static_rule(col)
            ],
            keep="last",  # it has the number of days passed in the actual state
            ignore_index=True,
        )
    logger.debug(
        f"{log_prefix} dropped {old_num_records - df.shape[0]:4d} records "
        f"in patient_df, which now is:\n" + df.to_string()
    )

    global _NEW_DF, _NEW_DF_LOCK
    _NEW_DF_LOCK.acquire()
    if _NEW_DF is None:
        _NEW_DF = df.copy()
    else:
        _NEW_DF = pd.concat([_NEW_DF, df.copy()], ignore_index=True, sort=True)
    _NEW_DF_LOCK.release()

    return df.sort_values(rename_helper("DataRef"))


def run():
    assert getattr(parsed_args(), "max_workers") > 0
    assert getattr(parsed_args(), "save_dir") is not None
    assert any(
        (
            getattr(parsed_args(), "light_meta_model_random_seeds", None)
            is not None,
            getattr(parsed_args(), "seeds_to_explore", None) is not None
            and getattr(parsed_args(), "seeds_to_explore", None) > 0,
        )
    )
    skip_first_seeds = getattr(parsed_args(), "skip_first_z_seeds", 0)
    assert skip_first_seeds >= 0

    initialize_logging(
        f"{__name__.replace('.', '_')}_{run.__name__}__debug.log",
        getattr(parsed_args(), "log_level", LOGGING_LEVEL),
        debug_mode=getattr(parsed_args(), "verbose", False),
    )

    input_file = getattr(parsed_args(), "input")
    patient_key_col = rename_helper(
        getattr(parsed_args(), "patient_key_col"), errors="warn"
    )
    df = clear_and_refill_state_transition_columns(
        input_file.name if input_file.name.endswith(".xlsx") else input_file,
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
        hexadecimal_patient_id=str(
            pd.api.types.infer_dtype(df.loc[:, patient_key_col])
        )
        == "string",
        ignore_transferred_state=str(
            getattr(parsed_args(), "ignore_transferred_state", True)
        )
        == str(True),
        observed_variables=list(getattr(parsed_args(), "observed_variables"))
        + list(
            ["UPDATED_CHARLSON_INDEX"]
            if str(getattr(parsed_args(), "update_charlson_index", True))
            == str(True)
            else []
        )
        + list(
            ["DAYS_IN_STATE"]
            if str(getattr(parsed_args(), "add_day_count", True)) == str(True)
            else []
        ),
        oxygen_states=getattr(
            parsed_args(), "oxygen_states", [state.name for state in State]
        ),
        random_seed=getattr(parsed_args(), "meta_model_random_seed", None),
        save_to_dir=getattr(parsed_args(), "save_dir"),
    )
    if False:
        heavy_mm.show(training_matrix=True)
        heavy_mm.show(testing_matrix=True)

    if getattr(parsed_args(), "seeds_to_explore", None) is not None:
        max_seeds = getattr(parsed_args(), "seeds_to_explore")
    else:
        max_seeds = 1 + max(
            getattr(parsed_args(), "light_meta_model_random_seeds")
        )
    num_workers = min(getattr(parsed_args(), "max_workers"), cpu_count())

    light_mm_df = heavy_mm.light_meta_model_df
    light_mm_kwargs = heavy_mm.light_meta_model_kwargs
    light_mm_wl_map = heavy_mm.light_meta_model_workload_mapping
    heavy_mm.save_to_disk()
    heavy_mm.debug(f"The first {skip_first_seeds} seeds will be skipped")

    heavy_mm.flush_logging_queue()  # blocking

    global _SIGNAL_QUEUE, _STATS_QUEUE, _WORKERS_POOL
    for sig_num in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig_num, _sig_handler)
    for i in range(num_workers):
        w = Process(
            target=_hmm_trainer,
            name=str(i),
            args=(
                HmmTrainerInput(
                    df=light_mm_df,
                    num_iterations=(max_seeds // num_workers)
                    + int(i < max_seeds % num_workers),
                    num_workers=num_workers,
                    signal_queue=_SIGNAL_QUEUE,
                    skip_first_seeds=(skip_first_seeds // num_workers)
                    + int(i < skip_first_seeds % num_workers),
                    worker_id=i,
                    workload_mapping=light_mm_wl_map,
                ),
            ),
            kwargs=light_mm_kwargs,
        )
        _WORKERS_POOL.append(w)
        w.start()
    while not _SIGNAL_QUEUE.empty():
        _SIGNAL_QUEUE.get()

    all_stats = list()
    for _ in range(num_workers):
        while True:
            stat = _STATS_QUEUE.get()
            if stat is None:
                break
            else:
                all_stats.append(stat)
    _join_all(_WORKERS_POOL, "Workers in charge of HMMs training")

    logging.debug(f"All epoch execution times: {repr(all_stats)}")
    logging.info(
        "Mean Epoch Time (s): "
        f"{np.mean(all_stats):.3f} ± {np.std(all_stats):.3f} "
        f"(over all {num_workers} workers)"
    )


def save_hidden_markov_model(
    dir_name,
    hmm,
    hmm_kwargs,
    validation_matrix,
    validation_labels,
    logger=logging,
    compress=False,
    pad=32,
):
    logger.debug(f"node_count:  {hmm.node_count():3d}")
    logger.debug(f"state_count: {hmm.state_count():3d}")

    hmm_result_dict = dict(
        log_probability=hmm.log_probability(validation_matrix),
        predict=hmm.predict(validation_matrix, algorithm="map"),
        score=float(hmm.score(validation_matrix, validation_labels)),
        state_mapping={
            state.name.replace(" ", "_"): i
            for i, state in enumerate(hmm.states)
            if not any(
                (state.name.endswith("-start"), state.name.endswith("-end"))
            )
        },
        validation_labels=validation_labels.tolist(),
    )
    logger.info()
    for k, v in hmm_result_dict.items():
        logger.info(f"{k}: {repr(v)}")
    logger.info()

    logger.info(
        f"Saving {type(hmm).__name__.ljust(pad-1)} to {abspath(dir_name)}"
    )

    with open(
        join_path(dir_name, "hmm_constructor_parameters.yaml"), "w"
    ) as f:
        dump(hmm_kwargs, f, Dumper=SafeDumper, default_flow_style=False)
        logger.info(f"Saved {'hmm_kwargs'.rjust(pad)} to {basename(f.name)}")

    for k, v in hmm_result_dict.items():
        with open(join_path(dir_name, f"{k}.yaml"), "w") as f:
            dump(v, f, Dumper=SafeDumper, default_flow_style=False)
            logger.info(f"Saved {k.rjust(pad)} to {basename(f.name)}")

    for k, v in dict(
        predict_proba=hmm.predict_proba(validation_matrix),
        predict_log_proba=hmm.predict_log_proba(validation_matrix),
        dense_transition_matrix=hmm.dense_transition_matrix(),
    ).items():
        np.savetxt(join_path(dir_name, f"{k}.txt"), v, fmt="%16.9e")
        np.save(join_path(dir_name, f"{k}.npy"), v, allow_pickle=False)
        logger.info(
            f"Saved {k.rjust(pad)} to {str(basename(join_path(dir_name, k)))}"
            + ".{txt,npy}"
        )

    if False:
        logger.info(
            "maximum_a_posteriori: "
            + repr(hmm.maximum_a_posteriori(validation_matrix))
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
            f" to {'.'.join(basename(f.name).split('.')[:-1])}"
            + ".{json,"
            + f.name.split(".")[-1]
            + "}"
        )

    if compress:
        compress_directory(dir_name, logger=logger)


def sort_cols_as_in_df(unsorted_cols, df):
    """return 'unsorted_cols' in the same order they appear in df.columns"""
    assert len(set(unsorted_cols)) == len(unsorted_cols), str(
        "'unsorted_cols' must be a unique set of columns"
    )
    ret, unsorted_columns = list(), set(unsorted_cols)
    for col in df.columns:
        if col in unsorted_columns:
            ret.append(col)
            unsorted_columns.remove(col)
    assert not len(unsorted_columns), str(
        f"The following columns were not found in self._df:\t"
        f"{', '.join(unsorted_columns)}."
    )
    return tuple(ret)


class StreamLogger(Callback):
    @property
    def log(self):
        assert callable(self._log_method)
        return self._log_method  # without actually calling it

    def __init__(self, log_method):
        self._all_epochs = list()
        self._log_method = log_method

    def on_training_begin(self):
        self._t_start = time()
        self.log()
        self.log("Training started")

    def on_epoch_end(self, logs):
        self.log(
            "  ".join(
                (
                    f"[iter {logs['epoch']:>4d}]",
                    f"Improved: {logs['improvement']:17.9f}",
                    f"in {logs['duration']:6.3f} s",
                )
            )
        )
        self._all_epochs.append(logs["duration"])

        global _STATS_QUEUE
        _STATS_QUEUE.put(logs["duration"])

    def on_training_end(self, logs):
        self._t_end = time()
        total_improvement = logs["total_improvement"]
        self.log(f"Total Improvement: {total_improvement:20.9f}")
        self.log(f"Training took (s): {self._t_end-self._t_start:20.3f}")
        self.log(
            "Average Epoch Time (s): "
            f"{sum(self._all_epochs)/len(self._all_epochs):15.3f}"
        )
        self.log()


class MetaModel(object):
    """Split input data in 'fake-training'/test set"""

    @property
    def has_logging_lock(self):
        return self._has_logging_lock

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

                date = date.normalize()  # truncate to midnight
                state_val = row[rename_helper("ActualState_val")]
                assert isinstance(state_val, (int, float)), str(
                    f"Patient '{repr(patient)}' has {repr(state_val)} "
                    f"in columnn '{rename_helper('ActualState_val')}'"
                    " instead of a float or integer."
                )
                actual_state = State(int(state_val))
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

        return self._input_data_dict

    @property
    def light_meta_model_df(self):
        assert not isinstance(self, LightMetaModel)
        if self._training_df is None:
            self._split_dataset()
        return self._training_df.copy(deep=True)

    @property
    def light_meta_model_kwargs(self):
        assert not isinstance(self, LightMetaModel)
        return dict(
            hexadecimal_patient_id=None,
            observed_variables=self.observed_variables,
            patient_key_col=self.patient_id,
            # random_seed will be taken from parsed_args()
            ratio=self.fixed_validation_set_ratio(),
            save_to_dir=self.worker_dir,
            # skip_preprocessing will be set by LightMetaModel.__init__()
        )

    @property
    def light_meta_model_workload_mapping(self):
        assert not isinstance(self, LightMetaModel)
        if getattr(parsed_args(), "train_little_hmm", False):
            self.info(
                "Each HMM will be trained over a subset of the oxygen states"
            )
            workload_mappings = (  # ad-hoc-little-HMMs
                {State.Discharged.value, State.No_O2.value, State.O2.value},
                {State.Deceased.value, State.Intubated.value, State.NIV.value},
                {State.No_O2.value, State.O2.value, State.HFNO.value},
                {State.Intubated.value, State.NIV.value, State.HFNO.value},
                {State.O2.value, State.HFNO.value, State.NIV.value},
                {State.No_O2.value, State.O2.value},
                {State.NIV.value, State.Intubated.value},
            )
        else:
            self.info("All HMMs will be trained over all the oxygen states")
            workload_mappings = (  # all-in-one-HMMs
                set(self.oxygen_states),
                set(self.oxygen_states),
            )
        for state_value in set(self.oxygen_states):
            if state_value == State.Transferred.value:
                continue
            assert any(
                state_value in workload for workload in workload_mappings
            ), str(
                f"State '{State(state_value)}' "
                "is not covered by any workload mapping"
            )
        return {i: wl for i, wl in enumerate(workload_mappings)}

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
                elif str(getattr(parsed_args(), "add_day_count", True)) == str(
                    False
                ):
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
            only_columns=self.enforced_columns_order(
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
            only_columns=self.enforced_columns_order(
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
            only_columns=self.enforced_columns_order(
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
        hexadecimal_patient_id=False,
        ignore_transferred_state=True,
        observed_variables=None,
        oxygen_states=None,
        outliers="ignore",
        postponed_logging_queue=list(),
        random_seed=None,
        ratio=None,
        save_to_dir=None,
        skip_preprocessing=False,
    ):
        """Prepare a MetaModel object ready to be used to train an HMM"""
        assert outliers in ("clip", "ignore")
        self._has_logging_lock = False
        self._ignore_transferred_state = ignore_transferred_state
        self._input_data_dict = None
        self._outliers_treatment = outliers
        self._patient_key_col = patient_key_col
        self._postponed_logging_queue = postponed_logging_queue
        self._previous_msg = random_string(length=80)
        if random_seed is None:
            self._random_seed = randint(0, 999999)
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
        self.debug(
            f"oxygen_states: {repr(self.oxygen_states)[1:-1]}\t(i.e.: "
            + str(
                ", ".join(
                    State(state_value).name
                    for state_value in self.oxygen_states
                )
            )
            + ")."
        )
        if not getattr(parsed_args(), "ignore_expert_knowledge", False):
            new_obs_var = set(self.observed_variables).intersection(
                observables_of_interest(self.oxygen_states)
            )
            if set(self.observed_variables) != set(new_obs_var):
                self.warning(
                    "Updated observed variables for states:\n\t("
                    + ", ".join(
                        State(state_value).name
                        for state_value in self.oxygen_states
                    )
                    + ")\nwith knowledge from the expert:\n\t"
                    + repr(tuple(sorted(new_obs_var, key=str.lower)))
                )
                self.observed_variables = new_obs_var
            else:
                self.debug(
                    "observed_variables were already "
                    "the same suggested by domain experts"
                )
        if (
            rename_helper("UPDATED_CHARLSON_INDEX", errors="quiet")
            in self.observed_variables
            and rename_helper("UPDATED_CHARLSON_INDEX", errors="quiet")
            not in df.columns
        ):
            df = df.assign(
                **{
                    rename_helper(
                        "UPDATED_CHARLSON_INDEX", errors="quiet"
                    ): pd.Series([np.nan for _ in range(df.shape[0])])
                }
            )
        if (
            rename_helper("DAYS_IN_STATE", errors="quiet")
            in self.observed_variables
            and rename_helper("DAYS_IN_STATE", errors="quiet")
            not in df.columns
        ):
            df = df.assign(
                **{
                    rename_helper("DAYS_IN_STATE", errors="quiet"): pd.Series(
                        [np.nan for _ in range(df.shape[0])]
                    )
                }
            )

        # enforce same column order to make it easier comparing
        # matrices during debugging
        df = df.loc[:, tuple(sorted(df.columns, key=str.lower))]
        if skip_preprocessing:
            self.info("Skipping preprocessing")
            self._old_df = None
            self._df = df.loc[
                #
                # cut away records about oxygen states not of interest (if any)
                df[rename_helper("ActualState_val")].isin(self.oxygen_states),
                #
                # drop unobserved columns (if any)
                sort_cols_as_in_df(
                    set(
                        rename_helper(
                            (patient_key_col, "DataRef", "ActualState_val")
                        )
                    ).union(
                        set(
                            rename_helper(
                                tuple(observed_variables), errors="quiet"
                            )
                        )
                    ),
                    df,
                ),
            ].sort_values(rename_helper("DataRef"))
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
                    rename_helper("DataRef"): pd.to_datetime(
                        self._df_old.loc[:, rename_helper("DataRef")]
                    ).dt.normalize(),  # truncate HH:MM:SS
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
            self._df = (
                _NEW_DF.loc[
                    # cut away records about oxygen state not of interest
                    _NEW_DF[rename_helper("ActualState_val")].isin(
                        self.oxygen_states
                    ),
                    sort_cols_as_in_df(_NEW_DF.columns, df),
                ]
                .sort_values(list(rename_helper((patient_key_col, "DataRef"))))
                .reset_index(drop=True)
            )
            _NEW_DF = None
            _NEW_DF_LOCK.release()

            self.plot_observed_variables_distributions(has_outliers=True)

            show_final_hint = False
            for col, data in minimum_maximum_column_limits(
                getattr(parsed_args(), "outlier_limits")
            ).items():
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
                lower_outliers = self._df.loc[
                    self._df[col] < data["min"], col
                ].count()
                upper_outliers = self._df.loc[
                    self._df[col] > data["max"], col
                ].count()
                if lower_outliers == 0 and upper_outliers == 0:
                    continue

                for outliers, kind, limit in (
                    (lower_outliers, "lower", data["min"]),
                    (upper_outliers, "upper", data["max"]),
                ):
                    if outliers > 0:
                        self.warning(
                            str(
                                f"Column {repr(col).rjust(34)} has: "
                                f"{outliers:5d} {kind} outliers ("
                                f"{dict(lower='<', upper='>')[kind]} "
                                f"{limit:9.3f})"
                            )
                        )
                show_final_hint = True
                if self._outliers_treatment == "clip":
                    self._df.loc[:, col] = self._df.loc[:, col].clip(
                        lower=data["min"], upper=data["max"]
                    )
                elif self._outliers_treatment == "ignore":
                    self._df.loc[self._df[col] < data["min"], col] = np.nan
                    self._df.loc[self._df[col] > data["max"], col] = np.nan
                else:
                    assert self._outliers_treatment in ("clip", "ignore")
            if show_final_hint:
                self.warning(
                    "The above outliers will be "
                    + str(
                        "clipped to the respective limits."
                        if self._outliers_treatment == "clip"
                        else "considered as nan."
                    )
                )
                self.warning(
                    "To change the above lower/upper limits please "
                    "consider the column percentiles in the debug log"
                )
            if self._outliers_treatment == "clip":
                self.warning(
                    "Clipping outlier values can be very dangerous! "
                    "Please compare the plot of the observed variables "
                    "distributions to be sure that no distortion has "
                    "been introduced"
                )
            self.info("Ended preprocessing")
        self.plot_observed_variables_distributions(has_outliers=False)
        self._split_dataset()

    def _log(self, level, msg):
        msg = str(
            f"{self.logger_prefix}"
            f"{' ' if not msg.startswith('[') else ''}"
            f"{msg}"
        )
        if msg == self._previous_msg:
            return
        self._previous_msg = msg
        if not self.has_logging_lock and self._postponed_logging_queue:
            # If there are pending messages try to get the logging
            # lock and flush them; but without wasting too much time
            self.flush_logging_queue(timeout=0.1, release_logging_lock=False)
        if self.has_logging_lock:
            logging.log(level, msg)
        else:
            self._postponed_logging_queue.append(tuple((level, msg)))

    def _split_dataset(self, pad=22):
        """Split dataset into training set and validation (or test) set"""
        if self._ignore_transferred_state:
            self.info("Transferred states will be ignored")
            df = self._df.loc[
                ~self._df[rename_helper("ActualState_val")].isin(
                    {
                        State.Transferred,
                        State.Transferred.name,
                        State.Transferred.value,
                        str(State.Transferred),
                    }
                ),
                :,
            ]
        else:
            df = self._df  # whole df

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
            and self._ratio >= 1 / df.shape[0]
            and self._ratio <= 1 - (1 / df.shape[0])
        ), str(
            "Ratio (CLI argument --ratio-{validation,test}-set) is not "
            f"in ({1 / df.shape[0]}, {1 - (1 / df.shape[0])})"
        )
        self.debug(
            "full dataset shape: ".rjust(pad) + f"{df.shape[0]:7d} rows, "
            f"{df.shape[1]:3d} columns"
        )
        target_df = None
        target_rows = max(1, round(df.shape[0] * self._ratio))
        self._ratio = target_rows / df.shape[0]
        self.debug(
            f"Splitting with ratio {self._ratio:.6f} "
            f"(target_rows: {target_rows})"
        )

        patients_left = [
            patient_id
            for patient_id, _ in Counter(
                df[self.patient_id].to_list()
            ).most_common()
        ]
        while target_df is None or target_df.shape[0] < target_rows:
            assert len(patients_left) >= 1, "No patient left"
            patient_id = patients_left.pop(0)
            patient_df = df[df[self.patient_id].isin([patient_id])].copy()
            if bool(self.random_state.randint(2)):  # toss a coin
                # try to add all the patient's records to the
                # validation/test set
                if (
                    target_df is None and patient_df.shape[0] <= target_rows
                ) or (
                    target_df is not None
                    and target_df.shape[0] + patient_df.shape[0] <= target_rows
                ):
                    if target_df is None:
                        self.debug(f"target_df has {0:6d} rows")
                        target_df = patient_df
                        self.debug(
                            f"A) target_df has {target_df.shape[0]:6d} rows"
                        )
                    else:
                        target_df = pd.concat([target_df, patient_df])
                        self.debug(
                            f"B) target_df has {target_df.shape[0]:6d} rows"
                        )
                    continue  # successfully added all patients records
            # try to add the last ratio of patient's records to the
            # validation/test set
            cut_row = max(
                1,
                min(
                    patient_df.shape[0],
                    round(patient_df.shape[0] * (1 - self._ratio)),
                ),
            )
            if target_df is not None and (
                patient_df.shape[0]
                - cut_row  # validation/test records
                + target_df.shape[0]
                > target_rows
            ):
                cut_row = patient_df.shape[0]
                -(target_rows - target_df.shape[0]),
            if self._training_df is None:
                self.debug(f"{' '*32}training_df has {0:6d} rows")
                self._training_df = patient_df.iloc[:cut_row, :]
                self.debug(
                    f"{' '*32}training_df has "
                    f"{self._training_df.shape[0]:6d} rows{' '*4}(E"
                )
            else:
                self._training_df = pd.concat(
                    [self._training_df, patient_df.iloc[:cut_row, :]]
                )
                self.debug(
                    f"{' '*32}training_df has "
                    f"{self._training_df.shape[0]:6d} rows{' '*4}(F"
                )
            if target_df is None:
                self.debug(f"target_df has {0:6d} rows")
                target_df = patient_df.iloc[cut_row:, :]
                self.debug(f"C) target_df has {target_df.shape[0]:6d} rows")
            elif patient_df.iloc[cut_row:, :].shape[0] > 0:
                target_df = pd.concat(
                    [target_df, patient_df.iloc[cut_row:, :]]
                )
                self.debug(f"D) target_df has {target_df.shape[0]:6d} rows")
        assert target_df.shape[0] == target_rows, str(
            f"{'validation' if isinstance(self, LightMetaModel) else 'test'} "
            f"matrix has {target_df.shape[0]} "
            f"rows instead of {target_rows}"
        )
        # add patients left to training set
        self._training_df = pd.concat(
            [
                self._training_df,
                df[df[self.patient_id].isin(patients_left)].copy(),
            ]
        )
        assert (
            self._training_df.shape[0] + target_df.shape[0] == df.shape[0]
        ), str(
            f"training matrix has {self._training_df.shape[0]} "
            "rows instead of "
            f"{df.shape[0] - target_df.shape[0]}"
        )

        self.debug(
            "training set shape: ".rjust(pad)
            + f"{self._training_df.shape[0]:7d} rows, "
            f"{self._training_df.shape[1]:3d} columns"
        )
        self.debug(
            str(
                str(
                    "validation"
                    if isinstance(self, LightMetaModel)
                    else "test"
                )
                + " set shape: "
            ).rjust(pad)
            + f"{target_df.shape[0]:7d} rows, "
            f"{target_df.shape[1]:3d} columns"
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

        for state_value in self.oxygen_states:
            state_name = State(state_value).name
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
                + repr(str(state)).ljust(2 + max(len(str(s)) for s in State))
                + f" has index {ret[state_name]} "
                f"while its default enum value is {state.value}"
            )
        return ret

    def debug(self, msg=""):
        """Log debug message."""
        self._log(logging.DEBUG, msg)

    def enforced_columns_order(self, cols):
        """return 'cols' in the same order they appear in self._df.columns"""
        return sort_cols_as_in_df(cols, self._df)

    def fixed_validation_set_ratio(self):
        """test_records : test_ratio = validation_records : validation_ratio"""
        test_records = self._test_df.shape[0]
        validation_records = max(
            1,
            round(
                test_records
                * getattr(parsed_args(), "validation_set_ratio_hmm")
                / getattr(parsed_args(), "test_set_ratio_composer")
            ),
        )
        new_validation_ratio = validation_records / self._training_df.shape[0]
        self.debug(
            f"{validation_records} records will be put in validation set;"
            " changing ratio accordingly ("
            f"{getattr(parsed_args(), 'validation_set_ratio_hmm')}"
            f"~> {new_validation_ratio})"
        )
        return new_validation_ratio

    def flush_logging_queue(self, timeout=None, release_logging_lock=True):
        global _LOGGING_LOCK
        if not self.has_logging_lock:
            if _LOGGING_LOCK.acquire(block=timeout is None, timeout=timeout):
                logging.debug("")
                logging.info(f"{self.logger_prefix} Took logging lock")
                self._has_logging_lock = True
        if self.has_logging_lock:
            while self._postponed_logging_queue:
                level, msg = self._postponed_logging_queue.pop(0)
                logging.log(level, msg)
            if release_logging_lock:
                logging.info(f"{self.logger_prefix} Released logging lock")
                logging.debug("")
                self._has_logging_lock = False
                _LOGGING_LOCK.release()
        return self._postponed_logging_queue

    def info(self, msg=""):
        """Log info message."""
        self._log(logging.INFO, msg)

    def plot_observed_variables_distributions(self, has_outliers):
        suptitle = None
        if getattr(parsed_args(), "debug_mode", False):
            specs = dict()
            if has_outliers:
                specs["has_outliers"] = True
            else:
                # explain how outliers were treated/removed
                specs["outliers_treatement"] = self._outliers_treatment
                specs["outliers_limits"] = getattr(
                    parsed_args(), "outlier_limits"
                )
            specs["oxygen_states"] = [
                str(State(i).name).replace("_", " ")
                for i in sorted(
                    self.oxygen_states,
                    key=lambda state_value: -1
                    if state_value == State.Discharged.value
                    else state_value,
                )
                if not (
                    self._ignore_transferred_state and i == State.Transferred
                )
            ]
            suptitle = f"{type(self).__name__}({repr(specs)[1:-1]})".replace(
                "[", "\n["
            )
        plot_histogram_distribution(
            self._df.loc[:, self.observed_variables],
            has_outliers,
            logger=self,
            save_to_dir=self.worker_dir,
            suptitle=suptitle,
        )

    def show(
        self,
        file_obj=None,
        occurrences_matrix=False,
        testing_matrix=False,
        training_matrix=False,
        transition_matrix=False,
        validation_matrix=False,
        **kwargs,
    ):
        assert (
            len(
                [
                    b
                    for b in (
                        occurrences_matrix,
                        testing_matrix,
                        training_matrix,
                        transition_matrix,
                        validation_matrix,
                    )
                    if bool(b)
                ]
            )
            == 1
        ), str(
            "Please set only one flag between "
            "{occurrences,transition,training}_matrix"
        )
        style = {
            k: kwargs.get(k, v)
            for k, v in dict(
                cell_pad=2, float_decimals=3, separators=True
            ).items()
        }
        print_buffer = str()

        col_names = State.values()
        if occurrences_matrix:
            matrix = self.occurrences_matrix
            title = "Occurrences"
        elif transition_matrix:
            matrix = self.transition_matrix
            title = "Transition"
        elif training_matrix or validation_matrix or testing_matrix:
            raise NotImplementedError()
        else:
            raise ValueError(
                "Please set only one flag between "
                "{occurrences,transition,training}_matrix"
            )
        title += " matrix (with"
        if str(getattr(parsed_args(), "add_day_count", True)) == str(True):
            title += "out"
        title += " self-transitions)"

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
            cell_size = max(cell_size, style["float_decimals"] + 2)

        header = " " * (3 + 3) + "From / to".center(legend_size) + " " * 3
        header += "".join(
            str(col).rjust(cell_size) + " " * style["cell_pad"]
            for col in col_names
        )
        if title:
            print_buffer += f"{title}\n"
        print_buffer += f"{header}\n"
        if style["separators"]:
            print_buffer += f"{'_' * len(header)}\n"
        for row in State.values():
            print_buffer += str(
                f"{row:3d} = " + str(State(row)).center(legend_size)
            )
            print_buffer += " | "
            for col in range(len(col_names)):
                cell_value = matrix[row][col]
                if isinstance(cell_value, (int, np.uint16)):
                    cell_str = str(cell_value)
                elif isinstance(cell_value, (float, np.float64)):
                    if not np.isnan(cell_value):
                        assert cell_value <= 1 and cell_value >= 0, str(
                            "This function expects float values "
                            "to be in "
                            "[0, 1] like probabilities; "
                            f"got {cell_value:g} instead."
                        )
                    cell_str = "{:.16f}".format(  # right pad with many zeros
                        round(cell_value, style["float_decimals"])
                    )
                    cell_str = cell_str[:cell_size]  # cut unneded "padding" 0
                else:
                    raise NotImplementedError(
                        f"Please add support to {type(cell_value)} cells"
                    )
                print_buffer += (
                    f"{cell_str.rjust(cell_size)}{' ' * style['cell_pad']}"
                )
            print_buffer += "\n"
        if style["separators"]:
            print_buffer += f"{'_' * len(header)}\n"
        print_buffer += "\n"

        if file_obj is not None:
            print(print_buffer, file=file_obj)
        else:
            for line in print_buffer.split("\n"):
                self.info(line)

    def show_start_probability(
        self, file_obj=None, float_decimals=3, cell_pad=2
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

    def save_to_disk(self, dir_name=None, compress=False, pad=32):
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

        if not isinstance(self, LightMetaModel):
            with open(join_path(dir_name, "flavour.yaml"), "w") as f:
                parsed_args().dump(f)
                self.info(
                    f"Saved {'CLI arguments'.rjust(pad)} to "
                    f"{basename(f.name)}\t\t\t(importable with -f/--flavour)"
                )

        with open(join_path(dir_name, "input_data.pickle"), "wb") as f:
            pickle.dump(self.input_data, f)
            self.info(
                f"Saved {'input_data'.rjust(pad)} to "
                f"{'.'.join(basename(f.name).split('.')[:-1])}"
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
                    self.info(
                        f"Saved {object_name.rjust(pad)} to {basename(f.name)}"
                    )
        with open(join_path(dir_name, "oxygen_states.yaml"), "a") as f:
            for state in self.oxygen_states:
                f.write(f"# State({state}).name == '{State(state).name}'\n")

        with open(join_path(dir_name, "outliers_treatement.yaml"), "w") as f:
            dump(
                dict(
                    _outliers_treatment=self._outliers_treatment,
                    min_max_column_limits=minimum_maximum_column_limits(
                        getattr(parsed_args(), "outlier_limits")
                    ),
                ),
                f,
                Dumper=SafeDumper,
                default_flow_style=False,
            )
            self.info(
                f"Saved {'min/max column limits dictionary'.rjust(pad)} "
                f"to {basename(f.name)}"
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
                f"{str(basename(join_path(dir_name,'start_prob')))}"
                + ".{txt,npy}"
            )

            np.save(
                join_path(dir_name, "occurrences_matrix.npy"),
                self.occurrences_matrix,
                allow_pickle=False,
            )
            with open(join_path(dir_name, "occurrences_matrix.txt"), "w") as f:
                self.show(file_obj=f, occurrences_matrix=True)
            self.info(
                f"Saved {'occurrences_matrix'.rjust(pad)} to "
                + f"{str(basename(join_path(dir_name, 'occurrences_matrix')))}"
                + ".{txt,npy}"
            )

            np.save(
                join_path(dir_name, "transition_matrix.npy"),
                self.transition_matrix,
                allow_pickle=False,
            )
            with open(join_path(dir_name, "transition_matrix.txt"), "w") as f:
                self.show(file_obj=f, transition_matrix=True, float_decimals=6)
            self.info(
                f"Saved {'transition_matrix'.rjust(pad)} to "
                + f"{str(basename(join_path(dir_name, 'transition_matrix')))}"
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
            header = self.enforced_columns_order(
                set(rename_helper(("ActualState_val",))).union(
                    set(self.observed_variables)
                )
            )
            col_size = max(16, max(len(repr(col)) + 1 for col in header))
            np.savetxt(
                join_path(dir_name, f"{matrix_name}.txt"),
                getattr(self, matrix_name),
                fmt=f"%{col_size}.9e",
                header=str(
                    "["
                    + str(
                        ", ".join(
                            (
                                repr(col).rjust(
                                    col_size - int(3 if i == 0 else 1)
                                )
                                for i, col in enumerate(header)
                            )
                        )
                    )
                    + "]"
                ),
            )
            np.save(
                join_path(dir_name, f"{matrix_name}.npy"),
                getattr(self, f"{matrix_name}"),
                allow_pickle=False,
            )
            self.info(
                f"Saved {matrix_name.rjust(pad)} to "
                + f"{str(basename(join_path(dir_name, matrix_name)))}"
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
                + str(basename(join_path(dir_name, f"{df_name}.csv")))
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
        if compress:
            compress_directory(dir_name, logger=self)

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
    """Split MetaModel.fake_training into training/validation set."""

    @property
    def _worker_dir(self):
        if super().worker_dir is None:
            return None
        ret = join_path(
            super().worker_dir,
            "__".join(
                State(state_value).name
                for state_value in sorted(
                    self.oxygen_states,
                    # fake discharge value in sorting critera to place it 1st
                    key=lambda v: v if v != State.Discharged.value else -1,
                )
            ),
            f"seed_{self._random_seed:0>6d}",
        )
        if not isdir(ret):
            makedirs(ret)
        return ret

    @property
    def hidden_markov_model_dir(self):
        if super().worker_dir is None:
            return None
        ret = join_path(self._worker_dir, "HiddenMarkovModel_class")
        if not isdir(ret):
            makedirs(ret)
        return ret

    @property
    def hidden_markov_model_kwargs(self):
        return dict(
            algorithm="baum-welch",
            max_iterations=getattr(parsed_args(), "max_iter", 1e8),
            min_iterations=max(1, getattr(parsed_args(), "min_iter", 1)),
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
        ret = join_path(self._worker_dir, f"{LightMetaModel.__name__}_class")
        if not isdir(ret):
            makedirs(ret)
        return ret

    def __init__(self, df, patient_key_col, log_msg_queue=list(), **kwargs):
        """LightMetaModel are 'light' because they skip preprocessing"""
        super().__init__(
            df=df,
            patient_key_col=patient_key_col,
            postponed_logging_queue=log_msg_queue,
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
