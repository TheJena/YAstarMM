#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020 Federico Motta <191685@studenti.unimore.it>
#               2021 Federico Motta <federico.motta@unimore.it>
#
# This file is part of YAstarMM
#
# YAstarMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# YAstarMM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YAstarMM.  If not, see <https://www.gnu.org/licenses/>.
"""
   Compose several Hidden Markov Models.

   Predict patients' future State by choosing the more probable State
   among the ones returned by the most performant HMM for each
   possible state transition.

   Usage:
            from  YAstarMM.composer  import  run as run_composer
"""


from .column_rules import rename_helper
from .constants import (
    FilterWorkerError,
    FilterWorkerInput,
    FilterWorkerOutput,
    InputOutputErrorQueues,
    MIN_PYTHON_VERSION,
    SubModelWorkerError,
    SubModelWorkerInput,
    SubModelWorkerOutput,
)
from .flavoured_parser import parsed_args
from .hmm import dataframe_to_numpy_matrix, sort_cols_as_in_df
from .model import State
from .utility import black_magic, initialize_logging
from collections import Counter, defaultdict
from io import BufferedIOBase, TextIOBase
from logging import critical, debug, info, INFO, warning
from math import pow as power, sqrt
from multiprocessing import cpu_count, Process, Queue
from os import walk
from os.path import abspath, isdir, join as join_path
from pomegranate import HiddenMarkovModel
from sys import version_info
from yaml import load as load_yaml, SafeLoader
from zipfile import ZipFile
import itertools
import numpy as np
import pandas as pd

META_MODEL_DATA = dict(
    # meta_model_path
    outliers_treatment="outliers_treatment.yaml",
    seed="_random_seed.yaml",
    test_df="_test_df.csv",
    test_matrix="test_matrix.npy",
)

LIGHT_MM_DATA = dict(
    observed_variables="observed_variables.yaml",
    oxygen_states="oxygen_states.yaml",
    seed="_random_seed.yaml",
)

HMM_DATA = dict(
    actual_validation_labels="validation_labels.yaml",
    init_kwargs="hmm_constructor_parameters.yaml",
    # log probability of observing sequences like those in validation matrix
    predicted_validation_labels="predict.yaml",  # list of integers
    score="score.yaml",  # scalar float (accuracy of the model)
    serialized_obj="hmm_trained_and_serialized.yaml",
    state_mapping="state_mapping.yaml",
)


def _filter_worker_body(
    mm_test_df, test_labels, input_queue, output_queue, error_queue
):
    debug("Filter worker started")

    stats = defaultdict(int)  # 0 is default value for missing keys
    while True:
        fw_input = input_queue.get()
        if fw_input is None:
            debug("Filter worker got a termination signal")
            break

        stats["total_hmm"] += 1
        light_mm_data, hmm_data = fw_input
        try:
            if model_is_inaccurate(hmm_data, light_mm_data):
                stats[
                    "dropped_inaccurate\t(< "
                    f"{getattr(parsed_args(), 'minimum_score_threshold'):.4f}"
                    ")"
                ] += 1
                continue
            if model_is_overfitted(hmm_data, light_mm_data):
                hmm_states = oxygen_states_description(hmm_data, light_mm_data)
                stats[
                    "dropped_overfitted\t("
                    f"predicts_only_one_of:_{hmm_states})"
                ] += 1
                continue
            test_df = test_set_feasible_subset(mm_test_df, light_mm_data)
            if test_df.empty:
                stats["dropped_unfeasible"] += 1
                continue

            hidden_markov_model = HiddenMarkovModel.from_dict(
                hmm_data["serialized_obj"]
            )
            pred_stress, stress_outcome_counter = under_stress_predictions(
                hidden_markov_model, mm_test_df, light_mm_data, hmm_data
            )
            pred_test, test_matrix = hmm_predictions(
                hidden_markov_model, hmm_data, test_df, mm_test_df
            )
            pred_validation = dict(
                enumerate(hmm_data["predicted_validation_labels"])
            )
            validation_performance = measure_performance(
                pred_validation,
                dict(enumerate(hmm_data["actual_validation_labels"])),
            )
            hmm_out_prob = Counter(
                list(stress_outcome_counter.elements())
                + list(pred_test.values())
                + list(pred_validation.values())
            )
            hmm_out_prob = {
                outcome: float(amount) / sum(hmm_out_prob.values())
                for outcome, amount in hmm_out_prob.most_common()
            }
        except Exception as e:
            error_queue.put(
                FilterWorkerError(
                    hmm_description(hmm_data, light_mm_data), str(e)
                )
            )
            raise e
        else:
            stats["useful_hmm"] += 1
            stats["total_outcomes"] += len(pred_test)
            hmm_info = hmm_description(hmm_data, light_mm_data)
            for (i, pred_state), future_state, current_state in zip(
                pred_test.items(),  # predicted tomorrow states
                test_labels.values(),  # ground-truth (tomorrow)
                test_df[rename_helper("ActualState_val")].tolist(),  # today
            ):
                model_ranking_data = dict(
                    banned=False,
                    oracle_flag=bool(int(pred_state) == int(future_state)),
                )
                if outcome_is_banned(pred_state, stress_outcome_counter):
                    model_ranking_data["banned"] = True
                    stats[
                        "banned\t\t(oracle_flag="
                        f"{model_ranking_data['oracle_flag']})_outcomes"
                    ] += 1
                if not model_ranking_data["banned"]:
                    model_ranking_data = dict(
                        banned=False,
                        inv_outcome_prob=1 - hmm_out_prob[pred_state],
                        log_prob=hidden_markov_model.log_probability(
                            test_matrix[i : i + 1]  # one line i-th matrix
                        ),
                        oracle_flag=bool(pred_state == future_state),
                        outcome_prob=hmm_out_prob[pred_state],
                        prediction=pred_state,
                        score=hmm_data["score"],
                        **validation_performance.get(
                            State(future_state).name, dict()
                        ),
                    )
                output_queue.put(
                    FilterWorkerOutput(
                        i,
                        dict(description=hmm_info, **model_ranking_data),
                        None,
                    )
                )
    output_queue.put(FilterWorkerOutput(None, None, stats))
    output_queue.put(None)
    error_queue.put(None)
    debug("Filter worker acked twice to termination signal")
    debug("Filter worker is ready to rest in peace")


def _submodel_worker_body(input_queue, output_queue, error_queue):
    while True:
        submodel_worker_input = input_queue.get()
        if submodel_worker_input is None:
            debug("Sub model worker got a termination signal")
            break
        try:
            sub_model = dict()
            meta_model_path, sm_path, sm_name, sm_files = submodel_worker_input
            debug(f"Started loading {repr(sm_path)}")
            if sm_path.endswith(".zip"):
                with ZipFile(sm_path) as sub_model_zip:
                    for k, filename in sm_files.items():
                        with sub_model_zip.open(
                            join_path(sm_name, filename)
                        ) as file_obj:
                            sub_model[k] = load(file_obj)
            else:
                for k, filename in sm_files.items():
                    sub_model[k] = load(sm_path, filename)
        except Exception as e:
            error_queue.put(SubModelWorkerError(sm_path, str(e)))
        else:
            if sub_model:
                output_queue.put(
                    SubModelWorkerOutput(meta_model_path, sm_name, sub_model)
                )
    output_queue.put(None)
    error_queue.put(None)
    debug("Sub model worker acked twice to termination signal")
    debug("Sub model worker is ready to rest in peace")


def accuracy(TP=0, TN=0, FP=0, FN=0):
    """https://en.wikipedia.org/wiki/Accuracy#In_binary_classification"""
    if sum((TP, TN, FP, FN)) < 1e-4:  # avoid division by zero
        return 1e-4
    return min(1 - 1e-4, max(1e-4, float((TP + TN) / (TP + TN + FP + FN))))


def add_votes_to_ranking_data(prediction_pool, test_labels):
    vote_counts = {
        ith_record: Counter(
            -1
            if ranking_data.get("banned", False)
            else ranking_data.get("prediction", -1)  # -1 means BANNED
            for ranking_data in all_models_ranking_data
        )
        for ith_record, all_models_ranking_data in prediction_pool.items()
    }
    for i, actual_state in test_labels.items():
        debug(
            f"test_set_sample = {i:6d} | actual_state = "
            + State(actual_state).name.rjust(12)
            + " | "
            + repr(
                {
                    "BANNED" if s < 0 else State(s).name: count
                    for s, count in vote_counts[i].most_common()
                }
            ).replace("'", "")
        )
    return {
        ith_record: [
            ranking_data
            if ranking_data["banned"]
            else dict(
                votes=vote_counts[ith_record][
                    ranking_data.get("prediction", -1)
                ],
                inv_votes=sum(vote_counts[ith_record].values())
                - vote_counts[ith_record][ranking_data.get("prediction", -1)],
                **ranking_data,
            )
            for ranking_data in all_models_ranking_data
        ]
        for ith_record, all_models_ranking_data in prediction_pool.items()
    }


def convert_hmm_predictions(predicted_state_indexes, state_mapping):
    reverse_state_mapping = {  # predicted_state_index: State(X).name
        v: k for k, v in state_mapping.items()
    }
    return [
        getattr(State, reverse_state_mapping[state_index]).value
        if state_index in reverse_state_mapping
        else -1
        for state_index in predicted_state_indexes
    ]


def e_measure(TP=0, FP=0, FN=0, beta=0.5, **kwargs):
    """Effectiveness measure

    As defined by van Rijsbergen at page 140 of Information Retrieval, 2nd ed.
    (Available here:
        http://openlib.org/home/krichel/courses/lis618/readings/
                                                rijsbergen79_infor_retriev.pdf
    )
                                1
    E = 1 - -----------------------------------------------
            (alpha / precision) + ( (1 - alpha) / recall )

    where:  alpha = 1 / (beta^2 + 1)

    iff
        beta = 1 (i.e. alpha = 0.5)  precision and recall are equally important

    e.g.                                                  (more interested in)
        beta = 0.5  weights recall less than precision    (      recall      )
        beta = 2.0  weights recall more than precision    (     precision    )
    """
    assert beta > 0
    _precision = precision(TP, FP)
    _recall = recall(TP, FN)
    if _precision * _recall < 1e-4:  # avoid division by zero
        return 1e-4
    # simplification taken from page 3 of
    # https://ccs.neu.edu/home/vip/teach/IRcourse/5_eval_userstudy/other_notes/
    #                                                               metrics.pdf
    return min(
        1 - 1e-4,
        max(
            1e-4,
            1
            - (
                ((power(beta, 2) + 1) * _precision * _recall)
                / (power(beta, 2) * _precision + _recall)
            ),
        ),
    )


def f1_score(TP=0, FP=0, FN=0, **kwargs):
    """https://en.wikipedia.org/wiki/F-score#Definition"""
    return 1 - e_measure(TP, FP, FN, beta=1)


def get_meta_model_path(dirpath, choices):
    for path in choices:
        if dirpath.startswith(path):
            return path
    raise KeyError(
        f"Could not find MetaModel path of {repr(dirpath)} "
        f"among {repr(choices)}"
    )


def hmm_description(hmm_data, light_mm_data):
    return (
        repr(
            dict(
                oxygen_states=oxygen_states_description(
                    hmm_data, light_mm_data
                ),
                seed=f"{light_mm_data['seed']:06d}",
            )
        )
        .strip("{}")
        .replace("'", "")
        .replace(": ", "=")
    )


def hmm_predictions(hmm, hmm_data, test_df, mm_test_df):
    test_matrix = dataframe_to_numpy_matrix(
        test_df,
        only_columns=sort_cols_as_in_df(test_df.columns, mm_test_df),
    )
    return (
        dict(
            enumerate(
                convert_hmm_predictions(
                    hmm.predict(test_matrix, algorithm="map"),
                    hmm_data["state_mapping"],
                )
            )
        ),
        test_matrix,
    )


def load(*args):
    load_method, load_kwargs = None, dict()
    if len(args) != 1 and all(isinstance(p, str) for p in args):
        input_file = join_path(*args)
    elif len(args) == 1 and isinstance(
        args[0], (str, BufferedIOBase, TextIOBase)
    ):
        input_file = args[0]
    else:
        raise TypeError(
            "Expected sequence of path to join, single path or "
            f"single file object; got {repr(args)} instead!"
        )
    read_mode = "r"
    if any(
        (
            isinstance(input_file, str) and input_file.endswith(".yaml"),
            hasattr(input_file, "name") and input_file.name.endswith(".yaml"),
        )
    ):
        load_method = load_yaml
        load_kwargs["Loader"] = SafeLoader
    elif any(
        (
            isinstance(input_file, str) and input_file.endswith(".npy"),
            hasattr(input_file, "name") and input_file.name.endswith(".npy"),
        )
    ):
        load_method = np.load
        read_mode = "rb"
    elif any(
        (
            isinstance(input_file, str) and input_file.endswith(".csv"),
            hasattr(input_file, "name") and input_file.name.endswith(".csv"),
        )
    ):
        load_method = pd.read_csv
    if load_method is None:
        raise NotImplementedError(
            f"Loading of {repr(input_file)} is not yet supported"
        )
    try:
        if isinstance(input_file, (BufferedIOBase, TextIOBase)):
            ret = load_method(input_file, **load_kwargs)
        else:
            with open(input_file, read_mode) as file_obj:
                ret = load_method(file_obj, **load_kwargs)
    except Exception as e:
        warning(f"Could not load {repr(input_file)} " f"because: {str(e)}")
        ret = None
    else:
        debug(
            "Successfully loaded "
            + str(
                repr(input_file)
                if not hasattr(input_file, "name")
                else str(
                    f"{type(input_file).__name__}"
                    f"(name={repr(input_file.name)})"
                ).ljust(96)
            )
            + str(f" ({repr(ret)})" if "\n" not in repr(ret) else "")
        )
        if isinstance(ret, pd.DataFrame):
            # drop any "index" columns, i.e. unnamed columns
            ret = ret.loc[
                :,
                [
                    col
                    for col in ret.columns
                    if not col.lower().startswith("unnamed")
                ],
            ]
    finally:
        return ret


@black_magic
def load_composer_models(composer_input_dir):
    assert isdir(composer_input_dir)
    ret = dict()

    smw_in_queue, smw_out_queue, smw_err_queue = InputOutputErrorQueues(
        Queue(), Queue(), Queue()
    )
    submodel_workers = [
        Process(
            target=_submodel_worker_body,
            args=(smw_in_queue, smw_out_queue, smw_err_queue),
        )
        for _ in range(cpu_count())
    ]
    for smw in submodel_workers:
        smw.start()

    for dirpath, dirnames, filenames in walk(
        composer_input_dir, followlinks=True
    ):
        debug(f"[DIR]\t{dirpath}")
        for f in filenames:
            debug(f"[FILE]\t\t{f}")
        meta_model_path = None
        if dirpath.endswith("MetaModel_class") and "flavour.yaml" in filenames:
            meta_model_path = dirpath
            ret[meta_model_path] = load_meta_model(
                meta_model_path, smw_in_queue
            )
        continue
    for _ in submodel_workers:
        smw_in_queue.put(None)  # send termination signals
    for _ in submodel_workers:
        smw_error = smw_err_queue.get()
        if smw_error is not None:
            warning(
                f"While loading sub model: {repr(smw_error.sub_model_path)}) "
                f"got the following exception {str(smw_error.exception)}"
            )
    for _ in submodel_workers:
        while True:
            smw_output = smw_out_queue.get()
            if smw_output is None:
                break  # receive ack to termination signal
            meta_model_path, sub_model_name, sub_model_dict = smw_output
            if sub_model_name not in ret[meta_model_path]:
                ret[meta_model_path][sub_model_name] = list()
            ret[meta_model_path][sub_model_name].append(sub_model_dict)
    debug(f"Waiting for all {len(submodel_workers)} sub_model_workers to join")
    for smw in submodel_workers:
        smw.join()
    debug(f"All {len(submodel_workers)} sub_model_workers joined")

    new_ret = dict()
    for meta_model_path, mm_data in ret.items():
        mm_seed = mm_data["seed"]
        new_ret[mm_seed] = new_ret.get(mm_seed, list())
        mm_data["meta_model_path"] = meta_model_path
        new_ret[mm_seed].append(mm_data)
    return new_ret


def load_meta_model(meta_model_path, workload_queue):
    ret = dict()
    debug(f"Started loading {repr(meta_model_path)}")
    for k, filename in META_MODEL_DATA.items():
        ret[k] = load(meta_model_path, filename)
    for dirpath, dirnames, filenames in walk(
        meta_model_path, followlinks=True
    ):
        if dirpath.endswith("MetaModel_class") and "flavour.yaml" in filenames:
            continue
        for sub_model_name, sub_model_data in (
            ("LightMetaModel_class", LIGHT_MM_DATA),
            ("HiddenMarkovModel_class", HMM_DATA),
        ):
            if f"{sub_model_name}.zip" in filenames:
                workload_queue.put(
                    SubModelWorkerInput(
                        meta_model_path,
                        join_path(dirpath, f"{sub_model_name}.zip"),
                        sub_model_name,
                        sub_model_data,
                    )
                )
            elif dirpath.endswith(sub_model_name):
                workload_queue.put(
                    SubModelWorkerInput(
                        meta_model_path,
                        dirpath,
                        sub_model_name,
                        sub_model_data,
                    )
                )
    return ret


def measure_performance(predicted_labels, real_labels, return_string_of=""):
    assert len(predicted_labels) == len(real_labels), str(
        "Please provide sequences of the same length"
        f"\t\t{len(predicted_labels)}\t{len(real_labels)}\n"
        f"{repr(predicted_labels)}\n"
        f"{repr(real_labels)}\n"
    )
    per_state_confusion_matrix = dict()
    for i, real_state_val in real_labels.items():
        real_state = State(real_state_val).name
        per_state_confusion_matrix[real_state] = update_confusion_matrix(
            per_state_confusion_matrix.pop(real_state, dict()),
            predicted_labels[i],
            real_state_val,
        )
    debug(repr(per_state_confusion_matrix))
    ret = {
        state_name: {
            metric.__name__: metric(**conf_mat)
            for metric in (accuracy, precision, recall, f1_score, e_measure)
        }
        for state_name, conf_mat in per_state_confusion_matrix.items()
    }
    if ret and return_string_of in ret.get(list(ret.keys())[0], dict()):
        return all(
            ret[state_name][return_string_of] < 1e-4
            for state_name in set(State.names()).intersection(set(ret.keys()))
        ), (
            "\t| "
            + str(
                " | ".join(
                    f"{state_name}="
                    + f"{ret[state_name][return_string_of]:6.4f}".rstrip("0")
                    .rstrip(".")
                    .ljust(6)
                    for state_name in State.names()
                    if state_name in ret
                )
            )
            + f" |{' '*4}[{return_string_of.capitalize().replace('_', '-'):9}]"
        )
    elif return_string_of != "":
        warning(
            f"Could not find {repr(return_string_of)} in"
            " measured performance dict "
            f"(keys: {sorted(ret.get(list(ret.keys())[0], dict()).keys())}"
        )
    return ret


def model_is_inaccurate(hmm_data, light_mm_data):
    if hmm_data["score"] is None or hmm_data["score"] < getattr(
        parsed_args(), "minimum_score_threshold"
    ):
        debug(
            "Ignoring HiddenMarkovModel("
            f"{hmm_description(hmm_data, light_mm_data)}"
            f") with accuracy: {hmm_data['score']:15.9f}"
        )
        return True
    return False


def model_is_overfitted(hmm_data, light_mm_data):
    if all(
        (
            str(getattr(parsed_args(), "ignore_overfitted_models", True))
            == str(True),
            len(set(hmm_data["predicted_validation_labels"])) == 1,
            len(set(hmm_data["actual_validation_labels"])) != 1,
        )
    ):
        debug(
            "Ignoring HiddenMarkovModel("
            f"{hmm_description(hmm_data, light_mm_data)}"
            f") always predicting: "
            + str(
                State(
                    convert_hmm_predictions(
                        hmm_data["predicted_validation_labels"],
                        hmm_data["state_mapping"],
                    )[0]
                ).name
            )
        )
        return True
    return False


def most_agreed_outcomes(prediction_pool, sum_term, coeff=1, **kwargs):
    ret = dict()
    for i, predictions in prediction_pool.items():
        outcome_partial_sum = defaultdict(float)
        for model_ranking_data in predictions:
            if model_ranking_data.get("banned", False):
                continue
            if "prediction" not in model_ranking_data:
                continue
            outcome_partial_sum[model_ranking_data["prediction"]] += power(
                model_ranking_data[sum_term]
                * model_ranking_data.get(coeff, 1),
                2,  # euclidean norm (step 1/2)
            )
        if len(outcome_partial_sum) < 1:
            ret[i] = -1
        else:
            ret[i] = sorted(  # outcome with highest partial sum
                outcome_partial_sum.items(),
                key=lambda t: sqrt(t[1]),  # euclidean norm (step 2/2)
                reverse=True,
            )[0][0]
    return ret


def oracle_cherry_picker(prediction_pool, test_labels, use_flag=True):
    good_models = set()
    predictions = [-1 for i in test_labels.keys()]
    for i, actual_state_val in test_labels.items():
        for model_ranking_data in prediction_pool[i]:
            if any(
                (
                    use_flag and model_ranking_data["oracle_flag"],
                    int(actual_state_val)
                    == int(model_ranking_data.get("prediction", -1)),
                )
            ):
                predictions[i] = actual_state_val
                good_models.add(model_ranking_data["description"])
    return predictions, good_models


def outcome_is_banned(
    outcome, prediction_counter, max_count=100, most_common=2
):
    if str(getattr(parsed_args(), "ban_too_learned_outcomes", False)) == str(
        False
    ):
        return False
    return all(
        (
            prediction_counter[outcome] > max_count,
            outcome in dict(prediction_counter.most_common(most_common)),
        )
    )


def oxygen_states_description(hmm_data=dict(), light_mm_data=dict()):
    assert hmm_data or light_mm_data, "Please provide at least one argument"
    if "state_names" in hmm_data["init_kwargs"]:
        ret = repr(hmm_data["init_kwargs"]["state_names"])
    ret = repr([State(i).name for i in light_mm_data["oxygen_states"]])
    return ret.strip("[]").replace("'", "")


def precision(TP=0, FP=0, **kwargs):
    """https://en.wikipedia.org/wiki/Precision_and_recall"""
    if sum((TP, FP)) < 1e-4:  # avoid division by zero
        return 1e-4
    return min(1 - 1e-4, max(1e-4, float(TP / (TP + FP))))


def recall(TP=0, FN=0, **kwargs):
    """https://en.wikipedia.org/wiki/Precision_and_recall"""
    if sum((TP, FN)) < 1e-4:  # avoid division by zero
        return 1e-4
    return min(1 - 1e-4, max(1e-4, float(TP / (TP + FN))))


def run():
    assert getattr(parsed_args(), "composer_input_dir", None) is not None
    assert (
        getattr(parsed_args(), "minimum_score_threshold") > 0
        and getattr(parsed_args(), "minimum_score_threshold") < 1
    ), str("--minimum-score-threshold must be in (0, 1)")

    initialize_logging(
        f"{__name__.replace('.', '_')}_{run.__name__}__debug.log",
        getattr(parsed_args(), "log_level", LOGGING_LEVEL),
        debug_mode=getattr(parsed_args(), "verbose", False),
    )

    data = load_composer_models(
        abspath(getattr(parsed_args(), "composer_input_dir"))
    )

    for mm_seed, mm_list in data.items():
        info(f"Started composing descendands of MetaModel(seed={mm_seed})")
        fw_in_queue, fw_out_queue, fw_err_queue = InputOutputErrorQueues(
            Queue(), Queue(), Queue()
        )
        meta_model_path = "Unknown"
        first_test_set, prediction_pool, test_labels = None, None, None
        while mm_list:
            mm_data = mm_list.pop()
            mm_test_df = mm_data.get("test_df", None).astype(
                {rename_helper("ActualState_val"): pd.Int64Dtype()}
            )
            if first_test_set is None:
                meta_model_path = mm_data.get("meta_model_path", "Unknown")
                first_test_set = test_set_skeleton(mm_test_df)
                test_labels = mm_test_df[
                    rename_helper("ActualState_val")
                ].to_dict()
                debug(
                    "test_labels = "
                    + repr({k: State(v) for k, v in test_labels.items()})
                )
                prediction_pool = {k: list() for k in test_labels.keys()}
                filter_workers = [
                    Process(
                        target=_filter_worker_body,
                        args=(
                            mm_test_df,
                            test_labels,
                            fw_in_queue,
                            fw_out_queue,
                            fw_err_queue,
                        ),
                    )
                    for _ in range(cpu_count())
                ]
                for fw in filter_workers:
                    fw.start()
            assert first_test_set == test_set_skeleton(mm_test_df), str(
                "test_labels must be the same if MetaModels were "
                f"built with the same seed ({mm_seed})\n\t"
                f"first_test_set_found = {repr(first_test_set)}\n\t"
                f"current_test_set     = {repr(test_set_skeleton(mm_test_df))}"
            )
            for light_mm_data, hmm_data in zip(
                mm_data.get("LightMetaModel_class", list()),
                mm_data.get("HiddenMarkovModel_class", list()),
            ):
                fw_in_queue.put(FilterWorkerInput(light_mm_data, hmm_data))
        for _ in filter_workers:
            fw_in_queue.put(None)

        stats = dict()
        for _ in filter_workers:
            fw_error = fw_err_queue.get()
            if fw_error is not None:
                warning(
                    f"While filtering HMM({repr(fw_error.hmm_description)}) "
                    f"got the following exception: {str(fw_error.exception)}."
                )
        for _ in filter_workers:
            while True:
                fw_output = fw_out_queue.get()
                if fw_output is None:
                    break  # receive ack to termination signal
                ith_record, model_ranking_data, fw_stats = fw_output
                if ith_record is not None and model_ranking_data is not None:
                    prediction_pool[ith_record].append(model_ranking_data)
                if fw_stats is not None:
                    for k1, v1 in fw_stats.items():
                        if isinstance(v1, int):
                            stats[k1] = stats.get(k1, 0) + v1
                        else:
                            for k2, v2 in v1.items():
                                stats[k1] = stats.get(k1, dict())
                                stats[k1][k2] = stats[k1].get(k2, 0) + v2
        debug(f"Waiting for all {len(filter_workers)} filter_workers to join")
        for fw in filter_workers:
            fw.join()
        debug(f"All {len(filter_workers)} filter_workers to joined")

        prediction_pool = add_votes_to_ranking_data(
            prediction_pool, test_labels
        )

        info("")
        if any(k.startswith("dropped_") and v for k, v in stats.items()):
            info("The following HMMs were dropped and NOT SEEN BY the ORACLE:")
            for reason, amount in sorted(
                stats.items(), key=lambda t: t[1], reverse=True
            ):
                if not reason.startswith("dropped_"):
                    continue
                info(
                    f"{stats[reason]:5d} because "
                    + reason.replace("dropped_", "").replace("_", " ")
                )
            info("")
        info(
            f"{stats['useful_hmm']:5d} useful HMMs were found "
            f"(among {stats['total_hmm']:5d}; "
            f"i.e. {100*stats['useful_hmm']/stats['total_hmm']:.2f}%)"
        )
        show_oracle_performance_in_log(
            prediction_pool, test_labels, stats["total_hmm"], use_flag=True
        )

        for reason, amount in sorted(
            stats.items(), key=lambda t: t[1], reverse=True
        ):
            if not reason.endswith("_outcomes") or reason.startswith("total_"):
                continue
            info(
                f"{100 * stats[reason] / stats['total_outcomes']:6.2f}%"
                " outcomes were on average ignored because "
                + reason.replace("_outcomes", "").replace("_", " ")
            )

        commutative_operands = set()
        info("")
        info(
            f"Descendants of MetaModel(seed={mm_seed}, path=results/"
            f"{meta_model_path.split('results/')[1].split('/')[0]}):"
        )
        for coeff in (
            "1",
            "log_prob",
            "outcome_prob",
            "inv_outcome_prob",
            "votes",
            "inv_votes",
        ):
            for by in (
                "accuracy",
                "precision",
                "recall",
                "e_measure",
                "f1_score",
                #
                # "votes",
                # "inv_votes",
                "log_prob",
                "outcome_prob",
                "inv_outcome_prob",
            ):
                if by == coeff or (
                    tuple(sorted([by.lower(), coeff.lower()]))
                    in commutative_operands
                ):
                    continue
                commutative_operands.add(
                    tuple(sorted([by.lower(), coeff.lower()]))
                )
                valid_predictions_ranked = {  # sort once
                    i: sorted(
                        [
                            {
                                k: d[k]
                                for k in ("prediction", by, coeff)
                                if k in d
                            }
                            for d in predictions
                            if not d["banned"] and by in d
                        ],
                        key=lambda d: d.get(by) * d.get(coeff, 1),
                        reverse=True,
                    )
                    for i, predictions in prediction_pool.items()
                }
                for mode, by, mult, fun, kwargs in (
                    (
                        "ranked",
                        f" Decreasing [{repr(by)}",
                        f" * {repr(coeff)}]",
                        top_ranked_outcomes,
                        dict(sort_by=by),
                    ),
                    (
                        "summed",
                        f" EuclidNorm ({repr(by)}",
                        f" * {repr(coeff)})",
                        most_agreed_outcomes,
                        dict(sum_term=by, coeff=coeff),
                    ),
                ):
                    show_performance_in_log(
                        ("e_measure",),
                        mode,
                        by,
                        mult,
                        fun,
                        valid_predictions_ranked,
                        test_labels,
                        **kwargs,
                    )
            info("")
        for sorting, sum_term in (
            ("Decreasing", "votes"),
            ("Increasing", "inv_votes"),
        ):
            show_performance_in_log(
                ("e_measure",),
                "summed",
                repr(f"{sorting} outcome votes"),
                "",
                most_agreed_outcomes,
                prediction_pool,
                test_labels,
                sum_term=sum_term,
            )

        show_oracle_performance_in_log(
            prediction_pool, test_labels, stats["total_hmm"], use_flag=False
        )

        if not getattr(parsed_args(), "verbose", False):
            debug(
                "Please provide -v/--verbose CLI argument to "
                "see ranked models in logs"
            )


def show_oracle_performance_in_log(
    prediction_pool, test_labels, total_hmm, use_flag
):
    info("")
    oracle_predictions, oracle_good_models = oracle_cherry_picker(
        prediction_pool, test_labels, use_flag=use_flag
    )
    show_performance_in_log(
        ("e_measure",),
        "picked",
        f"'ORACLE'{' (by flag)' if use_flag else ' (from pool)'}",
        "",
        lambda _: oracle_predictions,
        None,
        test_labels,
    )
    info("")
    info(
        f"{len(oracle_good_models):5d} "
        f"({100*len(oracle_good_models)/total_hmm:.2f}%) "
        f"good models were found by ORACLE"
    )
    info("")


def show_performance_in_log(
    metrics,
    mode,
    by,
    mult,
    fun,
    valid_predictions,
    test_labels,
    pad_by=32,
    pad_mult=25,
    **kwargs,
):
    criterion = by.ljust(pad_by) + mult.ljust(pad_mult)
    if mode.lower().strip() == "ranked":
        show_ranked_models_in_log(
            valid_predictions,
            test_labels,
            kwargs.get("sort_by", "UNKNOWN"),
            criterion,
        )
    for metric in metrics:
        all_zero, perf_str = measure_performance(
            fun(valid_predictions, **kwargs),
            test_labels,
            return_string_of=metric,
        )
        if (all_zero and mode != "picked") or metric != "e_measure":
            debug(f"{mode} by: {criterion} performed: {perf_str}")
            continue
        else:
            info(f"{mode} by: {criterion} performed: {perf_str}")


def show_ranked_models_in_log(
    prediction_pool, test_labels, sort_label, rank_key
):
    if not getattr(parsed_args(), "verbose", False):
        return
    debug(f"Ranking criteria is: {rank_key}")
    for i, predictions in prediction_pool.items():
        debug(
            " | ".join(
                (
                    f"test_set_sample = {i:6d}",
                    "actual_state = " + State(test_labels[i]).name.rjust(12),
                    "ranking = "
                    + str(
                        ", ".join(
                            [
                                f"({State(d['prediction']).name},"
                                f" {d[sort_label]:.6f})"
                                for d in predictions
                            ]
                        ).replace("'", "")
                        if predictions
                        else "empty list"
                    ),
                )
            )
        )


def test_set_feasible_subset(test_set_df, light_mm_data):
    return test_set_df.loc[
        test_set_df[rename_helper("ActualState_val")].isin(
            light_mm_data["oxygen_states"]
        ),
        set(test_set_df.columns).intersection(
            set(
                list(light_mm_data["observed_variables"])
                + [rename_helper("ActualState_val")]
            )
        ),
    ]


def test_set_skeleton(test_set_df):
    """Return the following columns: patient_id, date, oxygen_state.

    Since other observable variables may vary due to different outlier
    limits.
    """
    return test_set_df.loc[
        :,
        rename_helper(("", "date", "ActualState_val")),
    ].to_dict()


def top_ranked_outcomes(prediction_pool, **kwargs):
    return {
        i: predictions[0]["prediction"] if predictions else -1
        for i, predictions in prediction_pool.items()
    }


def under_stress_predictions(hmm, mm_test_df, light_mm_data, hmm_data):
    learned_columns = test_set_feasible_subset(
        mm_test_df, light_mm_data
    ).columns
    ret = {i: set() for i in range(mm_test_df.shape[0])}
    for oxygen_state_val in light_mm_data["oxygen_states"]:
        # let us perturbe all the observations in MetaModels' test
        # dataframe; not just those in the feasible test dataframe
        perturbed_test_df = mm_test_df.copy()
        perturbed_test_df.loc[
            :, rename_helper("ActualState_val")
        ] = oxygen_state_val
        for i, predicted_test_label in enumerate(
            tuple(
                convert_hmm_predictions(
                    hmm.predict(
                        dataframe_to_numpy_matrix(
                            perturbed_test_df,
                            only_columns=sort_cols_as_in_df(
                                learned_columns, mm_test_df
                            ),
                        ),
                        algorithm="map",
                    ),
                    hmm_data["state_mapping"],
                )
            )
        ):
            ret[i].add(predicted_test_label)
    return ret, Counter(itertools.chain.from_iterable(ret.values()))


def update_confusion_matrix(conf_mat, predicted_class, real_class):
    """https://en.wikipedia.org/wiki/Type_I_and_type_II_errors

    TP =  true positive
    TN =  true negative
    FP = false positive (or overestimation)
    FN = false negative (or underestimation)
    """
    conf_mat = defaultdict(
        lambda: 1e-4,  # missing keys start from (almost) zero
        **conf_mat,
    )
    if int(predicted_class) == int(real_class):
        conf_mat["TP"] += 1  # actually this is (TP + TN)
    elif predicted_class < 0:  # unknown prediction
        conf_mat["FP"] += 0.5
        conf_mat["FN"] += 0.5
    elif predicted_class > real_class:  # overestimation, type error I
        conf_mat["FP"] += 1
    else:  # underestimation, type error II
        conf_mat["FN"] += 1
    return conf_mat


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.composer",
    "YAstarMM.composer",
    "composer",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
