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
   Compose several Hidden Markov Models.

   Predict patients' future State by choosing the more probable State
   among the ones returned by the most performant HMM for each
   possible state transition.

   Usage:
            from  YAstarMM.composer  import  run as run_composer
"""


from .column_rules import rename_helper
from .constants import LOGGING_LEVEL, MIN_PYTHON_VERSION
from .flavoured_parser import parsed_args
from .hmm import dataframe_to_numpy_matrix, sort_cols_as_in_df
from .model import State
from .utility import black_magic, initialize_logging
from collections import Counter
from functools import lru_cache
from io import BufferedIOBase, TextIOBase
from logging import debug, info, warning
from os import walk
from os.path import basename, isdir, join as join_path
from pomegranate import HiddenMarkovModel
from sys import version_info
from yaml import load as load_yaml, SafeLoader
from zipfile import ZipFile
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
    init_kwargs="hmm_constructor_parameters.yaml",
    # log probability of observing sequences like those in validation matrix
    score="score.yaml",  # scalar float (accuracy of the model)
    serialized_obj="hmm_trained_and_serialized.yaml",
    state_mapping="state_mapping.yaml",
)


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


def get_meta_model_path(dirpath, choices):
    for path in choices:
        if dirpath.startswith(path):
            return path
    raise KeyError(
        f"Could not find MetaModel path of {repr(dirpath)} "
        f"among {repr(choices)}"
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
    for dirpath, dirnames, filenames in walk(
        composer_input_dir, followlinks=True
    ):
        debug(f"[DIR]\t{dirpath}")
        for f in filenames:
            debug(f"[FILE]\t\t{f}")
        meta_model_path = None
        if dirpath.endswith("MetaModel_class") and "flavour.yaml" in filenames:
            meta_model_path = dirpath
            ret[meta_model_path] = ret.get(meta_model_path, dict())
            for k, filename in META_MODEL_DATA.items():
                ret[meta_model_path][k] = load(meta_model_path, filename)
            continue
        for sub_model_name, sub_model_data in (
            ("LightMetaModel_class", LIGHT_MM_DATA),
            ("HiddenMarkovModel_class", HMM_DATA),
        ):
            try:
                meta_model_path = get_meta_model_path(
                    dirpath, choices=ret.keys()
                )
            except KeyError as e:
                if ret and filenames:
                    warning(str(e))
                continue
            debug(f"Relative MetaModel is: {repr(meta_model_path)}")
            try:
                seed = int(basename(dirpath).replace("seed_", "").lstrip("0"))
            except ValueError:
                pass
            else:
                debug(f"Current       seed is:  {seed:06d}")
            sub_model = dict()
            if f"{sub_model_name}.zip" in filenames:
                with ZipFile(
                    join_path(dirpath, f"{sub_model_name}.zip")
                ) as sub_model_zip:
                    for k, filename in sub_model_data.items():
                        with sub_model_zip.open(
                            join_path(sub_model_name, filename)
                        ) as file_obj:
                            sub_model[k] = load(file_obj)
            elif dirpath.endswith(sub_model_name):
                for k, filename in sub_model_data.items():
                    sub_model[k] = load(dirpath, filename)
            if sub_model:
                ret[meta_model_path][sub_model_name] = ret[
                    meta_model_path
                ].get(sub_model_name, list())
                ret[meta_model_path][sub_model_name].append(sub_model)
    new_ret = dict()
    for meta_model_path, mm_data in ret.items():
        mm_seed = mm_data["seed"]
        new_ret[mm_seed] = new_ret.get(mm_seed, list())
        mm_data["meta_model_path"] = meta_model_path
        new_ret[mm_seed].append(mm_data)
    return new_ret


@lru_cache(maxsize=None, typed=True)
def outcome_probability(label, all_labels, inverse=False):
    assert isinstance(all_labels, tuple)
    prob = float(Counter(all_labels)[label]) / len(all_labels)
    if inverse:
        return 1 - prob
    return prob


def run():
    assert getattr(parsed_args(), "composer_input_dir", None) is not None

    initialize_logging(
        f"{__name__.replace('.', '_')}_{run.__name__}__debug.log",
        getattr(parsed_args(), "log_level", LOGGING_LEVEL),
        debug_mode=getattr(parsed_args(), "verbose", False),
    )

    data = load_composer_models(getattr(parsed_args(), "composer_input_dir"))

    for mm_seed, mm_list in data.items():
        info(f"Started composing descendands of MetaModel(seed={mm_seed})")
        useful_hmm, total_hmm = 0, 0
        test_labels, prediction_pool = None, None
        for mm_data in mm_list:
            mm_test_df = mm_data.get("test_df", None).astype(
                {rename_helper("ActualState_val"): pd.Int64Dtype()}
            )
            if test_labels is None:
                test_labels = mm_test_df[
                    rename_helper("ActualState_val")
                ].to_dict()
                prediction_pool = {k: list() for k in test_labels.keys()}
                debug(
                    "test_labels = "
                    + repr({k: State(v) for k, v in test_labels.items()})
                )
            assert (
                mm_test_df[rename_helper("ActualState_val")].to_dict()
                == test_labels
            ), str(
                "test_labels must be the same if MetaModels were "
                f"built with the same seed ({mm_seed})\n\t"
                f"test_labels    = {repr(test_labels)}\n\t"
                f"mm_test_labels = "
                + repr(mm_test_df[rename_helper("ActualState_val")].to_dict())
            )
            mm_columns = sorted(mm_test_df.columns, key=str.lower)
            for light_mm_data, hmm_data in zip(
                mm_data.get("LightMetaModel_class", list()),
                mm_data.get("HiddenMarkovModel_class", list()),
            ):
                total_hmm += 1
                hmm_description = (
                    repr(
                        dict(
                            oxygen_states=hmm_data["init_kwargs"].get(
                                "state_names",
                                [
                                    State(i).name
                                    for i in light_mm_data["oxygen_states"]
                                ],
                            ),
                            seed=f"{light_mm_data['seed']:06d}",
                        )
                    )
                    .strip("{}")
                    .replace("'", "")
                    .replace(": ", "=")
                )
                if hmm_data["score"] is None or hmm_data["score"] < 0.5:
                    info(
                        f"Ignoring HiddenMarkovModel({hmm_description}) "
                        f"with accuracy: {hmm_data['score']:15.9f}"
                    )
                    continue
                test_df = mm_test_df.loc[
                    mm_test_df[rename_helper("ActualState_val")].isin(
                        light_mm_data["oxygen_states"]
                    ),
                    sorted(
                        set(mm_columns).intersection(
                            set(
                                list(light_mm_data["observed_variables"])
                                + [rename_helper("ActualState_val")]
                            )
                        ),
                        key=str.lower,
                    ),
                ]
                if test_df.empty:
                    continue
                useful_hmm += 1
                test_matrix = dataframe_to_numpy_matrix(
                    test_df,
                    only_columns=sort_cols_as_in_df(
                        test_df.columns, mm_test_df
                    ),
                )

                hidden_markov_model = HiddenMarkovModel.from_dict(
                    hmm_data["serialized_obj"]
                )
                predicted_test_labels = tuple(
                    convert_hmm_predictions(
                        hidden_markov_model.predict(
                            test_matrix, algorithm="map"
                        ),
                        hmm_data["state_mapping"],
                    )
                )

                for i, state_index in enumerate(predicted_test_labels):
                    prediction_pool[i].append(
                        {
                            "inv_outcome_prob_test": outcome_probability(
                                state_index,
                                predicted_test_labels,
                                inverse=True,
                            ),
                            "log_prob": hidden_markov_model.log_probability(
                                test_matrix[i : i + 1]  # one line i-th matrix
                            ),
                            "score": hmm_data["score"],
                            "seed": light_mm_data["seed"],
                            "outcome_prob_test": outcome_probability(
                                state_index,
                                predicted_test_labels,
                            ),
                            "prediction": state_index,
                        }
                    )

        info("")
        info(f"{useful_hmm} useful HMMs were found (among {total_hmm})")
        info("")
        for rank_criteria, sort_kwargs in {
            "Decreasing HMM accuracy": dict(by="score", reverse=True),
            "Decreasing HMM log-probability": dict(
                by="log_prob", reverse=True
            ),
            "Inverse of HMM outcome probability (test)": dict(
                by="inv_outcome_prob_test",
            ),
            "HMM outcome probability (test)": dict(
                by="outcome_prob_test",
            ),
        }.items():
            sort_label = sort_kwargs.pop("by")
            prediction_pool = {  # sort once
                i: sorted(
                    predictions, key=lambda d: d.get(sort_label), **sort_kwargs
                )
                for i, predictions in prediction_pool.items()
            }
            debug(f"Ranking criteria is: {repr(rank_criteria)}")
            for i, predictions in prediction_pool.items():
                debug(
                    " | ".join(
                        (
                            f"test_set_sample = {i:6d}",
                            "actual_state = "
                            + State(test_labels[i]).name.rjust(12),
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
            info(f"Ranking criteria is: {repr(rank_criteria)}")
            predicted_labels = {
                i: predictions[0]["prediction"] if predictions else -1
                for i, predictions in prediction_pool.items()
            }
            mm_score = np.mean(
                [
                    actual_state == predicted_labels[i]
                    for i, actual_state in test_labels.items()
                ]
            )
            info(
                f"Composer of HMMs descending from MetaModel(seed={mm_seed}) "
                f"got an {'accuracy'.center(16)} of = {mm_score:.6f}"
            )
            info("")

        info("Ranking criteria is: 'ORACLE'")
        oracle_mm_score = np.mean(
            [
                actual_state in [d["prediction"] for d in prediction_pool[i]]
                for i, actual_state in test_labels.items()
            ]
        )
        info(
            f"Composer of HMMs descending from MetaModel(seed={mm_seed}) "
            f"got an {'ORACLE-SCORE'.center(16)} of = {oracle_mm_score:.6f}"
        )
        info("")


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
