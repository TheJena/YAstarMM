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
   Compute Charlson-Index.
"""

from .column_rules import rename_helper
from .constants import TRUTH_DICT
from collections import Counter
import numpy as np
import pandas as pd

CHARLSON_COUNTER = Counter()


def reset_charlson_counter():
    global CHARLSON_COUNTER
    CHARLSON_COUNTER = Counter()


def most_common_charlson():
    for col, count in CHARLSON_COUNTER.most_common():
        yield col, count


def max_charlson_col_length():
    return max(len(c) for c in CHARLSON_COUNTER.keys())


def estimated_ten_year_survival(charlson_index):
    if pd.isna(charlson_index):
        return np.nan
    return 0.983 * np.exp(charlson_index * 0.9)


def compute_charlson_index(patient_df, valid_column_threshold=8):
    """Based on https://www.mdcalc.com/charlson-comorbidity-index-cci

    :param valid_column_threshold: is the number of
    returning-points-functions which must contain non-NaN information
    to consider the computed charlson-index valid.

    If you want to change the default value of the
    valid_column_threshold please have a look (in the log) to the
    report of the most common funtions-returning-points.

    Since more or less half of the patients' data in the dataset has
    been collected before the inclusion of the pieces of information
    necessary to compute the charlson-index, we need to understand if
    the given patient data is antecedent or consequent to that event.

    That way we can assume that the collection of the data needed to
    compute the charlson-index had already been started and that any
    missing information can be valued as zero points.

    The latter assumption is based on the fact that the diseases on
    which is based the charlson-index are quite irreversible and
    severe, so a doctor would never omit this piece of information
    given its importance in the choice of the patient therapy.
    """
    global CHARLSON_COUNTER

    cci, valid_columns = 0, 0
    for function_returning_points in (
        age_points,  # everybody has it
        myocardial_infarction_points,
        congestive_heart_failure_points,
        peripheral_vascular_disease_points,
        cerebrovascular_accident_or_transient_ischemic_attack_points,
        dementia_points,
        chronic_obstructive_pulmonary_disease_points,
        connective_tissue_disease_points,
        peptic_ulcer_disease_points,
        liver_disease_points,
        diabetes_mellitus_points,
        hemiplegia_points,
        moderate_to_severe_chronic_kidney_disease_points,
        solid_tumor_points,
        leukemia_points,
        lymphoma_points,
        aids_points,
    ):
        try:
            cci += function_returning_points(patient_df)
        except FileNotFoundError:  # necessary columns were all empty :(
            CHARLSON_COUNTER.update({function_returning_points.__name__: 0})
        else:
            valid_columns += 1
            CHARLSON_COUNTER.update({function_returning_points.__name__: 1})

    if valid_columns >= valid_column_threshold:
        # At least valid_column_threshold functions (out of 17)
        # successfully found the pieces of information necessary to
        # estimate the patient points for a given disease; we can
        # pretty safely assume that the collection of those pieces of
        # information had already been started; thus any missing
        # information can be considered as zero points.
        return cci
    else:
        # No assumption can be done on the missing data (because the
        # data collection of the pieces of information needed to
        # compute the charlson-index was not yet started).
        return np.nan


def age_points(patient_df):
    """1 point for every decade age 50 years and over, maximum 4 points.

    return 0 from  0 to 49 years
    return 1 from 50 to 59 years
    return 2 from 60 to 69 years
    return 3 from 70 to 79 years
    return 4 from 80 or more years
    """
    return max(
        0, min(4, (patient_df.loc[:, "age"].max() - 40) // 10)
    )


def myocardial_infarction_points(patient_df):
    """History of definite or probable MI (EKG changes and/or enzyme changes)

    https://en.wikipedia.org/wiki/Myocardial_infarction
    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def congestive_heart_failure_points(patient_df):
    """Exertional or paroxysmal nocturnal dyspnea and has responded to
    digitalis, diuretics, or afterload reducing agents

    https://en.wikipedia.org/w/index.php?title=Congestive_heart_failure

    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                }.get(str(value), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def peripheral_vascular_disease_points(patient_df):
    """Intermittent claudication or past bypass for chronic arterial
    insufficiency, history of gangrene or acute arterial insufficiency, or
    untreated thoracic or abdominal aneurysm (≥6 cm)

    https://en.wikipedia.org/wiki/Peripheral_artery_disease
    https://en.wikipedia.org/w/index.php?title=Peripheral_vascular_disease

    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(
            lambda value: {
                "": 0,
                "": 1,
            }.get(str(value).lower().replace(" ", ""), np.nan)
        )
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def cerebrovascular_accident_or_transient_ischemic_attack_points(patient_df):
    """History of a cerebrovascular accident with minor or no residua and
    transient ischemic attacks

    https://en.wikipedia.org/wiki/Transient_ischemic_attack
    https://en.wikipedia.org/w/index.php?title=Cerebrovascular_accident

    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(
            lambda value: {
                "": 0,
                "": 1,
            }.get(str(value), np.nan)
        )
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def dementia_points(patient_df):
    """Chronic cognitive deficit

    https://en.wikipedia.org/wiki/Dementia
    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def chronic_obstructive_pulmonary_disease_points(patient_df):
    """Pulmonary disease

    https://en.wikipedia.org/wiki/Chronic_obstructive_pulmonary_disease
    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper(("", ""))]
            .applymap(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max()
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                }.get(str(value).lower().replace(" ", ""), np.nan)
            )
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def connective_tissue_disease_points(patient_df):
    """Group of diseases affecting the bodies connective tissue such as
    fat, bone or cartilage

    https://en.wikipedia.org/wiki/Connective_tissue_disease

    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(
            lambda value: {
                "": 0,
                "": 1,
            }.get(str(value).lower().replace(" ", ""), np.nan)
        )
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def peptic_ulcer_disease_points(patient_df):
    """Any history of treatment for ulcer disease or history of ulcer bleeding

    https://en.wikipedia.org/wiki/Peptic_ulcer_disease
    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(
            lambda value: {
                "": 0,
                "": 1,
            }.get(str(value).lower().replace(" ", ""), np.nan)
        )
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Yes == 1


def liver_disease_points(patient_df):
    """Mild = chronic hepatitis (or cirrhosis without portal hypertension)
    Moderate = cirrhosis and portal hypertension but no variceal bleeding hist.
    Severe = cirrhosis and portal hypertension with variceal bleeding history

    https://en.wikipedia.org/wiki/Liver_disease
    """
    ret = pd.Series(
        [
            patient_df.loc[
                :, rename_helper(("", ""))
            ]
            .applymap(
                lambda value: {
                    "": 0,
                    "": 1,
                }.get(str(value).lower().replace(" ", ""), np.nan)
            )
            .max()
            .max(),
            patient_df.loc[
                :,
                rename_helper(
                    (
                    )
                ),
            ]
            .applymap(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max()
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    if all(
        (
            ret >= 1,  # mild leaver disease
            patient_df.loc[
                :,
                rename_helper(("", "")),
            ]
            .applymap(lambda value: TRUTH_DICT.get(str(value).lower(), 0))
            .max()
            .max()
            >= 1,  # plus hypertension
        )
    ):
        return 3  # Moderate-to-Severe == 3
    return ret  # No == 0, Mild == 1


def diabetes_mellitus_points(patient_df):
    """Uncomplicated (1) or with end-organ damage (2)

    https://en.wikipedia.org/w/index.php?title=Diabetes_mellitus
    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                    "": 2,
                }.get(str(value), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper(("", ""))]
            .applymap(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max()
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Uncomplicated == 1, End-organ-damage == 2


def hemiplegia_points(patient_df):
    """Weakness of one entire half side of the body, consequence or not of stroke

    https://en.wikipedia.org/wiki/Hemiparesis
    https://en.wikipedia.org/w/index.php?title=Hemiplegia
    """
    ret = (
        patient_df.loc[:, rename_helper("")]
        .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
        .max()
    )
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret * 2  # No == 0, Yes == 2


def moderate_to_severe_chronic_kidney_disease_points(patient_df):
    """Severe   = on dialysis, status post kidney transplant, uremia,
    Moderate = creatinine >3 mg/dL (0.27 mmol/L)

    https://en.wikipedia.org/wiki/Chronic_kidney_disease
    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                }.get(str(value), np.nan)
            )
            .max(),
            patient_df.loc[
                :,
                rename_helper(("", "")),
            ]
            .applymap(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max()
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    if ret == 0:
        max_creatinine = patient_df.loc[:, rename_helper("")].max()
        if pd.notna(max_creatinine) and max_creatinine > 3.0:  # mg/dL
            return 2  # moderate
    return ret * 2  # No == 0, Yes == 2


def solid_tumor_points(patient_df):
    """Localized solid tumor (2) or Metastatic solid tumor (6)

    https://en.wikipedia.org/wiki/Neoplasm
    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 2,
                    "": 6,
                }.get(str(value), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 2,
                    "": 6,
                }.get(str(value), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret  # No == 0, Localized == 2, Metastatic == 6


def leukemia_points(patient_df):
    """Malignancy consisting in increased number of immature and/or
    abnormal leucocytes

    https://en.wikipedia.org/wiki/Leukemia

    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                }.get(str(value).lower().replace(" ", ""), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                }.get(str(value).lower(), np.nan)
            )
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret * 2  # No == 0, Yes == 2


def lymphoma_points(patient_df):
    """Cancerous spread in the lymph system

    https://en.wikipedia.org/wiki/Lymphoma
    """
    # Unfortunately we do not have enough information to distinguish
    # between Leukemia and Lymphoma; we can only know if a (generical)
    # severe ematologic disease afflicts the patient
    raise FileNotFoundError()


def aids_points(patient_df):
    """Acquired Immune Deficiency Syndrome caused by the Human
    Immunodeficiency Virus (HIV)

    https://en.wikipedia.org/wiki/HIV/AIDS

    """
    ret = pd.Series(
        [
            patient_df.loc[:, rename_helper("")]
            .apply(
                lambda value: {
                    "": 0,
                    "": 1,
                    "": 1,
                    "": 1,
                }.get(str(value).lower().replace(" ", ""), np.nan)
            )
            .max(),
            patient_df.loc[:, rename_helper("")]
            .apply(lambda value: TRUTH_DICT.get(str(value).lower(), np.nan))
            .max(),
        ]
    ).max()
    if pd.isna(ret):
        raise FileNotFoundError()
    return ret * 6  # No == 0, Yes == 6
