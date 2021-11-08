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
   Compute Charlson-Index.

   Usage:
            from  YAstarMM.charlson_index  import  (
                compute_charlson_index,
                estimated_ten_year_survival,
                most_common_charlson,
                reset_charlson_counter,
            )

   ( or from within the YAstarMM package )

            from          .charlson_index  import  (
                compute_charlson_index,
                estimated_ten_year_survival,
                most_common_charlson,
                reset_charlson_counter,
            )
"""

from .column_rules import charlson_enum_rule, rename_helper
from .constants import BOOLEANIZATION_MAP, ICD9_CODES, MIN_PYTHON_VERSION
from collections import Counter, defaultdict
from multiprocessing import Lock
from sys import version_info
import logging
import numpy as np
import pandas as pd


_CHARLSON_COUNTER = Counter()
_COUNTER_LOCK = Lock()


class ColumnNotFoundError(Exception):
    """Required column was not found in patient dataframe"""

    pass


def non_negative_addend(fun):
    """Decorator to raise ColumnNotFoundError on negative returned values"""

    def wrapper(*args, **kwargs):
        logger = kwargs.pop("logger", logging)
        log_prefix = kwargs.pop("log_prefix", "").rstrip()

        addend_name = repr(
            " ".join(
                f"{fun.__name__}@@END@@".replace("_points@@END@@", "")
                .lstrip("_")
                .capitalize()
                .split("_")
            )
        ).ljust(56)
        ret = fun(*args, **kwargs)
        if ret < 0:
            logger.debug(f"{log_prefix} {addend_name} = not found")
            raise ColumnNotFoundError(addend_name)
        logger.debug(
            f"{log_prefix} {addend_name} = {ret:+.3f}".replace(".000", "")
        )
        return ret

    return wrapper


@non_negative_addend
def _age_points(patient_df):
    """1 point for every decade age 50 years and over, maximum 4 points.

    return -1 if no useful column was found

    return  0 from  0 to 49 years
    return  1 from 50 to 59 years
    return  2 from 60 to 69 years
    return  3 from 70 to 79 years
    return  4 from 80 or more years
    """
    return min(
        4,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_AGE"
            ).max(),
            -1
            if rename_helper("AGE") not in patient_df.columns
            or patient_df.loc[:, rename_helper("AGE")].fillna(-1).max() < 0
            else max(
                0,
                min(
                    4,
                    (patient_df.loc[:, rename_helper("AGE")].max() - 40) // 10,
                ),
            ),
        ),
    )


@non_negative_addend
def _aids_points(patient_df):
    """Acquired Immune Deficiency Syndrome caused by the Human
    Immunodeficiency Virus (HIV)

    https://en.wikipedia.org/wiki/HIV/AIDS

    return -1 if no useful column was found

    return  0 == No
    return  6 == Yes
    """
    return 6 * min(
        1,
        int(
            max(
                charlson_series_to_integer_values(
                    patient_df, "CHARLSON_AIDS"
                ).max(),
                _series_replaced_with_mapping_or_nan(
                    patient_df, "HIV", BOOLEANIZATION_MAP
                ).max(),
                _max_icd9_code(patient_df, "aids_hiv", weight=6),
            )
        ),
    )


@non_negative_addend
def _cerebrovascular_accident_or_transient_ischemic_attack_points(patient_df):
    """History of a cerebrovascular accident with minor or no residua and
    transient ischemic attacks

    https://en.wikipedia.org/wiki/Transient_ischemic_attack
    https://en.wikipedia.org/w/index.php?title=Cerebrovascular_accident

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_CVA_OR_TIA"
            ).max(),
            _max_icd9_code(patient_df, "cerebrovascular_disease"),
        ),
    )


@non_negative_addend
def _chronic_obstructive_pulmonary_disease_points(patient_df):
    """Pulmonary disease

    https://en.wikipedia.org/wiki/Chronic_obstructive_pulmonary_disease

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_COPD"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df,
                "CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE",
                BOOLEANIZATION_MAP,
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "COPD", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "chronic_pulmonary_disease"),
        ),
    )


@non_negative_addend
def _congestive_heart_failure_points(patient_df):
    """Exertional or paroxysmal nocturnal dyspnea and has responded to
    digitalis, diuretics, or afterload reducing agents

    https://en.wikipedia.org/w/index.php?title=Congestive_heart_failure

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_HEART_FAILURE"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "HEART_FAILURE", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "congestive_heart_failure"),
        ),
    )


@non_negative_addend
def _connective_tissue_disease_points(patient_df):
    """Group of diseases affecting the bodies connective tissue such as
    fat, bone or cartilage

    https://en.wikipedia.org/wiki/Connective_tissue_disease

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_CONNECTIVE_TISSUE_DISEASE"
            ).max(),
            _max_icd9_code(patient_df, "rheumatic_disease"),
        ),
    )


@non_negative_addend
def _dementia_points(patient_df):
    """Chronic cognitive deficit

    https://en.wikipedia.org/wiki/Dementia

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_DEMENTIA"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "DEMENTIA", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "dementia"),
        ),
    )


@non_negative_addend
def _diabetes_mellitus_points(patient_df):
    """Uncomplicated (1) or with end-organ damage (2)

    https://en.wikipedia.org/w/index.php?title=Diabetes_mellitus

    return -1 if no useful column was found

    return  0 == No
    return  1 == Uncomplicated
    return  2 == End organ damage
    """
    return min(
        2,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_DIABETES"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "DIABETES", BOOLEANIZATION_MAP
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "ANOTHER_DIABETES_COLUMN", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(
                patient_df, "diabetes_without_complication", weight=1
            ),
            _max_icd9_code(
                patient_df, "diabetes_with_chronic_complication", weight=2
            ),
        ),
    )


@non_negative_addend
def _hemiplegia_points(patient_df):
    """Weakness of one entire half side of the body, consequence or not of stroke

    https://en.wikipedia.org/wiki/Hemiparesis
    https://en.wikipedia.org/w/index.php?title=Hemiplegia

    return -1 if no useful column was found

    return  0 == No
    return  2 == Yes
    """
    return 2 * min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_HEMIPLEGIA"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "CEREBROVASCULAR_DISEASE", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "hemiplegia", weight=2),
        ),
    )


@non_negative_addend
def _leukemia_points(patient_df):
    """Malignancy consisting in increased number of immature and/or
    abnormal leucocytes

    https://en.wikipedia.org/wiki/Leukemia

    return -1 if no useful column was found

    return  0 == No
    return  2 == Yes
    """
    return 2 * min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_BLOOD_DISEASE"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "BLOOD_DISEASES", BOOLEANIZATION_MAP
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "BLOOD_NEOPLASMS", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "leukemia", weight=2),
        ),
    )


@non_negative_addend
def _liver_disease_points(patient_df):
    """Mild = chronic hepatitis (or cirrhosis without portal hypertension)
    Moderate = cirrhosis and portal hypertension but no variceal bleeding hist.
    Severe = cirrhosis and portal hypertension with variceal bleeding history

    https://en.wikipedia.org/wiki/Liver_disease

    return -1 if no useful column was found

    return  0 == No
    return  1 == Mild
    return  3 == Moderate to severe
    """
    mild_liver_disease = min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_LIVER_DISEASE"
            ).max(),
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_LIVER_FAILURE"
            ).max(),
            max(
                _series_replaced_with_mapping_or_nan(
                    patient_df, boolean_column, BOOLEANIZATION_MAP
                ).max()
                for boolean_column in (
                    "ALCOHOLIC_LIVER_DISEASE",
                    "AUTOIMMUNE_HEPATITIS",
                    "CIRRHOSIS",
                    "HEPATITIS_B",
                    "HEPATITIS_C",
                    "LIVER_FAILURE",
                    "OTHER_LIVER_PATOLOGIES",
                )
            ),
            _max_icd9_code(patient_df, "mild_liver_disease", weight=1),
        ),
    )
    if (
        all(
            (
                mild_liver_disease > 0,
                max(
                    _series_replaced_with_mapping_or_nan(
                        patient_df, "HYPERTENSION", BOOLEANIZATION_MAP
                    ).max(),
                    _series_replaced_with_mapping_or_nan(
                        patient_df, "PORTAL_HYPERTENSION", BOOLEANIZATION_MAP
                    ).max(),
                )
                > 0,  # plus hypertension
            )
        )
        or _max_icd9_code(
            patient_df, "moderate_to_severe_liver_disease", weight=3
        )
        > 0
    ):
        return 3  # Moderate-to-Severe == 3
    return mild_liver_disease


@non_negative_addend
def _lymphoma_points(patient_df):
    """Cancerous spread in the lymph system

    https://en.wikipedia.org/wiki/Lymphoma

    return -1 if no useful column was found

    return  0 == No
    return  2 == Yes
    """
    return _max_icd9_code(patient_df, "lymphoma", weight=2)


# DO NOT DECORATE
def _max_icd9_code(
    df,
    disease_name,
    icd9_code_column=rename_helper("icd9_code"),
    weight=1,
    **kwargs,
):
    return min(
        weight,
        max(
            0,
            max(
                set(
                    _series_replaced_with_mapping_or_nan(
                        df,
                        icd9_code_column,
                        {
                            k: weight
                            for k in (
                                float(code),
                                int(code),
                                str(code),
                                # In some countries comma is used in place of
                                # period to separate decimal places
                                str(code).replace(".", ","),
                                # These last three substitution rules are
                                # necessary because icd9 codes are often
                                # stored as '49121', which are obviously out
                                # of range [001-999], thus they are meant to
                                # signify '491.21' (e.g. Obstructive chronic
                                # bronchitis without exacerbation) as the
                                # icd9_description column confirms
                                float(str(code).replace(".", "")),
                                int(str(code).replace(".", "")),
                                str(code).replace(".", ""),
                            )
                        },
                        **kwargs,
                    ).max()
                    for code in ICD9_CODES.get(disease_name, tuple())
                )
            ),
        ),
    )


# DO NOT DECORATE
def _missing_as_nan():
    return np.nan


@non_negative_addend
def _moderate_to_severe_chronic_kidney_disease_points(patient_df):
    """Where:
    Severe   = on dialysis, status post kidney transplant, uremia
    Moderate = creatinine >3 mg/dL (0.27 mmol/L)

    https://en.wikipedia.org/wiki/Chronic_kidney_disease

    return -1 if no useful column was found

    return  0 == No
    return  2 == Yes
    """
    return 2 * min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_KIDNEY_DISEASE"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "CHRONIC_KIDNEY_DISEASE", BOOLEANIZATION_MAP
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "CKD", BOOLEANIZATION_MAP
            ).max(),
            -1
            if rename_helper("CREATININE") not in patient_df.columns
            else patient_df.loc[:, rename_helper("CREATININE")].max()
            > 3.0,  # mg/dL
            _max_icd9_code(patient_df, "renal_disease", weight=2),
        ),
    )


@non_negative_addend
def _myocardial_infarction_points(patient_df):
    """History of definite or probable MI (EKG changes and/or enzyme changes)

    https://en.wikipedia.org/wiki/Myocardial_infarction

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_MYOCARDIAL_ISCHEMIA"
            ).max(),
            _series_replaced_with_mapping_or_nan(
                patient_df, "CARDIAC_ISCHEMIA", BOOLEANIZATION_MAP
            ).max(),
            _max_icd9_code(patient_df, "myocardial_infarction"),
        ),
    )


@non_negative_addend
def _peptic_ulcer_disease_points(patient_df):
    """Any history of treatment for ulcer disease or history of ulcer bleeding

    https://en.wikipedia.org/wiki/Peptic_ulcer_disease

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_PEPTIC_ULCER_DISEASE"
            ).max(),
            _max_icd9_code(patient_df, "peptic_ulcer_disease"),
        ),
    )


@non_negative_addend
def _peripheral_vascular_disease_points(patient_df):
    """Intermittent claudication or past bypass for chronic arterial
    insufficiency, history of gangrene or acute arterial insufficiency, or
    untreated thoracic or abdominal aneurysm (≥6 cm)

    https://en.wikipedia.org/wiki/Peripheral_artery_disease
    https://en.wikipedia.org/w/index.php?title=Peripheral_vascular_disease

    return -1 if no useful column was found

    return  0 == No
    return  1 == Yes
    """
    return min(
        1,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_VASCULAR_DISEASE"
            ).max(),
            _max_icd9_code(patient_df, "peripheral_vascular_disease"),
        ),
    )


# DO NOT DECORATE
def _series_replaced_with_mapping_or_nan(
    df, col_name, mapping, default=0, column_not_found=-1
):
    col_name = rename_helper(col_name)
    if col_name in df.columns:
        return pd.Series(
            [
                defaultdict(_missing_as_nan, mapping).get(v)
                for v in df.loc[:, col_name].tolist()
            ]
        ).fillna(default)
    return pd.Series([column_not_found for _ in range(df.shape[0])])


@non_negative_addend
def _solid_tumor_points(patient_df):
    """Localized solid tumor (2) or Metastatic solid tumor (6)

    https://en.wikipedia.org/wiki/Neoplasm

    return -1 if no useful column was found

    return  0 == No
    return  2 == Localized
    return  6 == Metastatic
    """

    return min(
        6,
        max(
            charlson_series_to_integer_values(
                patient_df, "CHARLSON_SOLID_TUMOR"
            ).max(),  # -1 | 0 | 2 | 6
            _series_replaced_with_mapping_or_nan(
                patient_df,
                "SOLID_TUMOR",
                {"Any": 0, "Inactive": 2, "Active": 6},
            ).max(),
            2
            * _series_replaced_with_mapping_or_nan(
                patient_df, "NEOPLASMS", BOOLEANIZATION_MAP
            ).max(),  # -1 | 0 | 2
            # TODO: add icd9 codes of localized solid tumorswith weight=2
            _max_icd9_code(patient_df, "metastatic_solid_tumor", weight=6),
        ),
    )


def charlson_series_to_integer_values(df, col_name):
    """Return non-negative integer series if col_name is found

    Otherwise (col_name is not found) a negative series is returned
    """
    col_name = rename_helper(col_name)
    assert col_name.lower().startswith("charlson_"), str(
        f"'{col_name}' prefix should be 'Charlson_'"
    )
    if col_name in df.columns:
        try:
            dtype, conversion_map = charlson_enum_rule(
                set(v for v in df.loc[:, col_name].dropna())
            )
        except ValueError as e:
            if str(e) != "Bad guess, retry":
                raise e
        else:
            if dtype is not None and conversion_map is not None:
                return (
                    df.loc[:, col_name]
                    .replace(defaultdict(_missing_as_nan, conversion_map))
                    .astype(dtype)  # pd.CategoricalDtype(..., ordered=True)
                    .cat.codes.replace({-1: 0})  # replace Nan with zero
                    .astype(pd.Int64Dtype())
                )
    return pd.Series([-1 for _ in range(df.shape[0])]).astype(pd.Int64Dtype())


def compute_charlson_index(
    patient_df, valid_column_threshold=8, logger=logging, log_prefix=""
):
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

    cci, patient_counter, valid_columns = 0, Counter(), 0
    for function_returning_points in sorted(
        (
            "_age_points",
            "_aids_points",
            "_cerebrovascular_accident_or_transient_ischemic_attack_points",
            "_chronic_obstructive_pulmonary_disease_points",
            "_congestive_heart_failure_points",
            "_connective_tissue_disease_points",
            "_dementia_points",
            "_diabetes_mellitus_points",
            "_hemiplegia_points",
            "_leukemia_points",
            "_liver_disease_points",
            "_lymphoma_points",
            "_moderate_to_severe_chronic_kidney_disease_points",
            "_myocardial_infarction_points",
            "_peptic_ulcer_disease_points",
            "_peripheral_vascular_disease_points",
            "_solid_tumor_points",
        )
    ):
        try:
            cci += globals()[function_returning_points](
                patient_df, logger=logger, log_prefix=log_prefix
            )
        except ColumnNotFoundError:  # necessary columns were all empty :(
            patient_counter.update({function_returning_points: 0})
        else:
            assert cci >= 0 and cci <= 37, "Charlson-Index not in [0, 37]"
            valid_columns += 1
            patient_counter.update({function_returning_points: 1})

    global _CHARLSON_COUNTER, _COUNTER_LOCK
    _COUNTER_LOCK.acquire()
    _CHARLSON_COUNTER.update(patient_counter)
    _COUNTER_LOCK.release()

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


def estimated_ten_year_survival(charlson_index):
    if pd.isna(charlson_index):
        return np.nan
    return 0.983 * np.exp(charlson_index * 0.9)


def max_charlson_col_length():
    global _CHARLSON_COUNTER, _COUNTER_LOCK
    _COUNTER_LOCK.acquire()
    ret = max((0, *(len(c) for c in _CHARLSON_COUNTER.keys())))
    _COUNTER_LOCK.release()
    return ret


def most_common_charlson():
    global _CHARLSON_COUNTER, _COUNTER_LOCK
    _COUNTER_LOCK.acquire()
    most_common_columns = _CHARLSON_COUNTER.most_common()
    _COUNTER_LOCK.release()

    for col, count in most_common_columns:
        yield col, count


def reset_charlson_counter():
    global _CHARLSON_COUNTER, _COUNTER_LOCK
    _COUNTER_LOCK.acquire()
    _CHARLSON_COUNTER = Counter()
    _COUNTER_LOCK.release()


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.charlson_index",
    "YAstarMM.charlson_index",
    "charlson_index",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
