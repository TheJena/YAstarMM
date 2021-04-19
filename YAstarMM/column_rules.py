#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2021 Federico Motta <191685@studenti.unimore.it>
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
    New naming comvention, with rules to keep or drop columns

   Usage:
            from  YAstarMM.column_rules  import  (
                BOOLEANISATION_MAP,
                DAYFIRST_REGEXP,
                drop_rules,
                keep_rules,
                progressive_features,
                rename_helper,
                shift_features,
                summarize_features,
                switch_to_date_features,
                verticalize_features,
            )

   ( or from within the YAstarMM package )

            from          .column_rules  import  (
                BOOLEANISATION_MAP,
                DAYFIRST_REGEXP,
                drop_rules,
                keep_rules,
                progressive_features,
                rename_helper,
                shift_features,
                summarize_features,
                switch_to_date_features,
                verticalize_features,
            )
"""

from collections import namedtuple, OrderedDict
from datetime import datetime
from functools import lru_cache
from logging import debug
from random import random
from re import compile, IGNORECASE
from string import digits, punctuation
from sys import version_info
import numpy as np
import pandas as pd

BOOLEANISATION_MAP = {
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


#
# Some reminders about regexp:
#
# \d is equivalent to [0-9]
# \D is equivalent to [^0-9]
# \w is equivalent to [a-zA-Z0-9_]

DAYFIRST_REGEXP = compile(
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

NORMALIZED_TIMESTAMP_COLUMNS = [
    # These are the columns we are going to use in "db-like-join"
    # merge operations between sheets; for this reason it is important
    # that two record do not differ for a few seconds/minutes (since
    # we are interested in a daly time granularity). So we are going
    # to keep the date information and drop the time information.
    "date",
]


keep_rules = OrderedDict(
    {
        group_name: [
            OrderedDict(
                {
                    new_column_name: compile(
                        case_insensitive_regexp,
                        IGNORECASE,
                    )
                    for new_column_name, case_insensitive_regexp in sorted(
                        dictionary.items(),
                        key=lambda key__value: key__value[0].replace(
                            "end", "startzzzzend"  # sort start before end
                        ),
                    )
                }
            )
            for dictionary in list_of_dictionaries
        ]
        for group_name, list_of_dictionaries in OrderedDict(
            identifiers=[
                dict(
                    admission_code=r"",
                    admission_id=r"",
                    discharge_id=r"",
                    patient_id=r"",
                    provenance=r"",
                ),
            ],
            important_dates=[
                dict(
                    date=str(
                        r"",
                        r"",
                    ),
                ),
                dict(
                    admission_date=r"",
                    discharge_date=str(
                        r""
                        r""
                        r""
                    ),
                    discharge_mode=str(
                        r""
                    ),  # make black auto-formatting prettier
                ),
            ],
            immunological_therapies=[
                dict(
                    immunological_therapy=str(
                        r""
                    ),  # make black auto-formatting prettier
                    immunological_therapy_date=str(
                        r""
                    ),  # make black auto-formatting prettier
                ),
            ],
            antibiotics=[
                dict(
                    antibiotic=r"",
                    antibiotic_notes=r"",
                    antibiotic_therapy=r"",
                    roxithromycin=r"",
                ),
            ],
            anticoagulant_drugs=[
                {
                    "": str(  # heparin
                        r""
                        r""
                        r""
                        r""
                        r""
                    ),
                },
            ],
            antirheumatic_drugs=[
                dict(
                    anakinra=str(
                        r""
                        r"(?!"  # start of banned endings
                        r""
                        r")"  # end of banned endings
                    ),
                    anakinra_1st_dose=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_1st_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_2nd_dose=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_2nd_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_end_date=r"",
                    anakinra_sample_t0=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_sample_t2=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_sample_t7=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    anakinra_start_date=r"",
                    anakinra_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                ),
                dict(
                    plaquenil=r"",
                    plaquenil_1st_date=str(
                        r""
                    ),
                ),
                dict(
                    tocilizumab=str(
                        r"" r""
                    ),
                    tocilizumab_1st_dose=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_1st_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_2nd_dose=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_2nd_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_3rd_dose=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_3rd_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_sample_t0=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_sample_t2=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_sample_t7=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                    tocilizumab_via=str(
                        r""
                        r""  # banned middle string
                        r""
                    ),
                ),
            ],
            antiviral_drugs=[
                dict(
                    antiviral_therapy=str(
                        r""
                    ),
                    kaletra=r"",
                    rezolsta=r"",
                ),
            ],
            pro_nucleotide_drugs=[
                dict(
                    remdesivir=str(
                        r""
                        r""
                    ),
                    remdesivir_1st_date=str(
                        r""
                    ),
                    remdesivir_end_date=r"",
                    remdesivir_start_date=r"",
                ),
            ],
            other_drug_info=[
                dict(
                    atc_code=r"",
                    drug_amount=r"",
                    drug_amount_international_unit=r"",
                    drug_class=r"",
                    drug_daily_amount=r"",
                    drug_daily_amount_international_unit=r"",
                    drug_description=r"",
                    drug_end_date=r"",
                    drug_start_date=r"",
                ),
            ],
            cortisone_journey=[
                {
                    "": str(
                        r""
                        r""
                        r""
                        r""
                        r""
                    ),
                },
                dict(
                    cortisone_level=r"",
                    dexamethasone_end_date=r"",
                    dexamethasone_start_date=r"",
                    steroids=r"",
                    steroids_mg=r"",
                    steroids_mg_pro_kg=r"",
                    steroids_start=r"",
                    steroids_end=r"",
                ),
            ],
            oxygen_journey=[
                {
                    "": str(
                        r"("  # start of group
                        r""
                        r""
                        r""
                        r""
                        r")"  # end of group
                        r""
                        r""
                        r""
                    ),
                },
            ],
            respiration=[
                dict(
                    breathing_room_air=str(
                        r""
                        r"(?!"  # start banned endings
                        r""
                        r")"  # end banned endings
                    ),
                    breathing_room_air_notes=str(
                        r""
                    ),
                    fraction_inspired_oxygen=str(
                        r"(?!)"  # banned beginnings
                        r""
                        r""
                    ),
                    fraction_inspired_oxygen_notes=str(
                        r"(?!)"  # banned starts
                        r""  # allowed ends
                    ),
                    hacor_score=r"",
                    horowitz_index=str(
                        r"(?!)"
                        r""
                        r"(?!)"
                    ),
                    horowitz_index_notes=r"",
                    horowitz_index_under150_date=str(
                        r""
                    ),
                    horowitz_index_under250_date=str(
                        r""
                    ),
                    non_invasive_ventilation_exhaled_tidal_volume=str(
                        r""
                    ),  # make black auto-formatting prettier
                    non_invasive_ventilation_fraction_inspired_oxygen=str(
                        r""
                    ),
                    non_invasive_ventilation_positive_end_expiratory_pressure=str(
                        r""
                    ),
                    non_invasive_ventilation_ps=r"",
                    non_invasive_ventilation_vm=r"",
                    peak_inspiratory_pressure=r"",
                    peak_inspiratory_pressure_volume=r"",
                    oxygen_litres=str(
                        r"" r"" r""
                    ),
                    oxygen_partial_pressure=str(
                        r""
                    ),
                    oxygen_partial_pressure_notes=str(
                        r""  # allowed beginnings
                        r"(?!)"  # banned middle string
                        r""  # allowed ends
                    ),
                    oxygen_reservoirs_usage=r"",
                    oxygen_saturation=str(
                        r""  # allowed match
                        r"|"  # logic or
                        r""  # allowed match
                    ),
                    oxygen_saturation_notes=r"",
                    oxygen_therapy=str(
                        r"(?"  # start of banned beginnings
                        r""
                        r""
                        r")"  # end of banned beginnings
                        r""
                        r"(?!"  # start of banned endings
                        r""
                        r")"  # end of banned endings
                    ),
                    oxygen_therapy_notes=str(
                        r"(?!"  # start of banned beginnings
                        r""
                        r""
                        r")"  # end of banned beginnings
                        r""
                        r""
                        r""
                    ),
                    # oxygen_therapy_pressure=str(
                    #     r"(?!"
                    #     r""
                    #     r")"
                    #     r""
                    #     r"(?!"
                    #     r""
                    #     r")"
                    # ),
                ),
                {
                    "": str(  # respiratory_rate
                        r""
                        r""
                    ),
                },
                dict(
                    venturi_mask_litres=r"",
                ),
            ],
            vaccines=[
                dict(
                    influenza_vaccine=r"",
                    pneumococcal_vaccine=str(
                        r""
                    ),  # make black auto-formatting prettier
                ),
            ],
            swabs=[
                dict(
                    bronchoalveolar_lavage_date=r"",
                    bronchoalveolar_lavage_result=r"",
                    swab_1st_negative_date=r"",
                    swab_1st_positive_date=r"",
                    swab_all_dates=r"",
                    swab_check_date=str(
                        r""
                        r"|"  # logic or
                        r""
                    ),
                    swab_laboratory=r"",
                    swab_laboratory_notes=str(
                        r""
                    ),  # make black auto-formatting prettier
                    swab_result=r"",
                    swab_symptoms_start=str(
                        r""
                    ),  # make black auto-formatting prettier
                    swab_symptoms_start_notes=str(
                        r""
                    ),  # make black auto-formatting prettier
                ),
            ],
            ungrouped_rules=[
                dict(
                    age=str(
                        r"(?!"  # start banned beginnings
                        r""
                        r""
                        r")"  # end banned beginnings
                        r""
                    ),
                    body_mass_index=r"",
                    ceiling_effect_start_date=str(
                        r""
                    ),
                    ceiling_effect_notes=r"",
                    covid_19=r"",
                    covid_19_correlation=str(
                        r""
                    ),
                    covid_19_correlation_date=str(
                        r""
                    ),  # make black auto-formatting prettier
                    dopamine=r"",
                    height=r"",
                    infection_description=r"",
                    liver_controlled_attenuation_parameter=str(
                        r""
                    ),  # make black auto-formatting prettier
                    liver_controlled_attenuation_parameter_inter_quartile_range=str(
                        r""
                    ),
                    obesity=r"",
                    organ_transplant=r"",
                    plica_b=r"",
                    sex=r"",
                    waist_circumference=r"",
                    weight=r"",
                ),
            ],
            comorbidities=[
                dict(
                    basic_pathologies=r"",
                    blood_diseases=r"",
                    cardiovascular_disease=r"",
                    chronic_diseases=r"",
                    chronic_kidney_disease=str(
                        r""
                        r""  # optional middle string
                        r""
                    ),
                    chronic_obstructive_pulmonary_disease=str(
                        r""
                    ),  # make black auto-formatting prettier
                    copatologies=r"",
                    diabetes=r"",
                    hepatitis_b=r"",
                    hepatitis_c=r"",
                    hiv=r"",
                    hypertension=r"",
                    liver_failure=r"",
                    neoplasms=r"",
                    organ_damage=r"",
                    parkinson=r"",
                    parkinson_notes=r"",
                    staging_risk=r"",
                ),
                {
                    "": r"",  # charlson index
                },
            ],
            intensive_care_unit_scores=[
                dict(
                    fully_conscious_state=r"",
                    drg_code=r"",
                    drg_description=r"",
                    icd9_code=r"",
                    icd9_description=r"",
                    icd9_weight=r"",
                    nosocomial_pneumonia=r"",
                ),
                {
                    "": r"",  # APACHE II
                },
                dict(
                    glasgow_coma_scale=r"",
                ),
                {
                    "": r"",  # SAPS II
                },
                dict(
                    sofa_score=str(
                        r""
                        r""  # logic or
                        r""
                        r""  # logic or
                        r""
                    ),
                    sofa_score_bilirubin=r"",
                    sofa_score_creatinine=r"",
                    sofa_score_platelets=r"",
                    sofa_score_date=r"",
                    sofa_score_mean_arterial_pressure=r"",
                    sofa_score_horowitz_index=r"",
                    sofa_score_notes=r"",
                ),
            ],
            hospital_units_journey=[
                dict(
                    unit_code=r"",
                    unit_description=r"",
                ),
                dict(
                    actual_unit=r"",
                    gastroenterology_unit_covid_discharge_date=str(
                        r""
                    ),
                    infectious_disease_unit_transfer_date=str(
                        r""
                    ),
                    post_operative_recovery_unit_transfer_date=str(
                        r""
                    ),
                    previous_unit=r"",
                ),
                dict(
                    infectious_disease_unit_covid_date_range=str(
                        r""
                    ),
                    infectious_disease_unit_date_range=str(
                        r""
                    ),
                    internal_intensive_care_unit_critical_area_date_range=str(
                        r""
                    ),
                    internal_medicine_and_critical_care_unit_covid_admission_room_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_covid_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_covid_suspects_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_critical_area_covid_room_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_critical_area_covid_suspects_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_critical_area_date_range=str(
                        r""
                    ),
                    internal_medicine_unit_date_range=str(
                        r"" r""
                    ),
                    post_operative_covid_recovery_unit_date_range=str(
                        r""
                    ),
                    post_operative_recovery_unit_date_range=str(
                        r""
                        r""
                        r""
                    ),
                    respiratory_medicine_sub_intensive_unit_date_range=str(
                        r""
                    ),
                    respiratory_medicine_unit_covid_date_range=str(
                        r""
                    ),
                    respiratory_medicine_unit_date_range=str(
                        r""
                    ),
                ),
            ],
            traceability=[
                dict(
                    home_confinement=str(
                        r""
                        r""
                    ),
                    home_confinement_other_people=str(
                        r""
                    ),
                    home_confinement_start_date=str(
                        r""
                    ),
                ),
                {
                    "": r"",  # contacts
                },
                {
                    "": r"",  # expositions
                },
            ],
            signs_and_symptoms=[  # TODO split into signs and into symptoms
                {
                    "": r"",  # conjunctivitis
                },
                {
                    "": r"",  # cough
                },
                {
                    "": r"",  # diarrhea
                },
                {
                    "": r"",  # dyspnea
                },
                # dict(
                #     dyspnea_start_date=r"",
                #     dyspnea_when_sitting_lying=str(
                #         r""
                #     ),  # make black auto-formatting prettier
                #     dyspnea_when_walking=r"",
                #     dyspnea_when_washing_dressing=str(
                #         r""
                #     ),
                # ),
                {
                    "": r"",  # fatigue
                },
                {
                    "": r"",  # headache
                },
                {
                    "": r"",  # hemoptysis
                },
                {
                    "": r"",  # myalgia
                },
                {
                    "": r"",  # rash
                },
                {
                    "": r"",  # rhinorrhea
                },
                {
                    "": r"",  # rigors
                },
                {  # severe_lymphadenopathy
                    "": r"",
                },
                {  # sputum
                    "": r"",
                },
                dict(
                    stiffness=r"",
                    stiffness_inter_quartile_range=r"",
                    symptoms_list=str(
                        r""
                        r""
                        r""
                        r"",
                    ),
                    symptoms_start_date=str(
                        r""
                    ),
                ),
                {
                    "": r"",  # temperature
                },
                {
                    "": str(  # thorax_physical_exam
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                    ),
                },
                {
                    "": r"",  # throat_pain
                },
                {
                    "": r"",  # tracheotomy
                },
                {
                    "": r"",  # tonsil_oedema
                },
                dict(
                    other_symptoms=r"",
                ),
            ],
            blood=[
                dict(
                    hearth_rate=str(
                        r""
                        r""
                    ),
                ),
                dict(
                    diastolic_pressure=r"",
                    systolic_pressure=r"",
                ),
                dict(
                    arterial_blood=r"",
                    arterial_blood_notes=r"",
                    bicarbonate=r"",
                    bicarbonate_notes=r"",
                    carbon_dioxide_partial_pressure=str(
                        r""
                    ),  # make black auto-formatting prettier
                    carbon_dioxide_partial_pressure_notes=str(
                        r""
                    ),  # make black auto-formatting prettier
                    erythrocyte_sedimentation_rate=str(
                        r""
                    ),
                    erythrocyte_sedimentation_rate_notes=r"",
                    glucose=r"",
                    glucose_notes=r"",
                    hemoglobine=r"",
                    hemoglobine_notes=r"",
                    ph=r"",
                    ph_notes=r"",
                    procalcitonin_exam_date=r"",
                ),
            ],
            blood_tests_to_be_partitioned_into_ad_hoc_regexp=[
                # TODO divide some of the following into ad-hoc regexp
                {
                    "": str(
                        r"^(?!"  # start of banned beginnings
                        r""
                        r""
                        r""
                        r""
                        r")"  # end of banned beginnings
                        r""
                        r"("  # start of group
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r""
                        r")"  # end of group
                        r""
                        r""
                    )
                },
            ],
            unknown_stuff=[
                dict(
                    insert_date=r"",
                    validation_date=r"",
                ),
                dict(
                    edit_date=r"",
                    removal_date=r"",
                    report_version=r"",
                ),
                {
                    "": r"",  # Med_*
                },
                {
                    "": r"",  # PARVENT_*
                },
                {
                    "": r"",  # SARSCoV2_PCR_*
                },
                dict(
                    free_text_notes=str(
                        r""
                        r""
                    ),
                ),
            ],
        ).items()
    }
)


drop_rules = OrderedDict(
    {
        reason: [
            compile(case_insensitive_regexp, IGNORECASE)
            for case_insensitive_regexp in list_of_rules
        ]
        for reason, list_of_rules in sorted(
            dict(
                redundant_boolean_information=[
                    r"",
                    r"",
                ],
                sensitive_doctors_data=[
                    r"",
                    r"",
                    r"",
                    r"",
                    str(
                        r""
                        r""
                        r""
                    ),
                ],
                sensitive_patients_data=[
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                ],
                unnecessary_information=[
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"",
                    r"
                    str(  # fraction_inspired_oxygen_symptoms_id
                        r""
                    ),
                    str(  # horowitz_index_emogas_id
                        r""
                    ),
                    str(  # oxygen_symptoms_id
                        r""
                    ),
                    str(  # pab_symptoms_id
                        r""
                    ),
                    str(  # patient_journey_generation_date
                        r""
                    ),
                    str(  # respiratory_rate_symptoms_id
                        r""
                    ),
                    str(
                        r""
                        r""
                        r""
                    ),
                ],
            ).items()
        )
    }
)
SummarizeFeatureItem = namedtuple(
    "SummarizeFeatureItem", ["old_columns", "old_values_checker", "new_enum"]
)

SwitchToDateValue = namedtuple("SwitchToDateValue", ["true_val", "date_col"])

VerticalizeFeatureItem = namedtuple(
    "VerticalizeFeatureItem", ["date_column", "column_name", "related_columns"]
)




@lru_cache(maxsize=None)
def progressive_features():
    return iter(
        rename_helper(
            (
            )
        )
    )


@lru_cache(maxsize=None)
def rename_helper(columns):
    assert isinstance(columns, (str, tuple)), str(
        "Argument must be a string or a tuple of strings"
    )
    if not columns:
        return tuple()
    input_was_a_string = False
    if isinstance(columns, str):
        input_was_a_string = True
        columns = tuple([columns])
    ret = list()
    for old_col_name in columns:
        artificial_name = {  # columns not in extraction files
            "ActualState": "oxygen_therapy_state",
            "ActualState_val": "oxygen_therapy_state_value",
            "DayCount": "days_since_admission",
            "day_count": "days_since_admission",
        }.get(old_col_name, None)
        if artificial_name is not None:
            ret.append(artificial_name)
            continue
        try:
            for reason, list_of_rules in drop_rules.items():
                for rule in list_of_rules:
                    if rule.match(old_col_name) is not None:
                        raise StopIteration(
                            f"Skip column '{old_col_name}'"
                            " since it was dropped with reason: "
                            f"{repr(reason)}."
                        )
            #
            # any drop rule matched; let us look for a keep rule
            #
            for group, list_of_mappings in keep_rules.items():
                for mapping in list_of_mappings:
                    for new_col_name, rule in mapping.items():
                        if new_col_name == "":
                            continue
                        if rule.match(old_col_name) is not None:
                            ret.append(new_col_name)
                            raise StopIteration(
                                "New name found, continue with next column"
                                f"\t({old_col_name} ~> {new_col_name})"
                            )
        except StopIteration as e:
            debug(str(e))
            continue
        else:  # any keep rule matched == no new name
            ret.append(old_col_name)
    assert ret, str(
        f"Column '{old_col_name}' has probably been dropped;"
        " please adapt your code accordingly.."
    )
    if input_was_a_string:
        return ret.pop()  # return just a string
    return tuple(ret)


@lru_cache(maxsize=None)
def shift_features(sheet_name):
    return iter(
        rename_helper(
            dict(
                emogas=(
                ),
                symptoms=(
                ),
            ).get(
                sheet_name, tuple()
            )  # return empty tuple when sheet name not in dictionary
        )
    )


@lru_cache(maxsize=None)
def summarize_features():
    return {
        f"{new_categorical_col}_severity": [
            SummarizeFeatureItem(
                old_columns=rename_helper(tuple(sfi.old_columns)),
                old_values_checker=[
                    f if f not in [True, False] else lambda val: val is f
                    for f in sfi.old_values_checker
                ],
                new_enum=sfi.new_enum,
            )
            for sfi in summary_rule_list
        ]
        for new_categorical_col, summary_rule_list in {
            "dyspnea": [
                SummarizeFeatureItem([""], [False], "no dyspnea"),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "dyspnea during room walk",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "dyspnea while washing/dressing",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "dyspnea while sitting/lying",
                ),
            ],
            "cough": [
                SummarizeFeatureItem([""], [False], "no cough"),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "cough with weakness",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "persistent cough",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "persistent cough",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "persistent cough",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "persistent cough",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "persistent cough",
                ),
            ],
            "oxygen_therapy": [
                SummarizeFeatureItem(
                    [""],
                    [False],
                    "no oxygen used",
                ),
                SummarizeFeatureItem(
                    ["", ""],
                    [True, lambda rr: pd.notna(rr) and float(rr) < 30],
                    "oxygen used and respiratory rate < 30",
                ),
                SummarizeFeatureItem(
                    ["", ""],
                    [True, lambda rr: pd.notna(rr) and float(rr) >= 30],
                    "oxygen used and respiratory rate >= 30",
                ),
                SummarizeFeatureItem(
                    [""],
                    [True],
                    "non-invasive ventilation",
                ),
                SummarizeFeatureItem([""], [True], "intubated"),
            ],
        }.items()
    }


def switch_to_date_features(sheet_name):
    date = rename_helper("")
    anakinra, antibiotic, plaquenil, remdesivir, tocilizumab = rename_helper(
        ()
    )
    return dict(
        diary={
            anakinra: SwitchToDateValue("", date),
            antibiotic: SwitchToDateValue("", date),
            plaquenil: SwitchToDateValue("", date),
            remdesivir: SwitchToDateValue("", date),
            tocilizumab: SwitchToDateValue("", date),
        },
    ).get(
        sheet_name, dict()
    )  # return empty dict when sheet name not in dictionary


def verticalize_features():
    for item in [
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
        VerticalizeFeatureItem(
        ),
    ]:
        yield VerticalizeFeatureItem(
            date_column=rename_helper(item.date_column),
            column_name=rename_helper(item.column_name),
            related_columns=rename_helper(tuple(item.related_columns)),
        )


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.column_rules",
            "YAstarMM.column_rules",
            "column_rules",
        ),
        "BOOLEANISATION_MAP" in globals(),
        "DAYFIRST_REGEXP" in globals(),
        "drop_rules" in globals(),
        "keep_rules" in globals(),
        "rename_helper" in globals(),
        "shift_features" in globals(),
        "switch_to_date_features" in globals(),
        "verticalize_features" in globals(),
    )
), "Please update 'Usage' section of module docstring"
