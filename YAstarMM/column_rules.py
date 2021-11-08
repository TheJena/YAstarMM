#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2021 Federico Motta <federico.motta@unimore.it>
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
    New naming convention, with rules to keep or drop columns

   Usage:
            from  YAstarMM.column_rules  import  (
                does_not_match_categorical_rule,
                does_not_match_float_rule,
                drop_rules,
                keep_rules,
                matched_enumerated_rule,
                matches_boolean_rule,
                matches_date_time_rule,
                matches_integer_rule,
                matches_static_rule,
                rename_helper,
            )

   ( or from within the YAstarMM package )

            from          .column_rules  import  (
                does_not_match_categorical_rule,
                does_not_match_float_rule,
                drop_rules,
                keep_rules,
                matched_enumerated_rule,
                matches_boolean_rule,
                matches_date_time_rule,
                matches_integer_rule,
                matches_static_rule,
                rename_helper,
            )
"""

from .constants import (
    BOOLEANIZATION_MAP,
    HARDCODED_COLUMN_NAMES,
    MIN_PYTHON_VERSION,
    SummarizeFeatureItem,
    SwitchToDateValue,
    VerticalizeFeatureItem,
)
from .utility import hex_date_to_timestamp, timestamp_to_hex_date
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache
from logging import debug, DEBUG, log, warning, WARNING
from random import random
from re import compile, IGNORECASE
from string import digits, punctuation
from sys import version_info
import numpy as np
import pandas as pd

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
                    r"",
                    str(
                        r""
                    ),
                    str(
                        r""
                    ),
                    str(
                        r""
                    ),
                    str(
                        r""
                    ),
                    str(
                        r""
                    ),
                    str(
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
                dict(
                    hfno_state_end=r"",
                    hfno_state_start=r"",
                    intubation_state_end=r"",
                    intubation_state_start=r"",
                    niv_state_end=r"",
                    niv_state_start=r"",
                    oxygen_therapy_state_end=r"",
                    oxygen_therapy_state_start=r"",
                    post_hfno_state_end=r"",
                    post_hfno_state_start=r"",
                    post_niv_state_end=r"",
                    post_niv_state_start=r"",
                    post_oxygen_therapy_state_end=r"",
                    post_oxygen_therapy_state_start=r"",
                ),
                {
                    "": r"",
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
                    hypertension_notes=r"",
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
            has_respiratory_symptoms=[
                dict(
                    has_breathing_pain_symptom=r"",
                    has_chest_pain_symptom=r"",
                    has_cough_and_becomes_easily_exhausted_symptom=r"",
                    has_cough_and_breathing_causes_sleep_disorder_symptom=r"",
                    has_cough_and_coughing_causes_physical_tiredness_symptom=r"",
                    has_cough_and_feels_breathless_when_bending_symptom=r"",
                    has_cough_and_feels_breathless_when_talking_symptom=r"",
                    has_cough_and_feels_pain_when_coughing_symptom=r"",
                    has_cough_symptom=r"",
                    has_dyspnea_on_exertion_symptom=r"",
                    has_dyspnea_symptom=r"",
                    has_dyspnea_while_sitting_lying_symptom=r"",
                    has_dyspnea_while_walking_in_room_symptom=r"",
                    has_dyspnea_while_washing_dressing_symptom=r"",
                ),
            ],
            has_musculoskeletal_symptoms=[
                dict(
                    has_asthenia_symptom=r"",
                    has_falls_symptom=r"",
                    has_imbalance_disorder_symptom=r"",
                    has_inability_to_control_movement_symptom=r"",
                    has_joint_pain_symptom=r"",
                    has_muscle_pain_symptom=r"",
                ),
            ],
            has_neurocognitive_symptoms=[
                dict(
                    has_abdominal_pain_symptom=r"",
                    has_ageusia_symptom=r"",
                    has_altered_emotional_state_symptom=r"",
                    has_anosmia_symptom=r"",
                    has_behavior_alteration_symptom=r"",
                    has_clouded_mind_symptom=r"",
                    has_cognitive_attention_symptom=r"",
                    has_cognitive_concentration_symptom=r"",
                    has_cognitive_memory_symptom=r"",
                    has_diarrhea_symptom=r"",
                    has_dizziness_symptom=r"",
                    has_effluvium_symptom=r"",
                    has_epileptic_seizures_convulsions_symptom=r"",
                    has_erectile_dysfunction_symptom=r"",
                    has_eye_disorder_symptom=r"",
                    has_fainting_blackout_symptom=r"",
                    has_fear_of_the_future_symptom=r"",
                    has_headache_symptom=r"",
                    has_hearing_disorder_symptom=r"",
                    has_lost_appetite_symptom=r"",
                    has_menstrual_cycle_change_symptom=r"",
                    has_nausea_symptom=r"",
                    has_others_symptom=r"",
                    has_psychological_activation_symptom=r"",
                    has_skin_alteration_symptom=r"",
                    has_sleep_disorder_symptom=r"",
                    has_swallowing_troubles_symptom=r"",
                    has_sweating_symptom=r"",
                    has_tachycardia_symptom=r"",
                    has_throat_scraper_symptom=r"",
                    has_tremors_symptom=r"",
                    has_urinating_troubles_symptom=r"",
                    has_weak_legs_arms_symptom=r"",
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
		     urine_ph=r"",
		     urine_ph_notes=r"",
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


def charlson_enum_rule(column_values):
    charlson_age = {
        0: "(< 50)",
        1: "(50-59)",
        2: "(60-69)",
        3: "(70-79)",
        4: "(>= 80)",
    }
    charlson_map, unique_values = dict(), set(column_values)
    if any(
        "".join(age_suffix.split()) in "".join(str(val).split())
        for val in unique_values
        for age_suffix in charlson_age.values()
    ):
        # at least an age_suffix is contained in unique_values
        charlson_map = {k: set([f"{k} {v}"]) for k, v in charlson_age.items()}
        for val in unique_values:
            for i, age_suffix in charlson_age.items():
                if "".join(age_suffix.split()) in "".join(str(val).split()):
                    charlson_map[i] = charlson_map[i].union({val})
    else:
        for val in unique_values:
            if all(
                (
                    "0" in str(val),
                    "=" in str(val) or ("<" in str(val) and "50" in str(val)),
                    BOOLEANIZATION_MAP.get(
                        str(val)
                        .replace("50", "@@")
                        .strip("0.= ()")
                        .lower()
                        .replace("< @@", "no")
                        .replace("<@@", "no"),
                        True,
                    )
                    not in {np.nan, True},
                )
            ):  # we found a cell like '0 = No' or '0 (< 50)'
                charlson_map[0] = set([val])
    if not charlson_map:
        raise ValueError("Bad guess, retry")
    for val in unique_values:
        for i in range(10):
            if any(
                (
                    f"{i}=" in "".join(str(val).split()),
                    str(i) == str(val).strip(".0 "),
                    # the following line treats stuff in CHARLSON_AGE
                    str(i) == " ".join(str(val).split("(")[:1]).strip(".0 "),
                )
            ):
                charlson_map[i] = charlson_map.get(i, set()).union({val})
                break
    categories, conversion_map = list(), dict()
    for i, values in sorted(charlson_map.items(), key=lambda tup: tup[0]):
        elected_value = None
        if len(values) == 1:
            # The next line pops on purpose the only value in the set
            # but from a copy of it (i.e. a list) because otherwise the
            # loop below which populates the conversion map would not
            # add this value
            elected_value = str(list(values).pop()).strip(" ")
        else:
            for value in values:
                if str(i) in str(value) and (
                    any(
                        (
                            "-" in str(value),  # CHARLSON_AGE
                            "<" in str(value),  # CHARLSON_AGE
                            "=" in str(value),  # other CHARLSON_*
                            ">" in str(value),  # CHARLSON_AGE
                        )
                    )
                ):
                    elected_value = str(value).strip(" ")
                    break
        assert elected_value is not None, str(
            f"Expected a not None value for index '{i}'"
        )
        categories.append(elected_value)
        for v in values.union({int(i), float(i), str(int(i)), str(float(i))}):
            conversion_map[v] = " ".join(elected_value.split())
    for val in unique_values:
        assert val in conversion_map, str(
            f"Value {repr(val)} expected to be in Charlson conversion map"
        )
    dtype = pd.CategoricalDtype(categories=categories, ordered=True)
    return dtype, conversion_map


def covid_enum_rule(column_values):
    covid_map = {
        "": "",
        "": "",
    }

    if not all((str(res).lower() in covid_map for res in set(column_values))):
        raise ValueError("Bad guess, retry")

    covid_map = {k: v.capitalize() for k, v in covid_map.items()}
    for val in set(column_values):
        if val not in covid_map:
            covid_map[val] = covid_map[str(val).lower()]
    dtype = pd.CategoricalDtype(
        categories=sorted(set(covid_map.values())),
        ordered=False,
    )
    return dtype, covid_map


def does_not_match_categorical_rule(column_name, df):
    if any(
        (
            str(df.dtypes[column_name])
            in ("boolean", "datetime64[ns]"),  # already casted
            "" in column_name.lower(),  # free text notes
            "audit" in column_name.lower(),   # AUDIT-C alcool use
            "_description" in column_name.lower(),  # free text notes
            "" in column_name.lower(),  # exam results
            "_notes" in column_name.lower(),  # free text notes
            "_date_range" in column_name.lower(),
            "_vaccine" in column_name.lower(),
            "sapsii" in column_name.lower(),  # dates
            column_name.lower().startswith(""),
        )
    ):
        return True

    try:
        group_name = "blood_tests_to_be_partitioned_into_ad_hoc_regexp"
        first_dict_in_list = keep_rules[group_name][0]
        anonymous_group_regexp = first_dict_in_list[""]
        if anonymous_group_regexp.match(column_name) is not None:
            return True
    except Exception:
        raise Exception(
            "Please make this try-except piece of code "
            "return True if the column name matches "
            "the regexp against all the blood tests"
        )

    if column_name.lower().startswith("drug"):
        return False

    column_unique_values = set(df.loc[:, column_name].unique())
    if any(
        (
            len(column_unique_values) > 32,  # unit_name ~= 30
            all(
                pd.api.types.is_number(value)
                for value in column_unique_values
                if pd.notna(value)
            ),
            "absent" in (str(v).lower() for v in column_unique_values),
        )
    ):
        return True
    # ELSE
    return False


def does_not_match_float_rule(column_name, dtype):
    if matches_integer_rule(column_name):
        return False
    if any(
        (
            str(column_name).lower().startswith("drug"),
            str(column_name).lower().endswith("_all_dates"),
            str(column_name).lower().endswith(""),
            str(column_name).lower().endswith("_code"),
            str(column_name).lower().endswith("_date_range"),
            str(column_name).lower().endswith("_description"),
            str(column_name).lower().endswith(""),
            str(column_name).lower().endswith("_id"),
            str(column_name).lower().endswith("_list"),
            str(column_name).lower().endswith(""),
            str(column_name).lower().endswith("_notes"),
            str(column_name).lower().endswith(""),
            str(column_name).lower().endswith(""),
            str(column_name).lower().endswith("_version"),
            str(column_name).lower().endswith(""),
        )
    ):
        return True
    if column_name.lower() in (
        "antibiotic_therapy",
        "apacheii",  # APACHEII score + (date)
        "charlson",  # Charlson-Index value + (date)
        "",
        "",
        "",
        "sapsii",  # SAPSII score + (date)
        "sofa_score",  # SOFA score + (date)
    ):
        return True
    if str(dtype) in ("boolean", "category", "datetime64[ns]"):
        return True
    return False


def drug_enum_rule(column_values):
    drug_map = dict(
    )
    drug_map.update({f"{k}": f"{v}" for k, v in drug_map.items()})
    drug_map.update(
        {
            k: f"{'' if not v else ''}"
            for k, v in BOOLEANIZATION_MAP.items()
            if pd.notna(v)
        }
    )
    if not any(
        (
            " ".join(str(val).lower().split()) in drug_map.keys()
            for val in set(column_values)
        )
    ):
        raise ValueError("Bad guess, retry")

    drug_map = {k: v.capitalize() for k, v in drug_map.items()}
    for val in set(column_values):
        nice_val = " ".join(str(val).split()).capitalize()
        if nice_val.lower() in drug_map:
            drug_map[val] = drug_map[nice_val.lower()]
        elif nice_val not in set(drug_map.values()):
            drug_map[val] = nice_val
    dtype = pd.CategoricalDtype(
        categories=sorted(set(drug_map.values())), ordered=False
    )
    return dtype, drug_map


def fallback_enum_rule(column_values):
    categories, conversion_map = set(), dict()
    for val in set(column_values):
        nice_val = (
            " ".join(
                str(word).capitalize()
                if all(
                    (
                        pos == 0,
                        str(word)[0].lower() == str(word)[0],  # char is lower
                        not str(word).isupper(),  # whole word is in uppercase
                    )
                )
                else str(word)
                for pos, word in enumerate(str(val).split())
            )
            .replace("anakirna", "anakinra")  # typos are everywhere :_(
            .replace("Anakirna", "Anakinra")  # en.wikipedia.org/wiki/Anakinra
            .replace("ANAKIRNA", "ANAKINRA")
            .replace("", "")
        )
        categories.add(nice_val)
        conversion_map[val] = nice_val
    dtype = pd.CategoricalDtype(
        categories=sorted(categories, key=str.lower), ordered=False
    )
    return dtype, conversion_map


def matched_enumerated_rule(column_name, column_values):
    unique_values = set(v for v in column_values if pd.notna(v))
    for enum_rule in (
        oxygen_state_enum_rule,  # keep before oxygen_support !
        # ---------------------- #
        oxygen_support_enum_rule,  # very fragile, keep before Charlson !
        # ------------------------ # ---------------------------------- #
        charlson_enum_rule,
        covid_enum_rule,
        drug_enum_rule,
        sex_enum_rule,
        swab_enum_rule,
        # ------------------ # ------------ #
        fallback_enum_rule,  # keep as last !
    ):
        try:
            dtype, conversion_map = enum_rule(unique_values)
        except ValueError as e:
            if str(e) != "Bad guess, retry":
                raise e
        else:
            return dtype, conversion_map
    return None, None


def matches_boolean_rule(column_name, column_values):
    if column_name in (
        "influenza_vaccine",
        "pneumococcal_vaccine",
    ):
        return True
    unique_values = set(
        val.lower() if isinstance(val, str) else val
        for val in column_values
        if pd.notna(val)
    )
    return all(
        (
            not matches_date_time_rule(column_name)
            or (
                # already verticalized columns which were timestamps
                # and became booleans
                matches_date_time_rule(column_name)
                and len(unique_values.difference({0.0, 1.0})) == 0
            ),
            len(unique_values) <= len(BOOLEANIZATION_MAP),
            len(
                unique_values.difference(
                    set(BOOLEANIZATION_MAP.keys()),
                )
            )
            == 0,
            len(unique_values.difference({float(0.0), float(1.0)})) > 0,
        )
    )


@lru_cache(maxsize=None)
def matches_date_time_rule(column_name):
    c = column_name.lower()
    return any(
        (
            "" in c,
            c in ("date"),
            c.endswith(""),
            c.endswith(""),
            c.endswith(""),
            c.endswith("_end"),
            c.endswith(""),
            c.endswith("_from"),
            c.endswith(""),
            c.endswith("_start"),
            c.endswith("_to"),
            c.startswith("") and not c.endswith(""),
            c.startswith("") and not c.endswith(""),
        )
    )


@lru_cache(maxsize=None)
def matches_integer_rule(column_name):
    if column_name is None:
        return False
    c = column_name.lower()
    return any(
        (
            c == "age",
            not c.startswith("") and c.endswith("_code"),
            not c.startswith("") and c.endswith("_code"),
            not c.startswith("") and c.endswith("_id"),


@lru_cache(maxsize=None)
def matches_static_rule(column_name):
    return column_name in set(
        (
            "admission_date",
            "age",
            "birth_date",
            "DAYS_IN_STATE",
            "discharge_date",
            "discharge_mode",
            "height",
            "influenza_vaccine",
            "pneumococcal_vaccine",
            "sex",
            "UPDATED_CHARLSON_INDEX",
        )
    ).union(
        set(
            rename_helper(
                (
                    "BLOOD_DISEASES",
                    "CARDIOVASCULAR_DISEASE",
                    "CHARLSON_AGE",
                    "CHARLSON_AIDS",
                    "CHARLSON_BLOOD_DISEASE",
                    "CHARLSON_CONNECTIVE_TISSUE_DISEASE",
                    "CHARLSON_COPD",
                    "CHARLSON_CVA_OR_TIA",
                    "CHARLSON_DEMENTIA",
                    "CHARLSON_DIABETES",
                    "CHARLSON_HEART_FAILURE",
                    "CHARLSON_HEMIPLEGIA",
                    "CHARLSON_INDEX",
                    "CHARLSON_KIDNEY_DISEASE",
                    "CHARLSON_LIVER_DISEASE",
                    "CHARLSON_LIVER_FAILURE",
                    "CHARLSON_MYOCARDIAL_ISCHEMIA",
                    "CHARLSON_PEPTIC_ULCER_DISEASE",
                    "CHARLSON_SOLID_TUMOR",
                    "CHARLSON_VASCULAR_DISEASE",
                    "CHRONIC_KIDNEY_DISEASE",
                    "CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE",
                    "COPATOLOGIES",
                    "DIABETES",
                    "HEPATITIS_B",
                    "HEPATITIS_C",
                    "HIV",
                    "HYPERTENSION",
                    "LIVER_FAILURE",
                    "NEOPLASMS",
                    "OBESITY",
                    "ORGAN_TRANSPLANT",
                )
            )
        )
    )


@lru_cache(maxsize=None)
def minimum_maximum_column_limits(criteria):
    """:criteria: == 'strict' limits were built with reference ranges from:
        https://en.wikipedia.org/wiki/Reference_ranges_for_blood_tests and
        https://en.wikipedia.org/wiki/Clinical_urine_tests#Target_parameters
    whilst :criteria: == 'relaxed' limits were built
        considering the 0.03 and 0.97 percentiles of the respective columns;
    finally :criteria: == 'nonsense' limits were built
        without any particular logic in mind
    """
    assert criteria in ("strict", "relaxed", "nonsense")
    return {
        rename_helper(k, errors="quiet"): v
        for k, v in {  # units in comments
            "ActualState_val": dict(min=0, max=7),
            "AGE": dict(min=0, max=128),
            "CARBON_DIOXIDE_PARTIAL_PRESSURE": dict(
                min=dict(strict=33, relaxed=25, nonsense=20).get(criteria),
                max=dict(strict=51, relaxed=55, nonsense=80).get(criteria),
            ),  # mmHg
            "CHARLSON_INDEX": dict(min=0, max=37),
            "UPDATED_CHARLSON_INDEX": dict(min=0, max=37),
            "CREATININE": dict(
                min=dict(strict=0.5, relaxed=0.48, nonsense=0).get(criteria),
                max=dict(strict=1.6, relaxed=2.03, nonsense=15).get(criteria),
            ),  # mg/dL
            "DYSPNEA": dict(min=0, max=1),  # just a boolean
            "D_DIMER": dict(
                min=dict(strict=50, relaxed=40, nonsense=50).get(criteria),
                max=dict(strict=2000, relaxed=4000, nonsense=40000).get(
                    criteria
                ),
            ),  # ng/mL
            "GPT_ALT": dict(
                min=dict(strict=5, relaxed=0, nonsense=5).get(criteria),
                max=dict(strict=56, relaxed=255, nonsense=255).get(criteria),
            ),  # IU/L
            "HOROWITZ_INDEX": dict(
                min=dict(strict=100, relaxed=50, nonsense=50).get(criteria),
                max=dict(strict=450, relaxed=500, nonsense=450).get(criteria),
            ),
            "LDH": dict(
                min=dict(strict=50, relaxed=50, nonsense=50).get(criteria),
                max=dict(strict=150, relaxed=1200, nonsense=1550).get(
                    criteria
                ),
            ),  # U/L
            "LYMPHOCYTE": dict(min=0, max=100),  # % on total white blood cells
            "PH": dict(
                min=dict(strict=7.31, relaxed=7.3, nonsense=6).get(criteria),
                max=dict(strict=7.45, relaxed=7.6, nonsense=8).get(criteria),
            ),
            "URINE_PH": dict(
                min=dict(strict=5, relaxed=5, nonsense=6).get(criteria),
                max=dict(strict=7, relaxed=7, nonsense=8).get(criteria),
            ),
            "PHOSPHOCREATINE": dict(
                min=dict(strict=0, relaxed=0, nonsense=0).get(criteria),
                max=dict(strict=0.7, relaxed=30, nonsense=50).get(criteria),
            ),  # mg/dL
            "PROCALCITONIN": dict(
                min=dict(strict=0, relaxed=0, nonsense=0).get(criteria),
                max=dict(strict=0.5, relaxed=2.6201, nonsense=10).get(
                    criteria
                ),
            ),  # ng/mL
            "RESPIRATORY_RATE": dict(
                min=dict(strict=12, relaxed=12, nonsense=10).get(criteria),
                max=dict(strict=40, relaxed=40, nonsense=100).get(criteria),
            ),  # breaths per min
            "UREA": dict(
                min=dict(strict=7, relaxed=7, nonsense=5).get(criteria),
                max=dict(strict=21, relaxed=160, nonsense=155).get(criteria),
            ),  # mg/dL
        }.items()
    }


def new_key_col_value(admission_date, birth_date=None, discharge_date=None):
    """This function assumes that two people:
    1) born on the same year-month-day
    2) AND taken in charge at the same year-month-day+hour:minute
    3) AND discharged the same year-month-day+hour:minute
    are really rare and almost impossible to found.
    """
    if pd.isna(admission_date):
        return np.nan
    admission_date = pd.to_datetime(admission_date).to_pydatetime()
    assert admission_date.year >= 2000, str(
        "Please add more hex-digit to the admission_date.year below"
    )
    if admission_date.second != 0:
        debug(
            f"seconds in admission_date ({admission_date}) "
            "will be rounded to the closer minute"
        )
    admission_timedelta_sec = (
        round(
            timedelta(
                hours=admission_date.time().hour,
                minutes=admission_date.time().minute,
                seconds=admission_date.time().second,
            ).total_seconds()
            / 60.0
        )
        * 60
    )
    admission_date = pd.to_datetime(admission_date.date()) + timedelta(
        seconds=admission_timedelta_sec
    )

    days_in_hospital = int("EEE", base=16)  # still in charge
    discharge_timedelta_sec = 0.0  # midnight
    if pd.notna(discharge_date):
        days_in_hospital = (
            pd.to_datetime(discharge_date).to_pydatetime().date()
            - admission_date.date()
        ).days
        if discharge_date.second != 0:
            debug(
                f"seconds in discharge_date ({discharge_date}) "
                "will be rounded to the closer minute"
            )
        discharge_timedelta_sec = (
            round(
                timedelta(
                    hours=discharge_date.time().hour,
                    minutes=discharge_date.time().minute,
                    seconds=discharge_date.time().second,
                ).total_seconds()
                / 60.0
            )
            * 60
        )
        discharge_date = pd.to_datetime(discharge_date.date()) + timedelta(
            seconds=discharge_timedelta_sec
        )
        if days_in_hospital < 0:
            warning(
                f"admission_date '{str(admission_date)}' occurs after "
                f"discharge_date '{str(discharge_date)}'"
            )
            days_in_hospital = int("FFF", base=16)  # bad date range
            discharge_timedelta_sec = 0.0  # midnight

    if pd.notna(birth_date):
        birth_date = pd.to_datetime(birth_date).to_pydatetime()
    else:
        # fake a birthday in the future to distinguish these patients
        birth_date = datetime(
            year=1900 + int("F1", base=16),  # 2141
            month=int("1", base=16),  # January
            day=int("1F", base=16),  # 31th
        )
    ret = (
        timestamp_to_hex_date(
            admission_date.date(), year_offset=2000, desired_length=5
        )
        + timestamp_to_hex_date(
            birth_date.date(), year_offset=1900, desired_length=2 + 5
        )
        + f"{days_in_hospital:X}".rjust(2 + 3, "0")
        # next line counts the minutes since midnight of admission_date
        + f"{round(admission_timedelta_sec / 60.0):X}".rjust(1 + 3, "0")
        # next line counts the minutes since midnight of discharge_date
        + f"{round(discharge_timedelta_sec / 60.0):X}".rjust(3, "0")
    ).upper()
    debug(
        f"new key {repr(ret)} identifies the patient with ("
        + repr(
            {
                "admission_date": str(admission_date),
                "birth_date": str(birth_date),
                "days_in_hospital": str(days_in_hospital),
                "discharge_date": str(discharge_date),
            }
        )[1:-1].replace("': ", "'==")
        + ")."
    )
    assert revert_new_key_col_value(ret) == (
        admission_date,
        birth_date,
        None
        if discharge_date is None or days_in_hospital >= int("EEE", base=16)
        else discharge_date,
    ), f"revert returned: {repr(revert_new_key_col_value(ret))}"
    return ret


def oxygen_support_enum_rule(column_values):
    oxygen_map = {
        "": "hfno",
        "": "hfno",
        "": "absent",
        "niv": "niv",
        "no": "absent",
        "": "nasal cannula",
        "reservoir": "with reservoir bag",
        "": "nasal cannula",
        "ventimask": "venturi mask",
        "venturi": "venturi mask",
    }
    if not any(
        (
            word.strip(digits + punctuation)
            in set(oxygen_map.keys()).difference({"", "no", ""})
            for val in set(column_values)
            for word in str(val).lower().split()
        )
    ):
        raise ValueError("Bad guess, retry")
    oxygen_map = {
        k: v.capitalize() if v not in ("hfno", "niv") else v.upper()
        for k, v in oxygen_map.items()
    }
    for val in set(column_values):
        for word in str(val).lower().split():
            nice_word = word.strip(digits + punctuation)
            if nice_word not in oxygen_map.keys():
                continue
            if all(w not in nice_word for w in ("reserv", "", "")):
                nice_word = oxygen_map[nice_word]
                if "con" in str(val).lower():
                    nice_word += " with"
                elif "senza" in str(val).lower():
                    nice_word += " without"
                if "reserv" in str(val).lower():
                    nice_word += " reservoir bag"
                oxygen_map[val] = nice_word
            else:
                oxygen_map[val] = oxygen_map[nice_word]
            break
    dtype = pd.CategoricalDtype(
        categories=sorted(set(oxygen_map.values())), ordered=False
    )
    return dtype, oxygen_map


def oxygen_state_enum_rule(column_values):
    oxygen_states = [
        # this list should be equal to YAstarMM.model.State.names();
        # but unfortunately YAstarMM.model can not be imported because
        # it would generate a recursive dependency
        # ORDER DOES MATTER
        "No O2",
        "O2",
        "HFNO",
        "NIV",
        "Intubated",
        "Deceased",
        "Discharged",
        "Transferred",
    ]
    if set(
        str(v).strip().lower() for v in column_values if pd.notna(v)
    ).difference(set(s.lower() for s in oxygen_states)):
        raise ValueError("Bad guess, retry")
    oxygen_state_map = {
        val: state
        for val in set(column_values)
        for state in set(oxygen_states)
        if pd.notna(val) and val.strip().lower() == state.lower()
    }
    dtype = pd.CategoricalDtype(categories=oxygen_states, ordered=True)
    return dtype, oxygen_state_map


@lru_cache(maxsize=None)
def progressive_features():
    return iter(
        rename_helper(
            (
                "ANAKINRA",
                "ANAKINRA_1ST_DOSE",
                "ANAKINRA_SAMPLE_T0",
                "ANAKINRA_SAMPLE_T2",
                "ANAKINRA_SAMPLE_T7",
                "ANTIBIOTIC",
                "DYSPNEA_START",
                "HOROWITZ_INDEX_UNDER_150",
                "HOROWITZ_INDEX_UNDER_250",
                "ICU_TRANSFER",
                "IMMUNOLOGICAL_THERAPY",
                "INFECTIOUS_DISEASES_UNIT_TRANSFER",
                "INTUBATION_STATE_END",
                "INTUBATION_STATE_START",
                "NIV_STATE_END",
                "NIV_STATE_START",
                "OXYGEN_THERAPY_STATE_END",
                "OXYGEN_THERAPY_STATE_START",
                "PLAQUENIL",
                "PLAQUENIL_1ST_DATE",
                "REMDESIVIR",
                "REMDESIVIR_1ST_DATE",
                "SWAB",
                "SYMPTOMS_START",
                "TOCILIZUMAB",
                "TOCILIZUMAB_1ST_DOSE",
                "TOCILIZUMAB_2ND_DOSE",
                "TOCILIZUMAB_SAMPLE_T0",
                "TOCILIZUMAB_SAMPLE_T2",
                "TOCILIZUMAB_SAMPLE_T7",
            )
        )
    )


def rename_helper(columns, errors="warn"):
    assert columns is not None and (
        isinstance(columns, str)
        or all(isinstance(col, str) for col in columns)
    ), str(
        "Argument must be a string or a sequence of strings; "
        f"got {repr(columns)}"
    )
    assert errors in ("quiet", "raise", "warn")

    if isinstance(columns, str):
        return _rename_helper(columns, errors=errors)
    return tuple(
        _rename_helper(old_col_name, errors=errors) for old_col_name in columns
    )


@lru_cache(maxsize=None)
def _rename_helper(old_col_name, errors="warn"):
    assert isinstance(old_col_name, str)
    assert errors in ("quiet", "raise", "warn")

    if old_col_name.startswith("Has "):
        old_col_name = old_col_name.replace("Has ", "")
    new_col_name = HARDCODED_COLUMN_NAMES.get(old_col_name, None)
    if new_col_name is not None:
        debug(
            "Hardcoded column name found, replacing it"
            f"\t({old_col_name} ~> {new_col_name})"
        )
        return new_col_name

    if old_col_name.replace(" ", "_") in (
        "No_O2",
        "O2",
        "HFNO",
        "NIV",
        "Intubated",
        "Deceased",
        "Discharged",
        "Transferred",
    ):
        debug(f"Skipping state name {old_col_name}")
        return old_col_name

    for reason, list_of_rules in drop_rules.items():
        for rule in list_of_rules:
            if rule.match(old_col_name) is not None:
                raise NotImplementedError(
                    str(
                        f"Column '{old_col_name}' has probably been dropped;"
                        " please adapt your code accordingly.."
                    )
                )
    #
    # any drop rule matched; let us look for a keep rule
    #
    for group, list_of_mappings in keep_rules.items():
        for mapping in list_of_mappings:
            for new_col_name in mapping.keys():
                if new_col_name != "" and old_col_name.lower() == new_col_name:
                    debug(
                        "New name found:\t"
                        f"({old_col_name} ~> {new_col_name})"
                    )
                    return new_col_name
    for group, list_of_mappings in keep_rules.items():
        for mapping in list_of_mappings:
            for new_col_name, rule in mapping.items():
                if new_col_name != "" and rule.match(old_col_name) is not None:
                    debug(
                        "New name found:\t"
                        f"({old_col_name} ~> {new_col_name})"
                    )
                    return new_col_name
    if errors == "raise":
        raise Exception(f"No renaming was found for '{old_col_name}'")
    else:
        log(
            level=dict(quiet=DEBUG, warn=WARNING).get(errors),
            msg=str(
                "Any hardcoded or keep rule matched against "
                f"'{old_col_name}'; keeping it"
            ),
        )
        return old_col_name


def revert_new_key_col_value(new_key_col):
    assert isinstance(new_key_col, str) and len(new_key_col) == 24
    discharge_timedelta_sec = int(new_key_col[-3:], base=16) * 60
    admission_timedelta_sec = int(new_key_col[-1 - 3 - 3 : -3], base=16) * 60
    admission_date = hex_date_to_timestamp(
        new_key_col[:5],
        year_offset=2000,
        time_offset=timedelta(seconds=admission_timedelta_sec),
    )
    birth_date = hex_date_to_timestamp(
        new_key_col[5 : 5 + 2 + 5], year_offset=1900, drop_first_digits=2
    )
    days_in_hospital = int(new_key_col[5 + 2 + 5 : -1 - 3 - 3], base=16)
    if days_in_hospital >= int("EEE", base=16):
        discharge_date = None
    else:
        discharge_date = pd.to_datetime(
            admission_date.normalize().to_pydatetime()
            + timedelta(days=days_in_hospital, seconds=discharge_timedelta_sec)
        )
    return (admission_date, birth_date, discharge_date)


def sex_enum_rule(column_values, sex_map=None):
    if sex_map is None:
        sex_map = dict(
            f="female",
            female="female",
            m="male",
            male="male",
            xx="female",
            xy="male",
        )

    if not all((str(sex).lower() in sex_map for sex in set(column_values))):
        raise ValueError("Bad guess, retry")

    sex_map = {k: v.capitalize() for k, v in sex_map.items()}
    for val in set(column_values):
        if val not in sex_map:
            sex_map[val] = sex_map[str(val).lower()]
    dtype = pd.CategoricalDtype(
        categories=set(sorted(set(sex_map.values()), key=lambda _: random())),
        ordered=False,  # gender neutral
    )
    return dtype, sex_map


@lru_cache(maxsize=None)
def shift_features(sheet_name):
    return iter(
        rename_helper(
            dict(
                emogas=(
                    "BICARBONATE",
                    "CARBON_DIOXIDE_PARTIAL_PRESSURE",
                    "HOROWITZ_INDEX",
                    "LACTATES",
                    "OXYGEN_PARTIAL_PRESSURE",
                    "OXYGEN_SATURATION",
                    "PH",
                ),
                symptoms=(
                    "DIASTOLIC_PRESSURE",
                    "DYSPNEA",
                    "HEARTH_RATE",
                    "RESPIRATORY_RATE",
                    "SYSTOLIC_PRESSURE",
                    "TEMPERATURE",
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
                old_columns=rename_helper(
                    tuple(sfi.old_columns), errors="warn"
                ),
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
                SummarizeFeatureItem(["DYSPNEA"], [False], "no dyspnea"),
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
                    ["USE_OXYGEN"], [False], "no oxygen used"
                ),
                SummarizeFeatureItem(
                    ["USE_OXYGEN", "RESPIRATORY_RATE"],
                    [True, lambda rr: pd.notna(rr) and float(rr) < 30],
                    "oxygen used and respiratory rate < 30",
                ),
                SummarizeFeatureItem(
                    ["USE_OXYGEN", "RESPIRATORY_RATE"],
                    [True, lambda rr: pd.notna(rr) and float(rr) >= 30],
                    "oxygen used and respiratory rate >= 30",
                ),
                SummarizeFeatureItem(
                    ["NIV_STATE"], [True], "non-invasive ventilation"
                ),
                SummarizeFeatureItem(
                    ["INTUBATION_STATE"], [True], "intubated"
                ),
            ],
        }.items()
    }


def swab_enum_rule(column_values):
    swab_common_values = [
    ]
    if not any(
        (
            " ".join(str(val).lower().split()) in swab_common_values
            for val in set(column_values)
        )
    ):
        raise ValueError("Bad guess, retry")

    swab_common_values = [
        v.capitalize().replace("", "") for v in swab_common_values
    ]
    conversion_map = {val: val for val in swab_common_values}
    for val in set(column_values):
        nice_val = (
            " ".join(str(val).split())
            .capitalize()
            .replace(
                "",
                "",
            )
        )
        if nice_val not in swab_common_values:
            swab_common_values.append(nice_val)
            conversion_map[val] = nice_val
    dtype = pd.CategoricalDtype(
        categories=sorted(swab_common_values),
        ordered=False,
    )
    return dtype, conversion_map


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


@lru_cache(maxsize=None)
def translator_helper(old_col_name, bold=False, usetex=False, **kwargs):
    """Make column names pretty enough to be used as plot titles"""
    if bold and not usetex:
        warning("bold flag is useless without also usetex flag set")
    col_name = _rename_helper(old_col_name, errors="quiet")
    mapping = dict(**kwargs)
    mapping.update(
        {
            {
                True: "".join(
                    (
                        r"$\mathrm{",
                        r"\mathbf{" if bold else "",
                        "No~O",
                        r"}" if bold else "",
                        r"}_{\mathrm{",
                        r"\mathbf{" if bold else "",
                        "2",
                        r"}" if bold else "",
                        r"}}$",
                    )
                ),
                False: "No O2",
            }.get(usetex): ("no_o2",),
            {
                True: "".join(
                    (
                        r"$\mathrm{",
                        r"\mathbf{" if bold else "",
                        "O",
                        r"}" if bold else "",
                        r"}_{\mathrm{",
                        r"\mathbf{" if bold else "",
                        "2",
                        r"}" if bold else "",
                        r"}}$",
                    )
                ),
                False: "O2",
            }.get(usetex): ("o2",),
            "HFNO": ("hfno",),
            "NIV": ("niv",),
            "Intubated": ("intubated",),
            "Deceased": ("deceased",),
            "Discharged": ("discharged",),
            "Transferred": ("transferred",),
            "Alanine transaminase": ("gpt_alt",),
            "Age": ("age",),
            {
                True: "".join(
                    (
                        r"$\mathrm{",
                        r"\mathbf{" if bold else "",
                        "pCO",
                        r"}" if bold else "",
                        r"}_{\mathrm{",
                        r"\mathbf{" if bold else "",
                        "2",
                        r"}" if bold else "",
                        r"}}$",
                    )
                ),
                False: "pCO2",
            }.get(usetex): ("carbon_dioxide_partial_pressure",),
            "Charlson Comorbidity Index": ("",),
            "Charlson Comorbidity Index (updated)": (
                "updated_charlson_index",
            ),
            "Creatinine": ("",),
            "D-dimer": ("",),
            "Days in the same state": ("days_in_state",),
            "Dyspnea": ("",),
            "Horowitz-Index": ("horowitz_index",),
            "Lactate dehydrogenase": ("ldh",),
            "Lymphocytes": ("",),
            "pH (blood)": ("ph",),
            "pH (urine)": ("urine_ph",),
            "Phosphocreatine": ("pcr",),
            "Procalcitonin": ("",),
            "Respiratory rate": ("",),
            "Urea": ("urea",),
        }
    )
    for new_col_name, match_list in mapping.items():
        if col_name.lower().replace(" ", "_").replace("-", "_") in match_list:
            debug(
                "Translator helper hit a match:\t"
                f"{old_col_name} ~> {col_name} ~> {new_col_name}"
            )
            if usetex:
                new_col_name = new_col_name.replace("-", "--")
                if bold:
                    new_col_name = "".join((r"\textbf{", new_col_name, r"}"))
            return new_col_name

    return old_col_name
    raise KeyError(
        f"No translation was found for '{old_col_name}' ('{col_name}');"
        " you can provide a custom mapping to extend the hardcoded defaults"
    )


def verticalize_features():
    for item in [
        VerticalizeFeatureItem("ANAKINRA", "ANAKINRA", [""]),
        VerticalizeFeatureItem(
            "ANAKINRA_1ST_DOSE", "ANAKINRA_1ST_DOSE", ["ANAKINRA_1ST_VIA"]
        ),
        VerticalizeFeatureItem(
            "ANAKINRA_2ND_DOSE", "ANAKINRA_2ND_DOSE", ["ANAKINRA_2ND_VIA"]
        ),
        VerticalizeFeatureItem(
            "ANAKINRA_SAMPLE_T0", "ANAKINRA_SAMPLE_T0", list()
        ),
        VerticalizeFeatureItem(
            "ANAKINRA_SAMPLE_T2", "ANAKINRA_SAMPLE_T2", list()
        ),
        VerticalizeFeatureItem(
            "ANAKINRA_SAMPLE_T7", "ANAKINRA_SAMPLE_T7", list()
        ),
        VerticalizeFeatureItem(
            "ANTIBIOTIC", "ANTIBIOTIC", ["ANTIBIOTIC_NOTES"]
        ),
        VerticalizeFeatureItem("DYSPNEA_START", "DYSPNEA_START", list()),
        VerticalizeFeatureItem(
            "", "SYMPTOMS_START", [""]
        ),
        VerticalizeFeatureItem("PLAQUENIL", "PLAQUENIL", list()),
        VerticalizeFeatureItem(
            "PLAQUENIL_1ST_DATE", "", list()
        ),
        VerticalizeFeatureItem(
            "", "HOROWITZ_INDEX_UNDER_150", list()
        ),
        VerticalizeFeatureItem(
            "", "HOROWITZ_INDEX_UNDER_250", list()
        ),
        VerticalizeFeatureItem("REMDESIVIR", "REMDESIVIR", list()),
        VerticalizeFeatureItem(
            "REMDESIVIR_1ST_DATE", "", list()
        ),
        VerticalizeFeatureItem("SWAB_CHECK_DATE", "SWAB", ["SWAB_RESULT"]),
        VerticalizeFeatureItem(
            "IMMUNOLOGICAL_THERAPY_DATE",
            "IMMUNOLOGICAL_THERAPY",
            ["IMMUNOLOGICAL_THERAPY"],
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB", "TOCILIZUMAB", [""]
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB_1ST_DOSE",
            "TOCILIZUMAB_1ST_DOSE",
            ["TOCILIZUMAB_1ST_VIA"],
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB_2ND_DOSE",
            "TOCILIZUMAB_2ND_DOSE",
            ["TOCILIZUMAB_2ND_VIA"],
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB_SAMPLE_T0", "TOCILIZUMAB_SAMPLE_T0", list()
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB_SAMPLE_T2", "TOCILIZUMAB_SAMPLE_T2", list()
        ),
        VerticalizeFeatureItem(
            "TOCILIZUMAB_SAMPLE_T7", "TOCILIZUMAB_SAMPLE_T7", list()
        ),
        VerticalizeFeatureItem(
            "", "ICU_TRANSFER", list()
        ),
        VerticalizeFeatureItem(
            "",
            "INFECTIOUS_DISEASES_UNIT_TRANSFER",
            list(),
        ),
    ]:
        yield VerticalizeFeatureItem(
            date_column=rename_helper(item.date_column, errors="warn"),
            column_name=rename_helper(item.column_name, errors="warn"),
            related_columns=rename_helper(tuple(item.related_columns)),
        )


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.column_rules",
    "YAstarMM.column_rules",
    "column_rules",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
