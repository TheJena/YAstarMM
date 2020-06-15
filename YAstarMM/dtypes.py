#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020 Federico Motta <191685@studenti.unimore.it>
#                    Davide Ferrari <davideferrari@unimore.it>
#                    Francesco Ghinelli <213106@studenti.unimore.it>
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
   Rename Pandas DataFrame columns and set them the correct data type.

   Both the operations are defined in a vocabulary file (xlsx)
   containing a DataFrame with:
   - the rename criteria (e.g. a translation in another language)
   - the desired data type (e.g. float16 or float64 depending on the
     following learning method / architecture used)

   Usage:
            from  YAstarMM.dtypes  import  (
                set_dtype_vocabulary,
                set_name_vocabulary,
            )

   ( or from within the YAstarMM package )

            from           dtypes  import  (
                set_dtype_vocabulary,
                set_name_vocabulary,
            )
"""
import pandas as pd
from sys import version_info
from typing import Optional, TextIO, Union


def _get_feature_name(
    feature: Optional[str] = None,
    vocabulary_df: Optional[pd.DataFrame] = None,
    column_name: str = "rename",
) -> Optional[str]:
    """Return the new name of a requested variable according to the vocabulary.

       The vocabulary has a single column explaining the feature (aka
       variable) name in english.

       :param feature: Name of the requested feature.
       :param vocabulary_df: DataFrame from which information is extracted
       :param column_name: Column name containing the desired naming convention
       :return: Name of the requested feature in the desired naming convention
    """
    if feature is None or not feature:
        print("No feature passed, returning it as it is")
        return feature

    if vocabulary_df is None:
        print("Did not provide a vocabulary DataFrame, ", end="")
        print("returning 'feature' as it is")
        return feature

    try:
        name = vocabulary_df[vocabulary_df["feature"] == feature][column_name]
        name = name.dropna(axis="index")  # Drop rows contain missing values
    except IndexError:
        # print(f'The dtype for feature {feature} was not set')
        return feature
    except KeyError:
        print(f"Feature '{feature}' is not in the vocabulary DataFrame")
        return feature
    else:  # Success !
        return str(name.values[0])


def set_name_vocabulary(
    df: pd.DataFrame,
    vocabulary_file: Union[None, str, TextIO] = None,
    column_name: str = "rename",
) -> Optional[pd.DataFrame]:
    """Rename DataFrame columns.

       :param df: DataFrame containing columns to be renamed
       :param vocabulary_file: File containing DataFrame defining the rename
       :param column_name: Column (in vocabulary) defining the rename criteria
       :return: DataFrame with renamed columns or None
    """
    if vocabulary_file is None or not vocabulary_file:
        print("No vocabulary file passed.")
        # FIXME: Wouldn't it be better returning the DataFrame as it is here?
        return None

    vocabulary_df = pd.read_excel(vocabulary_file)

    for feature in list(df.columns):
        new_name = _get_feature_name(feature, vocabulary_df, column_name)
        if new_name is not None and new_name != feature:
            df.rename(
                columns={feature: new_name}, inplace=True,
            )
        else:
            print(f"Name not set for feature '{feature}'")

    return df


def _get_feature_dtype(
    feature: Optional[str] = None,
    vocabulary_df: Optional[pd.DataFrame] = None,
    column_name: str = "dtype",
) -> Optional[str]:
    """Return the dtype of a requested variable according to the vocabulary.

       The vocabulary has two different columns containing the type of
       the feature (aka variable).

       Since in datasets both type standards can occur, it is also
       provided the possibility to get the dtype from the second type
       column / convention.

       :param feature: Name of the requested feature.
       :param vocabulary_df: DataFrame from which information is extracted
       :param column_name: Column name containing the desired type convention
       :return: The dtype of the requested feature in the desired convention
    """
    if feature is None or not feature:
        print("No feature passed, returning 'object' by default")
        return "object"

    if vocabulary_df is None:
        print("Did not provide a vocabulary DataFrame. Returning 'object'")
        return "object"

    try:
        dtype = vocabulary_df[vocabulary_df["feature"] == feature][column_name]
        dtype = dtype.dropna(axis="index")  # Drop rows contain missing values
    except IndexError:
        # print(f'The dtype for feature {feature} was not set')
        return None
    except KeyError:
        print(f"Feature '{feature}' is not in the vocabulary DataFrame")
        return None
    else:  # Success !
        return str(dtype.values[0])


def set_dtype_vocabulary(
    df: pd.DataFrame,
    vocabulary_file: Union[None, str, TextIO] = None,
    column_name: str = "dtype",
) -> Optional[pd.DataFrame]:
    """Cast Dataframe columns.

       :param df: Dataframe containing columns to be casted
       :param vocabulary_file: File containing the DataFrame defining the cast
       :param column_name: Column (in vocabulary) defining the cast
       :return: DataFrame with renamed columns or None
    """
    if vocabulary_file is None or not vocabulary_file:
        print("No vocabulary file passed.")
        # FIXME: Wouldn't it be better returning the DataFrame as it is here?
        return None

    vocabulary_df = pd.read_excel(vocabulary_file)

    for feature in list(df.columns):
        data_type = _get_feature_dtype(feature, vocabulary_df, column_name)
        if data_type is not None:
            df[feature] = df[feature].astype(data_type)
        else:
            print(f"Dtype not set for feature '{feature}'")

    return df


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__ in ("YAstarMM.dtypes", "dtypes"),
        "set_dtype_vocabulary" in globals(),
        "set_name_vocabulary" in globals(),
    )
), "Please update 'Usage' section of module docstring"
