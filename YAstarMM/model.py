#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020 Federico Motta <191685@studenti.unimore.it>
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
   Modellize data from an input DataFrame into: States, Events and Journeys

   A State is the condition in which a patient is.
   An Event is the starting or ending timestamp of a State.
   A Journey is a sequence of Events.

   Usage:
            from  YAstarMM.model  import  (
                State, AdmissionEvent, ... , ReleaseEvent, HospitalJourney,
            )

   ( or from within the YAstarMM package )

            from          .model  import  (
                State, AdmissionEvent, ... , ReleaseEvent, HospitalJourney,
            )
"""

from .column_rules import rename_helper
from .constants import (  # without the dot notebook raises ModuleNotFoundError
    EXECUTING_IN_JUPYTER_KERNEL,
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    LOGGING_STREAM,
    LOGGING_STYLE,
    NASTY_SUFFIXES,
    ORDINARILY_HOME_DISCHARGED,
)
from datetime import datetime, timedelta
from enum import IntEnum, unique
from sys import version_info
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
import logging
import numpy as np
import pandas as pd


def _enforce_datetime_dtype(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cast DataFrame columns to Timestamp data type"""
    assert datetime.now().year <= 2029, str(
        "Please fix all tests using string '/202' to determine "
        "dayfirst/yearfirst boolean values to help Pandas parsing dates"
    )

    for idx, (col_name, series) in enumerate(dataframe.items()):
        if "datetime64" in repr(series.dtype):
            continue  # col_name is already casted correctly
        dayfirst, yearfirst = False, False
        for cell in series:
            if pd.isna(cell):
                continue
            if "/202" in repr(cell):  # 21st century
                # e.g. '25/04/2020 12:00:00'
                dayfirst = True
            else:
                # e.g. '2020-04-25 12:00:00'
                yearfirst = True
            break
        dataframe.iloc[:, idx] = pd.to_datetime(
            series, dayfirst=dayfirst, yearfirst=yearfirst
        )
    return dataframe


def _select_df_cols(
    dataframe: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:  # make black auto-formatting prettier
    """Return DataFrame columns

    Of course a simple selection like df[[col1, col2, ...]] would have
    done this fabulously.

    But since people are often not able to join Pandas DataFrames
    properly; this function allows to deal with renamed (by adding
    suffixes) columns (like pd.DataFrame.merge does in case of
    overlapping names)
    """
    suffixes = set(NASTY_SUFFIXES).union(set(("",)))  # add empty string
    columns_to_filter = [
        col + suffix
        for col in columns
        for suffix in suffixes
        if col + suffix in dataframe.columns
    ]
    if sorted(columns_to_filter) != sorted(columns):
        logging.debug(
            f" Using column filter [{', '.join(columns_to_filter)}, ] "
            f"instead of [{', '.join(columns)}, ]"
        )
    return dataframe[columns_to_filter]


def _select_df_rows_with_not_nan_values_in_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Return DataFrame rows with at least one not Nan value in given columns

    Of course a simple selection like df[df[col1, col2, ...].notna()]
    would have done this fabulously.

    But since people are often not able to join Pandas DataFrames
    properly; this function allows to deal with renamed (by adding
    suffixes) columns (like pd.DataFrame.merge does in case of
    overlapping names)
    """
    positions = [
        list(dataframe.index.values).index(data.Index)  # real index position
        for data in _select_df_cols(dataframe, columns)
        .notna()
        .itertuples(index=True)  # make black auto-formatting prettier
        if any((v for k, v in data._asdict().items() if k != "Index"))
    ]
    return dataframe.iloc[positions]


def new_columns_to_add(
    main_df_len: int,
) -> Iterator[Tuple[str, Union[pd.Categorical, pd.Series]]]:
    """Return iterator over (new columns name, new columns series)."""
    for col in ordered_state_transition_columns():
        if col not in rename_helper(("ActualState", "ActualState_val")):
            yield (col, pd.Series(data=[np.nan for _ in range(main_df_len)]))
    yield (
        rename_helper("ActualState"),
        pd.Categorical(
            values=[np.nan for i in range(main_df_len)],
            categories=State.names(),
            ordered=True,
        ),
    )
    yield (
        rename_helper("ActualState_val"),
        pd.Categorical(
            values=[np.nan for i in range(main_df_len)],
            categories=State.values(),
            ordered=True,
        ),
    )


def ordered_state_transition_columns() -> Tuple[str, ...]:
    # Order does matter, do not change it please
    return rename_helper(
        (
            "No_Ossigenoterapia_Inizio",
            "No_Ossigenoterapia",
            "No_Ossigenoterapia_Fine",
            "Ossigenoterapia_Inizio",
            "Ossigenoterapia",
            "Ossigenoterapia_Fine",
            "HFNO_Inizio",
            "HFNO",
            "HFNO_Fine",
            "NIV_Inizio",
            "NIV",
            "NIV_Fine",
            "Intubazione_Inizio",
            "Intubazione",
            "Intubazione_Fine",
            "ActualState",
            "ActualState_val",
        )
    )


@unique
class State(IntEnum):
    """Allowed states of a patient during its journey in the hospital."""

    No_O2 = 0
    """No oxygen therapy"""

    O2 = 1
    """Oxygen therapy"""

    HFNO = 2
    """High flow nasal oxygen therapy"""

    NIV = 3
    """Non-invasive ventilation therapy"""

    Intubated = 4
    """Invasive ventilation technique for patient in intensive care unit"""

    Deceased = 5
    """Release state: patient is dead"""

    Discharged = 6
    """Release state: patient is sent back home"""

    Transferred = 7
    """Release state: patient is moved to another hospital"""

    def __str__(self) -> str:
        return self.name.replace("_", " ")

    @property
    def start_col(self) -> str:
        """Column name in Pandas DataFrame denoting state start."""
        return dict(
            No_O2=rename_helper("No_Ossigenoterapia_Inizio"),
            O2=rename_helper("Ossigenoterapia_Inizio"),
            HFNO=rename_helper("HFNO_Inizio"),
            NIV=rename_helper("NIV_Inizio"),
            Intubated=rename_helper("Intubazione_Inizio"),
        )[
            self.name
        ]  # can raise a KeyError on release states

    @property
    def fill_col(self) -> str:
        """Column name in Pandas DataFrame denoting state persistence."""
        return dict(
            No_O2=rename_helper("No_Ossigenoterapia"),
            O2=rename_helper("Ossigenoterapia"),
            HFNO=rename_helper("HFNO"),
            NIV=rename_helper("NIV"),
            Intubated=rename_helper("Intubazione"),
        )[
            self.name
        ]  # can raise a KeyError on release states

    @property
    def end_col(self) -> str:
        """Column name in Pandas DataFrame denoting state end."""
        return dict(
            No_O2=rename_helper("No_Ossigenoterapia_Fine"),
            O2=rename_helper("Ossigenoterapia_Fine"),
            HFNO=rename_helper("HFNO_Fine"),
            NIV=rename_helper("NIV_Fine"),
            Intubated=rename_helper("Intubazione_Fine"),
        )[
            self.name
        ]  # can raise a KeyError on release states

    @classmethod
    def names(cls) -> List[str]:
        """Return list containing all the state names."""
        return [str(member) for _, member in State.__members__.items()]

    @classmethod
    def non_final_states_names(cls) -> Tuple[str]:
        return tuple(
            str(s)
            for s in (
                State.No_O2,
                State.O2,
                State.HFNO,
                State.NIV,
                State.Intubated,
            )
        )

    @classmethod
    def values(cls) -> List[int]:
        """Return list containing all the state's integer values."""
        return [int(member.value) for _, member in State.__members__.items()]


class Event(object):
    """General event denoting a state start or end."""

    text_pad = 48  # yet another magic number ;-)

    @property
    def index(self) -> int:
        """Chronological event position, starting from zero."""
        return int(self._index)

    @property
    def label(self) -> str:
        """String description of the event."""
        return str(self._label)

    @property
    def state(self) -> State:
        """State related to the event."""
        if self._state is None:
            raise NotImplementedError(
                "You probably forgot to override this property :D"
            )
        return self._state

    @property
    def filter_col(self) -> str:
        """Column to use in boolean indexing filter."""
        return str(self._filter_col)

    @property
    def timestamp_col(self) -> str:
        """Column from which extract timestamps of interest."""
        return str(self._timestamp_col)

    @property
    def is_start(self) -> bool:
        """Return whether state is starting."""
        return bool(self._start)

    @property
    def is_end(self) -> bool:
        """Return wheter state is ending."""
        return not self.is_start

    @property
    def is_nat(self) -> bool:
        """Return wheter the date of the event is Not-A-Timestamp."""
        return bool(pd.isna(self.value))

    @property
    def callable(self) -> Callable[..., Any]:  # type: ignore
        """Function to choose one of the desired timestamps."""
        return self._callable

    @property
    def date(self) -> pd.Timestamp:
        """Return timestamp of the date of the event."""
        return pd.to_datetime(self.value.date())

    @property
    def value(self) -> pd.Timestamp:
        """Return timestamp of the event."""
        if self._value is None:
            if self._dataframe is None:
                raise ValueError(
                    "You probably forgot to pass 'value' kwarg in __init__()"
                )
            filtered_df = _select_df_cols(
                _select_df_rows_with_not_nan_values_in_columns(
                    self._dataframe, (self.filter_col,)
                ),
                (self.timestamp_col,),
            )
            filtered_df = _enforce_datetime_dtype(filtered_df)
            if filtered_df.empty:
                self._value = pd.to_datetime(pd.NaT)
                return self._value
            filtered_series = self.callable(filtered_df)
            my_max = self.callable(filtered_series)
            my_series = my_max
            if not isinstance(
                my_series,
                (
                    pd.Timestamp,
                    type(pd.NaT),
                ),
            ):
                raise ValueError(
                    "Event.callable should return a Pandas.Timestamp; "
                    f"got a '{type(my_series)}' instead"
                )
            self._value = pd.to_datetime(my_series)
        return self._value

    @value.setter
    def value(self, new_value: Union[str, pd.Timestamp]) -> None:
        """Set a new timestamp to the event."""
        if isinstance(new_value, pd.Timestamp):
            self._value = new_value
            return
        if pd.isna(new_value):
            self._value = pd.NaT
            return
        if isinstance(new_value, str):
            if new_value.lower() in (
                "nat",
                "nan",
            ):
                self._value = pd.NaT
                return
            if new_value.lower() in (
                "now",
                "today",
            ):
                self._value = pd.to_datetime(datetime.now().date())
                return
        raise ValueError(
            "Allowed values are: pandas.Timestamps, "
            "str('now'), str('nat'), str('nan')"
        )

    def __init__(
        self,
        index: int,
        label: str,
        state: Optional[State],
        dataframe: pd.DataFrame,
        timestamp_col: Optional[str],
        filter_col: Optional[str] = None,
        start: Optional[bool] = None,
        start_callable: Callable[..., pd.Series] = pd.DataFrame.min,
        end: Optional[bool] = None,
        end_callable: Callable[..., pd.Series] = pd.DataFrame.max,
        value: Optional[pd.Timestamp] = None,
    ) -> None:
        """Save passed arguments in private instance variables.

        :param index: Chronological event position, starting from zero
        :param label: String description of the event
        :param state: Corresponding state (enum)
        :param dataframe: Pandas DataFrame containing desired timestamp
        :param filter_col: Column of the Pandas DataFrame used to
                           filter with boolean indexing the rows of interest
                           (if None it is assumed to be equal to timestamp_col)
        :param timestamp_col: Column of the Pandas DataFrame used to
                              extract from the rows of interest the
                              desired timestamps
        :param start: Boolean flag to denote the event as the state start
        :param start_callable: Function to choose one of the desired
                               timestamps (in case start flag is True)
        :param end: Boolean flag to denote the event as the state end
        :param end_callable: Function to choose one of the desired
                               timestamps (in case end flag is True)
        :param value: Optional timestamp of the event
        """
        super(Event, self).__init__()
        if bool(start) == bool(end):
            raise ValueError("Please set either the start or the end flag")
        self._index = index
        self._label = label
        self._state = state
        self._dataframe = dataframe
        self._timestamp_col = timestamp_col
        if filter_col is None:
            self._filter_col = timestamp_col
        else:
            self._filter_col = filter_col
        self._start = bool(start)
        assert self._start == (not bool(end)), "start_flag != not end_flag"
        self._callable = start_callable if self._start else end_callable
        assert datetime.now().year <= 2029, str(
            "Please fix all tests using string '/202' to determine "
            "dayfirst/yearfirst boolean values to help Pandas parsing dates"
        )
        self._value = value

    def __repr__(self) -> str:
        if len(f"{self.label}:") >= self.text_pad:
            raise ValueError(
                str("Please increase max label size " f"({self.text_pad})")
            )
        return f"{self.label}:".ljust(self.text_pad) + repr(self.value)

    def __str__(self) -> str:
        return self.label

    def day_offset(self, days: int) -> pd.Timestamp:
        """Return timestamp of the date of the event plus (or minus) offset."""
        return pd.to_datetime((self.value + timedelta(days=days)).date())


class AdmissionEvent(Event):
    """Patient admission to the hospital."""

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 0,
        label: str = "admission",
        start: bool = True,
        state: State = State.No_O2,
        timestamp_col: str = rename_helper("AdmissionTime"),
        **kwargs,
    ) -> None:
        super(AdmissionEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class NoO2StartEvent(Event):
    """Patient No O2 start."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        df: Optional[pd.DataFrame] = None,  # timestamp already determined
        index: int = 1,
        label: str = "no_oxygen_start",
        start: bool = True,
        state: State = State.No_O2,
        timestamp_col: Optional[str] = None,  # timestamp already determined
        **kwargs,
    ) -> None:
        super(NoO2StartEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            value=pd.to_datetime(timestamp),  # timestamp already determined
            **kwargs,
        )


class NoO2EndEvent(Event):
    """Patient No O2 end."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        df: Optional[pd.DataFrame] = None,  # timestamp already determined
        end: bool = True,
        index: int = 2,
        label: str = "no_oxygen_end",
        state: State = State.No_O2,
        timestamp_col: Optional[str] = None,  # timestamp already determined
        **kwargs,
    ) -> None:
        super(NoO2EndEvent, self).__init__(
            dataframe=df,
            end=end,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            value=pd.to_datetime(timestamp),  # timestamp already determined
        )


class O2StartEvent(Event):
    """Patient oxygen therapy start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = rename_helper("Ossigenoterapia_Inizio"),
        index: int = 3,
        label: str = "oxygen_therapy_start",
        start: bool = True,
        state: State = State.O2,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(O2StartEvent, self).__init__(
            dataframe=df,
            filter_col=filter_col,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class O2EndEvent(Event):
    """Patient oxygen therapy end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        filter_col: str = rename_helper("Ossigenoterapia_Fine"),
        index: int = 4,
        label: str = "oxygen_therapy_end",
        state: State = State.O2,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(O2EndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class HFNOStartEvent(Event):
    """Patient high flow nasal oxygen therapy start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = rename_helper("HFNO_Inizio"),
        index: int = 5,
        label: str = "hfno_start",
        start: bool = True,
        state: State = State.HFNO,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(HFNOStartEvent, self).__init__(
            dataframe=df,
            filter_col=filter_col,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class HFNOEndEvent(Event):
    """Patient high flow nasal oxygen therapy end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        filter_col: str = rename_helper("HFNO_Fine"),
        index: int = 6,
        label: str = "hfno_end",
        state: State = State.HFNO,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(HFNOEndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class NIVStartEvent(Event):
    """Patient non-invasive ventilation start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = rename_helper("NIV_Inizio"),
        index: int = 7,
        label: str = "niv_start",
        start: bool = True,
        state: State = State.NIV,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(NIVStartEvent, self).__init__(
            dataframe=df,
            filter_col=filter_col,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class NIVEndEvent(Event):
    """Patient non-invasive ventilation end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        filter_col: str = rename_helper("NIV_Fine"),
        index: int = 8,
        label: str = "niv_end",
        state: State = State.NIV,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(NIVEndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class IntubationStartEvent(Event):
    """Patient intubation start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = rename_helper("Intubazione_Inizio"),
        index: int = 9,
        label: str = "intubation_start",
        start: bool = True,
        state: State = State.Intubated,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(IntubationStartEvent, self).__init__(
            dataframe=df,
            filter_col=filter_col,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class IntubationEndEvent(Event):
    """Patient intubation end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        filter_col: str = rename_helper("Intubazione_Fine"),
        index: int = 10,
        label: str = "intubation_end",
        state: State = State.Intubated,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(IntubationEndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostNIVStartEvent(Event):
    """Patient post-intubation non-invasive ventilation start."""

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 11,
        label: str = "post_niv_start",
        start: bool = True,
        state: State = State.NIV,
        timestamp_col: str = rename_helper("NIV_Post_Inizio_Data"),
        **kwargs,
    ) -> None:
        super(PostNIVStartEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostNIVEndEvent(Event):
    """Patient post-intubation non-invasive ventilation end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        index: int = 12,
        label: str = "post_niv_end",
        state: State = State.NIV,
        timestamp_col: str = rename_helper("NIV_Post_Fine_Data"),
        **kwargs,
    ) -> None:
        super(PostNIVEndEvent, self).__init__(
            dataframe=df,
            end=end,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostHFNOStartEvent(Event):
    """Patient post-intubation high flow nasal oxygen therapy start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = rename_helper("HFNO_Post_Inizio"),
        index: int = 13,
        label: str = "post_hfno_start",
        start: bool = True,
        state: State = State.HFNO,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(PostHFNOStartEvent, self).__init__(
            dataframe=df,
            filter_col=filter_col,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostHFNOEndEvent(Event):
    """Patient post-intubation high flow nasal oxygen therapy end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        filter_col: str = rename_helper("HFNO_Post_Fine"),
        index: int = 14,
        label: str = "post_hfno_end",
        state: State = State.HFNO,
        timestamp_col: str = rename_helper("DataRef"),
        **kwargs,
    ) -> None:
        super(PostHFNOEndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostO2StartEvent(Event):
    """Patient post-intubation oxygen therapy start."""

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 15,
        label: str = "post_oxygen_therapy_start",
        start: bool = True,
        state: State = State.O2,
        timestamp_col: str = rename_helper("Ossigenoterapia_Post_Inizio_Data"),
        **kwargs,
    ) -> None:
        super(PostO2StartEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostO2EndEvent(Event):
    """Patient post-intubation oxygen therapy end."""

    def __init__(
        self,
        df: pd.DataFrame,
        end: bool = True,
        index: int = 16,
        label: str = "post_oxygen_therapy_end",
        state: State = State.O2,
        timestamp_col: str = rename_helper("Ossigenoterapia_Post_Fine_Data"),
        **kwargs,
    ) -> None:
        super(PostO2EndEvent, self).__init__(
            dataframe=df,
            end=end,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            **kwargs,
        )


class PostNoO2StartEvent(Event):
    """Patient post-intubation No O2 start."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        df: Optional[pd.DataFrame] = None,  # timestamp already determined
        index: int = 17,
        label: str = "post_no_oxygen_start",
        start: bool = True,
        state: State = State.No_O2,
        timestamp_col: Optional[str] = None,  # timestamp already determined
        **kwargs,
    ) -> None:
        super(PostNoO2StartEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            state=state,
            timestamp_col=timestamp_col,
            value=pd.to_datetime(timestamp),  # timestamp already determined
        )


class PostNoO2EndEvent(Event):
    """Patient post-intubation No O2 end."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        df: Optional[pd.DataFrame] = None,  # timestamp already determined
        end: bool = True,
        index: int = 18,
        label: str = "post_no_oxygen_end",
        state: State = State.No_O2,
        timestamp_col: Optional[str] = None,  # timestamp already determined
        **kwargs,
    ) -> None:
        super(PostNoO2EndEvent, self).__init__(
            dataframe=df,
            end=end,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestamp_col,
            value=pd.to_datetime(timestamp),  # timestamp already determined
        )


class ReleaseEvent(Event):
    """Patient discharge from the hospital."""

    @property
    def reason(self) -> State:
        """Return release reason, i.e. state."""
        if self._reason is None:  # type: ignore
            if self.value > datetime.now():
                raise ValueError("Patient still in charge")
            reasons = set(
                (
                    reason
                    for col_name, series in _select_df_cols(
                        self._dataframe, rename_helper(("ModalitaDimissione",))
                    ).items()
                    for reason in series.unique()
                    if not pd.isna(reason)
                )
            )

            # death wins over any possible other reason
            for r in reasons:
                if "deceduto" in r.lower():
                    self._reason = State.Deceased
                    return self._reason

            if (
                self._last_reason_between(reasons).lower()
                == ORDINARILY_HOME_DISCHARGED.lower()
            ):
                self._reason = State.Discharged
            else:
                self._reason = State.Transferred
        return self._reason

    @property
    def state(self) -> State:
        """Alias for reason property.

        Could raise ValueError("Patient still in charge")

        Overrides parent property to avoid NotImplementedError
        """
        return self.reason

    @property
    def value(self) -> pd.Timestamp:  # type: ignore
        """Return timestamp of the event."""
        if pd.isna(super().value):
            # patient has not yet been released; let's use a day in
            # the future
            tomorrow = datetime.now() + timedelta(days=+1)
            return pd.to_datetime(tomorrow.date())
        return super().value

    @value.setter
    def value(self, new_value: Union[str, pd.Timestamp]) -> None:
        super().value(new_value)

    def _last_reason_between(self, reasons: Iterable[str]) -> str:
        """Return which discharge reason is the last one (between multiple)

        Explore the filtered_df looking for rows containing only a date
        (or multiple copies of it) and only a reason (or multiple
        copies of it); then between the collected {date: reason}
        return the reason of the maximum date.
        """
        if len(reasons) == 1:
            return reasons.pop()

        saved_log_level = logging.getLogger().getEffectiveLevel()

        # let's abuse of the existing HospitalJourney class just to
        # get, through its property, the id of the patient having
        # several discharge reasons in order to make the logging more
        # meaningful
        patient_id = HospitalJourney(
            patient_df=self._dataframe,
            log_level=logging.CRITICAL,
        ).patient_id

        logging.getLogger().setLevel(saved_log_level)  # restore log level

        logging.warning(
            "".join(
                (
                    f" Patient '{patient_id}' has several discharge reasons: ",
                    f"\n{' ' * 10}" if EXECUTING_IN_JUPYTER_KERNEL else "",
                    f"\n{' ' * 10}".join(
                        (
                            str("- " + repr(reason)[:66] + " ...'").replace(
                                "' ...'", "'"
                            )
                            for reason in sorted(reasons)
                        )
                    )
                    if EXECUTING_IN_JUPYTER_KERNEL
                    else repr(sorted(reasons))[1:-1] + ".",
                )
            )
        )

        filtered_df = _select_df_cols(
            self._dataframe,
            rename_helper(
                (
                    "DataDimissione",
                    "ModalitaDimissione" 
                )
            ),
        )
        data = dict()
        for row in filtered_df.itertuples(index=False, name=None):
            possible_reason = None
            possible_date = None
            for cell in row:
                if pd.isna(cell):
                    continue
                if cell in reasons:
                    if possible_reason in (None, cell):
                        possible_reason = cell
                    else:  # got multiple reasons
                        possible_reason = None
                        break
                else:  # cell contains a date
                    assert datetime.now().year <= 2029, str(
                        "Please fix all tests using string '/202' to "
                        "determine dayfirst/yearfirst boolean values "
                        "to help Pandas parsing dates"
                    )
                    cell = pd.to_datetime(
                        cell,
                        dayfirst="/202" in repr(cell),  # 21st century
                        # e.g. '25/04/2020 12:00:00'
                    )
                    if possible_date in (None, cell):
                        possible_date = cell
                    else:  # got multiple dates
                        possible_date = None
                        break
            if possible_reason is not None and possible_date is not None:
                if (
                    possible_date in data
                    and data[possible_date] != possible_reason
                    and True  # make black auto-formatting prettier
                ):
                    # misleading data containing contraddictions,
                    # let's trigger the case: "no discharge reason found"
                    data = dict()
                    break
                data[possible_date] = possible_reason
        if not data:
            logging.critical(
                f" Patient '{patient_id}' last discharge reason is"
                " absent or not easy to guess (due to contraddictions"
                f" in the input data); '{str(State.Transferred)}'"
                " will be probably chosen by default."
            )
            return "no discharge reason found"

        last_discharge_date = max(data.keys())
        assert last_discharge_date == self.value, str(
            f"Last discharge date ({repr(last_discharge_date)}) found "
            f"in ReleaseEvent._last_reason_between() for "
            f"patient '{patient_id}' differs from the one in the "
            f"@property ReleaseEvent.value ({repr(self.value)})\t"
            "THAT IS DEFINITELY SOMETHING CRITICAL"
        )
        logging.warning(
            "".join(
                (
                    f" Patient '{patient_id}' last discharge date is ",
                    f"{last_discharge_date}, with reason: ",
                    f"\n{' ' * 10}" if EXECUTING_IN_JUPYTER_KERNEL else "",
                    f"* {repr(data[last_discharge_date])[:66]} ...'".replace(
                        "' ...'", "'"
                    )
                    if EXECUTING_IN_JUPYTER_KERNEL
                    else f"* {repr(data[last_discharge_date])}",
                )
            )
        )
        return data[last_discharge_date]

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 19,
        label: str = "release",
        start: bool = True,
        start_callable: Callable[..., Any] = pd.DataFrame.max,
        state: Optional[State] = None,  # release reason is not yet known
        timestamp_col: str = rename_helper("DataDimissione"),
        **kwargs,
    ) -> None:
        super(ReleaseEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            start_callable=start_callable,
            state=state,
            timestamp_col=timestamp_col,
        )
        self._reason = None  # type: ignore

    def __repr__(self) -> str:
        if len(f"{self.label}:") >= self.text_pad:
            raise ValueError(
                str(f"Please increase max label size " f"({self.text_pad})")
            )
        if self.value > datetime.now():
            return "".join(
                (  # make black auto-formatting prettier
                    f"{self.label}:".ljust(self.text_pad),
                    "Timestamp('in the future')",
                )
            )
        return super().__repr__()

    def day_offset(self, days: int) -> pd.Timestamp:
        """Return timestamp of the date of the event plus (or minus) offset."""
        if self.value > datetime.now():
            # patient is still in charge; let's cheat a bit to make
            # the rest of the logic correct
            days += 1
        return pd.to_datetime((self.value + timedelta(days=days)).date())


class HospitalJourney(object):
    """Patient journey from admission to the hospital untill the release."""

    @property
    def beginning(self) -> Event:
        """Return first event of the journey."""
        return self._journey[0]

    @property
    def ending(self) -> Event:
        """Return last event of the journey."""
        return self._journey[-1]

    @property
    def unnatural_FiO2_events(self) -> Iterator[Event]:
        """Return an iterator over the events with FiO2 > 21%

        FiO2 means https://en.wikipedia.org/wiki/Fraction_of_inspired_oxygen

        The returned iterator iterates over the events which in theory
        suppose a FiO2 greater than the natural one (i.e. 21%):
            [O2|HFNO|NIV|Intubation|PostNIV|PostO2][Start|End]Event
        """
        for event in self:
            if not isinstance(event, ReleaseEvent) and event.state in (
                State.O2,
                State.HFNO,
                State.NIV,
                State.Intubated,
            ):
                yield event

    @property
    def patient_id(self) -> int:
        """Return patient identifier."""
        if self._patient_id is None:
            possible_ids = list(
                set(int(i) for i in self._patient_df["IdPatient"].unique())
            )
            assert len(possible_ids) == 1, str(
                "Could not determine patient_id; "
                f"several IDs found: {repr(possible_ids)[1:-1]}."
            )
            self._patient_id: Optional[int] = possible_ids[0]
        return self._patient_id

    def __init__(
        self,
        patient_df: pd.DataFrame,
        log_level: int = LOGGING_LEVEL,
    ) -> None:
        """Populate patient journey from its Pandas DataFrame."""
        self._patient_df = patient_df
        self._patient_id = None
        if not EXECUTING_IN_JUPYTER_KERNEL:
            logging.basicConfig(
                filename=LOGGING_STREAM,
                format=LOGGING_FORMAT,
                level=log_level,  # LOGGING_LEVEL,
                style=LOGGING_STYLE,
            )
        else:  # logging should already be configured
            pass
            )
        self._journey = [
            AdmissionEvent(patient_df),
            NoO2StartEvent(pd.NaT),  # not yet known
            NoO2EndEvent(pd.NaT),  # not yet known
            O2StartEvent(patient_df),
            O2EndEvent(patient_df),
            HFNOStartEvent(patient_df),
            HFNOEndEvent(patient_df),
            NIVStartEvent(patient_df),
            NIVEndEvent(patient_df),
            IntubationStartEvent(patient_df),
            IntubationEndEvent(patient_df),
            PostNIVStartEvent(patient_df),
            PostNIVEndEvent(patient_df),
            PostHFNOStartEvent(patient_df),
            PostHFNOEndEvent(patient_df),
            PostO2StartEvent(patient_df),
            PostO2EndEvent(patient_df),
            PostNoO2StartEvent(pd.NaT),  # not yet known
            PostNoO2EndEvent(pd.NaT),  # not yet known
            ReleaseEvent(patient_df),
        ]
        for i, event in enumerate(self._journey):
            assert i == event.index, str(
                f"Please fix index of {type(event)} in its __init__(); "
                "expected '{i}' but got '{index}' instead."
            )

    def __iter__(self) -> Iterator[Event]:
        """Make hospital journey iterable."""
        return iter(self._journey)

    def next_event(self, ev: Event) -> Event:
        """Return next event in the hospital journey."""
        if ev not in self._journey:
            raise ValueError(  # make black auto-formatting prettier
                f"Could not find event '{ev.label}' in patient journey."
            )
        if ev == self.ending:
            raise StopIteration("No more events")
        return self._journey[ev.index + 1]

    def prev_event(self, ev: Event) -> Event:
        """Return previous event in the hospital journey."""
        if ev not in self._journey:
            raise ValueError(  # make black auto-formatting prettier
                f"Could not find event '{ev.label}' in patient journey."
            )
        if ev == self.beginning:
            raise StopIteration("No more events")
        return self._journey[ev.index - 1]

    def next_valid_event(self, ev: Event) -> Event:
        """Return next not NaT event in the hospital journey."""
        while True:
            next_ev = self.next_event(ev)  # can raise a StopIteration
            if not next_ev.is_nat:
                return next_ev
            ev = next_ev

    def prev_valid_event(self, ev: Event) -> Event:
        """Return previous not NaT event in the hospital journey."""
        while True:
            prev_ev = self.prev_event(ev)  # can raise a StopIteration
            if not prev_ev.is_nat:
                return prev_ev
            ev = prev_ev


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__ in ("analisi.src.YAstarMM.model", "YAstarMM.model", "model"),
        "State" in globals(),
        "AdmissionEvent" in globals(),
        "ReleaseEvent" in globals(),
        "HospitalJourney" in globals(),
    )
), "Please update 'Usage' section of module docstring"
