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
   Preprocess a DataFrame containing dates and the relatives states.

   Usage:
            from  YAstarMM.model  import  (
                State, AdmissionEvent, ... , ReleaseEvent, HospitalJourney,
            )

   ( or from within the YAstarMM package )

            from           model  import  (
                State, AdmissionEvent, ... , ReleaseEvent, HospitalJourney,
            )
"""

from datetime import datetime, timedelta
from enum import IntEnum, unique
from sys import version_info
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def new_columns_to_add(
    main_df_len: int,
) -> Iterator[Tuple[str, Union[pd.Categorical, pd.Series]]]:
    """Return iterator over (new columns name, new columns series)."""
    for col in ordered_state_transition_columns():
        if col not in ("ActualState", "ActualState_val"):
            yield (col, pd.Series(data=[np.nan for _ in range(main_df_len)]))
    yield (
        "ActualState",
        pd.Categorical(
            values=[np.nan for i in range(main_df_len)],
            categories=State.names(),
            ordered=True,
        ),
    )
    yield (
        "ActualState_val",
        pd.Categorical(
            values=[np.nan for i in range(main_df_len)],
            categories=State.values(),
            ordered=True,
        ),
    )


def ordered_state_transition_columns() -> Tuple[str, ...]:
    # Order does matter, do not change it please
    return (
        "No_Ossigenoterapia_Inizio",
        "No_Ossigenoterapia",
        "No_Ossigenoterapia_Fine",
        "Ossigenoterapia_Inizio",
        "Ossigenoterapia",
        "Ossigenoterapia_Fine",
        "NIV_Inizio",
        "NIV",
        "NIV_Fine",
        "Intubazione_Inizio",
        "Intubazione",
        "Intubazione_Fine",
        "ActualState",
        "ActualState_val",
    )


@unique
class State(IntEnum):
    """Allowed states of a patient during its journey in the hospital."""

    No_O2 = 0
    """No oxygen therapy"""

    O2 = 1
    """Oxygen therapy"""

    NIV = 2
    """Non-invasive ventilation therapy"""

    Intubated = 3
    """Invasive ventilation technique for patient in intensive care unit"""

    Deceased = 4
    """Release state: patient is dead"""

    Discharged = 5
    """Release state: patient is sent back home"""

    Transferred = 6
    """Release state: patient is moved to another hospital"""

    def __str__(self) -> str:
        return self.name.replace("_", " ")

    @property
    def start_col(self) -> str:
        """Column name in Pandas DataFrame denoting state start."""
        return dict(
            No_O2="No_Ossigenoterapia_Inizio",
            O2="Ossigenoterapia_Inizio",
            NIV="NIV_Inizio",
            Intubated="Intubazione_Inizio",
        )[
            self.name
        ]  # can raise a KeyError for release states

    @property
    def fill_col(self) -> str:
        """Column name in Pandas DataFrame denoting state persistence."""
        return dict(
            No_O2="No_Ossigenoterapia",
            O2="Ossigenoterapia",
            NIV="NIV",
            Intubated="Intubazione",
        )[
            self.name
        ]  # can raise a KeyError for release states

    @property
    def end_col(self) -> str:
        """Column name in Pandas DataFrame denoting state end."""
        return dict(
            No_O2="No_Ossigenoterapia_Fine",
            O2="Ossigenoterapia_Fine",
            NIV="NIV_Fine",
            Intubated="Intubazione_Fine",
        )[
            self.name
        ]  # can raise a KeyError for release states

    @classmethod
    def names(cls) -> List[str]:
        """Return list containing all the state names."""
        return [str(member) for _, member in State.__members__.items()]

    @classmethod
    def values(cls) -> List[int]:
        """Return list containing all the state's integer values."""
        return [int(member.value) for _, member in State.__members__.items()]


class Event(object):
    """General event denoting a state start or end."""

    text_pad = 32

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
            my_series = self.callable(
                self._dataframe[self._dataframe[self.filter_col].notna()][
                    self.timestamp_col
                ]
            )
            if not isinstance(my_series, (pd.Timestamp, type(pd.NaT),)):
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
            if new_value.lower() in ("nat", "nan",):
                self._value = pd.NaT
                return
            if new_value.lower() in ("now", "today",):
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
        timestamp_col: str = "AdmissionTime",
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
        filter_col: str = "Ossigenoterapia_Inizio",
        index: int = 3,
        label: str = "oxygen_therapy_start",
        start: bool = True,
        state: State = State.O2,
        timestamp_col: str = "DataRef",
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
        filter_col: str = "Ossigenoterapia_Fine",
        index: int = 4,
        label: str = "oxygen_therapy_end",
        state: State = State.O2,
        timestam_col: str = "DataRef",
        **kwargs,
    ) -> None:
        super(O2EndEvent, self).__init__(
            dataframe=df,
            end=end,
            filter_col=filter_col,
            index=index,
            label=label,
            state=state,
            timestamp_col=timestam_col,
            **kwargs,
        )


class NIVStartEvent(Event):
    """Patient non-invasive ventilation start."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter_col: str = "NIV_Inizio",
        index: int = 5,
        label: str = "niv_start",
        start: bool = True,
        state: State = State.NIV,
        timestamp_col: str = "DataRef",
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
        filter_col: str = "NIV_Fine",
        index: int = 6,
        label: str = "niv_end",
        state: State = State.NIV,
        timestamp_col: str = "DataRef",
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
        filter_col: str = "Intubazione_Inizio",
        index: int = 7,
        label: str = "intubation_start",
        start: bool = True,
        state: State = State.Intubated,
        timestamp_col: str = "DataRef",
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
        filter_col: str = "Intubazione_Fine",
        index: int = 8,
        label: str = "intubation_end",
        state: State = State.Intubated,
        timestamp_col: str = "DataRef",
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
        index: int = 9,
        label: str = "post_niv_start",
        start: bool = True,
        state: State = State.NIV,
        timestamp_col: str = "NIV_Post_Inizio_Data",
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
        index: int = 10,
        label: str = "post_niv_end",
        state: State = State.NIV,
        timestamp_col: str = "NIV_Post_Fine_Data",
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


class PostO2StartEvent(Event):
    """Patient post-intubation oxygen therapy start."""

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 11,
        label: str = "post_oxygen_therapy_start",
        start: bool = True,
        state: State = State.O2,
        timestamp_col: str = "Ossigenoterapia_Post_Inizio_Data",
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
        index: int = 12,
        label: str = "post_oxygen_therapy_end",
        state: State = State.O2,
        timestamp_col: str = "Ossigenoterapia_Post_Fine_Data",
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
        index: int = 13,
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
        index: int = 14,
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
            for r in self._dataframe["ModalitaDimissione"].unique():
                if not pd.isna(r):
                    if "deceduto" in r.lower():
                        self._reason = State.Deceased
                    elif "dimissione ordinaria al domicilio" in r.lower():
                        self._reason = State.Discharged
                    else:
                        self._reason = State.Transferred
                    break
        return self._reason  # type: ignore

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

    def __init__(
        self,
        df: pd.DataFrame,
        index: int = 15,
        label: str = "release",
        start: bool = True,
        start_callable: Callable[..., Any] = pd.DataFrame.max,
        state: Optional[State] = None,  # release reason is not yet known
        timestam_col: str = "DataDimissione",
        **kwargs,
    ) -> None:
        super(ReleaseEvent, self).__init__(
            dataframe=df,
            index=index,
            label=label,
            start=start,
            start_callable=start_callable,
            state=state,
            timestamp_col=timestam_col,
        )
        self._reason = None  # type: ignore

    def __repr__(self) -> str:
        if len(f"{self.label}:") >= self.text_pad:
            raise ValueError(
                str(f"Please increase max label size " f"({self.text_pad})")
            )
        if self.value > datetime.now():
            return str(
                f"{self.label}:".ljust(self.text_pad)  # force newline
                + "Timestamp('in the future')"
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
            [O2|NIV|Intubation|PostNIV|PostO2][Start|End]Event
        """
        for event in self:
            if not isinstance(event, ReleaseEvent) and event.state in (
                State.O2,
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
        self, patient_df: pd.DataFrame, patient_id: Optional[int] = None
    ) -> None:
        """Populate patient journey from its Pandas DataFrame."""
        self._patient_df = patient_df
        self._patient_id = patient_id
        self._journey = [
            AdmissionEvent(patient_df),
            NoO2StartEvent(pd.NaT),  # not yet known
            NoO2EndEvent(pd.NaT),  # not yet known
            O2StartEvent(patient_df),
            O2EndEvent(patient_df),
            NIVStartEvent(patient_df),
            NIVEndEvent(patient_df),
            IntubationStartEvent(patient_df),
            IntubationEndEvent(patient_df),
            PostNIVStartEvent(patient_df),
            PostNIVEndEvent(patient_df),
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
            raise ValueError(  # force newline
                f"Could not find event '{ev.label}' in patient journey."
            )
        if ev == self.ending:
            raise StopIteration("No more events")
        return self._journey[ev.index + 1]

    def prev_event(self, ev: Event) -> Event:
        """Return previous event in the hospital journey."""
        if ev not in self._journey:
            raise ValueError(  # force newline
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
