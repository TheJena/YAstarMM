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
   Reconstruct wrong or missing data in a DataFrame.

   Usage:
            from  YAstarMM.preprocessing  import  (
                clear_and_refill_state_transition_columns,
            )

   ( or from within the YAstarMM package )

            from          .preprocessing  import  (
                clear_and_refill_state_transition_columns,
            )
"""

from .column_rules import rename_helper
from .constants import (  # without the dot notebook raises ModuleNotFoundError
    EXECUTING_IN_JUPYTER_KERNEL,
    InputOutputErrorQueues,
    LOGGING_LEVEL,
    MIN_PYTHON_VERSION,
    GroupWorkerError,
    GroupWorkerInput,
    GroupWorkerOutput,
    NASTY_SUFFIXES,
)
from .model import (  # without the dot notebook raises ModuleNotFoundError
    Event,
    HospitalJourney,
    NoO2EndEvent,
    NoO2StartEvent,
    PostNoO2EndEvent,
    PostNoO2StartEvent,
    State,
    new_columns_to_add,
)
from .utility import black_magic
from datetime import timedelta
from io import BufferedIOBase
from multiprocessing import cpu_count, Lock, Process, Queue
from sys import version_info
from typing import Iterator, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

_LOGGING_LOCK = Lock()


class DumbyDog(object):
    """Dumb algorithm to deal with human collected time series.

    Since human errors in hand collected time series are difficult to
    fix, let's start with a dumb rule which consider as valid only
    patient's states with both start and end date, and with the start
    date occurring before the end date.
    """

    __name__ = "DumbyDog"
    __version__ = 0, 1, 0

    _stats = dict(used_dates=0)
    """Statistics collected globally, over all users."""

    @classmethod
    def show_statistics(cls, stats=None) -> None:
        """Log collected statistics."""
        if stats is not None:
            cls._stats = stats
        prefix = str(
            f"[{cls.__name__} "
            f"v{'.'.join(str(v) for v in cls.__version__)}]"
            f"[{'STATISTICS'.center(16)}] "
        )
        if EXECUTING_IN_JUPYTER_KERNEL:
            prefix = " "

        saved_log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)
        logging.info(prefix)  # empty line as separator
        fixed_dates = 0
        for k, v in cls._stats.items():
            if k == "used_dates":
                continue
            if k in (
                "filled_double_holes",
                "filled_no_oxygen_ends",
                "filled_patients_without_oxygen_therapy",
                "filled_no_oxygen_starts",
            ):
                v *= 2
            fixed_dates += v
            logging.info(prefix + f"{k}:".ljust(Event.text_pad) + f"{v:10d}")
            cls._stats[k] = 0  # reset already shown value

        if fixed_dates > 0:  # DumbyDog should never enter here
            logging.info(prefix + " " * Event.text_pad + "-" * 10)  # sum line
            logging.info(
                prefix
                + "Total fixed dates:".ljust(Event.text_pad)
                + f"{fixed_dates:10d}"
            )

        logging.info(
            prefix
            + "Total valid dates used:".ljust(Event.text_pad)
            + f"{cls._stats['used_dates']:10d}"
        )
        cls._stats["used_dates"] = 0  # reset already shown value
        logging.info(prefix)  # empty line as separator
        logging.getLogger().setLevel(saved_log_level)

    @property
    def date_range(self) -> pd.Series:
        """Pandas Series covering the range of dates of the patient journey."""
        date_range = set((ev.value for ev in self.journey if not ev.is_nat))
        min_date, max_date = min(date_range), max(date_range)
        return pd.date_range(
            start=min_date,
            end=max_date,
            freq="D",
            normalize=True,
        ).to_series()

    @property
    def journey(self) -> HospitalJourney:
        """Reference to the patient journey."""
        return self._journey

    @property
    def logger_prefix(self) -> str:
        """Return string prefix for logging messages."""
        return str(
            f"[{self.__name__} v{'.'.join(str(v) for v in self.__version__)}]"
            f"[patient {self.journey.patient_id}]"
        )

    @property
    def results(
        self,
    ) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, str, str, str, str, int]]:
        """Retrun an iterator over the patient 'explained' hospital journey.

        Each returned tuple has the form: (start_day, end_day,
            start_col, end_col, fill_col, state_name, state_value)
        """
        if self._results is None:
            assert all(
                (
                    hasattr(self, "run"),
                    callable(getattr(self, "run")),
                )
            ), "Please update the RuntimeWarning in the line after this one :D"
            raise RuntimeWarning("Please call the .run() method")

        self.debug(f"{self.__name__}.results = [")
        for start_day, end_day, state in self._results:
            if start_day > end_day:
                continue  # forbid unordered dates
            self._stats["used_dates"] += 2  # both start and end dates
            try:
                start_col = state.start_col
                end_col = state.end_col
                fill_col = state.fill_col
            except KeyError:
                start_col, end_col, fill_col = None, None, None
            finally:
                ret = (
                    start_day,
                    end_day,
                    start_col,
                    end_col,
                    fill_col,
                    str(state),
                    state.value,
                )
                self.debug(repr(ret) + ",")
                yield ret
        self.debug("]")
        self.flush_logging_queue()

    @property
    def temporary_patient_dataframe(self) -> pd.DataFrame:
        """Create and populate a DataFrame containing the patient journey."""
        patient_tmp_df = pd.concat(
            [
                self.date_range.reset_index(drop=True),
                pd.Series(
                    pd.Categorical(
                        values=[np.nan for i in range(len(self.date_range))],
                        categories=sorted(State, key=lambda stat: stat.value),
                        ordered=True,
                    ),
                ),
            ],
            axis=1,
            keys=list(rename_helper(("Date", "State"))),
        )

        # Populate temporary patient dataframe
        for state in sorted(State, key=lambda state: state.value):
            for start_ev in self.journey:
                if start_ev == self.journey.ending or any(
                    (
                        start_ev.is_nat,
                        not start_ev.is_start,
                        start_ev.state != state,
                        False,  # make black auto-formatting prettier
                    )
                ):
                    continue

                try:
                    end_ev = self.journey.next_event(start_ev)
                except StopIteration:
                    self.warning("At least discharge date should exist")

                if any(
                    (
                        end_ev.is_nat,
                        not end_ev.is_end,
                        end_ev.state != state,
                        False,  # make black auto-formatting prettier
                    )
                ):
                    continue

                patient_tmp_df.loc[
                    (patient_tmp_df[rename_helper("Date")] >= start_ev.date)
                    & (patient_tmp_df[rename_helper("Date")] <= end_ev.date),
                    [rename_helper("State")],
                ] = state.value

        # Add also discharge reason
        discharge_ev = self.journey.ending
        try:
            patient_tmp_df.loc[
                (patient_tmp_df[rename_helper("Date")] >= discharge_ev.date)
                & (
                    patient_tmp_df[rename_helper("Date")]
                    < discharge_ev.day_offset(+1)
                ),
                [rename_helper("State")],
            ] = discharge_ev.reason.value
        except ValueError as e:
            if str(e).lower() == "patient still in charge":
                cut_date = discharge_ev.date  # i.e. tomorrow
            else:
                raise e
        else:
            # Let's preserve discharge day by cutting the day after it
            cut_date = discharge_ev.day_offset(+1)

        # Cut dataframe to the real date range
        return patient_tmp_df.loc[
            (
                patient_tmp_df[rename_helper("Date")]
                >= self.journey.beginning.date
            )
            & (patient_tmp_df[rename_helper("Date")] <= cut_date)
        ]

    def __init__(
        self,
        journey: HospitalJourney,
        log_level: Optional[int] = None,
    ) -> None:
        """Initialize DumbyDog algorithm."""
        super(DumbyDog, self).__init__()
        self._journey: HospitalJourney = journey
        if log_level is not None:
            for hdlr in logging.getLogger().handlers:
                if not isinstance(hdlr, logging.FileHandler):
                    hdlr.setLevel(log_level)

        # let us try to avoid interleaving of logging records about
        # different patients due to parallelism
        self._postponed_logging_queue = list()

        self._results: Optional[  # make black auto-formatting prettier
            List[Tuple[pd.Timestamp, pd.Timestamp, State]]
        ] = None
        self.info("")
        for ev in self.journey:
            self.info(repr(ev))
        self.info("")

    def columns_to_wipe(
        self,
        df_len: int,
    ) -> Iterator[Tuple[str, pd.Series]]:
        """Return iterator over (columns, wiped_series) to wipe real
        patient df.
        """
        nan_series = pd.Series([np.nan for _ in range(df_len)])
        for state in State:
            for nasty_suffix in NASTY_SUFFIXES:
                for column_type in ("start_col", "fill_col", "end_col"):
                    try:
                        column = getattr(state, column_type)
                    except KeyError:
                        continue  # release states do not have these columns
                    else:
                        yield (f"{column}{nasty_suffix}", nan_series)

    def flush_logging_queue(self) -> None:
        global _LOGGING_LOCK
        _LOGGING_LOCK.acquire()
        while self._postponed_logging_queue:
            level, msg = self._postponed_logging_queue.pop(0)
            logging.log(level, msg)
        _LOGGING_LOCK.release()

    def run(self) -> None:
        """Prepare results to write back to the real patient dataframe."""
        self._stats["used_dates"] += 1  # admission date
        results = list()
        start_day, end_day, prev_state = None, None, None
        for (index, row) in self.temporary_patient_dataframe.iterrows():
            curr_date, curr_state = row.to_list()
            if any(
                (
                    start_day is None,  # put admission in start_day
                    curr_state != prev_state,  # usual change of state
                )
            ):
                if start_day is not None:
                    prev_state = State(prev_state)  # type: ignore
                    results.append(
                        (
                            start_day,
                            end_day,
                            prev_state,
                        )
                    )
                    if not pd.isna(curr_state):
                        curr_state = State(curr_state)
                        if curr_state in (  # test exit condition
                            State.Discharged,
                            State.Transferred,
                            State.Deceased,
                        ):
                            results.append(
                                (
                                    curr_date,
                                    curr_date,
                                    curr_state,
                                )
                            )
                            self._stats["used_dates"] += 1  # release date
                            break  # skip else statement of the for loop
                start_day, end_day = curr_date, curr_date
                if pd.isna(curr_state):
                    # emulate a restart from admission
                    start_day, end_day, prev_state = None, None, None
                else:
                    prev_state = State(curr_state)
            else:  # curr_state == prev_state
                end_day = curr_date
        else:  # patient is still in charge
            if prev_state is not None:
                prev_state = State(prev_state)
                if prev_state not in (
                    State.Discharged,
                    State.Transferred,
                    State.Deceased,
                ):
                    results.append(
                        (
                            start_day,
                            end_day,
                            prev_state,
                        )
                    )
                    self._stats["used_dates"] += 1  # future release date
        self._results = results
        self.flush_logging_queue()

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._postponed_logging_queue.append(
            tuple((logging.DEBUG, f"{self.logger_prefix} {message}"))
        )

    def info(self, message: str) -> None:
        """Log info message."""
        self._postponed_logging_queue.append(
            tuple((logging.INFO, f"{self.logger_prefix} {message}"))
        )

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._postponed_logging_queue.append(
            tuple((logging.WARNING, f"{self.logger_prefix} {message}"))
        )


class Insomnia(DumbyDog):
    """Algorithm to improve human collected time series.

    Since humans can be tired, sleepy, in a hurry and so on, this
    algorithm fixes the most common / easy to fix errors happening
    when hand-collecting a time series like an hospital patient
    journey; e.g:

    t_0) admission
    ...
    t_x) state_X_start_event
    t_y) state_X_end_event
    ...
    t_N) discharge

    The redundancy of information (i.e. having both start and end
    timestamps) together with the knowledge of the order in which a
    patient can cross the states, allow this algorithm to:

    - detect and remove wrong timestamps
    - infer and insert missing timestamps
    """

    __name__ = "Insomnia"
    __version__ = 0, 3, 0

    _stats = dict(
        deleted_dates=0,
        swapped_dates=0,
        filled_double_holes=0,
        filled_missing_starts=0,
        filled_missing_ends=0,
        filled_patients_without_oxygen_therapy=0,
        filled_no_oxygen_starts=0,
        filled_no_oxygen_ends=0,
        used_dates=0,  # Amount of used dates by the algorithm
    )
    """Statistics collected globally, over all users."""

    def __init__(
        self,
        journey: HospitalJourney,
        log_level: Optional[int] = None,
    ) -> None:
        """Initialize Insomnia algorithm."""
        super(Insomnia, self).__init__(
            journey=journey,
            log_level=log_level,
        )

    def fix_inverted_dates(self) -> None:
        """Detect couples of dates in reverse order and fix them.

        The fix actually happens if one of the dates is in the range
        (previous event, next event); thus the date in range is kept
        and the other one deleted.

        Otherwise, i.e. both the dates are not in the above range,
        both of them is deleted.
        """
        for ev in self.journey.unnatural_FiO2_events:
            if ev.is_nat or not ev.is_start:
                continue
            try:
                next_ev = self.journey.next_event(ev)
            except StopIteration:
                continue  # no next event available
            if any(
                (
                    next_ev.is_nat,
                    not next_ev.is_end,
                    ev.value <= next_ev.value,  # events are already in order
                    ev.state.fill_col != next_ev.state.fill_col,
                )
            ):
                continue
            # (current event date, next event date) are in reverse
            # order, let's guess if at least one of them is correct

            try:
                prev_ev = self.journey.prev_valid_event(ev)
            except StopIteration:
                self.warning(
                    "This should never happen "
                    "because at least the admission date should exists"
                )
                self.debug(f"deleted two dates: {ev.label}, {next_ev.label}")
                self._stats["deleted_dates"] += 2
                ev.value = "NaT"
                next_ev.value = "NaT"
                continue

            try:
                next_next_ev = self.journey.next_valid_event(next_ev)
            except StopIteration:
                # Since discharge date may not exist let's use today as bound
                next_next_ev = self.journey.ending
                next_next_ev.value = "Today"

            if all(
                (
                    prev_ev.date <= ev.date,
                    prev_ev.date <= next_ev.date,
                    next_next_ev.date >= ev.date,
                    next_next_ev.date >= next_ev.date,
                )
            ):
                # Both dates are in range, but in the wrong order, swap them
                self.debug(f"swapped two dates: {ev.label}, {next_ev.label}")
                self._stats["swapped_dates"] += 2
                ev.value, next_ev.value = next_ev.value, ev.value
                continue
            elif all(
                (
                    prev_ev.date <= ev.date,
                    next_next_ev.date >= ev.date,
                )
            ):
                # Current date is in range, delete the next one
                self.debug(f"deleted one date: {next_ev.label}")
                self._stats["deleted_dates"] += 1
                next_ev.value = "NaT"
                continue
            elif all(
                (
                    prev_ev.date <= next_ev.date,
                    next_next_ev.date >= next_ev.date,
                    True,  # make black auto-formatting prettier
                )
            ):
                # Next date is in range, delete current one
                self.debug(f"deleted one date: {ev.label}")
                self._stats["deleted_dates"] += 1
                ev.value = "NaT"
                continue

    def fix_missing_start_or_end_date(self) -> None:
        """Detect and fix states with only the start (or the end) event date.

        The fix actually happens if the event before (or after) has a
        valid date.
        """
        for ev in self.journey.unnatural_FiO2_events:
            if not ev.is_start:
                continue
            try:
                next_ev = self.journey.next_event(ev)
            except StopIteration:
                continue  # no end event available
            if any(
                (
                    not next_ev.is_end,
                    ev.is_nat == next_ev.is_nat,
                    ev.state.fill_col != next_ev.state.fill_col,
                )
            ):
                continue

            # Only date in (current event date, next event date) is
            # nan and they are respectively the start date and end
            # date of the same state
            if not ev.is_nat:
                # found a start date without its end, let's guess it
                try:
                    next_next_ev = self.journey.next_valid_event(next_ev)
                except StopIteration:
                    continue  # no valid date found after next event date
                else:
                    self.debug(f"filled missing end date: {next_ev.label}")
                    next_ev.value = max(next_next_ev.day_offset(-1), ev.value)
                    self.info(repr(next_ev))
                    self._stats["filled_missing_ends"] += 1
                    continue
            elif not next_ev.is_nat:
                # found an end date without its start, let's guess it
                try:
                    prev_ev = self.journey.prev_valid_event(ev)
                except StopIteration:
                    self.warning(
                        "This should never happen "
                        "because at least the admission date should exists"
                    )
                    continue
                else:
                    self.debug(f"filled missing start date: {ev.label}")
                    ev.value = min(prev_ev.day_offset(+1), next_ev.value)
                    self.info(repr(ev))
                    self._stats["filled_missing_starts"] += 1
                    continue

    def fix_missing_initial_no_oxygen_state(self) -> None:
        """Detect and fill missing No O2 state after admission."""
        admission_ev = self.journey.beginning
        if admission_ev.is_nat:
            patient_id = self.journey.patient_id
            logging.warning(f" Patient '{patient_id}' has nan admission date")
            return

        # Iterate forward excluding admission, discharge and No O2 dates
        for ev in self.journey.unnatural_FiO2_events:
            if ev.is_nat:
                continue

            if all(
                (
                    ev.is_start,
                    ev.date > admission_ev.date,
                )
            ):
                if admission_ev.date > ev.day_offset(-1):
                    break  # no time for No O2 state

                no_oxygen_start_ev = self.journey.next_event(admission_ev)
                no_oxygen_end_ev = self.journey.next_event(no_oxygen_start_ev)

                assert all(
                    (
                        isinstance(no_oxygen_start_ev, NoO2StartEvent),
                        isinstance(no_oxygen_end_ev, NoO2EndEvent),
                    )
                ), "Expected NoO2[Start|End]Events :("

                self.debug(f"filled missing initial {str(State.No_O2)} state")
                no_oxygen_start_ev.value = admission_ev.date
                no_oxygen_end_ev.value = ev.day_offset(-1)
                self._stats["filled_no_oxygen_starts"] += 1
                self.info(repr(no_oxygen_start_ev))
                self.info(repr(no_oxygen_end_ev))
            # Let's fix just the period involving the 1st date after admission
            break

    def fix_missing_final_no_oxygen_state(self) -> None:
        """Detect and fill missing No O2 state before discharge."""
        discharge_ev = self.journey.ending
        assert not discharge_ev.is_nat, str(
            "ReleaseEvent should never have NaT timestamp value; "
            "for still in charge patients a value in the future "
            "should have been returned instead"
        )

        # Iterate backward excluding discharge, admission and No O2 dates
        for ev in reversed(tuple(self.journey.unnatural_FiO2_events)):
            if ev.is_nat:
                continue

            if all(
                (
                    ev.is_end,
                    ev.date < discharge_ev.date,
                )
            ):
                if ev.day_offset(+1) > discharge_ev.day_offset(-1):
                    break  # no time for No O2 state

                post_no_oxygen_end_ev = self.journey.prev_event(discharge_ev)
                post_no_oxygen_start_ev = self.journey.prev_event(
                    post_no_oxygen_end_ev
                )  # make black auto-formatting prettier

                assert all(
                    (
                        isinstance(
                            post_no_oxygen_start_ev, PostNoO2StartEvent
                        ),
                        isinstance(post_no_oxygen_end_ev, PostNoO2EndEvent),
                    )
                ), "Expected PostNoO2[Start|End]Events :("

                self.debug(f"filled missing final {str(State.No_O2)} state")
                post_no_oxygen_start_ev.value = ev.day_offset(+1)
                post_no_oxygen_end_ev.value = discharge_ev.day_offset(-1)
                self._stats["filled_no_oxygen_ends"] += 1
                self.info(repr(post_no_oxygen_start_ev))
                self.info(repr(post_no_oxygen_end_ev))
            # Let's fix just the period involving last date before discharge
            break

    def fix_internal_double_holes(self) -> None:
        """Detect and fill missing states surrounded by valid states."""
        for ev in self.journey.unnatural_FiO2_events:
            if not ev.is_nat:
                continue

            try:
                prev_ev = self.journey.prev_event(ev)
                next_ev = self.journey.next_event(ev)
                next_next_ev = self.journey.next_event(next_ev)
            except StopIteration:
                continue  # not enough surrounding dates

            if any(
                (
                    not next_ev.is_nat,
                    prev_ev.is_nat,
                    next_next_ev.is_nat,
                )
            ):
                continue  # not valid surrounding dates

            # current date and the next one are missing (nan)
            # but previous date and the one after the next one are
            # not missing

            # i.e.:
            # a) date - 1 not missing
            # b) date         missing    <~    current date
            # c) date + 1     missing
            # d) date + 2 not missing

            new_start_date = prev_ev.day_offset(+1)  # b)
            new_end_date = next_next_ev.day_offset(-1)  # c)

            if new_start_date > new_end_date:
                continue  # unfillable hole, no time

            no_oxygen_start_ev = self.journey.next_event(
                self.journey.beginning
            )  # make black auto-formatting prettier
            no_oxygen_end_ev = self.journey.next_event(no_oxygen_start_ev)
            post_no_oxygen_end_ev = self.journey.prev_event(
                self.journey.ending
            )  # make black auto-formatting prettier
            post_no_oxygen_start_ev = self.journey.prev_event(
                post_no_oxygen_end_ev
            )  # make black auto-formatting prettier

            assert all(
                (
                    isinstance(no_oxygen_start_ev, NoO2StartEvent),
                    isinstance(no_oxygen_end_ev, NoO2EndEvent),
                    isinstance(post_no_oxygen_start_ev, PostNoO2StartEvent),
                    isinstance(post_no_oxygen_end_ev, PostNoO2EndEvent),
                )
            ), "Expected [Post|]NoO2[Start|End]Events :("

            if all(
                (
                    not no_oxygen_start_ev.is_nat
                    and abs(
                        no_oxygen_start_ev.date - new_start_date
                    )  # make black auto-formatting prettier
                    <= timedelta(days=1),
                    not no_oxygen_end_ev.is_nat
                    and abs(
                        no_oxygen_end_ev.date - new_end_date
                    )  # make black auto-formatting prettier
                    <= timedelta(days=1),
                )
            ):
                continue  # double hole coincident with No O2 state
            if all(
                (
                    not post_no_oxygen_start_ev.is_nat
                    and abs(post_no_oxygen_start_ev.date - new_start_date)
                    <= timedelta(days=1),
                    not post_no_oxygen_end_ev.is_nat
                    and abs(
                        post_no_oxygen_end_ev.date - new_end_date
                    )  # make black auto-formatting prettier
                    <= timedelta(days=1),
                )
            ):
                continue  # double hole coincident with post No O2 state

            self.debug(f"filled double hole: {ev.label}, {next_ev.label}")
            ev.value = new_start_date
            next_ev.value = new_end_date
            self._stats["filled_double_holes"] += 1
            self.info(repr(ev))
            self.info(repr(next_ev))

    def fix_just_admission_discharge_patients(self) -> None:
        """Set No O2 state for patients with only admission and
        discharge dates.
        """
        admission_ev = self.journey.beginning
        discharge_ev = self.journey.ending
        assert not discharge_ev.is_nat, str(
            "ReleaseEvent should never have NaT timestamp value; "
            "for still in charge patients a value in the future "
            "should have been returned instead"
        )

        if all(
            (
                not admission_ev.is_nat,
                not discharge_ev.is_nat,
                admission_ev.date <= discharge_ev.day_offset(-1),
            )
        ) and all(
            event.is_nat
            for event in self.journey
            if event not in (admission_ev, discharge_ev)
        ):
            no_oxygen_start_ev = self.journey.next_event(admission_ev)
            no_oxygen_end_ev = self.journey.next_event(no_oxygen_start_ev)

            assert all(
                (
                    isinstance(no_oxygen_start_ev, NoO2StartEvent),
                    isinstance(no_oxygen_end_ev, NoO2EndEvent),
                )
            ), "Expected NoO2[Start|End]Events :("

            self.debug(f"filled {str(State.No_O2)} patient")
            no_oxygen_start_ev.value = admission_ev.date
            no_oxygen_end_ev.value = discharge_ev.day_offset(-1)
            self.info(repr(no_oxygen_start_ev))
            self.info(repr(no_oxygen_end_ev))
            self._stats["filled_patients_without_oxygen_therapy"] += 1

    def run(self) -> None:
        """Run Insomnia algorithm and build super()._results property

        Overrides parent class method (DumbyDog.run())
        """
        self.fix_inverted_dates()
        self.fix_missing_start_or_end_date()
        self.fix_missing_initial_no_oxygen_state()
        self.fix_missing_final_no_oxygen_state()
        self.fix_internal_double_holes()
        self.fix_just_admission_discharge_patients()

        super().run()  # build super()._results property


@black_magic
def clear_and_refill_state_transition_columns(
    whole_df,
    patient_key_col,
    log_level=LOGGING_LEVEL,
    show_statistics=True,
    use_dumbydog=False,
    use_insomnia=False,
    *args,
    **kwargs,
):
    if isinstance(whole_df, str):
        whole_df = open(whole_df, "rb")
    if isinstance(whole_df, BufferedIOBase):  # df = open('file.xlsx', 'rb')
        logging.debug(f"Reading whole_df from '{whole_df.name}'")
        whole_df = pd.read_excel(whole_df)
    assert isinstance(whole_df, pd.DataFrame)
    assert bool(use_insomnia) != bool(use_dumbydog), str(
        "Please set either use_insomnia=True or use_dumbydog=True"
    )
    assert patient_key_col in whole_df.columns, str(
        "'patient_key_col' must be a column of the passed whole_df\n"
    ) + repr(sorted(whole_df.columns, key=str.lower))

    saved_log_level = logging.getLogger().getEffectiveLevel()

    # Add new columns (if missing)
    for new_col, new_nan_series in new_columns_to_add(len(whole_df)):
        if new_col not in whole_df.columns:
            whole_df[new_col] = new_nan_series

    if False:  # serial version
        # Sort df by DataRef and fill State Transition cols of each patient
        whole_df = (
            whole_df.sort_values(rename_helper("DataRef"))
            .groupby(patient_key_col)
            .apply(
                run_clear_and_refill_algorithm_over_patient_df,
                patient_key_col,
                log_level=log_level,
                use_dumbydog=use_dumbydog,
                use_insomnia=use_insomnia,
            )
        )
        all_stats = None
    else:  # parallel version
        input_queue, output_queue, error_queue = InputOutputErrorQueues(
            Queue(), Queue(), Queue()
        )
        parallel_workers = [
            Process(
                target=_run_clear_and_refill_algorithm_over_patient_df,
                args=(
                    input_queue,
                    output_queue,
                    error_queue,
                    patient_key_col,
                    log_level,
                    use_dumbydog,
                    use_insomnia,
                ),
            )
            for _ in range(cpu_count())
        ]
        for pw in parallel_workers:
            pw.start()
        # Sort whole_df by DataRef and fill State Transition cols of
        # each patient
        for group_name, patient_df in whole_df.sort_values(
            rename_helper("DataRef")
        ).groupby(patient_key_col):
            input_queue.put(GroupWorkerInput(group_name, patient_df))
        for _ in parallel_workers:
            input_queue.put(None)  # send termination signals

        whole_df, all_stats = list(), dict()
        output_ack, error_ack = len(parallel_workers), len(parallel_workers)
        while output_ack > 0:
            output = output_queue.get()
            if output is None:  # receive ack to termination signal
                output_ack -= 1
                continue
            new_patient_df, patient_stats = output
            whole_df.append(new_patient_df)
            all_stats = {
                k: patient_stats.get(k, 0) + all_stats.get(k, 0)
                for k in set(patient_stats.keys()).union(set(all_stats.keys()))
            }
        while error_ack > 0:
            error = error_queue.get()
            if error is None:  # receive ack to termination signal
                error_ack -= 1
                continue
            logging.warning(
                "While preprocessing patients dataframes; "
                f" worker {repr(error.group_name)} got the "
                f"follwing exception: {str(error.exception)}"
            )
        logging.debug(
            "".join(
                (
                    "Waiting for all preprocessing workers to join ",
                    f"({len(parallel_workers)})",
                )
            )
        )
        for pw in parallel_workers:
            pw.join()
        logging.debug(
            f"All ({len(parallel_workers)}) preprocessing workers joined"
        )
        whole_df = pd.concat(whole_df, join="outer", sort=True)

    if show_statistics and use_insomnia:
        Insomnia.show_statistics(all_stats)
    elif show_statistics and use_dumbydog:
        DumbyDog.show_statistics(all_stats)

    logging.getLogger().setLevel(saved_log_level)  # restore log level

    return whole_df


def _run_clear_and_refill_algorithm_over_patient_df(
    input_queue,
    output_queue,
    error_queue,
    patient_key_col,
    log_level,
    use_dumbydog,
    use_insomnia,
):
    worker_results, all_stats = list(), dict()
    while True:
        group_worker_input = input_queue.get()
        if group_worker_input is None:
            break

        group_name, old_patient_df = group_worker_input
        try:
            (
                new_patient_df,
                _stats,
            ) = run_clear_and_refill_algorithm_over_patient_df(
                old_patient_df,
                patient_key_col,
                log_level,
                use_dumbydog,
                use_insomnia,
            )
        except Exception as e:
            error_queue.put(GroupWorkerError(group_name, e))
            worker_results.append(old_patient_df)
        else:
            worker_results.append(new_patient_df)
            all_stats = {
                k: _stats.get(k, 0) + all_stats.get(k, 0)
                for k in set(_stats.keys()).union(set(all_stats.keys()))
            }

    output_queue.put(
        GroupWorkerOutput(
            pd.concat(worker_results, join="outer", sort=True),
            all_stats,
        )
    )
    error_queue.put(None)
    output_queue.put(None)


def run_clear_and_refill_algorithm_over_patient_df(
    patient_df,
    patient_key_col,
    log_level=LOGGING_LEVEL,
    use_dumbydog=False,
    use_insomnia=False,
):
    assert bool(use_insomnia) != bool(use_dumbydog), str(
        "Please set either use_insomnia=True or use_dumbydog=True"
    )

    journey = HospitalJourney(patient_df, patient_key_col, log_level=log_level)

    if use_insomnia:
        algorithm = Insomnia(journey, log_level=log_level)
    elif use_dumbydog:
        algorithm = DumbyDog(journey, log_level=log_level)
    else:
        raise ValueError(  # make black auto-formatting prettier
            "Please set either use_insomnia=True or use_dumbydog=True"
        )

    algorithm.run()

    # algorithm.results now contains all the meaningful pieces of
    # information; we can safely wipe original patient's events dates

    for col, nan_series in algorithm.columns_to_wipe(df_len=len(patient_df)):
        if col in patient_df.columns:
            patient_df[col] = nan_series
        elif all((not col.endswith(s) for s in NASTY_SUFFIXES)):
            print(
                f"Could not find column '{col}' in "
                f"patient '{journey.patient_id}' dataframe"
            )

    for (
        start_day,
        end_day,
        start_col,
        end_col,
        fill_col,
        state_name,
        state_value,
    ) in algorithm.results:
        if start_col is not None:
            patient_df.loc[
                patient_df[rename_helper("DataRef")] == start_day, [start_col]
            ] = True
        if end_col is not None:
            patient_df.loc[
                patient_df[rename_helper("DataRef")] == end_day, [end_col]
            ] = True
        if fill_col is not None:
            patient_df.loc[
                (patient_df[rename_helper("DataRef")] >= start_day)
                & (patient_df[rename_helper("DataRef")] <= end_day),
                [fill_col],
            ] = True
        if state_name in State.names():
            patient_df.loc[
                (patient_df[rename_helper("DataRef")] >= start_day)
                & (patient_df[rename_helper("DataRef")] <= end_day),
                [rename_helper("ActualState")],
            ] = state_name
        if state_value in State.values():
            patient_df.loc[
                (patient_df[rename_helper("DataRef")] >= start_day)
                & (patient_df[rename_helper("DataRef")] <= end_day),
                [rename_helper("ActualState_val")],
            ] = state_value
    algorithm.flush_logging_queue()
    return (patient_df, algorithm._stats)


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.preprocessing",
    "YAstarMM.preprocessing",
    "preprocessing",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
