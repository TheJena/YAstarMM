#!/usr/bin/env python3
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
   Preprocess a DataFrame read from an input file.

   Usage:
            from  YAstarMM.preprocessing  import  (
                DumbyDog, Insomnia,
            )

   ( or from within the YAstarMM package )

            from           preprocessing  import  (
                DumbyDog, Insomnia,
            )
"""

from datetime import timedelta
from .model import (
    State,
    Event,
    NoO2StartEvent,
    NoO2EndEvent,
    PostNoO2StartEvent,
    PostNoO2EndEvent,
    HospitalJourney,
)
from sys import version_info
from typing import Iterator, List, Optional, TextIO, Tuple
import logging
import numpy as np
import pandas as pd


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
    def show_statistics(cls, logger_prefix: Optional[str] = None) -> None:
        """Print collected statistics."""
        if logger_prefix is None:
            logger_prefix = str(
                f"[INFO][{__name__}.py][{cls.__name__}"
                f" v{'.'.join(str(v) for v in cls.__version__)}]"
                f"[{'STATISTICS'.center(16)}] "
            )
        print(logger_prefix)  # empty line as separator
        fixed_dates = 0
        for k, v in cls._stats.items():
            if k == "used_dates":
                continue
            if k in (
                "double_hole",
                "ending_no_oxygen",
                "no_o2_patients",
                "starting_no_oxygen",
            ):
                v *= 2
            fixed_dates += v
            print(logger_prefix + f"{k}:".ljust(Event.text_pad) + f"{v:10d}")
            cls._stats[k] = 0  # reset already printed value

        if fixed_dates > 0:  # DumbyDog should never enter here
            print(logger_prefix + " " * Event.text_pad + "-" * 10)  # sum line
            print(
                logger_prefix
                + "Total fixed dates:".ljust(Event.text_pad)
                + f"{fixed_dates:10d}"
            )

        print(
            logger_prefix
            + "Total valid dates used:".ljust(Event.text_pad)
            + f"{cls._stats['used_dates']:10d}"
        )
        cls._stats["used_dates"] = 0  # reset already printed value
        print(logger_prefix)  # empty line as separator

    @property
    def date_range(self) -> pd.Series:
        """Pandas Series covering the range of dates of the patient journey."""
        date_range = set((ev.value for ev in self.journey if not ev.is_nat))
        min_date, max_date = min(date_range), max(date_range)
        return pd.date_range(
            start=min_date, end=max_date, freq="D", normalize=True,
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
            f"[patient{self.journey.patient_id:9d}]"
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
                (hasattr(self, "run"), callable(getattr(self, "run")),)
            ), "Please update the RuntimeWarning in the line after this one :D"
            raise RuntimeWarning("Please call the .run() method")

        self.debug("{self.__name__}.results = [")
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
            keys=["Date", "State"],
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
                    (patient_tmp_df["Date"] >= start_ev.date)
                    & (patient_tmp_df["Date"] <= end_ev.date),
                    ["State"],
                ] = state.value

        # Add also discharge reason
        discharge_ev = self.journey.ending
        try:
            patient_tmp_df.loc[
                (patient_tmp_df["Date"] >= discharge_ev.date)
                & (patient_tmp_df["Date"] < discharge_ev.day_offset(+1)),
                ["State"],
            ] = discharge_ev.reason.value
        except ValueError as e:
            if str(e).lower() == "patient still in charge":
                cut_date = discharge_ev.date  # aka tomorrow
            else:
                raise e
        else:
            # Let's preserve discharge day by cutting the day after it
            cut_date = discharge_ev.day_offset(+1)

        # Cut dataframe to the real date range
        return patient_tmp_df.loc[
            (patient_tmp_df["Date"] >= self.journey.beginning.date)
            & (patient_tmp_df["Date"] <= cut_date)
        ]

    def __init__(
        self, journey: HospitalJourney, log_level: int = logging.WARNING,
    ) -> None:
        """Initialize DumbyDog algorithm."""
        super(DumbyDog, self).__init__()
        logging.basicConfig(
            format="[%(levelname)s][%(filename)s]%(message)s", level=log_level,
        )
        self._journey: HospitalJourney = journey
        self._results: Optional[  # make black auto-formatting prettier
            List[Tuple[pd.Timestamp, pd.Timestamp, State]]
        ] = None
        self.info("")
        for ev in self.journey:
            self.info(repr(ev))
        self.info("")

    def columns_to_wipe(
        self, patient_df_len: int,  # make black auto-formatting prettier
    ) -> Iterator[Tuple[str, pd.Series]]:
        """Return iterator over (columns, wiped_series) to wipe real patient df.
        """
        nan_series = pd.Series([np.nan for _ in range(patient_df_len)])
        for state in State:
            for column_type in ("start_col", "fill_col", "end_col"):
                try:
                    column = getattr(state, column_type)
                except KeyError:
                    continue  # release states do not have these columns
                else:
                    yield (column, nan_series)

    def run(self) -> None:
        """Prepare results to write back to the real patient dataframe."""
        self._stats["used_dates"] += 1  # admission date
        results = list()
        start_day, end_day, prev_state = None, None, None
        for index, row in self.temporary_patient_dataframe.iterrows():
            curr_date, curr_state = row.to_list()
            if any(
                (
                    start_day is None,  # put admission in start_day
                    curr_state != prev_state,  # usual change of state
                )
            ):
                if start_day is not None:
                    prev_state = State(prev_state)  # type: ignore
                    results.append((start_day, end_day, prev_state,))
                    if not pd.isna(curr_state):
                        curr_state = State(curr_state)
                        if curr_state in (  # test exit condition
                            State.Discharged,
                            State.Transferred,
                            State.Deceased,
                        ):
                            results.append((curr_date, curr_date, curr_state,))
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
                    results.append((start_day, end_day, prev_state,))
                    self._stats["used_dates"] += 1  # future release date
        self._results = results

    def debug(self, message: str) -> None:
        """Log debug message."""
        logging.debug(f"{self.logger_prefix} {message}")

    def info(self, message: str) -> None:
        """Log info message."""
        logging.info(f"{self.logger_prefix} {message}")

    def warning(self, message: str) -> None:
        """Log warning message."""
        logging.warning(f"{self.logger_prefix} {message}")


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
        double_hole=0,
        ending_no_oxygen=0,
        no_o2_patients=0,
        spared_end_case=0,
        spared_start_case=0,
        starting_no_oxygen=0,
        swapped_dates=0,
        used_dates=0,  # Amount of used dates by the algorithm
    )
    """Statistics collected globally, over all users."""

    def __init__(
        self, journey: HospitalJourney, log_level: int = logging.WARNING,
    ) -> None:
        """Initialize Insomnia algorithm."""
        super(Insomnia, self).__init__(
            journey=journey, log_level=log_level,
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
            elif all((prev_ev.date <= ev.date, next_next_ev.date >= ev.date,)):
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
                    self.debug(f"filled spared date: {next_ev.label}")
                    next_ev.value = max(next_next_ev.day_offset(-1), ev.value)
                    self.info(repr(next_ev))
                    self._stats["spared_start_case"] += 1
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
                    self.debug(f"filled spared date: {ev.label}")
                    ev.value = min(prev_ev.day_offset(+1), next_ev.value)
                    self.info(repr(ev))
                    self._stats["spared_end_case"] += 1
                    continue

    def fix_missing_initial_no_oxygen_state(self) -> None:
        """Detect and fill missing No O2 state after admission."""
        admission_ev = self.journey.beginning
        assert not admission_ev.is_nat, "Admission date should not be nan"

        # Iterate forward excluding admission, discharge and No O2 dates
        for ev in self.journey.unnatural_FiO2_events:
            if ev.is_nat:
                continue

            if all((ev.is_start, ev.date > admission_ev.date,)):
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
                self._stats["starting_no_oxygen"] += 1
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

            if all((ev.is_end, ev.date < discharge_ev.date,)):
                if ev.day_offset(+1) > discharge_ev.day_offset(-1):
                    break  # no time for No O2 state

                post_no_oxygen_end_ev = self.journey.prev_event(discharge_ev)
                post_no_oxygen_start_ev = self.journey.prev_event(  # force \n
                    post_no_oxygen_end_ev
                )

                assert all(
                    (
                        isinstance(
                            post_no_oxygen_start_ev,
                            PostNoO2StartEvent,  # force pretty auto-formatting
                        ),
                        isinstance(post_no_oxygen_end_ev, PostNoO2EndEvent),
                    )
                ), "Expected PostNoO2[Start|End]Events :("

                self.debug(f"filled missing final {str(State.No_O2)} state")
                post_no_oxygen_start_ev.value = ev.day_offset(+1)
                post_no_oxygen_end_ev.value = discharge_ev.day_offset(-1)
                self._stats["ending_no_oxygen"] += 1
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

            if any((not next_ev.is_nat, prev_ev.is_nat, next_next_ev.is_nat,)):
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

            no_oxygen_start_ev = self.journey.next_event(  # force newline
                self.journey.beginning
            )
            no_oxygen_end_ev = self.journey.next_event(no_oxygen_start_ev)
            post_no_oxygen_end_ev = self.journey.prev_event(  # force newline
                self.journey.ending
            )
            post_no_oxygen_start_ev = self.journey.prev_event(  # force newline
                post_no_oxygen_end_ev
            )

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
                    not no_oxygen_start_ev.is_nat,
                    not no_oxygen_end_ev.is_nat,
                    abs(no_oxygen_start_ev.date - new_start_date)  # force \n
                    <= timedelta(days=1),
                    abs(no_oxygen_end_ev.date - new_end_date)  # force newline
                    <= timedelta(days=1),
                )
            ):
                continue  # double hole coincident with No O2 state
            if all(
                (
                    not post_no_oxygen_start_ev.is_nat,
                    not post_no_oxygen_end_ev.is_nat,
                    abs(post_no_oxygen_start_ev.date - new_start_date)
                    <= timedelta(days=1),
                    abs(post_no_oxygen_end_ev.date - new_end_date)  # force \n
                    <= timedelta(days=1),
                )
            ):
                continue  # double hole coincident with post No O2 state

            self.debug(f"filled double hole: {ev.label}, {next_ev.label}")
            ev.value = new_start_date
            next_ev.value = new_end_date
            self._stats["double_hole"] += 1
            self.info(repr(ev))
            self.info(repr(next_ev))

    def fix_just_admission_discharge_patients(self) -> None:
        """Set No O2 state for patients with only admission and discharge dates.
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
            self._stats["no_o2_patients"] += 1

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


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.preprocessing",
            "YAstarMM.preprocessing",
            "preprocessing",
        ),
        "DumbyDog" in globals(),
        "Insomnia" in globals(),
    )
), "Please update 'Usage' section of module docstring"
