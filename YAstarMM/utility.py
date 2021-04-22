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
   Miscellaneous functions without external dependencies.

   Usage:
            from  YAstarMM.utility  import  (
                debug_start_timer, debug_stop_timer, initialize_logging,
            )

   ( or from within the YAstarMM package )

            from          .utility  import  (
                debug_start_timer, debug_stop_timer, initialize_logging,
            )
"""

from datetime import datetime, timedelta
from functools import lru_cache
from logging import (
    basicConfig,
    debug,
    DEBUG,
    Formatter,
    getLogger,
    INFO,
    info,
    StreamHandler,
    WARNING,
    warning,
)
from hashlib import blake2b
from os import mkdir
from os.path import abspath, expanduser, isdir, isfile, join as join_path
from multiprocessing import Lock
from psutil import virtual_memory, swap_memory
from sys import version_info
from tempfile import mkdtemp, NamedTemporaryFile
from time import time
import gzip
import numpy as np
import pandas as pd
import pickle
import re

_CACHE_DIR = None  # where do you prefer to save @black_magic ?
_DISABLE_BLACK_MAGIC_GLOBALLY = False

_CACHE_LOCK, _PERF_LOCK = Lock(), Lock()

# https://en.wikipedia.org/w/index.php?title=Bissextile_year&redirect=yes
AVERAGE_DAYS_PER_YEAR = 365 + 1 / 4 - 1 / 100 + 1 / 400

EXTRACTION_REGEXP = re.compile(
    r"(Estrazione|Extraction)"
    r"[ _-]*"  # separator
    r"(?P<year>2[01]\d\d)"  # valid until 2199
    r"[ _-]*"  # separator
    r"(?P<month>0[1-9]|1[012])"
    r"[ _-]*"  # separator
    r"(?P<day>[012][1-9]|30|31)"
    r".*",  # whatever
    re.IGNORECASE,
)
assert datetime.today().year < 2200, "Please fix the above regular expression"


def _hex_date_to_timestamp(
    hex_str,
    year_offset=0,
    time_offset=timedelta(seconds=0),
    drop_first_digits=0,
):
    assert isinstance(hex_str, str)
    while drop_first_digits > 0:
        d, hex_str = hex_str[0], hex_str[1:]
        assert d == "0", f"digits to drop should be zeros, got '{d}' instead"
        drop_first_digits -= 1
    assert len(hex_str) == 5
    return pd.to_datetime(
        datetime(
            year=year_offset + int(hex_str[:2], base=16),
            month=int(hex_str[2:-2], base=16),
            day=int(hex_str[-2:], base=16),
        )
        + time_offset
    )


def _timestamp_to_hex_date(date, year_offset=0, desired_length=5):
    date = pd.to_datetime(date)
    ret = (  # X means hexadecimal notation
        f"{date.year-year_offset:X}".rjust(desired_length - 3, "0")
        + f"{date.month:X}".rjust(1, "0")
        + f"{date.day:X}".rjust(2, "0")
    )
    assert len(ret) == desired_length, f"len({repr(ret)}) != {desired_length}"
    return ret


def black_magic(fun):
    """Like @lru_cache but persistent thank to compressed pickle files

    https://pythonbasics.org/decorators/#Real-world-examples
    """
    assert fun.__name__ != "wrapper", str(
        "Please use @black_magic before any other decorator"
    )

    global _CACHE_DIR, _CACHE_LOCK, _DISABLE_BLACK_MAGIC_GLOBALLY
    if _DISABLE_BLACK_MAGIC_GLOBALLY:
        return fun

    _CACHE_LOCK.acquire()
    if _CACHE_DIR is None:
        if not isdir(join_path(expanduser("~"), "RAMDISK")):
            _CACHE_DIR = mkdtemp(
                prefix=datetime.now().strftime("%Y_%m_%d__%H_%M_%S__"),
                suffix="__black_magic_decorator_CACHE",
            )
        else:
            _CACHE_DIR = join_path(expanduser("~"), "RAMDISK", "BLACK_MAGIC")
            if not isdir(_CACHE_DIR):
                mkdir(_CACHE_DIR)
        info(
            "@black_magic decorator is going to store some cache-files here:\n"
            f"\t{abspath(_CACHE_DIR)}"
        )
    _CACHE_LOCK.release()

    def wrapper(*args, **kwargs):
        fun_cache_dir = join_path(_CACHE_DIR, fun.__name__)
        if not isdir(fun_cache_dir):
            mkdir(fun_cache_dir)
        n = 4096  # consider only first N bytes of object representation
        if fun.__name__ == "merge_sheets":
            hashed_input = "CHECKPOINT_CHARLIE"
        else:
            hashed_input = blake2b(
                str.encode("".join((repr(args)[:n], repr(kwargs)[:n]))),
                digest_size=16,
            ).hexdigest()
        cache_file = join_path(fun_cache_dir, f"{hashed_input}.pkl.gz")
        if not isfile(cache_file) or not kwargs.get("use_black_magic", True):
            debug(
                f"Cache file {cache_file} not found; "
                f"calling function '{fun.__name__}'"
            )
            t0 = time()
            ret = fun(*args, **kwargs)
            msg = str(
                f"Execution of function '{fun.__name__}' ".ljust(80)
                + f"took {time() - t0:9.3f} seconds.\n"
            )
            with open(join_path(_CACHE_DIR, "performances.txt"), "a") as f:
                f.write(msg)
            with gzip.open(cache_file, "wb") as g:
                g.write(pickle.dumps(ret))
            debug(f"Cached result of '{fun.__name__}' in '{cache_file}'")
        else:
            with gzip.open(cache_file, "rb") as f:
                info(
                    f"\nLoading cached result of function '{fun.__name__}' "
                    f"from:\n\t{cache_file}"
                )
                ret = pickle.loads(f.read())
                info("Please pass 'use_black_magic=False' to avoid caching\n")
        return ret

    return wrapper


def debug_start_timer(event_name):
    global _PERF_LOCK, _PERF_TIMER

    t0 = time()
    _PERF_LOCK.acquire()
    if "_PERF_TIMER" not in globals():
        _PERF_TIMER = dict()
    _PERF_TIMER[event_name] = t0
    _PERF_LOCK.release()

    debug(f"Started execution of fragment labeled '{event_name}'")


def debug_stop_timer(event_name):
    global _PERF_LOCK, _PERF_TIMER

    t1 = time()
    assert "debug_start_timer" in globals(), str(
        "Please update the next assertion message"
    )

    _PERF_LOCK.acquire()
    assert event_name in _PERF_TIMER, str(
        f"Did you call: debug_start_timer({repr(event_name)}) ?"
    )
    t0 = _PERF_TIMER[event_name]
    _PERF_LOCK.release()

    debug(
        "".join(
            (
                f"Execution of fragment '{event_name}' ".ljust(80),
                f"took {t1 - t0:9.3f} seconds.",
            )
        )
    )


def extraction_date(filename):
    m = EXTRACTION_REGEXP.match(filename)
    if m is not None:
        return datetime(
            year=int(m.group("year")),
            month=int(m.group("month")),
            day=int(m.group("day")),
        )
    raise ValueError(
        "Extraction input files must have a name like:"
        "\n\t- Estrazione_20001231_whatever.xlsx"
        "\n\t- Estrazione_20001231_whatever.xlsx"
        "\n\t- Estrazione_2000_12_31_whatever.xlsx"
        "\n\t- Extraction_2000_12_31_whatever.xlsx"
        f"\n\n(Please fix {filename} accordingly)"
    )


def initialize_logging(level=INFO, debug_mode=False):
    logfile = NamedTemporaryFile(
        prefix=datetime.now().strftime("%Y_%m_%d__%H_%M_%S__"),
        suffix="__00_data_loading_and_manipulation__debugging_log.txt",
        delete=False,
    )
    basicConfig(
        filename=logfile.name,
        format="\t".join(
            (
                "[{levelname: ^9s}| {module}+L{lineno} | PID-{process}]",
                "{message}",
            )
        ),
        style="{",
        level=level if not debug_mode else DEBUG,
    )
    root_logger = getLogger()
    stderr_handle = StreamHandler()
    stderr_handle.setLevel(INFO if not debug_mode else DEBUG)
    stderr_handle.setFormatter(Formatter("{levelname}: {message}", style="{"))
    root_logger.addHandler(stderr_handle)

    numexpr_logger = getLogger("numexpr")
    numexpr_logger.setLevel(WARNING)

    info("Temporary file with debugging log will be available here:")
    info(f"{' ' * 4}{logfile.name}")
    debug(f"{'~' * 120}")
    debug("")
    debug("Logging initialized and temporary file created")
    debug("")


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
        # fake a birthday in the future to distinguish patients once sorted
        birth_date = datetime(
            year=1900 + int("F1", base=16),  # 2141
            month=int("1", base=16),  # January
            day=int("1F", base=16),  # 31th
        )
    ret = (
        _timestamp_to_hex_date(
            admission_date.date(), year_offset=2000, desired_length=5
        )
        + _timestamp_to_hex_date(
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


def revert_new_key_col_value(new_key_col):
    assert isinstance(new_key_col, str) and len(new_key_col) == 24
    discharge_timedelta_sec = int(new_key_col[-3:], base=16) * 60
    admission_timedelta_sec = int(new_key_col[-1 - 3 - 3 : -3], base=16) * 60
    admission_date = _hex_date_to_timestamp(
        new_key_col[:5],
        year_offset=2000,
        time_offset=timedelta(seconds=admission_timedelta_sec),
    )
    birth_date = _hex_date_to_timestamp(
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


def row_selector(default_bool=True, index=None, size=None):
    assert (index is not None or size is not None) and (
        index is None or size is None
    ), str("Please provide either 'index' or 'size' argument")
    size = index.size if size is None else size
    return pd.Series([default_bool for _ in range(size)], index=index)


def swap_month_and_day(date):
    ret = datetime(
        year=date.year,
        month=date.day,  # month <~ day
        day=date.month,  # day   <~ month
        hour=date.hour,
        minute=date.minute,
        second=date.second,
    )
    if ret > datetime.today():
        raise ValueError(f"{ret} is in the future")
    return pd.to_datetime(ret)


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert version_info >= (3, 6), "Please use at least Python 3.6"
assert all(
    (
        __name__
        in (
            "analisi.src.YAstarMM.utility",
            "YAstarMM.utility",
            "utility",
        ),
        "initialize_logging" in globals(),
        "debug_start_timer" in globals(),
        "debug_stop_timer" in globals(),
    )
), "Please update 'Usage' section of module docstring"
