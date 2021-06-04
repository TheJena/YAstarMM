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
                black_magic,
                debug_start_timer,
                debug_stop_timer,
                enough_ram,
                extraction_date,
                initialize_logging,
            )

   ( or from within the YAstarMM package )

            from          .utility  import  (
                black_magic,
                debug_start_timer,
                debug_stop_timer,
                enough_ram,
                extraction_date,
                initialize_logging,
            )
"""

from .constants import EXTRACTION_REGEXP, MIN_PYTHON_VERSION
from datetime import datetime, timedelta
from functools import lru_cache
from logging import (
    DEBUG,
    Formatter,
    INFO,
    StreamHandler,
    WARNING,
    basicConfig,
    debug,
    getLogger,
    info,
)
from hashlib import blake2b
from multiprocessing import Lock
from numpy.random import RandomState
from os import mkdir
from os.path import abspath, expanduser, isdir, isfile, join as join_path
from psutil import virtual_memory, swap_memory
from sys import version_info
from tempfile import mkdtemp, NamedTemporaryFile
from time import time
import gzip
import pandas as pd
import pickle

_CACHE_DIR = None  # where do you prefer to save @black_magic ?
_DISABLE_BLACK_MAGIC_GLOBALLY = True

_CACHE_LOCK, _PERF_LOCK = Lock(), Lock()


def black_magic(fun):
    """Like @lru_cache but persistent thank to compressed pickle files

    https://pythonbasics.org/decorators/#Real-world-examples
    """
    assert fun.__name__ != "wrapper", str(
        "Please use @black_magic before any other decorator"
    )

    global _CACHE_DIR, _CACHE_LOCK, _DISABLE_BLACK_MAGIC_GLOBALLY
    if _DISABLE_BLACK_MAGIC_GLOBALLY:
        debug(f"black_magic decorator is disabled; {fun.__name__} will be run")
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
        # elif fun.__name__ == "":
        #     hashed_input = "CHECKPOINT_BRAVO"
        # elif fun.__name__ == "":
        #     hashed_input = "CHECKPOINT_ALPHA"
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


def duplicated_columns(df):
    return sorted(df.loc[:, df.columns.duplicated()].columns)


@lru_cache(maxsize=None, typed=True)
def enough_ram(gb):
    if virtual_memory().available + swap_memory().free > gb * 1024 ** 3:
        debug(f"There is enough memory to allocate {gb} GiB")
        return True
    debug(f"There is not enough memory to allocate {gb} GiB")
    return False


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
        "\n\t- Extraction_20001231_whatever.xlsx"
        "\n\t- Estrazione_2000_12_31_whatever.xlsx"
        "\n\t- Extraction_2000_12_31_whatever.xlsx"
        "\n\t- *cleaned_extraction_20001231_whatever.xlsx"
        "\n\t- *cleaned_extraction_2000_12_31_whatever.xlsx"
        f"\n\n(Please fix {filename} accordingly)"
    )


def hex_date_to_timestamp(
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


def initialize_logging(suffix_filename, level=INFO, debug_mode=False):
    logfile = NamedTemporaryFile(
        prefix=datetime.now().strftime("%Y_%m_%d__%H_%M_%S____"),
        suffix="____"
        + str(
            suffix_filename.strip("_")
            + str(".txt" if "." not in suffix_filename else "")
        ),
        delete=False,
    )
    basicConfig(
        filename=logfile.name,
        force=True,
        format="\t".join(
            (
                "[{levelname: ^9s}| {module}+L{lineno} | PID-{process}]",
                "{message}",
            )
        ),
        level=DEBUG,  # use always the most verbose level for log file
        style="{",
    )
    root_logger = getLogger()
    stderr_handle = StreamHandler()
    stderr_handle.setLevel(level if not debug_mode else DEBUG)
    stderr_handle.setFormatter(Formatter("{levelname}: {message}", style="{"))
    root_logger.addHandler(stderr_handle)

    # make foreign modules quiet in logfile
    for module_name in ("matplotlib", "numexpr", "PIL"):
        getLogger(module_name).setLevel(WARNING)

    info("Temporary file with debugging log will be available here:")
    info(f"{' ' * 4}{logfile.name}")
    debug(f"{'~' * 120}")
    debug("")
    debug("Logging initialized and temporary file created")
    debug("")


def random_string(length):
    return "".join(
        chr(i)
        for i in RandomState(seed=None).choice(
            sorted(
                set(range(ord("0"), ord("9") + 1))
                .union(set(range(ord("A"), ord("Z") + 1)))
                .union(set(range(ord("a"), ord("z") + 1)))
            ),
            size=length,
        )
    )


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


def timestamp_to_hex_date(date, year_offset=0, desired_length=5):
    date = pd.to_datetime(date)
    ret = (  # X means hexadecimal notation
        f"{date.year-year_offset:X}".rjust(desired_length - 3, "0")
        + f"{date.month:X}".rjust(1, "0")
        + f"{date.day:X}".rjust(2, "0")
    )
    assert len(ret) == desired_length, f"len({repr(ret)}) != {desired_length}"
    return ret


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.utility",
    "YAstarMM.utility",
    "utility",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
