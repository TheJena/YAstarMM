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


def initialize_logging(level=INFO):
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
        level=level,
    )
    root_logger = getLogger()
    stderr_handle = StreamHandler()
    stderr_handle.setLevel(INFO)
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
