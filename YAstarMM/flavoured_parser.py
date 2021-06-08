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
   Parse arguments from 'flavour' file and command line.

   Accordingly to the following hierarchy, higher priority values will
   always have precedence over lower priority values:

   (higher priority)
   - values parsed from the Command Line Interface (CLI)
   - values parsed from the YAML 'flavour' file
   - hardcoded default values in parser code (used as fallback)
   (lower priority)

   Usage:
            from  YAstarMM.flavoured_parser  import  parsed_args

   ( or from within the YAstarMM package )

            from           flavoured_parser  import  parsed_args
"""

from .column_rules import rename_helper
from .constants import AVG_ITER_TIME, MIN_PYTHON_VERSION
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    FileType,
    Namespace,
    SUPPRESS,
)
from multiprocessing import cpu_count
from pprint import pformat as pretty_format
from sys import version_info
from textwrap import wrap
from typing import Any, Dict, Iterator, Optional, TextIO, Tuple, Union
from yaml import dump, load

_CLI_ARGUMENTS: Dict[Tuple[str, ...], Dict[str, Any]] = {
    ("-d", "--debug",): dict(
        action="store_true",
        default=False,
        dest="debug_mode",
        full_help="Add debug info to plots, disable results compression",
        help=SUPPRESS,
    ),
    ("--debug-parser",): dict(
        action="store_true",
        full_help="Print internal state of the CLI argument parser",
        help=SUPPRESS,
    ),
    ("--dump-flavoured-parser",): dict(
        default=None,
        metavar="file",
        full_help="Dump FlavouredNamespace to a file",
        help=SUPPRESS,
        type=FileType("w"),
    ),
    ("--ignore-expert-knowledge",): dict(
        action="store_true",
        default=False,
        full_help="Do not update observed variables "
        "following suggestions from experts on the domain",
        help=SUPPRESS,
    ),
    ("--ignore-transferred-state",): dict(
        choices=(str(False), str(True)),
        default=True,
        full_help="Whether to consider or not the transferred state "
        "in HMMs training",
        help=SUPPRESS,
    ),
    ("-i", "--input",): dict(
        help="Excel input file containing the DataFrame to parse",
        metavar="xlsx",
        type=FileType("rb"),
    ),
    ("-j", "--num-workers",): dict(
        default=cpu_count(),
        dest="max_workers",
        help="Split workload among several workers",
        type=int,
    ),
    ("-l", "--min-iter",): dict(
        default=max(1, round(30 / AVG_ITER_TIME)),  # 30 seconds
        full_help="Minimum number of iterations to run Baum-Welch training"
        f" for, on average an iteration takes {AVG_ITER_TIME:.3f} seconds.",
        help=SUPPRESS,
        type=int,
    ),
    ("-m", "--max-iter",): dict(
        default=min(2000, round(AVG_ITER_TIME * 5 * 60)),  # 5 minutes
        full_help="Maximum number of iterations to run Baum-Welch training"
        f" for, on average an iteration takes {AVG_ITER_TIME:.3f} seconds.",
        help=SUPPRESS,
        type=int,
    ),
    ("-n", "--explore-n-seeds",): dict(
        default=None,
        dest="seeds_to_explore",
        help="Train at most N HMMs with different seeds",
        metavar="N",
        mutex_group="hmm_seeds_definition",
        type=int,
    ),
    ("--observed-variables",): dict(
        choices=list(
            rename_helper(
                (
                    "AGE",
                    "CARBON_DIOXIDE_PARTIAL_PRESSURE",
                    "CHARLSON_INDEX",
                    "CREATININE",
                    "DYSPNEA",
                    "D_DIMER",
                    "GPT_ALT",
                    "HOROWITZ_INDEX",
                    "LDH",
                    "LYMPHOCYTE",
                    "PH",
                    "PHOSPHOCREATINE",
                    "PROCALCITONIN",
                    "RESPIRATORY_RATE",
                    "UREA",
                )
            )
        ),
        default=list(),
        help="Take into account only given observed variables",
        nargs="+",
        type=str,
    ),
    ("--outlier-limits",): dict(
        choices=("strict", "relaxed", "nonsense"),
        default="strict",
        full_help="Preset of outlier limits; "
        "strict = what theoretically should be used, "
        "relaxed = what 0.03 and 0.97 percentiles suggest, "
        "nonsense = what a random monkey suggested",
        help=SUPPRESS,
    ),
    ("--oxygen-states",): dict(
        choices=[
            "No_O2",
            "O2",
            "HFNO",
            "NIV",
            "Intubated",
            "Deceased",
            "Discharged",
            "Transferred",
        ],
        default=list(),
        help="Take into account only given oxygen states",
        nargs="+",
        type=str,
    ),
    ("--patient-key-col",): dict(
        dest="patient_key_col",
        full_help="Column containing patients' identifiers",
        help=SUPPRESS,
        type=str,
    ),
    ("-r", "--random-seed",): dict(
        default=None,
        dest="meta_model_random_seed",
        help="Initialize random generator used to "
        "split the test set for the composer",
        metavar="int",
        type=int,
    ),
    ("--ratio-test-set",): dict(
        default=0.1,
        dest="test_set_ratio_composer",
        help="Size of the test set for the composer",
        type=float,
    ),
    ("--ratio-validation-set",): dict(
        default=0.1,
        dest="validation_set_ratio_hmm",
        help="Size of the validation set for the HMM",
        type=float,
    ),
    ("-s", "--hmm-seeds",): dict(
        default=None,
        dest="light_meta_model_random_seeds",
        help="Initialize random generator used to "
        "split the training/validation set for the HMMs "
        "(and exit after exausting the list)",
        mutex_group="hmm_seeds_definition",
        nargs="+",
        type=int,
    ),
    ("--save-dir",): dict(
        default=None,
        help="Where results will be stored",
        metavar="path",
        type=str,
    ),
    ("--threshold", "--stop-threshold",): dict(
        default=1e-9,
        dest="stop_threshold",
        full_help="Stop fitting when this threshold improvement ratio is reached",
        help=SUPPRESS,
        metavar="float",
        type=float,
    ),
    ("--train-little-hmm",): dict(
        action="store_true",
        default=False,
        full_help="Auto spawn ad-hoc-little-HMM workloads "
        "among the available workers in a load balanced way",
        help=SUPPRESS,
    ),
    ("--update-charlson-index",): dict(
        choices=(str(False), str(True)),
        default=True,
        full_help="Compute and fix Charlson-Index scores",
        help=SUPPRESS,
    ),
    ("--use-latex",): dict(
        action="store_true",
        default=False,
        full_help="Render text in plots with LaTeX",
        help=SUPPRESS,
    ),
    ("-v", "--verbose"): dict(
        action="store_true",
        default=False,
        full_help="Increase logging verbosity",
        help=SUPPRESS,
    ),
}
"""Command line interface arguments to parse."""

_DESCRIPTION: str = """
    The 'flavour' / behaviour of %(prog)s can be personalized by
    providing a YAML file containing key-value pairs.

    This way the amount of arguments passed via Command Line Interface
    can be reduced and in the near future deploying the code to a web
    server 'in production' should be easier.

    In any case CLI arguments will always overwrite the respectives
    ones read from the 'flavour' file."""


def _get_cli_parser(
    flavour_dict: Dict[str, Any], help_value: Union[bool, None, str] = False
) -> ArgumentParser:
    """Return parser of all arguments except the one about the 'flavour'."""
    global _CLI_ARGUMENTS, _DESCRIPTION

    hardcoded_default_values = {
        name_or_flag[-1].lstrip("-").replace("-", "_"): kwargs["default"]
        for name_or_flag, kwargs in _CLI_ARGUMENTS.items()
        if "default" in kwargs
    }
    # Overwrite hardcoded default values with the existing ones in the
    # 'flavour' dictionary because of their higher pripority
    hardcoded_default_values.update(flavour_dict)

    cli_parser = ArgumentParser(
        add_help=False,
        allow_abbrev=True,
        description=_DESCRIPTION,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    cli_parser.add_argument(
        "-h",
        "--help",
        default=SUPPRESS,
        help="Show this help message and exit",
        nargs="?",
        metavar="full",
    )

    # WARNING: what follows is not the real -f / --flavour argument !!!
    # The only purpose of this dummy argument is showing an help
    # message for -f / --flavour
    cli_parser.add_argument(
        "-f",
        "--flavour",
        default=hardcoded_default_values.pop("flavour", None),
        help="If present, first of all the 'flavour' file is parsed",
        metavar="yaml",
        type=FileType("r"),
    )
    # DO NOT EDIT the above -f / --flavour argument !!!
    # Please edit the one in _get_flavour_parser()
    assert (
        "_get_flavour_parser" in globals()
    ), "Please update the comment in the line before this one :D"

    abolish_required = help_value in (None, "full")
    mutex_groups = dict()
    for args, kwargs in _CLI_ARGUMENTS.items():
        longest_name = args[-1].lstrip("-").replace("-", "_")
        if longest_name == "input":
            kwargs["required"] = "input" not in hardcoded_default_values
        kwargs["default"] = hardcoded_default_values.pop(
            longest_name, kwargs.get("default", None)
        )
        if help_value == "full" and kwargs.get("help", SUPPRESS) == SUPPRESS:
            kwargs["help"] = kwargs.pop("full_help", SUPPRESS)
        else:
            kwargs.pop("full_help", None)  # drop this unused kwarg
        if "metavar" not in kwargs and "type" in kwargs:
            kwargs["metavar"] = kwargs["type"].__name__
        if abolish_required:
            kwargs["required"] = False
        if "mutex_group" not in kwargs:
            cli_parser.add_argument(*args, **kwargs)
        else:
            mutex_group = kwargs.pop("mutex_group")
            if mutex_group not in mutex_groups:
                mutex_groups[
                    mutex_group
                ] = cli_parser.add_mutually_exclusive_group(
                    required=not abolish_required
                )
            mutex_groups[mutex_group].add_argument(*args, **kwargs)
    return cli_parser


def _get_flavour_parser(
    default_flavour_file: Optional[str] = None,
) -> ArgumentParser:  # make black auto-formatting prettier
    """Return parser of only the 'flavour' argument."""
    flavour_parser = ArgumentParser(add_help=False, allow_abbrev=False)
    flavour_parser.add_argument(
        "-h",
        "--help",
        choices=("full",),
        default=False,
        help=SUPPRESS,
        nargs="?",
    )
    flavour_parser.add_argument(
        # This is the real 'flavour' argument to edit !!!
        "-f",
        "--flavour",
        #
        # Next line is the only right place to set a default 'flavour' filename
        default=default_flavour_file,
        #
        dest="flavour_file",
        help=SUPPRESS,
        type=FileType("r"),
    )
    return flavour_parser


def _get_flavour_dict(flavour_file: Optional[TextIO]) -> Dict[str, Any]:
    """Return parsed 'flavour' file or an empty dictionary."""
    flavour: Dict[str, Any] = dict(flavour=None)
    if flavour_file is not None:
        try:
            # faster compiled (safe) Loader
            from yaml import CSafeLoader as SafeLoader
        except ImportError:
            # fallback, slower interpreted (safe) Loader
            from yaml import SafeLoader  # type: ignore
        finally:
            loaded_flavour: Union[Dict[str, Any], None] = load(
                flavour_file, Loader=SafeLoader
            )
            # Add 'flavour' file name to non-empty 'flavour' dict
            if loaded_flavour is not None:
                loaded_flavour["flavour"] = flavour_file.name
                flavour = loaded_flavour
    return flavour


class FlavouredNamespace(object):
    """Wrap CLI arguments and 'flavour' arguments."""

    _initialized: bool = False
    """Flag to run __init__ once."""

    _instance = None  # ignore: type
    """Singleton instance to return in __new__."""

    def _debug(self) -> str:
        """Return internal state representation and prioritized fields view."""
        global _PARSED_ARGS
        ret = "FlavouredNamespace internal state:\n\n"
        ret += f"{str(_PARSED_ARGS)}\n\n\n"
        ret += "FlavouredNamespace fields values (accordingly to priority):\n"

        longest_field = max(len(field) for field in self)
        indent_size = longest_field
        indent_size += 9  # size of '\t.' (with expanded tab)
        indent_size += 3  # size of ' = '

        # Prioritized fields view
        for field in sorted(iter(self)):
            ret += f"\n\t.{field.ljust(longest_field)} = "
            ret += pretty_format(getattr(self, field)).replace(
                "\n", f"\n{' ' * (indent_size)}"
            )
        return ret

    def __getattr__(self, name: str) -> Any:
        """Lookup for name following the priority hierarchy."""
        if all(
            (
                name != "flavour",  # skip because it is in self._flavour
                hasattr(self._namespace, name),
            )
        ):
            return getattr(self._namespace, name)  # higher priority
        elif name in self._flavour:
            return self._flavour[name]  # lower priority
        else:
            raise AttributeError(
                f"'FlavouredNamespace' object has no attribute '{name}'"
            )

    def __getitem__(self, key: str) -> Any:
        """Unify dictionary interface with the attributes' one."""
        if not isinstance(key, str):
            raise TypeError("FlavouredNamespace only admits str keys")
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e).replace("attribute", "key"))

    def __init__(
        self,
        flavour: Dict[str, Any],
        namespace: Namespace,
        dump_filename: Optional[TextIO] = None,
    ) -> None:
        """Save passed arguments in private instance variables.

        :param flavour: Dictionary read from 'flavour' file.
        :param namespace: Namespace built from CLI arguments.
        :param dump_filename: TextIO object to which self will be dumped
        """
        if not FlavouredNamespace._initialized:
            self._flavour = flavour
            self._namespace = namespace
            FlavouredNamespace._initialized = True
            if dump_filename is not None:
                self.dump(dump_filename)

    def __iter__(self) -> Iterator[str]:
        """Return iterator over 'flavour' keys and namespace attributes."""
        return iter(set(self._flavour.keys()) | set(vars(self._namespace)))

    def __len__(self) -> int:
        """Return the amount of 'flavour' keys or namespace attributes."""
        return len(set(self._flavour.keys()) | set(vars(self._namespace)))

    def __new__(cls, *args, **kwargs):  # type: ignore
        """Pythonic implementation of the singleton design pattern.

        https://python-patterns.guide/gang-of-four/singleton/#a-more-pythonic-implementation
        """
        if cls._instance is None:
            cls._instance = super(FlavouredNamespace, cls).__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return dictionary representation of the internal state."""
        return repr(vars(self))

    def __setitem__(self, key: str, value: Any) -> None:
        """Forbid key assignements."""
        raise AttributeError(
            "FlavouredNamespace object does not support attribute assignements"
        )

    def __str__(self) -> str:
        """Return cute representation of the internal state."""
        return pretty_format(vars(self))

    def dump(self, filename: TextIO) -> None:
        """Save object as a 'flavour' file in yaml format."""
        data: Dict[str, Any] = dict()
        for field in sorted(iter(self)):
            value = getattr(self, field)
            if field == "input" and hasattr(value, "name"):
                data["input"] = str(value.name)  # store input filename
            elif field == "flavour":
                # skip original 'flavour' file name
                continue
            elif field.startswith("debug") and not field.endswith("mode"):
                # skip debugging attributes with suppressed help message
                continue
            else:
                data[field] = value

        try:
            # faster compiled (safe) Dumper
            from yaml import CSafeDumper as SafeDumper
        except AttributeError:
            # fallback, slower interpreted (safe) Dumper
            from yaml import SafeDumper  # type: ignore
        finally:
            dump(
                data,
                filename,
                Dumper=SafeDumper,
                default_flow_style=False,
            )


_PARSED_ARGS: Optional[FlavouredNamespace] = None
"""Namespace-like object, hiding 'flavour' complexity."""


def parsed_args(
    default_flavour_file: Optional[str] = None,
) -> FlavouredNamespace:  # make black auto-formatting prettier
    """Return parsed arguments from CLI and YAML file hiding complexity."""
    global _PARSED_ARGS
    if _PARSED_ARGS is None:
        # 1st parser just parses -f / --flavour cli argument (if present)
        _flavour_namespace, _args_left = _get_flavour_parser(
            default_flavour_file
        ).parse_known_args()
        if getattr(_flavour_namespace, "help", False) in (None, "full"):
            _args_left.extend(["-h", getattr(_flavour_namespace, "help")])

        # Parse 'flavour' file (if absent or empty, a dictionary is returned)
        _flavour_dict = _get_flavour_dict(_flavour_namespace.flavour_file)

        # 2nd parser is the real one and parses the remaining cli arguments
        cli_parser = _get_cli_parser(
            _flavour_dict,
            help_value=getattr(_flavour_namespace, "help", False),
        )
        _namespace = cli_parser.parse_args(_args_left)

        # Populate the global variable which other modules will import
        _PARSED_ARGS = FlavouredNamespace(
            flavour=_flavour_dict,
            namespace=_namespace,
            dump_filename=_namespace.dump_flavoured_parser,
        )
        if getattr(_flavour_namespace, "help", False) in (None, "full"):
            raise SystemExit(pretty_tabulate(cli_parser.format_help()))
        if _PARSED_ARGS.debug_parser:
            raise SystemExit(_PARSED_ARGS._debug())
    return _PARSED_ARGS


def pretty_tabulate(help_message, indent="  ", sep="  "):
    max_line_length = max(len(line) for line in help_message.split("\n"))
    usage, description, body = help_message.split("\n\n", maxsplit=2)
    body = "\n".join(body.split("\n")[1:])
    table = {
        f"-{key.strip()}": " ".join(value.split())
        for key, value in dict(
            arg.split("  ", maxsplit=1)
            for arg in f"\n{body}".split("\n  -")
            if arg
        ).items()
    }
    left_col_size = max(len(k) for k in table.keys())
    right_col_size = max_line_length - len(indent) - len(sep) - left_col_size
    ret = "\n".join((usage, "", description, ""))
    for k, v in sorted(
        table.items(),
        key=lambda t: t[0].replace("-h", r"\0").lstrip("-").lower(),
    ):
        ret += f"\n{indent}{k.ljust(left_col_size)}{sep}"
        ret += f"\n{indent}{' '*left_col_size}{sep}".join(
            wrap(v, right_col_size)
        )
    return ret


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.flavoured_parser",
    "YAstarMM.flavoured_parser",
    "flavoured_parser",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
