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
   Plot several plots.

   Usage:
            from  YAstarMM.plot  import  (
                create_new_axes,
            )

   ( or from within the YAstarMM package )

            from          .plot  import  (
                create_new_axes,
            )

"""

from .column_rules import minimum_maximum_column_limits, translator_helper
from .constants import (
    MIN_PYTHON_VERSION,
)
from .flavoured_parser import parsed_args
from hashlib import blake2b
from matplotlib import colors, rc
from os.path import join as join_path
from sys import version_info
import itertools
import logging
import matplotlib.pyplot as plt

_USETEX = getattr(parsed_args(), "use_latex", False)
TRANSPARENT_WHITE = colors.to_rgba("w", alpha=0)


def create_new_axes(
    figsize=(11.02, 8.27),  # 4/3 aspect ratio (beamer friendly)
    nrows=1,
    ncols=1,
    constrained_layout=False,
    dpi=300,
    tight_layout=False,
    **kwargs,
):
    global _USETEX
    plt.cla()
    plt.clf()
    if _USETEX:
        # https://matplotlib.org/3.1.0/tutorials/text/usetex.html
        rc("text", usetex=True)
        rc("font", **{"family": "serif", "serif": "Computer Modern Roman"})

    fig = plt.figure(
        constrained_layout=constrained_layout,
        dpi=dpi,
        figsize=figsize,
        tight_layout=tight_layout,
        **kwargs,
    )
    return fig.subplots(nrows, ncols)  # Axes or array of Axes


def plot_histogram_distribution(
    df,
    has_outliers,
    dpi=600,
    extensions=("pdf", "svg"),
    figsize=(15.35, 11.42),  # size of a 19" display with ratio 4:3
    logger=logging,
    save_to_dir=None,
    suptitle=None,
):
    assert has_outliers in (True, False)
    global _USETEX
    nrows, ncols = sorted(
        [
            (rows, cols)
            for rows in range(len(df.columns))
            for cols in range(len(df.columns))
            if rows * cols >= len(df.columns)
        ],
        key=lambda tup: (
            tup[0] * tup[1],  # rows * cols
            abs(  # penalize ratios too much different from figsize
                tup[1] / tup[0] - figsize[0] / figsize[1]
            ),
            tup[figsize.index(max(figsize))],
            tup[figsize.index(min(figsize))],
        ),
    ).pop(0)
    logger.debug(f"Creating a figure with {nrows} rows x {ncols} subplots")
    all_axes = create_new_axes(figsize, nrows, ncols, dpi=dpi)
    num_axes = len(df.columns)
    for col, ax in zip(
        sorted(
            df.columns,
            key=lambda col: translator_helper(
                col, usetex=_USETEX, bold=_USETEX
            )
            .lower()
            .replace(r"$", "")
            .replace(r"\mathbf", "")
            .replace(r"\mathrm", "")
            .replace(r"\textbf", "")
            .strip(r"{ }"),
        ),
        list(itertools.chain.from_iterable(all_axes.tolist()))[:num_axes],
    ):
        logger.debug(f"Plotting histogram distribution of '{col}'")
        ax.set_title(
            translator_helper(col, usetex=_USETEX, bold=_USETEX),
            fontsize=10.5,
            pad=-14,
            y=1.0,
        )
        max_hex_digits = 6
        n, bins, _ = ax.hist(
            df.loc[:, col].dropna(),
            color=plt.cm.gnuplot(
                max(
                    0.05,
                    min(
                        0.95,
                        int(
                            blake2b(  # hash of the title
                                str.encode(
                                    translator_helper(
                                        col, usetex=False, bold=False
                                    )
                                )
                            ).hexdigest()[:max_hex_digits],
                            base=16,
                        )
                        / float(16 ** max_hex_digits),
                    ),
                )
            ),
            rwidth=0.8,
        )
        for i, (height, x_start, x_end) in enumerate(
            zip(n, bins[:-1], bins[1:])
        ):
            logger.debug(
                f"Bin-{i} is {round(height):5d} tall, "
                f"starts from {x_start:9.3f} and ends to {x_end:9.3f}"
            )
            if has_outliers and height > 0:
                ax.text(
                    x_start + (x_end - x_start) * 0.65,  # more centered
                    height,
                    str(round(height)),
                    fontsize="x-small",
                    ha="center",
                    va="bottom",
                )
        if has_outliers:
            ax.set_yticks(list())  # already written on top of each bin
        else:
            lower_limit = (
                minimum_maximum_column_limits(
                    getattr(parsed_args(), "outlier_limits")
                )
                .get(col, dict())
                .get("min", None)
            )
            upper_limit = (
                minimum_maximum_column_limits(
                    getattr(parsed_args(), "outlier_limits")
                )
                .get(col, dict())
                .get("max", None)
            )
            logger.debug(f"limits are: [{lower_limit}, {upper_limit}]")
            old_ticks = list(ax.get_xticks())
            logger.debug(f"previous x-ticks: {repr(old_ticks)}")
            ax.vlines(
                x=[
                    lim
                    for lim in (lower_limit, upper_limit)
                    if lim is not None
                ],
                color="k",
                label="limits",
                linestyles="dashdot",
                linewidths=1,
                transform=ax.get_xaxis_transform(),
                ymin=0,
                ymax=0.88,
            )
            if lower_limit is not None or upper_limit is not None:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[-1:],  # only limits, no hist
                    labels[-1:],  # only limits, no hist
                    bbox_to_anchor=(1, 0.92),
                    borderpad=0.1,
                    borderaxespad=0.1,
                    edgecolor=TRANSPARENT_WHITE,
                    facecolor=TRANSPARENT_WHITE,
                    loc="upper right",
                )
                ax.get_xaxis().reset_ticks()
                ax.set_xticks(
                    np.linspace(
                        lower_limit
                        if lower_limit is not None
                        else old_ticks[0],
                        upper_limit
                        if upper_limit is not None
                        else old_ticks[-1],
                        5,
                    )
                )
                logger.debug(f"new x-ticks: {repr(list(ax.get_xticks()))}")
        # enlarge by 10% to the left, top and right
        x_start, x_end = ax.get_xlim()
        pad = abs(x_end - x_start) / 10.0
        ax.set_xlim(x_start - pad, x_end + pad)
        ax.set_ylim(-0.02 * ax.get_ylim()[1], ax.get_ylim()[1] * 1.10)
    for ax in list(itertools.chain.from_iterable(all_axes.tolist()))[
        num_axes:
    ]:
        ax.set_axis_off()  # do not draw anything

    fig = plt.gcf()
    fig.set_tight_layout(dict(rect=(0, 0, 1, 1 if suptitle is None else 0.98)))
    if suptitle is not None:
        logger.debug(f"suptitle: {repr(suptitle)}")
        fig.suptitle(
            "".join(
                (
                    r"\textbf{" if _USETEX else "",
                    " ".join(
                        suptitle.replace("_", " " if _USETEX else " ")
                        .replace("\n", r"\,\\\," if _USETEX else "")
                        .split(" ")
                    ),
                    r"}" if _USETEX else "",
                )
            ),
            fontweight="bold" if not _USETEX else "normal",
            fontsize="xx-large",
        )
    if save_to_dir is not None:
        for extension in extensions:
            save_path = join_path(
                save_to_dir,
                "observed_variables_distribution_"
                + str("with" if has_outliers else "without")
                + f"_outliers.{extension}",
            )
            fig.savefig(save_path, dpi=dpi)
            logger.debug(f"Saved plot '{save_path}'")
    else:
        logger.warning("save_to_dir is None")
    fig.clf()


if __name__ == "__main__":
    raise SystemExit("Please import this script, do not run it!")
assert (
    version_info >= MIN_PYTHON_VERSION
), f"Please use at least Python {'.'.join(str(n) for n in MIN_PYTHON_VERSION)}"
assert __name__ in (
    "analisi.src.YAstarMM.plot",
    "YAstarMM.plot",
    "plot",
), "Wrong module name; please update 'Usage' section of module docstring"
for usage_docstring in __doc__.split("import")[1:]:
    for fun in "".join(
        usage_docstring.split(")")[0].lstrip(" (").split()
    ).split(",")[:-1]:
        assert fun in globals(), str(
            f"Function {fun} not found in module;"
            " please update 'Usage' section of module docstring"
        )
