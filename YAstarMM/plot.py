#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2020 Federico Motta <191685@studenti.unimore.it>
#               2021 Federico Motta <federico.motta@unimore.it>
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
                plot_histogram_distribution,
                plot_transition_probabilities,
            )

   ( or from within the YAstarMM package )

            from          .plot  import  (
                plot_histogram_distribution,
                plot_transition_probabilities,
            )

"""

from .column_rules import minimum_maximum_column_limits, translator_helper
from .constants import (
    APPLE_GREEN,
    CAPRI_BLUE,
    EPSILON,
    MIN_PYTHON_VERSION,
    Point,
)
from .flavoured_parser import parsed_args
from .model import State
from hashlib import blake2b
from math import ceil
from matplotlib import colors, rc
from os.path import expanduser, join as join_path
from sys import version_info
import itertools
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import missingno as msno
import networkx as nx
import numpy as np
import pandas as pd

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


def polygon_layout(graph, scale=1, swap_transferred_and_discharged=True):
    start_angle = np.pi / 2  # 12:00
    sum_angle = -2 * np.pi / graph.number_of_nodes()  # clockwise

    ret = {
        node: Point(
            x=scale * np.cos(start_angle + i * sum_angle),
            y=scale * np.sin(start_angle + i * sum_angle),
        )
        for i, node in enumerate(graph.nodes)
    }

    if all(
        (
            swap_transferred_and_discharged,
            State.Transferred in graph.nodes,
            State.Discharged in graph.nodes,
        )
    ):
        ret[State.Discharged], ret[State.Transferred] = (
            ret[State.Transferred],
            ret[State.Discharged],
        )
    return ret


def plot_histogram_distribution(
    df,
    has_outliers,
    dpi=600,
    extensions=("pdf", "svg"),
    figsize=(15.35, 11.42),  # size of a 19" display with ratio 4:3
    logger=logging,
    max_subplots=24,
    prefix="",
    save_to_dir=None,
    suptitle=None,
):
    tot_pages = ceil(len(df.columns) / max_subplots)
    for page in range(tot_pages):
        _plot_histogram_distribution(
            df.loc[
                :,
                sorted(df.columns, key=str.lower)[
                    max_subplots * page : max_subplots * (page + 1)
                ],
            ],
            has_outliers,
            dpi=dpi,
            extensions=extensions,
            figsize=figsize,
            logger=logging,
            page_num=f"page {page+1} of {tot_pages}".replace(" ", "_"),
            prefix=prefix,
            save_to_dir=save_to_dir,
            suptitle=suptitle,
        )


def _plot_histogram_distribution(
    df,
    has_outliers,
    dpi,
    extensions,
    figsize,
    logger,
    page_num,
    prefix,
    save_to_dir,
    suptitle,
):
    assert has_outliers in (True, False)
    global _USETEX
    nrows, ncols = sorted(
        [
            (rows, cols)
            for rows in range(1 + len(df.columns))
            for cols in range(1 + len(df.columns))
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
    all_axes = create_new_axes(figsize, nrows, ncols, dpi=dpi).tolist()
    if not isinstance(all_axes[0], list):
        all_axes = [all_axes]
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
        list(itertools.chain.from_iterable(all_axes))[:num_axes],
    ):
        logger.debug(f"Plotting histogram distribution of '{col}'")
        ax.set_title(
            translator_helper(col, usetex=_USETEX, bold=_USETEX),
            fontsize=10.5,
            y=None,
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
                    bbox_to_anchor=(1, 0.98),
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
        ax.grid(False)
        # enlarge by 10% to the left, top and right
        x_start, x_end = ax.get_xlim()
        pad = abs(x_end - x_start) / 10.0
        ax.set_xlim(x_start - pad, x_end + pad)
        ax.set_ylim(-0.02 * ax.get_ylim()[1], ax.get_ylim()[1] * 1.10)

    for ax in list(itertools.chain.from_iterable(all_axes))[num_axes:]:
        ax.set_axis_off()  # do not draw anything

    fig = plt.gcf()
    fig.set_tight_layout(
        dict(rect=(0, 0, 1, 1 if suptitle is None and not page_num else 0.98))
    )
    if suptitle is not None or page_num:
        if suptitle is None:
            suptitle = str(page_num)
        elif page_num:
            suptitle += f" ({str(page_num).lstrip(' ')})"
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
                f"{prefix.rstrip('_')}__feature_distribution_"
                + str("with" if has_outliers else "without")
                + f"_outliers_{page_num}.{extension}",
            )
            fig.savefig(save_path, dpi=dpi)
            logger.debug(f"Saved plot '{save_path}'")
    else:
        logger.warning("save_to_dir is None")
    fig.clf()
    plt.close(fig)


def plot_sparsity(
    df,
    dpi=600,
    extensions=("pdf", "svg"),
    figsize=(15.35, 11.42),  # size of a 19" display with ratio 4:3
    fontsize=16,
    logger=logging,
    prefix="",
    save_to_dir=None,
):
    if parsed_args():
        input_file = getattr(parsed_args(), "plot_dataset_sparsity", None)
        assert input_file is not None
        df = pd.read_csv(input_file).drop(columns="Unnamed: 0")
        df2 = df.rename(
            columns={
                c: translator_helper(
                    c,
                    bold=_USETEX,
                    usetex=_USETEX,
                    **{"Oxygen Therapy State": "oxygen_therapy_state_value"},
                )
                for c in df.columns
                if c not in ("date", "", "UPDATED_CHARLSON_INDEX")
            }
        ).drop(columns=["date", "", "UPDATED_CHARLSON_INDEX"])
    else:
        df2 = df.copy(deep=True)
    msno.bar(
        df2,
        ax=create_new_axes(tight_layout=True, dpi=dpi),
        figsize=figsize,
        fontsize=fontsize,
        sort="ascending",
    )
    fig = plt.gcf()
    if save_to_dir is not None:
        for extension in extensions:
            save_path = join_path(
                save_to_dir,
                f"{prefix.rstrip('_')}__feature_sparsity.{extension}",
            )
            fig.savefig(save_path, dpi=dpi)
            logger.debug(f"Saved plot '{save_path}'")
    else:
        logger.warning("save_to_dir is None")
    fig.clf()
    plt.close(fig)


def plot_transition_probabilities(transition_matrix, selfedges=True):
    founded_states = [
        state for state in State if np.any(transition_matrix[:, state.value])
    ]

    graph = nx.DiGraph()
    for state in founded_states:
        graph.add_node(
            state,
            color={"Deceased": "k", "Discharged": APPLE_GREEN}.get(
                str(state), CAPRI_BLUE
            ),
        )
    for state_from in founded_states:
        for state_to in founded_states:
            probability = round(
                transition_matrix[state_from.value][state_to.value], 3
            )
            if any(
                (
                    probability <= 0.007,
                )
            ):
                logging.info(
                    f"{state_from:^13} ~> {state_to:^12} "
                    f"has prob {probability:.3f}\t[ SKIPPED ]"
                )
                continue
            graph.add_edge(state_from, state_to, probability=probability)
            logging.debug(
                f"{state_from:^12} ~> {state_to:^12} "
                f"has prob {probability:.3f}"
                + str("\t(SELF-EDGE)" if state_from == state_to else "")
            )
    node_positions = polygon_layout(graph)

    ax = create_new_axes(tight_layout=True)

    # Draw nodes
    nx.draw(
        graph,
        pos=node_positions,
        ax=ax,
        edgelist=[],
        linewidths=3,
        node_color=[color for _, color in graph.nodes(data="color")],
        node_size=0.85 * 2 ** 11,
        with_labels=False,
    )

    # Draw node labels
    for node, (x, y) in node_positions.items():
        ax.text(
            fontweight="bold",
            ha="center",
            s=translator_helper(str(node), bold=_USETEX, usetex=_USETEX),
            size=24,
            va="center",
            x=x + sign(x) / 7,
            y=y + 10 * sign(y) / 40,
        )
        # Draw self-loop label
        if not selfedges and graph.has_edge(node, node):
            prob = graph.edges[node, node]["probability"]
            ax.text(
                fontweight="bold",  # that's readable in a IEEE 2cols US-letter
                ha="center",
                s=str(
                    r"\textbf{(self--trans.:\:"
                    f"{round(prob*100):.0f}"
                    r"\%)}"
                    if _USETEX
                    else f"(self-trans.: {round(prob*100):.0f}%)"
                ),
                size=24,
                va="center",
                x=x + 1.25 * sign(x) / 7,
                y=y + 16 * sign(y) / 40,
            )
            # since we wrote the probability in the label there is no
            # reason to also draw a fat self-edge; let us remove it
            graph.remove_edge(node, node)

    # Draw edges between different nodes
    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        alpha=0.75,
        arrowsize=25,
        arrowstyle="->",
        ax=ax,
        connectionstyle="arc3,rad=0.15",
        edgelist=[(a, b) for a, b in graph.edges if a != b],  # a -> b
        min_source_margin=45 if selfedges else 32,
        min_target_margin=45 if selfedges else 24,
        width=[
            probability * int(64 if selfedges else 100)
            for a, b, probability in graph.edges(data="probability")
            if a != b
        ],
    )

    # Draw self loops
    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        alpha=0.75,
        arrowstyle="-",
        ax=ax,
        connectionstyle=", ".join(
            (
                "arc",
                "angleA=170",
                "angleB=190",
                "armA=16",
                "armB=16",
                "rad=0.1",
            )
        ),
        edgelist=[(a, b) for a, b in graph.edges if a == b],  # a -> a
        width=[
            64 * probability
            for a, b, probability in graph.edges(data="probability")
            if a == b
        ],
    )
    for extension in ("svg",):
        plt.savefig(
            f"./plots/transition_probabilities.{extension}"
        )


def sign(num):
    if num < -EPSILON:
        return -1
    elif num > EPSILON:
        return +1
    return 0


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
