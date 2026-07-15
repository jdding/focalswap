#!/usr/bin/env python3
"""Render the checkpoint control-sensitivity forest plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR / "source_data" / "fig4_control_sensitivity.csv"

COLORS = {
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#D1D5DB",
    "focal": "#475569",
    "C0 neutral-mask": "#2563EB",
    "C1 profile-prioritized donor": "#94A3B8",
    "C2 category/profile-matched donor": "#0F766E",
    "paired C0-C2 difference": "#D97706",
}

MARKERS = {
    "C0 neutral-mask": "o",
    "C1 profile-prioritized donor": "s",
    "C2 category/profile-matched donor": "^",
    "paired C0-C2 difference": "D",
}

DISPLAY_CONDITIONS = {
    "C0 neutral-mask": "C0 neutral-mask",
    "C1 profile-prioritized donor": "C1 profile-prioritized donor",
    "C2 category/profile-matched donor": "C2 category/profile donor",
    "paired C0-C2 difference": "Paired C0-C2 difference",
}


def read_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def point_range(
    ax,
    row: dict[str, str],
    y: float,
    color: str,
    marker: str = "o",
    annotate: bool = True,
) -> None:
    estimate = float(row["estimate_x1e3"])
    low = float(row["ci_low_x1e3"])
    high = float(row["ci_high_x1e3"])
    ax.plot([low, high], [y, y], color=color, lw=1.8, solid_capstyle="round", zorder=2)
    ax.scatter([estimate], [y], s=34, marker=marker, color=color, edgecolor="white", linewidth=0.65, zorder=3)
    if annotate:
        ax.annotate(
            f"{estimate:+.3f}",
            xy=(estimate, y),
            xytext=(5 if estimate >= 0 else -5, 3),
            textcoords="offset points",
            ha="left" if estimate >= 0 else "right",
            va="bottom",
            fontsize=6.0,
            color=color,
            clip_on=False,
        )


def style_axis(ax, xlabel: str) -> None:
    ax.axvline(0, color=COLORS["ink"], lw=0.9, zorder=1)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=COLORS["muted"])
    ax.set_xlabel(xlabel)


def draw_focal(ax, rows: list[dict[str, str]]) -> None:
    labels = ["SESRec", "UniSAR S1", "UniSAR S2", "UniSAR S3"]
    y_values = list(range(len(rows) - 1, -1, -1))
    for row, y in zip(rows, y_values):
        point_range(ax, row, y, COLORS["focal"])

    ax.set_yticks(y_values, labels)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.text(0.0, 1.23, "(a) Latest-slot intervention", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=COLORS["ink"])
    style_axis(ax, r"Dependence margin $\Delta$RR ($\times 10^{-3}$)")


def draw_controls(ax, rows: list[dict[str, str]]) -> None:
    condition_order = [
        "C0 neutral-mask",
        "C1 profile-prioritized donor",
        "C2 category/profile-matched donor",
        "paired C0-C2 difference",
    ]
    seed_order = ["S1", "S2", "S3"]
    seed_y = {"S1": 2.0, "S2": 1.0, "S3": 0.0}
    offsets = {
        "C0 neutral-mask": 0.27,
        "C1 profile-prioritized donor": 0.09,
        "C2 category/profile-matched donor": -0.09,
        "paired C0-C2 difference": -0.27,
    }

    for seed in seed_order:
        for condition in condition_order:
            row = next(r for r in rows if r["seed_label"] == seed and r["condition"] == condition)
            point_range(
                ax,
                row,
                seed_y[seed] + offsets[condition],
                COLORS[condition],
                marker=MARKERS[condition],
                annotate=condition == "paired C0-C2 difference",
            )

    for separator in (0.5, 1.5):
        ax.axhline(separator, color=COLORS["grid"], lw=0.6, zorder=0)

    ax.set_yticks([2.0, 1.0, 0.0], ["Seed 1", "Seed 2", "Seed 3"])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(-3.1, 8.35)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.text(0.0, 1.23, "(b) All-source controls", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=COLORS["ink"])

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[condition],
            marker=MARKERS[condition],
            lw=1.5,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=DISPLAY_CONDITIONS[condition],
        )
        for condition in condition_order
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    style_axis(ax, r"Dependence margin $\Delta$RR ($\times 10^{-3}$)")


def main() -> None:
    setup_style()
    rows = read_rows()
    focal = [row for row in rows if row["panel"] == "focal"]
    controls = [row for row in rows if row["panel"] == "control"]

    fig = plt.figure(figsize=(7.35, 2.55), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.80, 1.55], wspace=0.48)
    ax_focal = fig.add_subplot(grid[0, 0])
    ax_controls = fig.add_subplot(grid[0, 1])
    draw_focal(ax_focal, focal)
    draw_controls(ax_controls, controls)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.76, bottom=0.22)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"fig4_control_sensitivity.{extension}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
