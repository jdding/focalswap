#!/usr/bin/env python3
"""Generate Figure 4: Qilin swap-tightness and random-placebo specificity."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from focalswap_figure_style import FONT, TITLE_COLOR, apply_figure_style


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR / "source_data" / "fig4_swap_specificity.csv"

COLORS = {
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#E5E7EB",
    "qilin": "#2563EB",
    "local": "#0F766E",
    "sig": "#B45309",
    "zero": "#94A3B8",
}

ORDER = [
    "same_user/all",
    "time_near/all",
    "time_near/<=300s",
    "time_near/<=120s",
    "time_near/<=60s",
    "time_near/<=30s",
    "tax_near/share>=1",
    "tax_near/share>=2",
    "tax_near/share>=3",
    "tax_near/share>=4",
]

LABELS = {
    "same_user/all": "same-user donor",
    "time_near/all": "time-near donor (all)",
    "time_near/<=300s": "within 5 min",
    "time_near/<=120s": "within 2 min",
    "time_near/<=60s": "within 1 min",
    "time_near/<=30s": "within 30 s",
    "tax_near/share>=1": r"tax overlap $\geq$ 1",
    "tax_near/share>=2": r"tax overlap $\geq$ 2",
    "tax_near/share>=3": r"tax overlap $\geq$ 3",
    "tax_near/share>=4": r"tax overlap $\geq$ 4",
}


def fmt3(x: float, _pos: int | None = None) -> str:
    if abs(x) < 0.0005:
        return "0"
    return f"{x:.3f}"


def fmt3_signed(x: float) -> str:
    if abs(x) < 0.0005:
        return "0"
    return f"{x:+.3f}"


def setup_style() -> None:
    apply_figure_style(COLORS["grid"])


def load_rows() -> list[dict[str, object]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        by_match = {row["match"]: row for row in csv.DictReader(source)}
    if set(by_match) != set(ORDER):
        raise ValueError("Expected all ten Qilin comparison buckets.")

    rows = []
    for match in ORDER:
        row = by_match[match]
        rows.append(
            {
                "match": match,
                "coverage": float(row["coverage"]),
                "half_tie": float(row["half_tie"]),
                "half_tie_ci": (float(row["half_tie_low"]), float(row["half_tie_high"])),
                "delta_vs_random_placebo": float(row["delta_vs_random_placebo"]),
                "delta_vs_random_placebo_ci": (float(row["delta_low"]), float(row["delta_high"])),
            }
        )
    return rows


def panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.06,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT["panel"],
        fontweight="bold",
        color=TITLE_COLOR,
    )


def point_label(ax, x: float, y: float, text: str, color: str, alpha: float) -> None:
    xmin, xmax = ax.get_xlim()
    offset = 0.035 * (xmax - xmin)
    right_guard = xmax - 0.18 * (xmax - xmin)
    if x < 0:
        ax.annotate(
            text,
            xy=(x - 0.010 * (xmax - xmin), y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=FONT["annotation"],
            color=color,
            alpha=alpha,
            clip_on=False,
        )
    elif x >= right_guard:
        ax.annotate(
            text,
            xy=(x - offset, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=FONT["annotation"],
            color=color,
            alpha=alpha,
            clip_on=True,
        )
    else:
        ax.annotate(
            text,
            xy=(x + offset, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FONT["annotation"],
            color=color,
            alpha=alpha,
            clip_on=True,
        )


def main() -> None:
    setup_style()
    rows = load_rows()
    labels = [LABELS[r["match"]] for r in rows]
    y = np.arange(len(rows))[::-1]
    half = np.array([r["half_tie"] for r in rows], dtype=float)
    half_lo = np.array([r["half_tie_ci"][0] for r in rows], dtype=float)
    half_hi = np.array([r["half_tie_ci"][1] for r in rows], dtype=float)
    delta = np.array([r["delta_vs_random_placebo"] for r in rows], dtype=float)
    delta_lo = np.array([r["delta_vs_random_placebo_ci"][0] for r in rows], dtype=float)
    delta_hi = np.array([r["delta_vs_random_placebo_ci"][1] for r in rows], dtype=float)
    coverage = np.array([r["coverage"] for r in rows], dtype=float)

    # Ten rows are required by the multiplicity contract. Compress their
    # vertical rhythm instead of dropping intermediate buckets.
    fig = plt.figure(figsize=(7.35, 3.30), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    panel_label(ax0, "(a)", "Matched-swap half-tie")
    ax0.set_xlim(0.478, 0.614)
    ax0.set_ylim(-0.40, len(rows) - 0.25)
    ax0.axvline(0.5, color=COLORS["zero"], lw=1.0, linestyle="--")
    for yi, label, x, lo, hi, cov in zip(y, labels, half, half_lo, half_hi, coverage):
        color = COLORS["local"] if "tax" in label else COLORS["qilin"]
        alpha = 0.9 if cov >= 0.20 else 0.55
        ax0.plot([lo, hi], [yi, yi], color=color, lw=1.8, alpha=alpha)
        ax0.scatter([x], [yi], s=46, color=color, edgecolor="white", linewidth=0.7, zorder=3, alpha=alpha)
        point_label(ax0, x, yi, f"{x:.3f}", color, alpha)
    ax0.set_yticks(y, labels)
    ax0.set_xlabel("True-search half-tie rate (0.5 = symmetry)")
    ax0.xaxis.set_major_formatter(FuncFormatter(fmt3))

    panel_label(ax1, "(b)", "Specificity beyond placebo")
    ax1.set_xlim(-0.036, 0.146)
    ax1.set_ylim(-0.40, len(rows) - 0.25)
    ax1.axvline(0, color=COLORS["ink"], lw=0.9)
    for yi, label, x, lo, hi, cov in zip(y, labels, delta, delta_lo, delta_hi, coverage):
        if "tax" in label:
            color = COLORS["local"]
        else:
            color = COLORS["qilin"]
        alpha = 0.9 if cov >= 0.20 else 0.55
        ax1.plot([lo, hi], [yi, yi], color=color, lw=1.8, alpha=alpha)
        ax1.scatter([x], [yi], s=46, color=color, edgecolor="white", linewidth=0.7, zorder=3, alpha=alpha)
        point_label(ax1, x, yi, fmt3_signed(x), color, alpha)
    ax1.set_yticks(y, [""] * len(rows))
    ax1.set_xlabel("Half-tie lift over same-subset random")
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt3))

    for ax in (ax0, ax1):
        ax.tick_params(axis="both", length=3, color=COLORS["muted"])
        ax.spines["left"].set_color(COLORS["muted"])
        ax.spines["bottom"].set_color(COLORS["muted"])

    fig.subplots_adjust(left=0.17, right=0.985, top=0.84, bottom=0.20)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"fig4_swap_specificity.{ext}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
