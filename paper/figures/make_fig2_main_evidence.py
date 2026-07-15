#!/usr/bin/env python3
"""Generate Figure 2: two-regime FocalSwap main evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "source_data"


COLORS = {
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#E5E7EB",
    "kuai": "#0F766E",
    "qilin": "#2563EB",
    "warn": "#D97706",
    "bad": "#DC2626",
    "neutral": "#94A3B8",
    "gate": "#EA580C",
    "oracle": "#059669",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "").strip()
    return None if val == "" else float(val)


def fmt3(x: float, _pos: int | None = None) -> str:
    if abs(x) < 0.0005:
        return "0"
    return f"{x:.3f}"


def fmt3_signed(x: float) -> str:
    if abs(x) < 0.0005:
        return "0"
    return f"{x:+.3f}"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.08,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_panel_a(ax, rows: list[dict[str, str]]) -> None:
    panel_label(ax, "(a)", "Calibrated opportunity")
    y = np.arange(len(rows))[::-1]
    labels = [r["regime"] for r in rows]
    colors = [COLORS["kuai"], COLORS["qilin"]]
    margins = [f(r, "margin_vs_p95") for r in rows]

    ax.axvline(0, color=COLORS["ink"], lw=0.9)
    for yi, row, color, margin in zip(y, rows, colors, margins):
        assert margin is not None
        if row["regime"] == "KuaiSAR":
            ax.scatter([margin], [yi], s=54, color=color, zorder=3, edgecolor="white", linewidth=0.7)
            ax.text(margin + 0.002, yi + 0.10, fmt3_signed(margin), ha="left", va="center", fontsize=7.0, color=color, fontweight="bold")
        else:
            ax.scatter([margin], [yi], s=58, facecolor="white", edgecolor=color, linewidth=1.4, zorder=3)
            ax.text(
                margin - 0.0012,
                yi + 0.12,
                f"p95 {fmt3_signed(margin)}",
                ha="right",
                va="center",
                fontsize=6.8,
                color=color,
                fontweight="bold",
            )
        if row["regime"] == "Qilin":
            p99_margin = f(row, "margin_vs_p99")
            assert p99_margin is not None
            ax.scatter(
                [p99_margin],
                [yi - 0.22],
                s=32,
                marker="D",
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=4,
            )
            ax.text(
                p99_margin - 0.0048,
                yi - 0.21,
                f"p99 {fmt3_signed(p99_margin)}",
                ha="right",
                va="center",
                fontsize=6.5,
                color=color,
            )

    ax.set_yticks(y, labels)
    ax.set_xlabel("Observed gain - null threshold")
    ax.set_xlim(-0.038, 0.008)
    ax.set_ylim(-0.42, 1.34)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt3))


def draw_panel_b(ax, rows: list[dict[str, str]]) -> None:
    panel_label(ax, "(b)", "Focal-search specificity")
    labels = []
    for row in rows:
        regime = "KuaiSAR" if row["regime"] == "KuaiSAR" else "Qilin"
        contrast = (
            row["contrast"]
            .replace("taxonomy share>=1", r"tax overlap $\geq$ 1")
            .replace("taxonomy share>=4 (cov 0.061)", r"tax overlap $\geq$ 4" + "\n(6% cov.)")
            .replace("category-broken source", "category-broken")
            .replace("same-category source", "same-category")
        )
        if regime == "KuaiSAR":
            labels.append(f"{regime}\n{contrast}")
        else:
            labels.append(f"{regime} {contrast}")
    y = np.arange(len(rows))[::-1]

    ax.axvline(0, color=COLORS["ink"], lw=0.9)
    for yi, row in zip(y, rows):
        color = COLORS["kuai"] if row["regime"] == "KuaiSAR" else COLORS["qilin"]
        margin = f(row, "margin")
        lo = f(row, "margin_low")
        hi = f(row, "margin_high")
        assert margin is not None and lo is not None and hi is not None
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.0, alpha=0.85)
        ax.scatter([margin], [yi], s=44, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        label_x = hi + 0.006
        label_y = yi
        if abs(margin) < 0.0015:
            label_x = max(label_x, 0.010)
        if row["regime"] == "Qilin" and "share>=4" in row["contrast"]:
            label_x = margin + 0.014
            label_y = yi + 0.18
        ax.text(
            label_x,
            label_y,
            fmt3_signed(margin),
            ha="left",
            va="center",
            fontsize=6.8,
            color=color,
        )

    ax.set_yticks(y, labels)
    ax.set_xlabel("Specificity margin")
    ax.set_xlim(-0.030, 0.146)
    ax.set_ylim(-0.55, len(rows) - 0.38)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt3))


def draw_panel_c(ax, rows: list[dict[str, str]]) -> None:
    panel_label(ax, "(c)", "Input-side recovery")
    labels = [r["setting"] for r in rows]
    scores = [f(r, "score") for r in rows]
    kinds = [r["kind"] for r in rows]
    assert all(s is not None for s in scores)
    y = np.arange(len(rows))[::-1]

    color_map = {
        "upper": COLORS["oracle"],
        "baseline": COLORS["neutral"],
        "gate": COLORS["gate"],
        "control": COLORS["bad"],
    }
    ax.axvline(0.242, color=COLORS["neutral"], lw=1.1, linestyle="--")

    for yi, label, score, kind in zip(y, labels, scores, kinds):
        color = color_map[kind]
        ax.scatter([score], [yi], s=44, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(score + 0.006, yi, f"{score:.3f}", ha="left", va="center", fontsize=6.8, color=color)

    ax.set_yticks(y, labels)
    ax.set_xlabel("RR on stress surface")
    ax.set_xlim(0.13, 0.310)
    ax.set_ylim(-0.9, len(rows) - 0.38)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt3))


def main() -> None:
    setup_style()
    calibration = read_csv(DATA_DIR / "fig2_calibration.csv")
    specificity = read_csv(DATA_DIR / "fig2_specificity.csv")
    gate = read_csv(DATA_DIR / "fig2_gate.csv")

    fig = plt.figure(figsize=(7.35, 2.35), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.33, 1.48], wspace=0.74)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]

    draw_panel_a(axes[0], calibration)
    draw_panel_b(axes[1], specificity)
    draw_panel_c(axes[2], gate)

    for ax in axes:
        ax.tick_params(axis="both", length=3, color=COLORS["muted"])
        ax.spines["left"].set_color(COLORS["muted"])
        ax.spines["bottom"].set_color(COLORS["muted"])

    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.26)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"fig2_main_evidence.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
