"""Shared typography and export settings for FocalSwap data figures."""

from __future__ import annotations

import matplotlib as mpl


FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
TITLE_COLOR = "#0B1F78"

FONT = {
    "base": 6.8,
    "axis": 6.8,
    "tick": 6.4,
    "legend": 6.0,
    "panel": 8.2,
    "annotation": 6.0,
    "note": 5.6,
}


def apply_figure_style(grid_color: str = "#E5E7EB") -> None:
    """Apply the common full-width-paper figure typography."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT["base"],
            "axes.labelsize": FONT["axis"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": grid_color,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
