"""Shared figure style (static figures for print).

Palette, marks and chrome follow one small system: categorical hues are used in a fixed slot
order (validated for colour-vision deficiency on a white surface), every series also has its own
marker shape as a second, colour-independent channel, text is always ink-coloured, grids are
solid hairlines, and every figure has a table twin in the paper.
"""
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#898781"                       # context / reference series
# categorical slots, fixed order - never reordered, never cycled
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def setup() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def line(ax, x, y, slot: Optional[int], label: str, yerr=None, zorder: int = 3):
    """2px line with ring-separated markers; ``slot=None`` draws the neutral reference style."""
    color = NEUTRAL if slot is None else SLOTS[slot]
    marker = None if slot is None else MARKERS[slot]
    if yerr is not None:
        y_, e_ = np.asarray(y, dtype=float), np.asarray(yerr, dtype=float)
        ax.fill_between(x, y_ - e_, y_ + e_, color=color, alpha=0.10, linewidth=0, zorder=zorder - 1)
    return ax.plot(x, y, color=color, marker=marker, markersize=6.5, markeredgecolor=SURFACE,
                   markeredgewidth=1.5, label=label, zorder=zorder,
                   linewidth=1.4 if slot is None else 2.0)[0]


def end_label(ax, x, y, text: str, dy: float = 0.0, dx: float = 4.0) -> None:
    """Direct label at a line end, in ink (never in the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha="left",
                fontsize=8, color=INK2, annotation_clip=False)


def spread_labels(ys: Sequence[float], min_gap: float) -> np.ndarray:
    """Vertical positions for end labels so that neighbours are at least ``min_gap`` apart."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (pos - np.array(ys, dtype=float)[order]).mean()
    out = np.empty(len(ys))
    out[order] = pos - shift
    return out


def rounded_bar(ax, x: float, height: float, width: float, color: str, base: float = 0.0,
                radius_pt: float = 3.0, horizontal: bool = False, zorder: int = 3) -> None:
    """A bar with a rounded data end and a square baseline end."""
    fig = ax.figure
    fig.canvas.draw_idle()
    inv = ax.transData.inverted()
    px = radius_pt * fig.dpi / 72.0
    o = ax.transData.transform((0, 0))
    rx = abs(inv.transform((o[0] + px, o[1]))[0] - inv.transform(o)[0])
    ry = abs(inv.transform((o[0], o[1] + px))[1] - inv.transform(o)[1])
    if horizontal:                       # x is the category position (on the y axis)
        lo, hi, end = x - width / 2, x + width / 2, base + height
        s = 1 if height >= 0 else -1
        r_len, r_w = min(rx, abs(height)), min(ry, width / 2)
        verts = [(base, lo), (end - s * r_len, lo), (end, lo), (end, lo + r_w), (end, hi - r_w), (end, hi),
                 (end - s * r_len, hi), (base, hi), (base, lo)]
    else:
        lo, hi, end = x - width / 2, x + width / 2, base + height
        s = 1 if height >= 0 else -1
        r_len, r_w = min(ry, abs(height)), min(rx, width / 2)
        verts = [(lo, base), (lo, end - s * r_len), (lo, end), (lo + r_w, end), (hi - r_w, end), (hi, end),
                 (hi, end - s * r_len), (hi, base), (lo, base)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def legend(ax, **kw):
    kw.setdefault("handlelength", 1.8)
    kw.setdefault("labelcolor", INK2)
    return ax.legend(**kw)


def save(fig, path: str) -> None:
    if path.endswith(".pdf"):
        # no creation date: regenerating a figure from unchanged results must give the same bytes
        fig.savefig(path, metadata={"CreationDate": None})
        fig.savefig(path[:-4] + ".png", dpi=200)
    else:
        fig.savefig(path)
    plt.close(fig)
