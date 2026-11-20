"""Generate the benchmark figures from `benchmarks/average_benchmark.md`.

The markdown file is the single source of truth: every figure in `images/` is
derived from it. To update the figures after adding a device, just run:

    python mlx_benchmark/run_viz.py

Other options:

    python mlx_benchmark/run_viz.py --check   # only parse & validate the markdown
    python mlx_benchmark/run_viz.py --out DIR # write figures somewhere else

Only matplotlib is required (`pip install matplotlib`; it ships numpy and Pillow).

Outputs (each in a light and a `-dark` variant):
    images/mlx_vs_mps_heatmap.png   MLX GPU vs PyTorch MPS, every chip x every op
    images/chip_ranking.png         Overall MLX GPU speed of each chip vs the M1
    images/ops/<Operation>.png      Runtime of one op across every chip / GPU
    images/README.md                Gallery page linking all of the above
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "benchmarks" / "average_benchmark.md"
DEFAULT_OUTPUT = REPO_ROOT / "images"

DPI = 160
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

# Shorter display names for long operation names.
OP_ALIASES = {"ScaledDotProductAttention": "SDPA"}

# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"mlx_gpu": "#2a78d6", "mps": "#eb6834", "cuda": "#1baf7a"},
        # Diverging scale: MPS faster (red) <- neutral gray -> MLX faster (blue)
        "diverging": ["#9c2b2b", "#e34948", "#f0efec", "#2a78d6", "#104281"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"mlx_gpu": "#3987e5", "mps": "#d95926", "cuda": "#199e70"},
        "diverging": ["#f19a9a", "#c94848", "#383835", "#2f74c9", "#9ec5f4"],
    },
}

SERIES_LABELS = {
    "mlx_gpu": "MLX (GPU)",
    "mps": "PyTorch (MPS)",
    "cuda": "PyTorch (CUDA)",
}

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


class BenchmarkFormatError(ValueError):
    pass


@dataclass
class Device:
    header: str
    kind: str  # "apple" | "cuda"
    name: str
    line: int
    gpu_cores: int | None = None
    mlx_version: str | None = None
    results: dict[str, dict[str, float]] = field(default_factory=dict)
    label: str = ""

    def value(self, op: str, backend: str) -> float | None:
        return self.results.get(op, {}).get(backend)


APPLE_NAME_RE = re.compile(r"\b(M\d+)(?:\s+(Pro|Max|Ultra))?\b")
GPU_RE = re.compile(r"(\d+)\s*GPU", re.IGNORECASE)
MLX_RE = re.compile(r"mlx\s*:?\s*v?(\d+(?:\.\d+)*)", re.IGNORECASE)
TIERS = {None: 0, "Pro": 1, "Max": 2, "Ultra": 3}


def _clean_header(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("**", "").replace("`", "")).strip()


def _parse_device(header: str, section: str, line_no: int) -> Device:
    if section == "cuda":
        name = header.split("(")[0].strip()
        # Collapse accidental repeated words, e.g. "A100 80GB 80GB PCIe".
        words: list[str] = []
        for w in name.split():
            if not words or words[-1] != w:
                words.append(w)
        name = " ".join(words) or header
        return Device(header=header, kind="cuda", name=name, line=line_no)

    m = APPLE_NAME_RE.search(header)
    if not m:
        raise BenchmarkFormatError(
            f"line {line_no}: could not find an Apple chip name (e.g. 'M3 Max') "
            f"in table title {header!r}"
        )
    gpu = GPU_RE.search(header)
    mlx = MLX_RE.search(header)
    return Device(
        header=header,
        kind="apple",
        name=" ".join(p for p in m.groups() if p),
        line=line_no,
        gpu_cores=int(gpu.group(1)) if gpu else None,
        mlx_version=mlx.group(1) if mlx else None,
    )


def _parse_value(cell: str, line_no: int) -> float | None:
    cell = cell.strip()
    if cell.lower() in {"", "nan", "-", "n/a"}:
        return None
    try:
        v = float(cell)
    except ValueError:
        raise BenchmarkFormatError(
            f"line {line_no}: expected a number, got {cell!r}"
        ) from None
    # A runtime that rounds to 0.00 ms is not a usable measurement.
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_benchmark(text: str) -> list[Device]:
    devices: list[Device] = []
    section = "apple"
    pending_header: tuple[str, int] | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        line_no = i + 1

        if stripped.startswith("## "):
            section = "cuda" if "cuda" in stripped.lower() else "apple"
            pending_header = None
        elif stripped.startswith("|"):
            columns = _split_row(stripped)
            if columns[0].lower() != "operation":
                raise BenchmarkFormatError(
                    f"line {line_no}: table must start with an '| Operation | ...' header row"
                )
            if pending_header is None:
                raise BenchmarkFormatError(
                    f"line {line_no}: table has no device title above it "
                    "(e.g. '**M4 Pro (4E+10P+20GPU+24GB)** mlx: 0.24.1')"
                )
            device = _parse_device(pending_header[0], section, pending_header[1])
            backends = {
                idx: col
                for idx, col in enumerate(columns)
                if idx > 0 and "speedup" not in col.lower()
            }
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_no = i + 1
                cells = _split_row(lines[i])
                i += 1
                if set("".join(cells)) <= set("-: "):
                    continue  # separator row
                if len(cells) != len(columns):
                    raise BenchmarkFormatError(
                        f"line {row_no}: expected {len(columns)} cells, got {len(cells)}"
                    )
                device.results[cells[0]] = {
                    col: v
                    for idx, col in backends.items()
                    if (v := _parse_value(cells[idx], row_no)) is not None
                }
            devices.append(device)
            pending_header = None
            continue
        elif stripped.startswith("**"):
            # Bold line: the usual device title, e.g. "**M4 Pro (...)** mlx: 0.24.1"
            pending_header = (_clean_header(stripped), line_no)
        elif pending_header is None and stripped and not stripped.startswith(
            ("#", "*", "-", ">")
        ):
            # Plain-text title, only if no bold title is waiting for its table
            # (so stray notes like "Average benchmark:" don't replace it).
            pending_header = (_clean_header(stripped), line_no)
        i += 1

    if not devices:
        raise BenchmarkFormatError("no benchmark tables found")
    _assign_labels(devices)
    return _sorted(devices)


def _version_key(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")) if v else (0,)


def _sorted(devices: list[Device]) -> list[Device]:
    def key(d: Device):
        if d.kind == "cuda":
            return (1, 0, 0, 0, (0,), d.line)
        m = APPLE_NAME_RE.search(d.name)
        gen = int(m.group(1)[1:])
        return (
            0,
            gen,
            TIERS[m.group(2)],
            d.gpu_cores or 0,
            _version_key(d.mlx_version),
            d.line,
        )

    return sorted(devices, key=key)


def _assign_labels(devices: list[Device]) -> None:
    for d in devices:
        if d.kind == "cuda":
            d.label = d.name
            continue
        parts = [d.name + (f" ({d.gpu_cores} GPU)" if d.gpu_cores else "")]
        if d.mlx_version:
            parts.append(f"mlx {d.mlx_version}")
        d.label = " · ".join(parts)
    seen: dict[str, int] = {}
    for d in devices:
        seen[d.label] = seen.get(d.label, 0) + 1
        if seen[d.label] > 1:
            d.label += f" #{seen[d.label]}"


def all_operations(devices: list[Device]) -> list[str]:
    ops: list[str] = []
    for d in devices:
        for op in d.results:
            if op not in ops:
                ops.append(op)
    return sorted(ops, key=str.lower)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def _style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 9,
            "text.color": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "axes.edgecolor": theme["axis"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["ink2"],
            "xtick.labelcolor": theme["muted"],
            "ytick.labelcolor": theme["ink2"],
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
        }
    )


def _fmt_ms(v: float) -> str:
    if v < 10:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def _fmt_ratio(r: float) -> str:
    return f"{r:.1f}×" if r < 10 else f"{r:.0f}×"


def _header(fig, theme, title: str, subtitle: str, x: float) -> None:
    h = fig.get_figheight()
    fig.text(x, 1 - 0.32 / h, title, fontsize=15, fontweight="bold",
             color=theme["ink"], va="baseline")
    fig.text(x, 1 - 0.58 / h, subtitle, fontsize=9.5, color=theme["ink2"],
             va="baseline")


def _footer(fig, theme, text: str, x: float) -> None:
    fig.text(x, 0.18 / fig.get_figheight(), text, fontsize=7.5,
             color=theme["muted"], va="baseline")


def _label_width_in(fig, labels) -> float:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer).width for t in labels if t.get_text()]
    return (max(widths) if widths else 0) / fig.dpi


def _rounded_hbar(ax, y_top: float, height: float, width: float, color: str,
                  rx: float, ry: float) -> None:
    """Horizontal bar from x=0, square at the baseline, rounded at the data end."""
    if width <= 0:
        return
    rx = min(rx, width)
    ry = min(ry, height / 2)
    k = 0.5523  # cubic Bezier quarter-circle constant
    y0, y1 = y_top, y_top + height
    verts = [
        (0, y0), (width - rx, y0),
        (width - rx + k * rx, y0), (width, y0 + ry - k * ry), (width, y0 + ry),
        (width, y1 - ry),
        (width, y1 - ry + k * ry), (width - rx + k * rx, y1), (width - rx, y1),
        (0, y1), (0, y0),
    ]
    C = MplPath
    codes = [C.MOVETO, C.LINETO, C.CURVE4, C.CURVE4, C.CURVE4, C.LINETO,
             C.CURVE4, C.CURVE4, C.CURVE4, C.LINETO, C.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=3))


def _px_per_unit(ax) -> tuple[float, float]:
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)


def _save(fig, path: Path) -> None:
    """Save as a palette PNG: flat chart colors quantize losslessly-looking and
    the files stay ~4x smaller, which matters since they live in git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        path, optimize=True
    )


# --------------------------------------------------------------------------- #
# Figure 1: one chart per operation
# --------------------------------------------------------------------------- #


def plot_operation(devices: list[Device], op: str, theme: dict, path: Path) -> bool:
    rows: list[tuple[str, Device | None, list[str]]] = []
    for kind, title in (("apple", "Apple Silicon"), ("cuda", "CUDA GPUs")):
        group = []
        for d in devices:
            if d.kind != kind:
                continue
            backends = ["mlx_gpu", "mps"] if kind == "apple" else ["cuda"]
            present = [b for b in backends if d.value(op, b) is not None]
            if present:
                group.append((d.label, d, backends))
        if group:
            rows.append((title, None, []))
            rows.extend(group)
    if not any(d for _, d, _ in rows):
        return False

    _style(theme)
    row_h = 0.36
    top_in, bottom_in = 1.25, 0.85
    width = 10.0
    height = top_in + bottom_in + row_h * len(rows)
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = fig.add_axes([0.2, 0.1, 0.6, 0.8])

    n = len(rows)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=8.5)
    for tick, (_, d, _) in zip(ax.get_yticklabels(), rows):
        if d is None:
            tick.set_fontweight("bold")
            tick.set_color(theme["ink"])
            tick.set_fontsize(9)
    ax.tick_params(axis="y", length=0, pad=8)

    values = [d.value(op, b) for _, d, bs in rows if d for b in bs]
    vmax = max(v for v in values if v is not None)

    # Layout: measured left margin for labels, right column for "faster" text.
    label_w = _label_width_in(fig, ax.get_yticklabels())
    left = 0.25 + label_w + 0.12
    right_col = 1.15
    ax.set_position([
        left / width,
        bottom_in / height,
        (width - left - right_col - 0.25) / width,
        (height - top_in - bottom_in) / height,
    ])
    ax.set_xlim(0, vmax * 1.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(theme["axis"])
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, labelsize=8, pad=4)
    ax.set_xlabel("Runtime (ms) · lower is better", fontsize=8.5,
                  color=theme["muted"], labelpad=8)

    px_x, px_y = _px_per_unit(ax)
    gap = 2 / px_y  # 2px surface gap between paired bars
    bar_h = min(0.34, (0.78 - gap) / 2)
    rx, ry = 4 / px_x, 4 / px_y

    right_x = 1 + 0.15 / (ax.get_position().width * width)
    ax.text(right_x, -0.5 + 0.0, "Faster backend", transform=ax.get_yaxis_transform(),
            fontsize=8, color=theme["muted"], va="bottom", ha="left")

    for y, (_, d, backends) in enumerate(rows):
        if d is None:
            continue
        slots = len(backends)
        total = slots * bar_h + (slots - 1) * gap
        for j, b in enumerate(backends):
            v = d.value(op, b)
            y_top = y - total / 2 + j * (bar_h + gap)
            if v is None:
                ax.text(4 / px_x, y_top + bar_h / 2, "n/a", fontsize=7,
                        color=theme["muted"], va="center", ha="left")
                continue
            _rounded_hbar(ax, y_top, bar_h, v, theme["series"][b], rx, ry)
            ax.text(v + 5 / px_x, y_top + bar_h / 2, _fmt_ms(v), fontsize=7,
                    color=theme["ink2"], va="center", ha="left", zorder=4)

        mlx, mps = d.value(op, "mlx_gpu"), d.value(op, "mps")
        if mlx and mps:
            winner, ratio = ("MLX", mps / mlx) if mlx <= mps else ("MPS", mlx / mps)
            ax.text(right_x, y, f"{winner} {_fmt_ratio(ratio)}",
                    transform=ax.get_yaxis_transform(), fontsize=8,
                    fontweight="bold" if winner == "MLX" else "normal",
                    color=theme["ink"] if winner == "MLX" else theme["ink2"],
                    va="center", ha="left")

    x0 = 0.25 / width
    _header(fig, theme, op, "Average runtime of each backend, per chip / GPU", x0)
    used = [b for b in ("mlx_gpu", "mps", "cuda")
            if any(b in bs for _, d, bs in rows if d)]
    handles = [Patch(facecolor=theme["series"][b], label=SERIES_LABELS[b]) for b in used]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               bbox_to_anchor=(x0 - 0.008, 1 - 0.72 / height), fontsize=8.5,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6,
               labelcolor=theme["ink2"])
    _footer(fig, theme, "Source: benchmarks/average_benchmark.md · generated by "
            "mlx_benchmark/run_viz.py", x0)
    _save(fig, path)
    return True


# --------------------------------------------------------------------------- #
# Figure 2: MLX GPU vs MPS heatmap
# --------------------------------------------------------------------------- #


def plot_heatmap(devices: list[Device], theme: dict, path: Path) -> None:
    apple = [d for d in devices if d.kind == "apple"]
    ops = [
        op for op in all_operations(apple)
        if any(d.value(op, "mlx_gpu") and d.value(op, "mps") for d in apple)
    ]
    clip = 3.0  # log2 -> 8x
    grid = []
    for d in apple:
        row = []
        for op in ops:
            mlx, mps = d.value(op, "mlx_gpu"), d.value(op, "mps")
            row.append(max(-clip, min(clip, math.log2(mps / mlx))) if mlx and mps else None)
        grid.append(row)

    _style(theme)
    cell_w, cell_h = 0.5, 0.3
    top_in, bottom_in = 1.95, 0.5
    fig = plt.figure(figsize=(10, 10), dpi=DPI)
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.7])
    ax.set_yticks(range(len(apple)))
    ax.set_yticklabels([d.label for d in apple], fontsize=8.5)
    label_w = _label_width_in(fig, ax.get_yticklabels())

    left = 0.25 + label_w + 0.12
    width = left + cell_w * len(ops) + 0.3
    height = top_in + bottom_in + cell_h * len(apple)
    fig.set_size_inches(width, height)
    ax.set_position([left / width, bottom_in / height,
                     cell_w * len(ops) / width, cell_h * len(apple) / height])

    cmap = LinearSegmentedColormap.from_list("div", theme["diverging"])
    surface = to_rgb(theme["surface"])
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v is None:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=surface,
                                           edgecolor=theme["grid"], linewidth=0.6,
                                           zorder=1))
                continue
            color = cmap((v + clip) / (2 * clip))
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color,
                                       edgecolor=surface, linewidth=1.4, zorder=2))
            r, g, b = color[:3]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "#ffffff" if lum < 0.45 else "#0b0b0b"
            ratio = 2 ** abs(v)
            label = _fmt_ratio(ratio) if abs(v) < clip else f"≥{_fmt_ratio(2**clip)}"
            ax.text(x, y, label, ha="center", va="center", fontsize=6.5,
                    color=text_color, zorder=3)

    ax.set_xlim(-0.5, len(ops) - 0.5)
    ax.set_ylim(len(apple) - 0.5, -0.5)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([OP_ALIASES.get(o, o) for o in ops], rotation=50,
                       ha="left", rotation_mode="anchor", fontsize=8.5,
                       color=theme["ink2"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)

    x0 = 0.25 / width
    _header(fig, theme, "MLX vs PyTorch MPS on Apple Silicon GPUs",
            "Runtime ratio per operation · each cell names the faster backend's "
            "speedup", x0)

    # Diverging legend bar
    lw, lh = 3.2, 0.11
    cax = fig.add_axes([x0, 1 - 1.0 / height, lw / width, lh / height])
    steps = 256
    for i in range(steps):
        cax.add_patch(plt.Rectangle((i / steps, 0), 1 / steps + 1e-3, 1,
                                    facecolor=cmap(i / (steps - 1)), edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    cax.set_xticks([(t + clip) / (2 * clip) for t in ticks])
    cax.set_xticklabels([f"{2**abs(t):.0f}×" for t in ticks], fontsize=7.5)
    cax.tick_params(length=0, pad=3, labelcolor=theme["muted"])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(x0, 1 - 0.80 / height, "MPS faster", fontsize=8,
             color=theme["ink2"], va="baseline")
    fig.text(x0 + lw / width, 1 - 0.80 / height, "MLX faster", fontsize=8,
             color=theme["ink2"], va="baseline", ha="right")

    _footer(fig, theme, "Empty cell: op not benchmarked on that entry, or runtime "
            "rounded to 0.00 ms · source: benchmarks/average_benchmark.md", x0)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 3: chip ranking (geometric-mean MLX GPU speedup vs the first chip)
# --------------------------------------------------------------------------- #


def plot_ranking(devices: list[Device], theme: dict, path: Path) -> None:
    apple = [d for d in devices if d.kind == "apple"]
    common = [op for op in all_operations(apple)
              if all(d.value(op, "mlx_gpu") for d in apple)]
    if not common:
        return
    base = apple[0]
    scores = []
    for d in apple:
        logs = [math.log(base.value(op, "mlx_gpu") / d.value(op, "mlx_gpu"))
                for op in common]
        scores.append((d, math.exp(sum(logs) / len(logs))))
    scores.sort(key=lambda s: s[1], reverse=True)

    _style(theme)
    row_h = 0.3
    top_in, bottom_in = 1.0, 0.85
    width = 10.0
    height = top_in + bottom_in + row_h * len(scores)
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
    n = len(scores)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d.label for d, _ in scores], fontsize=8.5)
    for tick, (d, _) in zip(ax.get_yticklabels(), scores):
        if d is base:
            tick.set_color(theme["muted"])
    ax.tick_params(axis="y", length=0, pad=8)
    label_w = _label_width_in(fig, ax.get_yticklabels())
    left = 0.25 + label_w + 0.12
    ax.set_position([left / width, bottom_in / height,
                     (width - left - 0.4) / width,
                     (height - top_in - bottom_in) / height])
    vmax = max(s for _, s in scores)
    ax.set_xlim(0, vmax * 1.1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(theme["axis"])
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, labelsize=8, pad=4)
    ax.set_xlabel(f"Speedup vs {base.name} · higher is better", fontsize=8.5,
                  color=theme["muted"], labelpad=8)

    px_x, px_y = _px_per_unit(ax)
    bar_h = 0.62
    for y, (d, s) in enumerate(scores):
        color = theme["muted"] if d is base else theme["series"]["mlx_gpu"]
        _rounded_hbar(ax, y - bar_h / 2, bar_h, s, color, 4 / px_x, 4 / px_y)
        ax.text(s + 5 / px_x, y, _fmt_ratio(s) if s >= 10 else f"{s:.2f}×",
                fontsize=7.5, color=theme["ink2"], va="center", ha="left")

    x0 = 0.25 / width
    _header(fig, theme, "MLX GPU speed by chip",
            f"Geometric-mean speedup over {base.label} across the {len(common)} "
            "operations every chip ran", x0)
    _footer(fig, theme, "MLX and PyTorch versions differ between entries · "
            "source: benchmarks/average_benchmark.md", x0)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Gallery page
# --------------------------------------------------------------------------- #


def _picture(src: str, alt: str) -> str:
    dark = src.replace(".png", "-dark.png")
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
        f'  <img alt="{alt}" src="{src}">\n'
        "</picture>"
    )


def write_gallery(ops: list[str], out_dir: Path) -> None:
    lines = [
        "# Benchmark figures",
        "",
        "<!-- Generated by mlx_benchmark/run_viz.py. Do not edit by hand. -->",
        "",
        "All figures are generated from "
        "[average_benchmark.md](../benchmarks/average_benchmark.md) by running "
        "`python mlx_benchmark/run_viz.py`.",
        "",
        "## Overview",
        "",
        _picture("mlx_vs_mps_heatmap.png", "MLX vs MPS heatmap"),
        "",
        _picture("chip_ranking.png", "MLX GPU speed by chip"),
        "",
        "## Per operation",
        "",
        " · ".join(f"[{op}](#{op.lower()})" for op in ops),
        "",
    ]
    for op in ops:
        lines += [f"### {op}", "", _picture(f"ops/{op}.png", f"{op} benchmark"), ""]
    (out_dir / "README.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate(devices: list[Device], out_dir: Path) -> list[Path]:
    ops_dir = out_dir / "ops"
    if ops_dir.exists():
        shutil.rmtree(ops_dir)  # drop charts of operations that no longer exist
    written: list[Path] = []
    plotted_ops: list[str] = []
    for mode, theme in THEMES.items():
        suffix = "" if mode == "light" else "-dark"
        p = out_dir / f"mlx_vs_mps_heatmap{suffix}.png"
        plot_heatmap(devices, theme, p)
        written.append(p)
        p = out_dir / f"chip_ranking{suffix}.png"
        plot_ranking(devices, theme, p)
        written.append(p)
        for op in all_operations(devices):
            p = ops_dir / f"{op}{suffix}.png"
            if plot_operation(devices, op, theme, p):
                written.append(p)
                if mode == "light":
                    plotted_ops.append(op)
    write_gallery(plotted_ops, out_dir)
    written.append(out_dir / "README.md")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="only parse and validate the markdown")
    args = parser.parse_args()

    try:
        devices = parse_benchmark(args.input.read_text())
    except BenchmarkFormatError as e:
        print(f"error: {args.input.name}: {e}", file=sys.stderr)
        return 1

    print(f"Parsed {len(devices)} devices, {len(all_operations(devices))} operations:")
    for d in devices:
        print(f"  {d.kind:5}  {d.label:40}  {len(d.results):2d} ops  (line {d.line})")
    if args.check:
        return 0

    written = generate(devices, args.out)
    print(f"Wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
