"""Plot GR00T multi-sample latency and peak VRAM vs N from validate_multisample.py output.

uv run python experiments/aac/plot_latency.py results/aac/validation/multisample_validation.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

BLUE, MUTED, GRID, INK = "#2a78d6", "#6b6a63", "#e4e3dc", "#1a1a19"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    lat = json.loads(path.read_text())["F_latency_vs_N"]
    ns = sorted(int(k) for k in lat if k.isdigit())
    out = path.parent / "plots"
    out.mkdir(exist_ok=True)
    # Two measures with different units -> two charts (no dual axis).
    for key, ylabel, fname, title in (
        ("sample_ms_mean", "ms per query", "latency_vs_N.png", "Multi-sample GR00T latency vs N (RTX 4090)"),
        ("peak_mem_mb", "peak VRAM (MB)", "vram_vs_N.png", "Peak VRAM vs N"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ys = [lat[str(n)][key] for n in ns]
        ax.plot(ns, ys, color=BLUE, linewidth=2, marker="o", markersize=6)
        for n, y in zip(ns, ys, strict=True):
            if n in (1, 20, ns[-1]):
                ax.annotate(f"{y:.0f}", (n, y), xytext=(4, 6), textcoords="offset points", fontsize=8, color=INK)
        _style(ax, title, "number of candidate chunks N", ylabel)
        ax.set_xticks(ns)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
    print(f"wrote {out}/latency_vs_N.png, vram_vs_N.png")


if __name__ == "__main__":
    main()
