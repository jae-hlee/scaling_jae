"""Time vs system size for ALIGNN-FF with energy + forces + stress (GB10, H200).

Run from this directory: `python scaling_forces.py`. Writes `scaling_forces.png`
from ../{gb10,h200}/alignn_ff/bench_*.json: one GPU, Si diamond supercells,
smooth 52-neighbour `matpes_r2scan` force field, median of three single points
per size, until the GPU ran out of memory (last plotted size = largest that fit).
Kept separate from scaling_overview.png, whose ALIGNN-FF curves are energy only.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_ff(path):
    r = json.load(open(path))["results"]
    return (np.array([x["natoms"] for x in r], float),
            np.array([x["t_median_s"] for x in r]))


gn, gt = load_ff(ROOT / "gb10" / "alignn_ff" / "bench_GB10_13474.json")
hn, ht = load_ff(ROOT / "h200" / "alignn_ff" / "bench_NVIDIA_H200_NVL_913733.json")

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
ax.loglog(gn, gt, "-D", ms=3.5, lw=1.8, color="#2ca02c", mfc="white",
          label="ALIGNN-FF, 1 GB10, energy+forces+stress")
ax.loglog(hn, ht, "-P", ms=4, lw=1.8, color="#7b3294", mfc="white",
          label="ALIGNN-FF, 1 H200, energy+forces+stress")
ax.set_xlabel("Number of atoms")
ax.set_ylabel("Wall time (s)")
ax.grid(True, which="major", color="#e4e3df")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), frameon=False,
          fontsize=8.5, ncol=1)
fig.tight_layout()
fig.savefig(HERE / "scaling_forces.png", facecolor="white", bbox_inches="tight")
