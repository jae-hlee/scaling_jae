"""Two-panel cross-GPU figure for the manuscript (ALIGNN-FF wall time vs size).

Run from this directory: `python cross_gpu_figure.py`. Writes `cross_gpu.png`.
(a) energy only with the B200 settings (Cu FCC, 12 neighbours,
v12.2.2024_dft_3d_307k): B200 from ../b200/alignn_ff/scaling_alignn_v6.npz, GB10
and H200 from ../{gb10,h200}/alignn_ff_b200settings/*.npz; time per call is
times_graph + times_inference, first row (warm-up) dropped.
(b) energy + forces + stress (Si diamond, matpes_r2scan, 52 neighbours, median
of three calls) from ../{gb10,h200}/alignn_ff/bench_*.json. There is no B200 run.
The panels use different models and structures and are not comparable.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STYLE = {"B200": ("#2a78d6", "o"), "GB10": ("#2ca02c", "D"), "H200": ("#7b3294", "P")}


def load_energy(path):
    z = np.load(path)
    return z["natoms"][1:].astype(float), (z["times_graph"] + z["times_inference"])[1:]


def load_forces(path):
    r = json.load(open(path))["results"]
    return (np.array([x["natoms"] for x in r], float),
            np.array([x["t_median_s"] for x in r]))


energy = {
    "B200": load_energy(ROOT / "b200" / "alignn_ff" / "scaling_alignn_v6.npz"),
    "GB10": load_energy(next((ROOT / "gb10" / "alignn_ff_b200settings").glob("scaling_alignn_pure_*.npz"))),
    "H200": load_energy(next((ROOT / "h200" / "alignn_ff_b200settings").glob("scaling_alignn_pure_*.npz"))),
}
forces = {
    "GB10": load_forces(ROOT / "gb10" / "alignn_ff" / "bench_GB10_13474.json"),
    "H200": load_forces(ROOT / "h200" / "alignn_ff" / "bench_NVIDIA_H200_NVL_913733.json"),
}

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=300)
for ax, data, title in [(axes[0], energy, "(a) Energy only, Cu FCC"),
                        (axes[1], forces, "(b) Energy + forces + stress, Si")]:
    for gpu, (n, t) in data.items():
        color, marker = STYLE[gpu]
        ax.loglog(n, t, "-" + marker, ms=3, lw=1.5, color=color, label=gpu)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Number of atoms")
    ax.grid(True, which="major", color="#e4e3df")
    ax.legend(loc="upper left", frameon=False)
axes[0].set_ylabel("Wall time (s)")
fig.tight_layout()
fig.savefig(HERE / "cross_gpu.png", facecolor="white", bbox_inches="tight")
