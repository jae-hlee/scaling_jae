"""All seven scaling runs on one plot: VASP (1 GPU and 4-8 GPUs), ALIGNN-FF
energy only (B200, GB10, H200; B200 settings) and ALIGNN-FF energy+forces+stress
(GB10, H200).

Run from this directory: `python scaling_all.py`. Writes `scaling_all.png` from
the same data as scaling_overview.py and scaling_forces.py, with the same colours
and markers (energy only: filled markers, dashed for the GB10/H200 reruns;
energy+forces+stress: hollow markers, solid lines).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root (analysis/ sits at the top level)
B200 = ROOT / "b200"

# ALIGNN-FF: total time per energy evaluation (times_graph already includes nl)
d = np.load(B200 / "alignn_ff" / "scaling_alignn_v6.npz")
an = d["natoms"][1:].astype(float)  # drop N=4 (warm-up)
at = (d["times_graph"] + d["times_inference"])[1:]

# VASP: time per SCF cycle. 1-GPU runs exist only up to 432 atoms; the larger
# cells were run once each, on 4 GPUs (2,000 atoms) or 8 GPUs (3,456+).
m = json.load(open(B200 / "vasp_dft" / "analysis" / "metrics.json"))
one, big = {}, {}
for r in m["all_runs"]:
    if r["scf_done"]:
        n = 2 * r["n"] ** 3
        t = r["elapsed_s"] / r["scf_done"]
        if r["ngpu"] == 1:
            one[n] = t
        elif n > 432:  # larger than the 6x6x6 strong-scaling cell
            big[n] = t
v1n = np.array(sorted(one), dtype=float)
v1t = np.array([one[k] for k in sorted(one)])
vbn = np.array([v1n[-1]] + sorted(big), dtype=float)  # bridge from last 1-GPU point
vbt = np.array([v1t[-1]] + [big[k] for k in sorted(big)])

# The energy+forces+stress sweeps (GB10, H200) are plotted separately by
# scaling_forces.py.

# ALIGNN-FF with the B200 run's own settings (energy only, Cu, 12 neighbours,
# same checkpoint), rerun on the pure-PyTorch model; first row = warm-up, dropped
# as for the B200 series. The H200 file is added once its run has finished.
def load_b200set(gpu_dir):
    fs = sorted((ROOT / gpu_dir / "alignn_ff_b200settings").glob("scaling_alignn_pure_*.npz"))
    if not fs:
        return None
    z = np.load(fs[-1])
    return z["natoms"][1:].astype(float), (z["times_graph"] + z["times_inference"])[1:]
eg = load_b200set("gb10")
eh = load_b200set("h200")


def load_ff(path):
    r = json.load(open(path))["results"]
    return (np.array([x["natoms"] for x in r], float),
            np.array([x["t_median_s"] for x in r]))
gn, gt = load_ff(ROOT / "gb10" / "alignn_ff" / "bench_GB10_13474.json")
hn, ht = load_ff(ROOT / "h200" / "alignn_ff" / "bench_NVIDIA_H200_NVL_913733.json")

fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
ax.loglog(v1n, v1t, "-s", ms=4.5, lw=1.8, color="#eb6834", label="VASP, 1 GPU, time per SCF cycle")
ax.loglog(vbn[1:], vbt[1:], "s", ms=4.5, mfc="white", mew=1.5, color="#eb6834")
ax.loglog(vbn, vbt, "--", lw=1.8, color="#eb6834", label="VASP, 4–8 GPUs (larger cells)")
ax.loglog(gn, gt, "-D", ms=3.5, lw=1.8, color="#2ca02c", mfc="white",
          label="ALIGNN-FF, 1 GB10, energy+forces+stress")
ax.loglog(hn, ht, "-P", ms=4, lw=1.8, color="#7b3294", mfc="white",
          label="ALIGNN-FF, 1 H200, energy+forces+stress")
ax.loglog(an, at, "-o", ms=3.5, lw=1.8, color="#2a78d6", label="ALIGNN-FF, 1 B200, energy only")
if eg is not None:
    ax.loglog(*eg, "--D", ms=3.5, lw=1.5, color="#2ca02c", label="ALIGNN-FF, 1 GB10, energy only")
if eh is not None:
    ax.loglog(*eh, "--P", ms=4, lw=1.5, color="#7b3294", label="ALIGNN-FF, 1 H200, energy only")
ax.set_xlabel("Number of atoms")
ax.set_ylabel("Wall time (s)")
ax.grid(True, which="major", color="#e4e3df")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=8.5, ncol=1)
fig.tight_layout()
fig.savefig(HERE / "scaling_all.png", facecolor="white", bbox_inches="tight")
