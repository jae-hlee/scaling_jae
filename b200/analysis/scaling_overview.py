"""Time vs system size for ALIGNN-FF (Cu) and VASP (Si) on B200.

Run from this directory: `python scaling_overview.py`. Writes `scaling_overview.png`
from ../alignn_ff/scaling_alignn_v6.npz and ../vasp_dft/analysis/metrics.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
B200 = HERE.parent

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

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
ax.loglog(an, at, "-o", ms=3.5, lw=1.8, color="#2a78d6", label="ALIGNN-FF, 1 GPU, time per energy")
ax.loglog(v1n, v1t, "-s", ms=4.5, lw=1.8, color="#eb6834", label="VASP, 1 GPU, time per SCF cycle")
ax.loglog(vbn[1:], vbt[1:], "s", ms=4.5, mfc="white", mew=1.5, color="#eb6834")
ax.loglog(vbn, vbt, "--", lw=1.8, color="#eb6834", label="VASP, 4–8 GPUs (larger cells)")
ax.set_xlabel("Number of atoms")
ax.set_ylabel("Wall time (s)")
ax.grid(True, which="major", color="#e4e3df")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8), frameon=False)
fig.tight_layout()
fig.savefig(HERE / "scaling_overview.png", facecolor="white")
