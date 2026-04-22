"""
analyze.py — scaling analysis of scale5b_v4 benchmark results.

Reads ../scaling_alignn_v4.npz and produces:
  - breakdown.png     stacked timing breakdown vs system size
  - scaling.png       log-log scaling with fitted power laws
  - energy.png        per-atom energy stability vs size
  - metrics.json      machine-readable summary
  - summary.md        written findings
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = np.load(HERE.parent / "scaling_alignn_v4.npz")

n = DATA["natoms"]
t_nl = DATA["times_nl"]
t_line = DATA["times_line"]
t_graph = DATA["times_graph"]  # t_nl + t_line
t_inf = DATA["times_inference"]
E = DATA["energies"]
t_total = t_graph + t_inf

# Size 1 (n=4) is a warmup outlier: first CUDA kernel launches + JIT.
# Use a mask for fits and summary stats, but include in plots with annotation.
warmup_mask = np.arange(len(n)) > 0
# Fit only on the linear-scaling regime (skip the very-small-N noise).
fit_mask = n >= 1000


def power_law_fit(x, y):
    """Least-squares fit y = a * x**b. Returns (a, b, r2)."""
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    a = np.exp(la)
    yhat = a * x**b
    ss_res = np.sum((np.log(y) - np.log(yhat)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


fits = {
    "neighbor_list_matscipy": power_law_fit(n[fit_mask], t_nl[fit_mask]),
    "dgl_graph_and_line": power_law_fit(n[fit_mask], t_line[fit_mask]),
    "alignn_inference": power_law_fit(n[fit_mask], t_inf[fit_mask]),
    "total": power_law_fit(n[fit_mask], t_total[fit_mask]),
}

# Where does line-graph construction start dominating total time?
line_share = t_line / t_total
crossover_idx = int(np.argmax(line_share > 0.5))  # first index where >50%
crossover_n = int(n[crossover_idx]) if line_share[crossover_idx] > 0.5 else None

# Per-atom stats (exclude warmup).
per_atom_inf_us = t_inf[warmup_mask] / n[warmup_mask] * 1e6  # microseconds / atom
per_atom_line_us = t_line[warmup_mask] / n[warmup_mask] * 1e6

# Energy stability.
E_mean = float(E[warmup_mask].mean())
E_std = float(E[warmup_mask].std())
E_drift = float(E[-1] - E[0])  # end-to-end drift

# -----------------------------------------------------------------------------
# Plot 1: timing breakdown
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, t_nl, "-o", ms=4, label=f"Neighbor list (matscipy CPU) — {fits['neighbor_list_matscipy'][1]:.2f}")
ax.plot(n, t_line, "-s", ms=4, label=f"DGL graph + line graph — {fits['dgl_graph_and_line'][1]:.2f}")
ax.plot(n, t_inf, "-^", ms=4, label=f"ALIGNN-FF inference — {fits['alignn_inference'][1]:.2f}")
ax.plot(n, t_total, "--", color="k", lw=1, label=f"Total — {fits['total'][1]:.2f}")
ax.set(xlabel="N atoms", ylabel="Time per iteration (s)",
       title="ALIGNN-FF scaling breakdown (legend shows fitted exponent b in t ∝ N^b)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.axvline(n[0], color="grey", ls=":", lw=0.8, alpha=0.5)
ax.annotate("warmup\n(JIT + 1st CUDA launch)", xy=(n[0], t_inf[0]),
            xytext=(20, 1.0), fontsize=8, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))
fig.tight_layout()
fig.savefig(HERE / "breakdown.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 2: share of total time by stage (stacked fraction)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
frac_nl = t_nl / t_total
frac_line = t_line / t_total
frac_inf = t_inf / t_total
ax.stackplot(n[warmup_mask], frac_line[warmup_mask], frac_nl[warmup_mask], frac_inf[warmup_mask],
             labels=["DGL graph + line graph", "Neighbor list (matscipy)", "ALIGNN-FF inference"],
             colors=["#d62728", "#1f77b4", "#2ca02c"], alpha=0.85)
ax.set_xscale("log")
ax.set(xlabel="N atoms", ylabel="Fraction of total time",
       title="Time share by stage (log-x)", ylim=(0, 1))
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "share.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 3: per-atom cost (where flat == linear scaling)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n[warmup_mask], per_atom_inf_us, "-^", ms=4, label="Inference (μs/atom)")
ax.plot(n[warmup_mask], per_atom_line_us, "-s", ms=4, label="Line-graph build (μs/atom)")
ax.plot(n[warmup_mask], t_nl[warmup_mask] / n[warmup_mask] * 1e6, "-o", ms=4,
        label="Neighbor list (μs/atom)")
ax.set(xlabel="N atoms", ylabel="Time per atom (μs)",
       title="Per-atom cost (flat line = O(N); rising = super-linear)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "per_atom.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 4: energy stability
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, E, "-o", ms=4)
ax.axhline(E_mean, color="grey", ls="--", lw=0.8, label=f"mean = {E_mean:.5f} eV")
ax.fill_between(n, E_mean - E_std, E_mean + E_std, color="grey", alpha=0.15,
                label=f"±1σ = ±{E_std:.5f} eV")
ax.set(xlabel="N atoms", ylabel="Model output (eV)",
       title="Energy output stability across supercell size (Cu FCC, same density)")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "energy.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Projection: what would size-99 take?
# -----------------------------------------------------------------------------
a_tot, b_tot, _ = fits["total"]
a_line, b_line, _ = fits["dgl_graph_and_line"]
a_inf, b_inf, _ = fits["alignn_inference"]
n_target = 4 * 99**3  # declared --max-size default
proj_total_s = a_tot * n_target**b_tot
proj_line_s = a_line * n_target**b_line
proj_inf_s = a_inf * n_target**b_inf

# -----------------------------------------------------------------------------
# Save metrics.json
# -----------------------------------------------------------------------------
metrics = {
    "n_points": int(len(n)),
    "n_min": int(n[0]),
    "n_max": int(n[-1]),
    "run_terminated_reason": "SLURM 24h time limit (see alignn_1306345.err)",
    "fits_excluding_n_lt_1000": {
        name: {"prefactor_a": float(a), "exponent_b": float(b), "r2": float(r2)}
        for name, (a, b, r2) in fits.items()
    },
    "line_graph_dominance_crossover_n": crossover_n,
    "line_graph_share_at_n_max": float(line_share[-1]),
    "inference_share_at_n_max": float(t_inf[-1] / t_total[-1]),
    "neighbor_list_share_at_n_max": float(t_nl[-1] / t_total[-1]),
    "per_atom_inference_us_median": float(np.median(per_atom_inf_us)),
    "per_atom_line_us_at_n_max": float(per_atom_line_us[-1]),
    "per_atom_line_us_at_n_1e4": float(
        t_line[np.argmin(np.abs(n - 10000))] / n[np.argmin(np.abs(n - 10000))] * 1e6
    ),
    "energy_mean_ev": E_mean,
    "energy_std_ev": E_std,
    "energy_drift_start_to_end_ev": E_drift,
    "projection_to_size_99": {
        "n_atoms": int(n_target),
        "total_seconds": float(proj_total_s),
        "total_hours": float(proj_total_s / 3600),
        "line_graph_seconds": float(proj_line_s),
        "inference_seconds": float(proj_inf_s),
    },
    "warmup_outlier_size_1": {
        "n_atoms": int(n[0]),
        "inference_s": float(t_inf[0]),
        "line_graph_s": float(t_line[0]),
        "note": "First-iteration JIT + CUDA kernel cache miss; excluded from fits.",
    },
}

with open(HERE / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Wrote {HERE / 'metrics.json'}")
print(f"Wrote {HERE / 'breakdown.png'}")
print(f"Wrote {HERE / 'share.png'}")
print(f"Wrote {HERE / 'per_atom.png'}")
print(f"Wrote {HERE / 'energy.png'}")
print()
print("Headline numbers:")
print(f"  total scaling exponent:        {fits['total'][1]:.3f}")
print(f"  line-graph scaling exponent:   {fits['dgl_graph_and_line'][1]:.3f}")
print(f"  inference scaling exponent:    {fits['alignn_inference'][1]:.3f}")
print(f"  neighbor-list exponent:        {fits['neighbor_list_matscipy'][1]:.3f}")
print(f"  line-graph % of total at N={n[-1]}: {line_share[-1]*100:.2f}%")
print(f"  projection to size 99 (N={n_target}):   {proj_total_s/3600:.1f} h "
      f"(line graph alone: {proj_line_s/3600:.1f} h)")
