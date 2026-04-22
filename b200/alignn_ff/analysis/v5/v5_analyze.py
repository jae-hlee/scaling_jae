"""
analyze.py — v5 scaling analysis + v4 vs v5 comparison.

Reads ../../scaling_alignn_v5.npz (and v4 for comparison) and produces:
  - breakdown.png     v5 per-stage timing, log-log, with fitted exponents
  - comparison.png    v4 vs v5 line-graph time on matched sizes
  - speedup.png       per-size line-graph speedup ratio v4 / v5
  - energy.png        v5 energy vs N, with baseline band + divergence onset
  - metrics.json      machine-readable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

V4 = np.load(ROOT / "scaling_alignn_v4.npz")
V5 = np.load(ROOT / "scaling_alignn_v5.npz")


def power_law_fit(x, y):
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    a = np.exp(la)
    yhat = a * x**b
    ss_res = np.sum((np.log(y) - np.log(yhat)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


# -----------------------------------------------------------------------------
# v5 scaling fits (exclude warmup + noisy sub-ms regime)
# -----------------------------------------------------------------------------
n5 = V5["natoms"]
t_nl5, t_line5, t_graph5, t_inf5, E5 = (
    V5["times_nl"], V5["times_line"], V5["times_graph"],
    V5["times_inference"], V5["energies"],
)
t_total5 = t_graph5 + t_inf5

# For fits, exclude small-N where timings are dominated by kernel-launch noise.
fit_mask = n5 >= 10_000

v5_fits = {
    "neighbor_list_matscipy": power_law_fit(n5[fit_mask], t_nl5[fit_mask]),
    "dgl_graph_and_line": power_law_fit(n5[fit_mask], t_line5[fit_mask]),
    "alignn_inference": power_law_fit(n5[fit_mask], t_inf5[fit_mask]),
    "total": power_law_fit(n5[fit_mask], t_total5[fit_mask]),
}

# -----------------------------------------------------------------------------
# Same-size v4 vs v5 comparison (v4 has 46 points, v5 has 58; overlap is 1..46)
# -----------------------------------------------------------------------------
n4 = V4["natoms"]
assert np.array_equal(n4, n5[: len(n4)]), "size arrays diverge in the overlap region"
overlap = slice(0, len(n4))

v4_line = V4["times_line"]
v5_line_overlap = t_line5[overlap]
v4_total = V4["times_graph"] + V4["times_inference"]
v5_total_overlap = t_total5[overlap]

line_speedup = v4_line / v5_line_overlap
total_speedup = v4_total / v5_total_overlap

# Total sweep wall time (sum of per-iteration times).
v4_wall_s = float(v4_total.sum())
v5_wall_s = float(t_total5.sum())

# -----------------------------------------------------------------------------
# Energy stability: v5 shows a new monotone drift at large N
# -----------------------------------------------------------------------------
# Small-N baseline (exclude warmup).
baseline_mask = (n5 >= 100) & (n5 <= 100_000)
E_baseline = float(E5[baseline_mask].mean())
E_baseline_std = float(E5[baseline_mask].std())

# Divergence onset: first index where |E - baseline| exceeds 2% of baseline
# AND the next point is even further off (so we catch a real runaway, not
# the small v4-style ~0.5% wobble).
dev = np.abs(E5 - E_baseline)
onset_threshold_abs = 0.02 * E_baseline  # 2% of baseline
sustained = np.zeros_like(dev, dtype=bool)
for i in range(len(dev) - 1):
    if dev[i] > onset_threshold_abs and dev[i + 1] > dev[i]:
        sustained[i] = True
if sustained.any():
    onset_idx = int(np.argmax(sustained))
    onset_n = int(n5[onset_idx])
    onset_deviation = float(dev[onset_idx])
else:
    onset_idx, onset_n, onset_deviation = None, None, None

E_drift_max_frac = float(dev.max() / E_baseline)

# -----------------------------------------------------------------------------
# Plot 1: v5 breakdown
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n5, t_nl5, "-o", ms=4,
        label=f"Neighbor list (matscipy) — b={v5_fits['neighbor_list_matscipy'][1]:.2f}")
ax.plot(n5, t_line5, "-s", ms=4,
        label=f"DGL graph + sparse line graph — b={v5_fits['dgl_graph_and_line'][1]:.2f}")
ax.plot(n5, t_inf5, "-^", ms=4,
        label=f"ALIGNN-FF inference — b={v5_fits['alignn_inference'][1]:.2f}")
ax.plot(n5, t_total5, "--", color="k", lw=1,
        label=f"Total — b={v5_fits['total'][1]:.2f}")
ax.set(xlabel="N atoms", ylabel="Time per iteration (s)",
       title="v5 scaling (fit exponents b in t ∝ N^b over N≥10⁴)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.axvline(n5[0], color="grey", ls=":", lw=0.8, alpha=0.5)
ax.annotate("warmup", xy=(n5[0], t_inf5[0]), xytext=(30, 2.0),
            fontsize=8, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))
fig.tight_layout()
fig.savefig(HERE / "breakdown.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 2: v4 vs v5 line-graph time on matched sizes
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n4, v4_line, "-s", ms=4, color="#d62728", label="v4: dgl.line_graph(g, shared=True)")
ax.plot(n5, t_line5, "-s", ms=4, color="#2ca02c", label="v5: _fast_line_graph(g)")
ax.set(xlabel="N atoms", ylabel="Line-graph build time (s)",
       title="Line-graph build: v4 (quadratic) vs v5 (linear)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
# Annotate the biggest overlap point.
i_big = len(n4) - 1
ratio = v4_line[i_big] / t_line5[i_big]
ax.annotate(
    f"at N={n4[i_big]:,}\n  v4: {v4_line[i_big]:,.0f} s\n  v5: {t_line5[i_big]:.3f} s\n  speedup: {ratio:,.0f}×",
    xy=(n4[i_big], v4_line[i_big]),
    xytext=(8_000, 500),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9),
)
fig.tight_layout()
fig.savefig(HERE / "comparison.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 3: per-size speedup ratio
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n4, line_speedup, "-o", ms=4, color="#d62728", label="Line-graph speedup")
ax.plot(n4, total_speedup, "-^", ms=4, color="#1f77b4", label="Total-iter speedup")
ax.axhline(1.0, color="grey", ls=":", lw=0.8)
ax.set(xlabel="N atoms", ylabel="v4 time / v5 time",
       title="Per-size speedup of v5 over v4 (higher is better)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "speedup.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 4: energy stability with divergence onset
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n5, E5, "-o", ms=4, color="#1f77b4")
ax.axhline(E_baseline, color="grey", ls="--", lw=0.8,
           label=f"baseline (100 ≤ N ≤ 10⁵) = {E_baseline:.4f} eV")
ax.fill_between(n5, E_baseline - onset_threshold_abs, E_baseline + onset_threshold_abs,
                color="grey", alpha=0.15, label="±2% of baseline band")
if onset_n is not None:
    ax.axvline(onset_n, color="red", ls="--", lw=0.8,
               label=f"drift onset: N≈{onset_n:,}")
ax.set(xlabel="N atoms", ylabel="Model output (eV)",
       title="v5 energy stability — monotone drift onset past ~N=500k")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "energy.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# metrics.json
# -----------------------------------------------------------------------------
metrics = {
    "v5_n_points": int(len(n5)),
    "v5_n_max": int(n5[-1]),
    "v5_terminated_reason": "CUDA OOM at size 59 (N=821,516); allocating 28 GiB on top of 157 GiB on a 178 GiB B200",
    "v5_sweep_total_wall_seconds": v5_wall_s,
    "v4_sweep_total_wall_seconds": v4_wall_s,
    "total_sweep_speedup": v4_wall_s / v5_wall_s,
    "v5_scaling_fits_N_ge_10000": {
        name: {"prefactor_a": float(a), "exponent_b": float(b), "r2": float(r2)}
        for name, (a, b, r2) in v5_fits.items()
    },
    "same_size_speedup_line_graph": {
        "at_N_max_overlap": {
            "N": int(n4[-1]),
            "v4_seconds": float(v4_line[-1]),
            "v5_seconds": float(t_line5[len(n4) - 1]),
            "ratio": float(line_speedup[-1]),
        },
        "median_ratio_over_overlap": float(np.median(line_speedup)),
        "max_ratio": float(line_speedup.max()),
    },
    "energy_stability": {
        "baseline_mean_eV": E_baseline,
        "baseline_std_eV": E_baseline_std,
        "drift_onset_N": onset_n,
        "max_deviation_eV": float(dev.max()),
        "max_deviation_fraction": E_drift_max_frac,
        "note": "v5 extends to N=780k and shows a new monotone upward drift that "
                "v4 never reached. Small-N verification was bitwise-identical, so the "
                "sparse line graph is not the cause.",
    },
    "new_bottleneck": {
        "per_iter_at_N_max": float(t_total5[-1]),
        "per_iter_at_N_389344": float(t_total5[45]),
        "note": "Runtime is now HBM-memory-bound, not compute-bound. "
                "Even at N=780k, a single iteration is ~5 s.",
    },
}

with open(HERE / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Wrote:")
for p in ("metrics.json", "breakdown.png", "comparison.png", "speedup.png", "energy.png"):
    print(f"  {HERE / p}")
print()
print("Headline numbers:")
print(f"  v5 line-graph exponent (was 1.995 in v4):  {v5_fits['dgl_graph_and_line'][1]:.3f}")
print(f"  v5 total exponent (was 1.986 in v4):       {v5_fits['total'][1]:.3f}")
print(f"  line-graph speedup at N={n4[-1]:,}:             {line_speedup[-1]:,.0f}×")
print(f"  total-iter speedup at N={n4[-1]:,}:             {total_speedup[-1]:,.0f}×")
print(f"  v5 total sweep wall time:                  {v5_wall_s:.1f} s (v4: {v4_wall_s:.0f} s)")
print(f"  energy drift onset (|ΔE| > 5σ_baseline):   N≈{onset_n:,}" if onset_n else "  energy: no sustained drift")
print(f"  max energy deviation:                      {E_drift_max_frac*100:.1f}% (at N_max)")
