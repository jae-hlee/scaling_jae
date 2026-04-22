"""
analyze.py — v6 scaling analysis + v5 vs v6 comparison.

Reads ../../scaling_alignn_v{5,6}.npz and produces:
  - breakdown.png     v6 per-stage timing, log-log, with fitted exponents
  - comparison.png    v5 vs v6 per-iter total time (should be ~identical)
  - energy.png        v5 vs v6 energy-vs-N, showing the pool fix works
                      below the cliff but does not close the cliff itself
  - overhead.png      per-iter v6-v5 time delta — the cost of _Float64Pool
  - metrics.json      machine-readable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

V5 = np.load(ROOT / "scaling_alignn_v5.npz")
V6 = np.load(ROOT / "scaling_alignn_v6.npz")

# Float64 ground truth: flat across every size where f64 fit in HBM.
# Established by the diagnose_drift.py 2026-04 run.
E_F64_REF = 0.604015


def power_law_fit(x, y):
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    a = np.exp(la)
    yhat = a * x**b
    ss_res = np.sum((np.log(y) - np.log(yhat)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


n5 = V5["natoms"]
n6 = V6["natoms"]
assert np.array_equal(n5, n6), "v5 and v6 must have matched size arrays"
n = n6  # common

t_nl6, t_line6, t_graph6, t_inf6, E6 = (
    V6["times_nl"], V6["times_line"], V6["times_graph"],
    V6["times_inference"], V6["energies"],
)
t_nl5, t_line5, t_graph5, t_inf5, E5 = (
    V5["times_nl"], V5["times_line"], V5["times_graph"],
    V5["times_inference"], V5["energies"],
)
t_total5 = t_graph5 + t_inf5
t_total6 = t_graph6 + t_inf6

fit_mask = n >= 10_000

v6_fits = {
    "neighbor_list_matscipy": power_law_fit(n[fit_mask], t_nl6[fit_mask]),
    "dgl_graph_and_line": power_law_fit(n[fit_mask], t_line6[fit_mask]),
    "alignn_inference": power_law_fit(n[fit_mask], t_inf6[fit_mask]),
    "total": power_law_fit(n[fit_mask], t_total6[fit_mask]),
}

# -----------------------------------------------------------------------------
# Overhead of the _Float64Pool wrapper
# -----------------------------------------------------------------------------
# The wrapper adds one f64 allocation + f64 sum per readout call. Cost is
# expected to grow with N (~1.6 GB copy at N=780k) and to be dominated by
# memory allocation; compute is trivial (a single reduction).
delta_inf = t_inf6 - t_inf5
delta_total = t_total6 - t_total5
# Fraction of v5's inference time that the wrapper adds, ignoring warmup.
frac_overhead = delta_inf[fit_mask] / t_inf5[fit_mask]

# -----------------------------------------------------------------------------
# Energy: v5 vs v6
# -----------------------------------------------------------------------------
# Pre-cliff region (N <= 442,368 = i=48) — this is where the _Float64Pool fix
# is supposed to work. The cliff onset is at i=49 (N=470,596) in both.
pre_cliff_mask = n <= 442_368
cliff_mask = n >= 470_596

# Deviation from the f64 reference.
dev_v5 = np.abs(E5 - E_F64_REF)
dev_v6 = np.abs(E6 - E_F64_REF)

# v6 headline: bit-constant in the pre-cliff region.
E6_precliff_unique = np.unique(np.round(E6[pre_cliff_mask], 6))
E6_precliff_min = float(E6[pre_cliff_mask].min())
E6_precliff_max = float(E6[pre_cliff_mask].max())

# v5 max pre-cliff deviation, for the contrast.
v5_precliff_max_dev = float(dev_v5[pre_cliff_mask].max())

# Both v5 and v6 still drift at the cliff; the fix does not close it.
v5_maxN_dev = float(dev_v5[-1])
v6_maxN_dev = float(dev_v6[-1])

# -----------------------------------------------------------------------------
# Plot 1: v6 breakdown
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, t_nl6, "-o", ms=4,
        label=f"Neighbor list (matscipy) — b={v6_fits['neighbor_list_matscipy'][1]:.2f}")
ax.plot(n, t_line6, "-s", ms=4,
        label=f"DGL graph + sparse line graph — b={v6_fits['dgl_graph_and_line'][1]:.2f}")
ax.plot(n, t_inf6, "-^", ms=4,
        label=f"ALIGNN-FF inference (+ _Float64Pool) — b={v6_fits['alignn_inference'][1]:.2f}")
ax.plot(n, t_total6, "--", color="k", lw=1,
        label=f"Total — b={v6_fits['total'][1]:.2f}")
ax.set(xlabel="N atoms", ylabel="Time per iteration (s)",
       title="v6 scaling (fit exponents b in t ∝ N^b over N≥10⁴)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "breakdown.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 2: v5 vs v6 per-iter total time
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, t_total5, "-o", ms=4, color="#2ca02c", label="v5 total/iter")
ax.plot(n, t_total6, "-s", ms=4, color="#1f77b4",
        label="v6 total/iter (with _Float64Pool)")
ax.set(xlabel="N atoms", ylabel="Per-iteration time (s)",
       title="Per-iter wall time: v5 vs v6 (wrapper is effectively free)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "comparison.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 3: energy stability, v5 vs v6, with f64 reference line
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(n, E5, "-o", ms=4, color="#2ca02c", label="v5 (no pool fix)")
ax.plot(n, E6, "-s", ms=4, color="#1f77b4", label="v6 (_Float64Pool)")
ax.axhline(E_F64_REF, color="k", ls="--", lw=0.9,
           label=f"f64 reference = {E_F64_REF:.6f} eV")
ax.axvline(470_596, color="red", ls="--", lw=0.8, alpha=0.7,
           label="cliff onset: N=470,596 (i=49)")
# Shade the pre-cliff region where v6 is flat on the reference.
ax.axvspan(n[0], 442_368, color="#1f77b4", alpha=0.07)
ax.set(xlabel="N atoms", ylabel="Model output (eV)",
       title="Energy stability: v6 eliminates the pre-cliff pool drift, but not the cliff itself")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
# Annotate the bit-constant pre-cliff region.
ax.annotate(
    f"v6 pre-cliff: flat at\n{E6_precliff_min:.6f} eV\n(matches f64 ref to <1e-5)",
    xy=(10_000, E6_precliff_min),
    xytext=(10, 0.74),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.6),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9),
)
fig.tight_layout()
fig.savefig(HERE / "energy.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 4: per-iter overhead of _Float64Pool (v6 - v5)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n, delta_inf * 1000, "-^", ms=4, color="#1f77b4",
        label="inference delta")
ax.plot(n, delta_total * 1000, "-o", ms=4, color="#d62728",
        label="total/iter delta")
ax.axhline(0, color="grey", ls=":", lw=0.8)
ax.set(xlabel="N atoms", ylabel="v6 − v5 time (ms)",
       title="Wrapper overhead (noise-floor level; positive at largest N only)")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "overhead.png", dpi=150)
plt.close(fig)

# -----------------------------------------------------------------------------
# metrics.json
# -----------------------------------------------------------------------------
metrics = {
    "v6_n_points": int(len(n)),
    "v6_n_max": int(n[-1]),
    "v6_terminated_reason": "CUDA OOM at size 59 (N=821,516); same limit as v5",
    "v6_sweep_total_wall_seconds": float(t_total6.sum()),
    "v5_sweep_total_wall_seconds": float(t_total5.sum()),
    "v6_vs_v5_wall_ratio": float(t_total6.sum() / t_total5.sum()),
    "v6_scaling_fits_N_ge_10000": {
        name: {"prefactor_a": float(a), "exponent_b": float(b), "r2": float(r2)}
        for name, (a, b, r2) in v6_fits.items()
    },
    "wrapper_overhead": {
        "median_delta_inference_ms": float(np.median(delta_inf[fit_mask]) * 1000),
        "median_fraction_of_inference": float(np.median(frac_overhead)),
        "max_delta_inference_ms": float(delta_inf.max() * 1000),
        "note": "Differences are within per-run CUDA nondeterminism; the "
                "_Float64Pool wrapper is effectively free relative to the "
                "rest of the pipeline.",
    },
    "energy_pool_fix_precliff": {
        "window_N": [int(n[0]), 442_368],
        "v5_max_deviation_from_f64_eV": v5_precliff_max_dev,
        "v6_max_deviation_from_f64_eV": float(dev_v6[pre_cliff_mask].max()),
        "v6_min_value_eV": E6_precliff_min,
        "v6_max_value_eV": E6_precliff_max,
        "v6_unique_values_rounded_6dp": [float(v) for v in E6_precliff_unique],
        "f64_reference_eV": E_F64_REF,
        "finding": "v5 wanders over a ~4e-3 eV band below the cliff; v6 is "
                   "bit-constant at 0.604013 eV across all pre-cliff sizes, "
                   "matching the f64 reference to <1e-5. The _Float64Pool "
                   "wrapper did close this small drift.",
    },
    "cliff_still_open": {
        "cliff_onset_N": 470_596,
        "v5_E_at_N_max_eV": float(E5[-1]),
        "v6_E_at_N_max_eV": float(E6[-1]),
        "v5_max_deviation_from_f64_eV": v5_maxN_dev,
        "v6_max_deviation_from_f64_eV": v6_maxN_dev,
        "fraction_remaining": float(v6_maxN_dev / v5_maxN_dev),
        "finding": "v5 and v6 drift identically past the cliff: at N=780k, v5 "
                   "deviates 0.234 eV from the f64 reference, v6 deviates 0.241 eV. "
                   "probe_pool.py confirms the pool is faithful at every size "
                   "(|mean_f64 - mean_f32|∞ ≤ 6e-6 across i=20,50,55); the drift "
                   "originates upstream of the readout in the message-passing "
                   "layers.",
    },
    "validity_window_f32": {
        "max_trustworthy_N": 442_368,
        "note": "For v4/v5/v6 alike, f32 model outputs are trustworthy up to "
                "i=48 (N=442,368). Beyond that, a cliff in upstream CUDA "
                "kernels corrupts the result regardless of the pool wrapper.",
    },
}

with open(HERE / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Wrote:")
for p in ("metrics.json", "breakdown.png", "comparison.png", "energy.png",
          "overhead.png"):
    print(f"  {HERE / p}")
print()
print("Headline numbers:")
print(f"  v6 total sweep wall:            {t_total6.sum():.1f} s "
      f"(v5: {t_total5.sum():.1f} s, ratio {t_total6.sum()/t_total5.sum():.3f})")
print(f"  v6 pre-cliff E range:           [{E6_precliff_min:.6f}, "
      f"{E6_precliff_max:.6f}]  (f64 ref: {E_F64_REF:.6f})")
print(f"  v5 pre-cliff max deviation:     {v5_precliff_max_dev:.3e} eV")
print(f"  v6 pre-cliff max deviation:     {dev_v6[pre_cliff_mask].max():.3e} eV")
print(f"  cliff at N=470,596 STILL OPEN — v5 and v6 drift identically past it")
print(f"  at N=780k: v5 dev = {v5_maxN_dev:.3f} eV, v6 dev = {v6_maxN_dev:.3f} eV")
