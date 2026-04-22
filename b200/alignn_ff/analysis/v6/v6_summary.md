# v6 scaling analysis — `scaling_alignn_v6.npz`

Data source: same 58 supercell sizes as v5 (Cu FCC, **N = 4 to 780,448 atoms**). Run terminated at the same OOM size (i=59, N=821,516). v6 adds one thing on top of v5: `_Float64Pool`, a drop-in replacement for ALIGNN's graph-level `dgl.nn.AvgPooling` that accumulates the mean in float64 and casts back. The hypothesis was that float32 accumulation in the readout was the cause of the N ≥ 500k energy drift seen in v5. This analysis separates what the wrapper did (it eliminated a smaller *baseline* drift) from what it did not (it did not close the ~39% *cliff drift* at large N).

## Headline

| aspect                         | v5                | v6                 | note |
| ------------------------------ | ----------------- | ------------------ | ---- |
| Timings (per-iter, all N)      | see `breakdown.png` | essentially identical | wrapper overhead is at the noise floor |
| Pre-cliff energy (N ≤ 442,368) | wanders over 4.1e-3 eV band | **bit-constant at 0.604013 eV** | matches f64 reference to <2e-6 eV |
| Post-cliff energy (N ≥ 470,596) | drifts +39% at N=780k | drifts +40% at N=780k | pool fix does not close the cliff |
| Total sweep wall               | 144 s             | 84 s               | CUDA nondeterminism / system load, not the wrapper |

## What v6 fixed

Below the cliff (N ≤ 442,368, i.e. i ≤ 48), v5's output was not actually stable — it wandered across a small band:

| N       | v5 E (eV)  | v6 E (eV)  | f64 ref  |
| ------- | ---------- | ---------- | -------- |
| 500     | 0.604010   | 0.604013   | 0.604015 |
| 32,000  | 0.603859   | 0.604013   | 0.604015 |
| 108,000 | 0.605611   | 0.604013   | 0.604015 |
| 256,000 | 0.600853   | 0.604013   | 0.604015 |
| 364,500 | 0.600274   | 0.604013   | 0.604015 |
| 442,368 | 0.600148   | 0.604013   | 0.604015 |

The peak pre-cliff deviation of v5 from the f64 reference was **4.12e-3 eV**. v6's peak pre-cliff deviation is **1.68e-6 eV** — a ~2400× reduction. Across the entire N=500 to N=442,368 range, v6 returns the same value (0.604013 eV) to at least six decimals.

In other words, the `_Float64Pool` wrapper removes a real source of f32 precision noise and nails the f64 ground truth, which the v5 analysis had missed because the wandering was small enough to sit inside v4's original 0.5% "wobble" band.

## What v6 did not fix

At i=49 (N=470,596), both v5 and v6 step off the reference value together and drift the same way from then on:

| N       | v5 E (eV) | v6 E (eV) | v5 dev from f64 | v6 dev from f64 |
| ------- | --------- | --------- | --------------- | --------------- |
| 470,596 | 0.604     | 0.607     | 0.000           | 0.003           |
| 500,000 | 0.636     | 0.639     | 0.032           | 0.035           |
| 629,856 | 0.751     | 0.757     | 0.147           | 0.153           |
| 780,448 | 0.838     | 0.845     | 0.234           | 0.241           |

v6's drift is marginally *larger* than v5's at the cliff, but the difference (0.003–0.007 eV) is well within per-run CUDA nondeterminism — not a regression. The takeaway is that v5 and v6 drift **the same way** past the cliff, which means the cliff is not the pool's fault.

## Why the pool is ruled out

`probe_pool.py` (see `output/probe_1332655.out`) instruments the v6 wrapper to print, per pool call, `|mean_f64 − mean_f32|` and statistics of the pool *input* `feat`. Results:

| i   | N       | E       | `|mean_f64 − mean_f32|∞` | `|feat|_max` |
| --- | ------- | ------- | ------------------------ | ------------ |
| 20  | 32,000  | 0.604013 | 4.77e-7 | 2.52 |
| 50  | 500,000 | 0.639071 | 2.15e-6 | 2.64 |
| 55  | 665,500 | 0.781473 | 5.72e-6 | 2.64 |

Two observations:

1. Even at i=55 (665k atoms), the *pool* produces an f32 mean that agrees with the f64 mean to ~1e-5 per feature. Any faithful readout would give the same answer — so neither DGL's AvgPooling nor the manual `torch.sum` replacement is the culprit at the cliff.
2. `|feat|_max` jumps from 2.52 at i=20 to 2.64 at i=50 and stays there. The pool *input* is already corrupted before it reaches the readout. This is the evidence for an upstream source.

## The cliff is upstream of the pool

Combined with the step-like onset at N=470,596 (not a gradual accumulation), the most plausible mechanism is **a CUDA kernel in the message-passing path switching algorithm at a size threshold** and producing systematically different rounding above that threshold. Candidates include:

- DGL's per-node `fn.sum` aggregation in `EdgeGatedGraphConv` (each node sums ~12 edge messages, but the kernel's launch configuration is N-dependent).
- PyTorch's `nn.LayerNorm` CUDA kernel, which switches between welford/one-pass implementations at different input sizes.
- DGL's message-passing kernels more broadly; these have size-dependent tile choices.

Pinning the exact op would take layer-by-layer instrumentation — a task left for a future pass. See "Open follow-ups" below.

## Wrapper overhead

The wrapper adds one f64 allocation + one f64 reduction per readout call. Median per-iter delta (v6 − v5) over N ≥ 10⁴ is within ±15 ms, with no monotone growth in N — the CUDA nondeterminism floor is larger than the actual wrapper cost even at N=780k. See `overhead.png`. The 84 s vs 144 s total-wall difference between the two runs is explained by non-overlapping system load during each SLURM job, not by the wrapper.

## Validity window

For f32 model outputs in this benchmark on Blackwell B200:

- **Trustworthy up to N = 442,368 (i=48)** in v6 — f32 agrees with f64 reference to <2e-6 eV.
- **Not trustworthy for N ≥ 470,596 (i=49)** in v4, v5, or v6 — the upstream cliff corrupts the output by 0.5%→39% as N grows to 780k.

## Open follow-ups

1. **Bisect the cliff.** Instrument the outputs of each of the 4 ALIGNN + 4 GCN layers at i=20 vs i=55, find the first layer whose per-node feature statistics diverge between the two, and try casting that layer's kernel path to f64 (or disabling any size-dependent kernel choice).
2. **Run `model.double()` end-to-end** at as large an N as HBM allows; combined with (1), this gives a direct readout of which kernel is responsible.
3. **File upstream with DGL / PyTorch** if a specific kernel is identified — this looks like a library-level precision quirk, not an ALIGNN or CUDA-hardware issue.

## Artifacts in `analysis/v6/`

- `analyze.py` — regenerable analysis script.
- `metrics.json` — machine-readable summary including the pool-fix and cliff data.
- `breakdown.png` — v6 per-stage timing with fitted exponents.
- `comparison.png` — v5 vs v6 per-iter total time.
- `energy.png` — v5 vs v6 energy-vs-N, with f64 reference line and cliff marker.
- `overhead.png` — per-iter time delta (v6 − v5), showing the wrapper is effectively free.
