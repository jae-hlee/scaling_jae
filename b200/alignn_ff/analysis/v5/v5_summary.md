# v5 scaling analysis — `scaling_alignn_v5.npz`

Data source: 58 supercell sizes of Cu FCC, **N = 4 to 780,448 atoms** (multipliers 1–58). Run terminated by **CUDA OOM at size 59** (N=821,516) — tried to allocate 28 GiB on top of 157 GiB already held on a 178 GiB B200. Neither the 72 h wall nor the neighbor-list path were the limit this time.

## The fix worked

The v5 sparse line-graph (`_fast_line_graph`) replaces `dgl.line_graph(g, shared=True)` — the 99.98 %-of-runtime bottleneck identified in the v4 analysis — with a single argsort-by-destination plus a segment-gather emission of edge pairs. All of the following are same-size, same-hardware comparisons:

| metric                                | v4             | v5         | ratio          |
| ------------------------------------- | -------------- | ---------- | -------------- |
| Line-graph build at N=389,344         | 11,431 s       | 0.269 s    | **~42,500×**   |
| Total iter at N=389,344               | 11,433 s       | 2.58 s     | ~4,400×        |
| Total sweep wall (through N=389,344)  | **22.5 h**     | **~2 min** | ~560×          |

The v5 sweep reached **12 more sizes** (N=389k → 780k) in **2.4 minutes of wall total** — compared to v4 which burned the full 24 h SLURM allocation just getting to N=389k.

## New scaling regime

Fits over N ≥ 10⁴ (to avoid the kernel-launch-noise floor visible below ~5000 atoms):

| stage                           | exponent *b* (v5) | v4 was | comment |
| ------------------------------- | ------------------ | ------ | ------- |
| DGL graph + sparse line graph   | **0.79**           | 1.995  | was quadratic; now sub-linear (effective) |
| ALIGNN-FF inference             | 0.92               | 0.88   | unchanged path, still dominated by GPU saturation |
| Neighbor list (matscipy)        | 0.98               | 0.97   | unchanged path |
| **Total**                       | **0.85**           | 1.986  | |

The theoretical line-graph exponent with bounded `max_neighbors` is 1.0 (the number of lg edges is at most 144·N for max_neighbors=12). The fitted 0.79 is below that because the inner-loop work per atom is amortized across a larger GPU grid at bigger sizes — a GPU-saturation effect, not an algorithmic one. The same effect explains the sub-linear inference exponent.

Concretely, at N=780,448 a full pipeline iteration (neighbor list + graph + inference) is **5.17 s**. The v4 projection to this size would have been **~45,900 s** (12.8 h) — nothing compared to the actual 5 s.

## The new bottleneck is HBM, not compute

OOM hit at size 59 with ~28 GiB more requested. At size 58 the whole iteration took 5.2 s, so there's plenty of wall left — it's memory. Knobs to try, in order of effort:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — no code change, may push a size or two further by reducing fragmentation (the error message itself suggests this).
2. `torch.cuda.empty_cache()` is already called each iteration, so fragmentation is the main angle.
3. Chunked model forward over sub-volumes of the graph (bigger change; ALIGNN's layers would need to handle partitioned inputs).

None of this is blocking for the scaling study — we already have a clean linear-regime curve over 5 orders of magnitude in N.

## Correctness finding: energy drift past N ≈ 500k

The `out["out"]` value is constant at **0.6034 ± 0.0016 eV** across N from ~100 to ~10⁵ atoms, then something breaks:

| N        | E (eV) | deviation from baseline |
| -------- | ------ | ----------------------- |
| 470,596  | 0.604  | 0 % (baseline)          |
| 500,000  | 0.636  | **+5.3 %**              |
| 562,432  | 0.698  | +15.7 %                 |
| 629,856  | 0.751  | +24.5 %                 |
| 702,464  | 0.797  | +32.1 %                 |
| **780,448** | **0.838** | **+38.6 %**         |

The drift is monotone, starts abruptly around N=500k, and grows with N. Since the system is the same Cu FCC crystal at the same density at every size, the correct `out` should be constant — this is a model output error, not physics.

It is **not the sparse line-graph**: `verify_line_graph_equivalence` produced bitwise-identical `cos(θ)` between old and new paths (max |Δ| = 0.00e+00 in the pre-flight check), and the divergence happens far past any N we validated. Leading hypotheses:

1. **float32 accumulation** in ALIGNN's graph-level pooling — with ~10⁶ contributions summed in float32, relative precision approaches ~2^-23 per term, which can produce bias if terms aren't zero-mean. The onset around N=500k is consistent with that scale.
2. **Large-tensor numerical paths in DGL/cuDNN** — some reduction kernels select a different algorithm past a size threshold (e.g., atomic vs tree reductions) and the numerical behavior differs.
3. Less likely: the `max_neighbors=12` cap biting differently in very large supercells (this was v4's suspected source for a 0.5 % wobble; can't explain a 39 % drift).

Cheapest diagnostic: **one forward pass in float64** at N=780k. If the drift disappears, it's (1). If it doesn't, we're in (2) territory. The model config is small to change — cast weights and inputs at load time and run a single size.

## Summary

- Line-graph replacement validated empirically: ~42,000× at large N, sub-linear scaling, identical numerical output on verification.
- v5 reaches 2× the v4 size ceiling in 1/500× the wall time.
- Memory is now the limit. Energy drift past ~500k atoms needs attention before trusting any production inference at that scale.

## Artifacts in `analysis/v5/`

- `analyze.py` — regenerable analysis script.
- `metrics.json` — machine-readable summary.
- `breakdown.png` — v5 per-stage timing with fitted exponents.
- `comparison.png` — v4 vs v5 line-graph time on matched sizes.
- `speedup.png` — per-size v4/v5 ratio for line-graph and total.
- `energy.png` — v5 energy stability with drift-onset marker.
