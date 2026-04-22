# ALIGNN-FF scaling analysis — `scaling_alignn_v4.npz`

Data source: 46 supercell sizes of Cu FCC, **N = 4 to 389,344 atoms** (multipliers 1–46). Run terminated at multiplier 46 by the **SLURM 24-hour time limit**, not OOM — see `alignn_1306345.err`: `CANCELLED ... DUE TO TIME LIMIT`. The loop's `try/except` for OOM was never triggered.

Power-law fits below use N ≥ 1000 only (smaller N is dominated by kernel-launch overhead and warmup). All fits have R² > 0.988.

## Headline finding: the wrong stage was optimized

The script's header docstring celebrates beating the v3 pure-PyTorch neighbor list (10.6 s at N=13,500) by switching to matscipy. That optimization worked — matscipy neighbor search is now **0.01 %** of total runtime at N = 389,344. The problem is that **DGL graph + line-graph construction now accounts for 99.98 %** of total runtime at the same size, and scales as **N^1.995** (essentially quadratic, R² = 0.99998).

| stage                          | exponent *b* in t ∝ N^b | % of total at N=389,344 | wall time at N=389,344 |
| ------------------------------ | ----------------------- | ----------------------- | ---------------------- |
| DGL graph + line graph         | **1.995**               | **99.98 %**             | **11,431 s (3.2 h)**   |
| ALIGNN-FF inference            | 0.879                   | 0.008 %                 | 0.96 s                 |
| Neighbor list (matscipy, CPU)  | 0.965                   | 0.012 %                 | 1.36 s                 |
| **Total**                      | **1.986**               | —                       | **11,433 s**           |

Crossover where line-graph construction exceeds 50 % of total time: already by **N ≈ 500** (multiplier 5). Past that, every further size is almost pure DGL line-graph cost.

## Why it's quadratic

The line graph of a nearest-neighbor molecular graph has one edge per (edge, edge) pair sharing a node, i.e. `Σᵥ deg(v)²`. With a bounded `max_neighbors` cap, this sum should be **O(N)**, not O(N²). The observed N² scaling therefore points at the **DGL implementation** of `dgl.line_graph(g, shared=True)` — likely a dense/materialized construction path rather than a sparse one — not at the underlying combinatorics. This is a library-level issue, not an algorithmic one.

Per-atom cost makes this concrete:
- Line-graph build at N=10,000:   **0.83 ms/atom**
- Line-graph build at N=389,344:  **29.4 ms/atom**  (35× worse per atom)
- Inference:                      **2.6 μs/atom**, essentially flat across the whole range.

## Projection: the advertised `--max-size 99` is not feasible

Extrapolating the fitted power laws to the script's default max size:

- N = 4·99³ = **3,881,196 atoms**
- Projected total per-iteration: **~300 hours (12.5 days)**, ~99 % of which is the line graph
- Projected inference alone: **~6.4 seconds** — the model itself would be fine

To reach size 99 within the 24 h SLURM wall, total runtime would need a **~12× reduction** — effectively eliminating the line-graph bottleneck.

## Energy sanity

Model output is near-constant at ~0.6034 eV across all sizes (std 0.00157 eV, ≈ 0.26 % of the mean), confirming it's reported as energy-per-atom and that the same material at the same density gives consistent answers. There is a small monotone drift over the run: **end-to-end −3.9 mmeV** from N=4 to N=389,344 (−0.65 %). The shape is not random noise — it trends downward past N≈100,000. Plausible causes:

1. `max_neighbors=12` cap truncates more candidate neighbors in larger supercells (where each atom really does have more equidistant neighbors than in a small supercell, since there are fewer PBC duplicates). This is the most likely explanation and is a property of the input, not a bug.
2. float32 accumulation across the GNN message passing.

Either way, the drift is small enough to not invalidate the timing conclusions. If it matters scientifically, worth cross-checking by raising `max_neighbors` to see if large-N values converge back upward.

## Warmup outlier

Size 1 (N=4) shows **37.7 s of "inference" time** — JIT compilation and first-touch CUDA kernel caching, not real work. It's excluded from fits; in plots it's annotated. The neighbor-list column for size 1 (0.148 s) is similarly first-call overhead — matscipy's C extension loads on first invocation.

## Recommendations (if continuing this benchmark)

1. **Replace `dgl.line_graph`** with a sparse hand-rolled construction: for each edge `(u,v)` with incident edges `(v,w)`, emit a line-graph edge. This is a single `edge_ids` + gather — O(Σ deg²) = O(N) work. This alone would likely move the total-time exponent from ~2 toward ~1 and unblock sizes far past 46.
2. **Cache and reuse `g`, `lg` across sizes** when the material and density are fixed — the topology up to a shift is identical for an FCC supercell expansion. Not a fair apples-to-apples change for a scaling study, but useful for production inference.
3. **Stop timing size 1** in future sweeps, or dedicate an explicit warmup pass before the loop (one forward on a tiny graph) so the reported timings are steady-state.
4. **Checkpointing worked**: the 24 h cancellation lost no data. Keep the per-iteration `np.savez`.

## Artifacts

- `analyze.py` — the analysis script; re-run after updating `../scaling_alignn_v4.npz`.
- `metrics.json` — machine-readable summary (fits, shares, projections).
- `breakdown.png` — log-log timing per stage with fitted exponents.
- `share.png` — fractional time share by stage vs N.
- `per_atom.png` — μs/atom for each stage (flat = linear scaling).
- `energy.png` — energy output vs N with mean ± 1σ band.
