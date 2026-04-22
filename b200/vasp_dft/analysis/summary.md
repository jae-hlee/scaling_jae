# VASP DFT scaling analysis — Si supercells on B200

Data source: `b200/vasp_dft/<n>x<n>x<n>/<ngpu>/{INCAR,OSZICAR,OUTCAR}`. Silicon diamond **2-atom primitive cell**, replicated n×n×n (so N_atoms = 2·n³). Single-point SCF on the B200 partition (`b200`, QOS `blackwell_test`). Analysis regenerable via `analyze.py`.

## Experiment layout

Two experiments are mixed in the tree and are **not apples-to-apples**:

| regime            | sizes                           | ngpu    | NELM | ALGO    | EDIFF | parallel mode              |
| ----------------- | ------------------------------- | ------- | ---- | ------- | ----- | -------------------------- |
| strong scaling    | 3³, 4³                          | 1,2,4,8 | 20   | default | 1e-6  | `KPAR=<ngpu>`, NKPTS=3     |
| strong scaling    | 5³, 6³                          | 1,2,4,8 | 20   | default | 1e-6  | `NCORE=1` only, NKPTS=1 (Γ)|
| convergent timing | 10³                             | 4       | 5    | Fast    | 1e-5  | `NCORE=1`, NKPTS=1         |
| truncated timing  | 12³, 14³                        | 8       | 2    | Fast    | 1e-5  | `NCORE=1`, NKPTS=1         |
| failed            | 15³, 16³                        | 8       | 2    | Fast    | 1e-5  | 15³: `NCORE=1`; 16³: `NCORE=4` |

Important consequence: the 3³/4³ strong-scaling rows mostly measure **k-point parallelism saturation** (only 3 k-points — KPAR=8 over-partitions), while 5³/6³ measure **band/pw parallelism** on a pure Γ calculation. Don't read them as a single curve.

## Headline

| aspect                         | result |
| ------------------------------ | ------ |
| Best strong-scaling speedup    | **1.27× on 4 GPUs at 6³** (432 atoms) — 32% parallel efficiency |
| 8 GPU > 4 GPU?                 | **Never** across any tested size — 8 GPUs is always slower than 4 GPUs |
| Size-scaling exponent (3456 → 5488 atoms, 8 GPU) | **T ∝ N^1.80** |
| Largest completed SCF          | 14³ = 5488 atoms, 8 GPU, 2 cycles, ~781 s/cycle avg |
| Largest failed                 | 15³ (6750 atoms) and 16³ (8192 atoms) — 0 SCF cycles, see below |

## Strong scaling — full numbers

Elapsed time (s) from `OUTCAR`, all 20/20 SCF cycles completed:

| n³ (atoms) | NKPTS | 1 GPU          | 2 GPU          | 4 GPU          | 8 GPU          |
| ---------- | ----- | -------------- | -------------- | -------------- | -------------- |
| 3³ (54)    | 3     | 15.28 (1.00×)  | 17.81 (0.86×)  | 18.28 (0.84×)  | 27.30 (0.56×)  |
| 4³ (128)   | 3     | 39.56 (1.00×)  | 38.34 (1.03×)  | 36.46 (1.08×)  | 42.53 (0.93×)  |
| 5³ (250)   | 1     | 57.64 (1.00×)  | 62.44 (0.92×)  | 75.25 (0.77×)  | 108.10 (0.53×) |
| 6³ (432)   | 1     | 113.69 (1.00×) | 96.24 (1.18×)  | 89.30 (1.27×)  | 123.65 (0.92×) |

Parallel efficiency at 8 GPUs: 7%, 12%, 7%, 11% respectively.

Observations:
- **KPAR-capped rows (n=3,4).** With 3 k-points and KPAR=4 or 8, the extra partitions have nothing to do and just add sync cost. At 3³/8, CPU time is only 16 s while real time is 27 s — over a third of the wall is communication/launch overhead, not compute.
- **Γ-only rows (n=5,6).** Pure band-parallel over tiny problems: 5³ degrades monotonically past 1 GPU, and 6³ tops out at 4 GPUs with only a 1.27× speedup.
- **The problem is size, not VASP.** Single B200 (180 GiB, 2.3 PFLOPS fp16) can hold and saturate these small Si cells without help. Adding GPUs subdivides already-small work and pays communication on top.

See `strong_scaling.png` (speedup + efficiency curves) and `per_scf_bar.png` (per-SCF breakdown across ngpu).

## Size scaling

Per-SCF cost (`elapsed / scf_done`):

| system       | ngpu | atoms | s/SCF  | notes |
| ------------ | ---- | ----- | ------ | ----- |
| 5³           | 1    | 250   | 2.88   | fully converged |
| 6³           | 1    | 432   | 5.68   | fully converged |
| 10³          | 4    | 2000  | 103.69 | 5 cycles, init-heavy; first cycle = 240 s, remaining 4 avg ≈ 70 s |
| 12³          | 8    | 3456  | 339.29 | 2 cycles only — first 249 s, second 430 s |
| 14³          | 8    | 5488  | 780.69 | 2 cycles only — first 600 s, second 961 s |

Fitting on the 12³→14³ pair (same ngpu, same settings): **T ∝ N^1.80**, consistent with band-projection-dominated Davidson on a GPU where the formal O(N³) asymptote isn't yet reached. Below ~500 atoms the per-SCF cost is init-dominated and the exponent looks sub-linear — not a useful trend point.

**Caveat on 12³/14³.** NELM=2 is too few cycles to separate steady-state per-SCF cost from initialization, and notably the second SCF is longer than the first in both (VASP often switches diag strategy after the first cycle). Treat the 1.80 exponent as a rough lower bound on the real steady-state scaling.

See `time_vs_size.png` (log-log, all ngpu overlaid) and `large_systems.png`.

## Failures: 15³ and 16³

Both crashed before completing any SCF step. `OSZICAR` is **empty** in both cases; `OUTCAR` shows:

- **15³ (6750 atoms, 8 GPU, NBANDS=16880):** stops at `Broyden mixing: mesh for mixing (old mesh)` — killed during charge-mixer setup, almost certainly OOM in mixer arrays.
- **16³ (8192 atoms, 8 GPU, NBANDS=20488):** truncated mid-coordinate write during POSCAR parsing — killed extremely early, before any electronic setup. This is the one run with `NCORE=4` instead of `NCORE=1`; unclear if intentional.

No in-tree stderr to distinguish OOM from wall-time kill, but the 15³ stopping point and the monotone growth of `NBANDS` make OOM the strong prior.

Per-GPU wavefunction memory at Γ (rough):  
N_pw ≈ (grid/2)³ ≈ 5–6e5 plane waves at 16³. `complex f64 × NBANDS × N_pw / ngpu` ≈ 16 B × 20488 × 6e5 / 8 ≈ **24 GiB per GPU just for ψ**, before mixers, projectors, and FFT buffers. That's within a 180 GiB B200 in isolation but evidently over the headroom with the current band/pw partitioning.

## Recommendations for a follow-up sweep

1. **Fill in strong scaling where the GPU is actually working.** Re-run 10³ / 12³ / 14³ at ngpu = 1, 2, 4, 8 with a consistent NELM ≥ 4 and ALGO=Fast. This is the regime where efficiency should rise — the current tree has exactly one data point per large size.
2. **Decouple k-point from band parallelism.** For 3³/4³, re-run with `KPAR=min(ngpu, 3)` and let the remaining ranks do band/pw, so the small sizes are directly comparable with 5³/6³ instead of measuring KPAR saturation.
3. **Recover 15³ and 16³.** Try higher `NCORE` (4 or 8) and/or `NSIM=1` to shrink per-rank wavefunction footprint; or bump to 16 GPUs if the partition permits. 15³'s crash at the Broyden step is a clear memory-pressure signal, not a logic bug.
4. **Use NELM ≥ 4 everywhere.** NELM=2 conflates init with steady-state cost; the 12³/14³ second-cycle-longer-than-first pattern shows the 2-cycle average is not representative.
5. **Commit the job script, POSCAR, POTCAR, and stderr** alongside the INCAR/OSZICAR/OUTCAR. Right now the VASP build version, MPI/GPU binding setup, and crash reasons for 15³/16³ are not recoverable from the tree.

## Artifacts in `analysis/`

- `analyze.py` — regenerable. Parses every `OUTCAR/OSZICAR` under `b200/vasp_dft/`, writes the PNGs and `metrics.json`. Idempotent.
- `metrics.json` — full per-run dump + derived strong-scaling and large-size metrics.
- `strong_scaling.png` — speedup and parallel efficiency vs ngpu for 3³/4³/5³/6³.
- `time_vs_size.png` — log-log per-SCF time vs N_atoms, all ngpu overlaid.
- `per_scf_bar.png` — per-SCF bar chart for the four small sizes × four ngpu values.
- `large_systems.png` — per-SCF cost for 10³/12³/14³ (blue) and failures 15³/16³ (red).
