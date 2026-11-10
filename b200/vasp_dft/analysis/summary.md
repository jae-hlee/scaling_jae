# VASP DFT scaling analysis: Si supercells on B200

Data source: `b200/vasp_dft/<n>x<n>x<n>/<ngpu>/{INCAR,OSZICAR,OUTCAR}`. Silicon diamond **2-atom primitive cell**, replicated n×n×n (so N_atoms = 2·n³). Single-point SCF on the B200 partition (`b200`, QOS `blackwell_test`). Analysis regenerable via `analyze.py`.

## Reading the timings

`OUTCAR` has three kinds of timing, and they tell different stories:

- `LOOP:` is one electronic (SCF) step.
- `LOOP+:` is one ionic step: all its SCF steps plus forces and stress.
- `Elapsed time` adds everything outside the ionic step: setup (reading input, building projectors, initial wavefunctions) and teardown.

Before 2026-11-06, `analyze.py` matched only `LOOP+:` and reported it as the first SCF cycle. That made the 12³ and 14³ runs look as if their second SCF cycle took longer than the first. It does not: the second cycle is shorter than the first in every run. `elapsed / scf_done`, the per-SCF cost used below, still includes setup and teardown (28-79% of elapsed time in the strong-scaling runs, 62-63% at 12³/14³), and at 12³/14³ forces and stress take another 23-24%, so the SCF cycles there are only 13-14% of elapsed time.

## Experiment layout

Two experiments are mixed in the tree and are **not apples-to-apples**:

| regime            | sizes                           | ngpu    | NELM | ALGO    | EDIFF | parallel mode              |
| ----------------- | ------------------------------- | ------- | ---- | ------- | ----- | -------------------------- |
| strong scaling    | 3³, 4³                          | 1,2,4,8 | 20   | default | 1e-6  | `KPAR=<ngpu>`, NKPTS=3     |
| strong scaling    | 5³, 6³                          | 1,2,4,8 | 20   | default | 1e-6  | `NCORE=1` only, NKPTS=1 (Γ)|
| convergent timing | 10³                             | 4       | 5    | Fast    | 1e-5  | `NCORE=1`, NKPTS=1         |
| truncated timing  | 12³, 14³                        | 8       | 2    | Fast    | 1e-4  | `NCORE=1`, NKPTS=1         |
| failed            | 15³, 16³                        | 8       | 2    | Fast    | 1e-4  | 15³: `NCORE=1`; 16³: `NCORE=4` |

Important consequence: the 3³/4³ strong-scaling rows mostly measure **k-point parallelism saturation** (only 3 k-points, so KPAR=4 or 8 over-partitions), while 5³/6³ measure **band/pw parallelism** on a pure Γ calculation. Don't read them as a single curve.

## Headline

| aspect                         | result |
| ------------------------------ | ------ |
| Best speedup in elapsed time   | **1.27× on 4 GPUs at 6³** (432 atoms), 32% parallel efficiency |
| 8 GPU faster than 4 GPU in elapsed time? | **Never** for these 20-cycle runs |
| Steady-state SCF cycle on 8 GPUs | **2.63× faster than 1 GPU at 6³**, 1.76× at 5³; the 8-GPU cycle is the fastest at both Γ-only sizes |
| Size-scaling exponent, 3456 → 5488 atoms, 8 GPU | **N^1.80** for elapsed time per SCF (mostly setup); **N^2.42** for the second SCF cycle alone |
| Largest completed SCF          | 14³ = 5488 atoms, 8 GPU, 2 cycles (114 s and 96 s) inside 1561 s elapsed |
| Largest failed                 | 15³ (6750 atoms) and 16³ (8192 atoms), 0 SCF cycles, see below |

## Strong scaling: elapsed time

Elapsed time (s) from `OUTCAR`, all 20/20 SCF cycles completed:

| n³ (atoms) | NKPTS | 1 GPU          | 2 GPU          | 4 GPU          | 8 GPU          |
| ---------- | ----- | -------------- | -------------- | -------------- | -------------- |
| 3³ (54)    | 3     | 15.28 (1.00×)  | 17.81 (0.86×)  | 18.28 (0.84×)  | 27.30 (0.56×)  |
| 4³ (128)   | 3     | 39.56 (1.00×)  | 38.34 (1.03×)  | 36.46 (1.08×)  | 42.53 (0.93×)  |
| 5³ (250)   | 1     | 57.64 (1.00×)  | 62.44 (0.92×)  | 75.25 (0.77×)  | 108.10 (0.53×) |
| 6³ (432)   | 1     | 113.69 (1.00×) | 96.24 (1.18×)  | 89.30 (1.27×)  | 123.65 (0.92×) |

Parallel efficiency at 8 GPUs: 7%, 12%, 7%, 11% respectively. On 8 GPUs every size is slower than on 1 GPU. The best GPU count is 1 for 3³ and 5³, and 4 for 4³ and 6³.

## Strong scaling: steady-state SCF cycle

Median real time of SCF cycles 2-20 (s), its speedup over 1 GPU, and the two costs that hide it in elapsed time: the first SCF cycle and the time outside the ionic step (setup and teardown):

| n³ (atoms) | ngpu | steady cycle (s) | speedup | first cycle (s) | outside ionic step (s) |
| ---------- | ---- | ---------------- | ------- | --------------- | ---------------------- |
| 3³ (54)    | 1 / 2 / 4 / 8 | 0.43 / 0.31 / 0.22 / 0.22 | 1.00 / 1.37 / 1.96 / 1.97 | 0.66 / 0.53 / 0.42 / 0.43 | 4.8 / 9.9 / 12.5 / 21.5 |
| 4³ (128)   | 1 / 2 / 4 / 8 | 1.18 / 0.87 / 0.66 / 0.67 | 1.00 / 1.36 / 1.79 / 1.76 | 1.40 / 1.08 / 0.86 / 0.87 | 10.9 / 16.6 / 19.2 / 24.7 |
| 5³ (250)   | 1 / 2 / 4 / 8 | 1.50 / 1.21 / 0.93 / 0.85 | 1.00 / 1.24 / 1.60 / 1.76 | 1.17 / 7.09 / 14.40 / 34.43 | 22.3 / 27.3 / 35.9 / 53.6 |
| 6³ (432)   | 1 / 2 / 4 / 8 | 2.71 / 1.92 / 1.26 / 1.03 | 1.00 / 1.41 / 2.14 / 2.63 | 2.57 / 8.58 / 14.56 / 35.16 | 44.5 / 38.2 / 40.2 / 60.2 |

Observations:
- **KPAR-capped rows (n=3,4).** The steady cycle stops improving at 4 GPUs (3 k-points), as expected. Setup time grows with GPU count, which is what makes 8 GPUs slowest in elapsed time. At 3³/8, CPU time is only 16 s while real time is 27 s.
- **Γ-only rows (n=5,6).** The SCF cycle itself does get faster with more GPUs, up to 2.63× on 8 GPUs at 6³. Elapsed time hides this because the first SCF cycle grows with GPU count (2.6 s on 1 GPU, 35 s on 8 at 6³) and setup adds 38-60 s. Both are paid once per run.
- **What this means for longer runs.** From these numbers, 8 GPUs would overtake 1 GPU at 6³ after roughly 26 SCF cycles, and overtake 4 GPUs after roughly 170 (estimates assuming the steady cycle time holds). A relaxation or MD run, with hundreds of SCF cycles, would favor more GPUs; a 20-cycle single point does not.

See `strong_scaling.png` (elapsed-time speedup and efficiency), `strong_scaling_steady.png` (elapsed against steady-cycle speedup) and `per_scf_bar.png` (elapsed / SCF across ngpu).

## Size scaling

Per-SCF cost (`elapsed / scf_done`) against the SCF cycles themselves:

| system       | ngpu | atoms | elapsed / SCF (s) | SCF cycles (s) | outside ionic step (s) |
| ------------ | ---- | ----- | ----------------- | -------------- | ---------------------- |
| 5³           | 1    | 250   | 2.88   | first 1.17, steady 1.50 | 22.3 |
| 6³           | 1    | 432   | 5.68   | first 2.57, steady 2.71 | 44.5 |
| 10³          | 4    | 2000  | 103.69 | first 35.2, steady 24.4 (cycles 2-5) | 278.1 |
| 12³          | 8    | 3456  | 339.29 | 60.5, then 31.3 | 430.0 |
| 14³          | 8    | 5488  | 780.69 | 113.8, then 95.8 | 961.2 |

The ionic step (`LOOP+`) is longer than the sum of its SCF cycles (12³: 248.6 s against 91.8 s), the rest being forces and stress.

Fitting elapsed / SCF on the 12³→14³ pair (same ngpu, same settings) gives **N^1.80**, but 86-87% of that time is outside the SCF cycles, so the exponent mostly measures setup. The second SCF cycle alone scales as **N^2.42** (31.3 s → 95.8 s), and the first as N^1.37. Two points at NELM=2 cannot pin down the steady-state exponent; the second-cycle value is the better estimate, and it is closer to the formal O(N³) than 1.80 suggested.

See `time_vs_size.png` (log-log, all ngpu overlaid) and `large_systems.png`.

## Failures: 15³ and 16³

Both crashed before completing any SCF step (`OSZICAR` empty). The SLURM stderr for both runs shows the same failure: every rank segfaulted (signal 11) inside `MPI_Alltoall`, called from `redis_pw` ← `choleski_orthch`, i.e. during the first orthonormalization of the initial wavefunctions. This happened with `NCORE=1` (15³) and `NCORE=4` (16³) alike.

- **15³ (6750 atoms, 8 GPU, NBANDS=16880):** `OUTCAR` ends at `Broyden mixing: mesh for mixing (old mesh)`; VASP's own memory estimate is ~46 GB per rank.
- **16³ (8192 atoms, 8 GPU, NBANDS=20488):** `OUTCAR` is cut off mid-coordinate list.

The `OUTCAR` end points are where buffered output stopped, not where the runs died. There is no CUDA or allocation error, and the estimated memory is well under a 180 GiB B200, so these are not out-of-memory failures. The likely cause is the MPI all-to-all itself at this size (MPI ran over UCX TCP, `UCX_TLS=tcp,self`; a message count exceeding 32-bit limits is one candidate). Not confirmed.

## Recommendations for a follow-up sweep

The B200 cluster these ran on has been decommissioned, so these apply to any rerun on other hardware.

1. **Report the steady-state SCF cycle, not only elapsed time.** Parse `LOOP:` lines; elapsed time on short runs is dominated by setup and the first cycle.
2. **Fill in strong scaling where the GPU is actually working.** Re-run 10³ / 12³ / 14³ at ngpu = 1, 2, 4, 8 with a consistent NELM ≥ 4 and ALGO=Fast.
3. **Decouple k-point from band parallelism.** For 3³/4³, re-run with `KPAR=min(ngpu, 3)` and let the remaining ranks do band/pw, so the small sizes are directly comparable with 5³/6³ instead of measuring KPAR saturation.
4. **Recover 15³ and 16³.** The crash is in MPI communication, not memory: first try a GPU-aware transport instead of `UCX_TLS=tcp,self` (NCCL / UCX over NVLink), then more ranks or a different band distribution to shrink each all-to-all message.
5. **Find out why the first SCF cycle grows with GPU count** on the Γ-only cells (6³: 2.6 s on 1 GPU, 35 s on 8). It is the main reason more GPUs lose in elapsed time.
6. **Commit the job script, POSCAR, POTCAR, and stderr** alongside the INCAR/OSZICAR/OUTCAR. (The POSCAR, job scripts and stderr of these runs survive locally in `dsai_archive/`.)

## Artifacts in `analysis/`

- `analyze.py`: regenerable. Parses every `OUTCAR/OSZICAR` under `b200/vasp_dft/`, writes the PNGs and `metrics.json`. Idempotent.
- `metrics.json`: full per-run dump (per-SCF-cycle times in `scf_real_s`, ionic steps in `ionic_step_real_s`, `outside_ionic_s`) plus derived strong-scaling, steady-state strong-scaling and large-size metrics.
- `strong_scaling.png`: elapsed-time speedup and parallel efficiency vs ngpu for 3³/4³/5³/6³.
- `strong_scaling_steady.png`: elapsed-time speedup next to steady-state SCF-cycle speedup.
- `time_vs_size.png`: log-log elapsed / SCF vs N_atoms, all ngpu overlaid.
- `per_scf_bar.png`: elapsed / SCF bar chart for the four small sizes × four ngpu values.
- `large_systems.png`: elapsed / SCF for 10³/12³/14³ (blue) and failures 15³/16³ (red).
