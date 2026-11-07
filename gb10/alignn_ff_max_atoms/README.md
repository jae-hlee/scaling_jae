# ALIGNN-FF maximum atoms on NVIDIA GB10

How many atoms ALIGNN-FF can evaluate on GB10 nodes (121.7 GiB unified memory), with and
without forces, for four models: directly on one GB10, chunked on one GB10, and chunked across three to six.
Single points only, not MD. Measured 2026-11-01 to 03 on atomgptlab with alignn at commit `6a20fb5`.

Models:

- AFF-r2SCAN: the paper's force field (results folder `matpes_r2scan_paper`), 2+2, hidden 128, figshare file 67360764, md5 `92cfe295`, job 19126
- `alignn_ff_mptrj`: 4+4, hidden 256, figshare file 70840341, md5 `3e70af9b`, job 19127
- `alignn_ff_matpes_pbe`: 4+4, hidden 256, figshare file 70840662, md5 `27010124`, job 19128
- `alignn_ff_matpes_r2scan`: 4+4, hidden 256, figshare file 70840338, md5 `92bd57fb`, job 19129

## Largest single evaluation on one GB10

Pristine Si diamond (8 i^3 atoms), stepping i up until the next size is killed by the host
(exit 137), one process per path. Each cell shows the largest size that fit. Time is the median of
three calls. Through ASE, each call is three model passes, because the calculator reruns the model
for energy, forces and stress. The energy-only path sums per-atom energies with no autograd, rather
than using the model's readout.

Differences among the `alignn_ff_*` models are where the host killed each run, as memory per atom is
identical. i steps by 2 below 15, so 13,824 (i=12) is the last ASE size tried below the kill at
21,952, and 17,576 (i=13) was not tried.

| Path | AFF-r2SCAN | `alignn_ff_mptrj` | `alignn_ff_matpes_pbe` | `alignn_ff_matpes_r2scan` |
|---|---|---|---|---|
| energy only | 551,368 (0.195 MB/atom, 9.7 s) | 238,328 (0.386 MB/atom, 13.1 s) | 238,328 (0.386 MB/atom, 13.1 s) | 262,144 (0.386 MB/atom, 15.0 s) |
| E+F+S, ASE calculator | 54,872 (1.953 MB/atom, 7.6 s) | 13,824 (7.019 MB/atom, 6.5 s) | 13,824 (7.019 MB/atom, 6.5 s) | 13,824 (7.019 MB/atom, 6.5 s) |
| E+F+S, `forward_tensors` | 140,608 (0.749 MB/atom, 4.6 s) | 39,304 (2.481 MB/atom, 4.7 s) | 39,304 (2.481 MB/atom, 4.8 s) | 46,656 (2.481 MB/atom, 5.6 s) |
| E+F+S, `forward_tensors` + layer checkpointing | 195,112 (0.526 MB/atom, 8.5 s) | 97,336 (1.139 MB/atom, 16.0 s) | 97,336 (1.139 MB/atom, 16.0 s) | 97,336 (1.139 MB/atom, 15.9 s) |

The ASE row for AFF-r2SCAN matches `../alignn_ff/` (job 13474) at the largest cell: the same
size and energy, 107.15 against 107.13 GB, 7.60 against 7.59 s. On the energy-only and checkpointed
paths, energy per atom stays within 1.1e-5 eV of the small-cell value for all four models. Only
AFF-r2SCAN passed the 470k atoms where `b200/alignn_ff/` hit its cliff. It shows no cliff up to
551,368 atoms, but this is weak evidence: it is a different model and energy path, and this Si cell
has 8.8M triplets against about 68M for Cu at the onset. The `alignn_ff_*` models stopped below 470k. The ASE path
wanders more, up to 1.2e-3 eV/atom (AFF-r2SCAN, 32,768 atoms).

## Chunked runs

The cell is cut into cubes. Each cube is evaluated with a surrounding halo on one GPU, and only
the atoms it owns count. Rattled Si (0.05 A), energy and forces, no stress. None of these runs
reached a memory limit, so these are the largest sizes tried rather than maxima. Force runs use layer
checkpointing. Time is for the slowest task. Peak is the GPU memory of the largest piece. Every task
also holds the whole cell on the host, peaking at 5.6 to 6.0 GB per task at 64M and 73.9 GB at 1B, in the
memory the GB10 shares between its CPU and GPU (`host_peak_gb` in the JSONs divides kB by 1e6, so it
reads 72.2 at 1B).

| Model | Atoms | GB10s | Computes | Cubes | Largest piece | Peak (GB) | Time (s) | E (eV/atom) |
|---|---|---|---|---|---|---|---|---|
| AFF-r2SCAN | 8M | 1 | energy | 64 | 261,205 | 49.3 | 264 | -8.69961 |
| AFF-r2SCAN | 8M | 1 | forces | 216 | 106,067 | 45.0 | 979 | -8.69961 |
| AFF-r2SCAN | 64M | 4 | energy | 512 | 261,206 | 49.3 | 351 | -8.69959 |
| AFF-r2SCAN | 64M | 5 | forces | 1,728 | 106,054 | 45.0 | 1,379 | -8.69959 |
| AFF-r2SCAN | 1B | 3 | energy | 8,000 | 261,222 | 49.3 | 3,235 | -8.69960 |
| AFF-r2SCAN | 1B | 4 | forces | 46,656 | 71,579 | 30.2 | 14,522 | -8.69960 |
| `alignn_ff_mptrj` | 8M | 1 | energy | 216 | 106,002 | 39.3 | 1,178 | -5.35097 |
| `alignn_ff_mptrj` | 8M | 1 | forces | 512 | 57,375 | 49.9 | 4,620 | -5.35097 |
| `alignn_ff_mptrj` | 64M | 4 | energy | 1,331 | 128,175 | 47.6 | 1,473 | -5.35095 |
| `alignn_ff_mptrj` | 64M | 6 | forces | 4,096 | 57,379 | 49.9 | 6,231 | -5.35095 |
| `alignn_ff_matpes_pbe` | 8M | 1 | energy | 216 | 106,002 | 39.3 | 1,176 | -5.38137 |
| `alignn_ff_matpes_pbe` | 8M | 1 | forces | 512 | 57,375 | 49.9 | 4,621 | -5.38137 |
| `alignn_ff_matpes_r2scan` | 8M | 1 | energy | 216 | 106,002 | 39.3 | 1,175 | -8.72789 |
| `alignn_ff_matpes_r2scan` | 8M | 1 | forces | 512 | 57,375 | 49.9 | 4,608 | -8.72789 |

AFF-r2SCAN uses its exact 20.01 A halo. The `alignn_ff_*` models use a 20 A halo instead of their exact
40 A. Cutting the halo to 20 A changed forces by 2e-9 to 2e-8 eV/A in a float64 check on 8,000 atoms of rattled
Si (`results/*/halo_*.json`). At the exact halo in float32, chunked results on 8,000 atoms of SiGe
differ from the full cell by at most 6e-7 eV/atom and 1e-3 eV/A (`results/*/check_*.json`), which is
roundoff: the same SiGe cells in float64 agree to 2e-12 eV/A and 2e-15 eV/atom (`results/*/f64_sige_exact_19291.json`).
On that SiGe cell in float64, the 20 A halo changes forces by 1.2e-8 to 1.3e-7 eV/A (`results/*/f64_sige_h20_19291.json`),
up to 8x the Si figure but far below float32 roundoff. A 64M cell cannot be checked against a full
evaluation on a GB10.

Each 64M force task did 1.33x the 8M workload and took 1.32-1.41x as long. AFF-r2SCAN's six tasks ran
on five nodes, with atomgptlab05 running two in turn (2,689 s in total). `alignn_ff_mptrj` ran on six nodes. Tasks
started as nodes freed up, so the 64M force runs took 45 min (AFF-r2SCAN) and 1 h 59 min (`alignn_ff_mptrj`) from first start to last
finish. The 1B runs were 10 tasks each. With forces, atomgptlab06, 07 and 08 ran three tasks each in turn and
atomgptlab04 ran one, 13 h from first start to last finish. Without forces, the 10 tasks ran on 06, 07 and 08.

## Files

- `ff_max_atoms.py`: `sweep`, `chunk` and `reduce` modes
The scripts expect alignn at commit `6a20fb5` and the four checkpoints under `/data/jlee859/ffmax`.

- `stage1.sbatch`: sweeps, chunk check, 8M timing samples and halo scan for one model (jobs 19126-19129)
- `stage2_submit.sh`, `chunk.sbatch`, `reduce.sbatch`: chunked runs (jobs 19140-19151)
- `stage3_submit.sh`: 64M without forces on 4 nodes, 6 tasks each (jobs 19182-19184, 19200). Tasks 4 and 5 of `alignn_ff_mptrj` were rerun after failing on a node without `/data`, and energy per atom agrees with the force runs to 8e-9 eV
- `stage4_submit.sh`: 1B atoms with AFF-r2SCAN, 10 tasks with and without forces, after a 3-cube pilot (jobs 19218-19222)
- `stage5.sbatch`: the 8,000-atom SiGe chunk check in float64, exact halo for all four models and 20 A for the `alignn_ff_*` models (job 19291)
- `logs/`: Slurm output of every job above (stage1_, chunk_<job>_<task>, reduce_, stage5_), with start and end times,
  one line per cube, and exit status. atomgptlab04 prints times in UTC, the other nodes in US Eastern time.
- `results/<model>/`: result JSONs. The per-task force arrays of the 64M and 1B runs (768 MB and 12 GB each) are
  not kept, and the `*_reduced.json` files hold their force statistics.
