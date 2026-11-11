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
- `cells.py`: structures other than the built-in Si cells (FCC Al and Cu with an optional lattice constant, ideal beta-cristobalite SiO2, grid-placed water at 1.0 g/cm^3, Voronoi polycrystals, and Si/SiO2/Al gate stacks), written as `.pos.npy`/`.Z.npy`/`.json`, plus the cube-and-halo geometry for any orthorhombic box and crystal pieces generated from the lattice for `ff_stream.py`. `python cells.py test` checks the geometry against brute force and the builders, with numpy and scipy only. Building a polycrystal takes about 86 bytes per atom of RAM (about 90 GB at one billion atoms)
- `ff_any.py`: `sweep` and `chunk` for any `cells.py` structure, importing the model and evaluation code from `ff_max_atoms.py` (unchanged, since stages 4 and 5 check its md5). Records bonds and bond angles per atom, and whether any atom reaches the neighbor cap (`neighbor_cap_binds`)
- `stage6.sbatch`, `summarize_cost.py`: AFF-r2SCAN on Al, Cu, water and a Si polycrystal: exactness in float64, largest single graph, and chunked throughput on 1M-atom cells (job 19326, results in `results/matpes_r2scan_paper/stage6/`, log `logs/stage6_19326.out`; see "Other structures" below). `summarize_cost.py` prints the tables from the JSONs and needs only the Python standard library
- `relax.py`: FIRE relaxation at fixed cell of a `cells.py` structure, as one graph or chunked, resumable; writes the relaxed structure and per-atom energies
- `ff_stream.py`: chunked evaluation of a crystal whose pieces are generated from the lattice, so no task holds the cell; with forces each task keeps only its x slab, and `reduce` adds slabs block by block
- `prep_v12.py`, `cliff_ops.py`: the B200 runs' checkpoint as a model folder for `ff_any.py` (5 A, 12 neighbors), and tests of the 2^26 line-graph threshold (graph construction against a CPU reference, layer operations past 2^26 rows)
- `energy_maps.py`, `maps.sbatch`: per-atom energy maps for figures from the stage-7 and stage-10 runs (a 5 A slab, each atom's energy relative to the median of its material, atom by atom or binned), written as small files to `results/<model>/maps/` for `analysis/gb10_energy_maps_figure.py`
- `stacking_fault.py`, `stage14_gsf.sbatch`: stacking-fault and surface energies of FCC Al or Cu (stage 14, below)
- `stage15_check.sbatch`, `stage15_submit.sh`: the billion-atom runs with the retrained model (stage 15, below). `chunk_any.sbatch` and `stream.sbatch` take `SMOOTH=1` from the environment for a model with smooth_messages, and `SMOOTH=1 sbatch maps.sbatch` makes the energy maps of the retrained model's runs
- `build.sbatch`, `chunk_any.sbatch`, `reduce_any.sbatch`, `stream.sbatch`, `stream_reduce.sbatch`: generic wrappers (build a structure; one array task of `ff_any.py` or `ff_stream.py`; reduce)
- `stage7_poly.sbatch`, `stage7_1B_submit.sh`, `stage8_cliff.sbatch`, `stage9_check.sbatch`, `stage9_submit.sh`, `stage10_check.sbatch`, `stage10_submit.sh`: stages 7 to 10, below
- `relax_check.py`, `relax_check.sbatch`: forces of a relaxed cell, which atoms keep large forces, and a test for force jumps at the graph cutoffs
- `map_profile.py`, `maps_figure.sbatch`: how far the energy disturbance reaches into each layer of the gate stack, and the energy-map figure on the cluster
- `stmv_convert.py`, `stage11_stmv.sbatch`: the NAMD STMV benchmark as a `cells.py` structure and one evaluation of it (stage 11, below)
- `smooth_messages.patch`, `smooth_eval.py`, `smooth_check.py`, `prep_smooth.py`, `smooth_train_config.py`, `smooth_train_base.json`, `stage12_check.sbatch`, `stage12_probe.sbatch`, `stage12_train.sbatch`: switch weights on the bond and angle messages of ALIGNN, their checks, and retraining (stage 12, below)
- `compare_models.py`, `stage13_compare.sbatch`, `stage13_rerun.sbatch`, `stage13_dryrun.sbatch`: stage 13 (below). Errors of AFF-r2SCAN and the retrained model on the run's test frames and a sample of its training frames, energy conservation in float64 (Si at 1500 K, water at 300 K with and without the 52-neighbor cap, same start for both), then the stage-7 relaxation, the bond-crossing test and the STMV evaluation with the retrained model. The dry run tests the script on the 2-epoch probe model

## Stages 7 to 10 (run 2026-11-08 to 09)

AFF-r2SCAN on GB10 nodes unless noted, single precision, exact 20.01 A halo. Result JSON, maps and logs are in `results/matpes_r2scan_paper/{stage7,stage9,stage10,maps}/`, `results/cliff/` (stage 8) and `logs/`. Force arrays, per-atom energies and structures stay on the cluster. Host memory is `host_peak_gb` times 1.024, because that field divides kB by 1e6.

| Cell | Atoms | Computes | Cubes | Tasks (GB10s) | Largest piece | GPU (GB) | Host (GB) | Slowest (s) | GB10-hours | E (eV/atom) | Jobs |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Si, rattled 0.05 A, `ff_stream.py` | 2,000,376,000 | energy | 12,167 | 10 (7) | 328,336 | 62.0 | 2.5 | 8,017 | 17.0 | -8.699596 | 21590-91 |
| Si, rattled 0.05 A, `ff_stream.py` | 2,000,376,000 | forces | 54,872 | 20 (9) | 104,194 | 44.2 | 5.7 | 12,988 | 67.4 | -8.699596 | 21592-93 |
| Si polycrystal, 4,800 grains, as built | 1,021,299,267 | forces | 50,653 | 10 (9) | 69,056 | 29.2 | 62.1 | 15,395 | 41.9 | -8.696316 | 21583-85 |
| Si/SiO2/Al gate stack, as built | 102,361,540 | forces | 15,925 | 6 (6) | 39,059 | 49.0 | 8.3 | 6,034 | 9.9 | -8.073493 | 21578-80 |

GB10-hours sum the tasks' wall time and leave out the structure build and the reduce step. The two 2-billion-atom runs used different pieces and their energies differ by 1.6e-10 eV/atom. The polycrystal is 51.2 meV/atom above perfect Si (-8.747558), against 112.4 for the as-built 8-grain cell below, whose grains are 2.2 times smaller.

- **Stage 7, relaxation (job 20863).** An 8-grain Si polycrystal of 155,281 atoms (146.61 A cube), one periodic graph with checkpointing, FIRE at fixed cell, 2,000 steps at 6.8 s. It did not reach the 0.05 eV/A target. Energy -8.635123 to -8.684895 eV/atom (97% of the drop in 100 steps), largest force 23.6 to 0.496 eV/A, RMS per component 0.874 to 0.00704 eV/A. At the end 1,934 atoms (1.2%) are above 0.05 eV/A, 37 above 0.2 and 3 above 0.4.
- **Why it stalled (job 21603, `relax_check.py`).** The line graph keeps a bond only up to 3.5 A with no smoothing of the angle terms, so forces are discontinuous there. Moving a bond across 3.5 A by 0.0004 A changed forces by 0.47-0.51 eV/A and energy by 0.02-0.13 eV (ratio to the same-side change 16-38, against 0.23 for a control bond). At the 5 A cutoff, where bond features taper to zero, the change on the one bond tested was 0.004 eV/A. 18 of the 20 atoms with the largest forces have a bond within 0.01 A of 3.5 A, against 2.3% of all atoms. Energy conservation in MD was not tested.
- **Stage 8, the 2^26 cliff (job 20862).** v12 checkpoint, Cu FCC, energy only. Chunked on a GB10 (27 cubes, one for 256 atoms, at most 104,551 atoms per piece) gives 0.604014 eV/atom at 256; 442,368; 470,596; 562,432 and 780,448 atoms, against 0.6040, 0.6040, 0.6070, 0.7027 and 0.8449 for the B200 single graph. On the GB10 the line graphs at 442,368 and 470,596 atoms (0.949 and 1.010 x 2^26 edges) matched the same construction in 64-bit integers on the CPU, and 0 of 36 layer-operation tests failed (27 past 2^26 rows, 9 just below). The error is in evaluating single graphs, not in the model, but the operation responsible was not found. The GB10 cannot hold the full single graph at 470,596 atoms.
- **Stage 9 checks (job 20864).** In float64 on 8,000 atoms of rattled Si, lattice-generated pieces against the full cell agree to 1.8e-13 eV/A, with identical energy per atom in all printed digits. One task against two: 3.3e-14 eV/A and 1.8e-15 eV/atom.
- **Stage 10 checks (job 20865).** In float64 on a 772-atom stack, chunked against full cell agree to 1.8e-15 eV/atom and 1.1e-12 eV/A. The neighbor cap does not bind there or in 40 pieces sampled through the height of an 897,184-atom stack, which gave 110 GB10-hours per 10^9 atoms.
- **Energy maps (jobs 21737, 21739-40, `energy_maps.py`, `map_profile.py`).** A 5 A slab through each cell, each atom relative to the median of its material (in the stack, of its element in its layer). The median Si atom matches perfect Si within 1 meV in all three. In the relaxed polycrystal, 10% of the atoms are more than 0.24 eV above the median and 1% more than 0.64 eV. In the stack, the mean energy differs by more than 1 meV only within 5.0 A of the interface in Si, 6.3 A in Al, and 7.0 and 7.7 A in the oxide. `analysis/gb10_energy_maps_figure.py` draws the figure from these maps.

A halo of the model's reach is exact only if no atom has more neighbors within the 5 A cutoff than the model's cap of 52. Si and SiGe have 28 and FCC Al and Cu 42, so the cap does not affect any run above (the 20 A halo of the `alignn_ff_*` runs is a separate, measured truncation). Liquid water has about 52. When the cap binds, whether bond j-i exists depends on j's other neighbors, which reaches past the receptive field `reach()` computes, so a halo of that size is no longer exact.

With AFF-r2SCAN in float64 (job 19326), the 20 A halo changes forces on 1,536 atoms of water by at most 8.2e-8 eV/A and energies by 1.8e-11 eV/atom, four orders of magnitude below float32 roundoff. A 25 A halo (4.8e-13 eV/A) or lifting the cap (4.1e-13 eV/A) makes it exact again, which confirms the cap as the cause.

## Stage 11: a complete virus (job 21741, 2026-11-09)

One evaluation of energy and forces with AFF-r2SCAN on one GB10, single precision, of the all-atom NAMD STMV benchmark (capsid, RNA, water and ions), as distributed. Result in `results/matpes_r2scan_paper/stage11/`, log in `logs/stage11_21741.out`. This is a test of size and cost only: the model's accuracy on biomolecules has not been tested.

| Atoms | Box (A) | Water | Cubes | Largest piece | Evaluated/owned | GPU (GB) | Host (GB) | Wall (s) | GB10-hours | Atoms/s | E (eV/atom) | Largest force component (eV/A) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,066,628 | 216.832 cube | 84% of atoms | 729 | 22,263 | 14.9 | 53.7 | 2.4 | 3,084 | 0.86 | 346 | -5.544803 | 20.84 |

The rate is 28% below the 481 atoms/s sampled for water in stage 6, in part because these pieces are smaller (14.9 atoms evaluated per atom owned against 12.1). The 52-neighbor cap binds in every piece, as in water. At this rate the 305-million-atom SARS-CoV-2 envelope would take about 245 GB10-hours per force evaluation; that structure was not run.

## Stage 12: removing the force jump (2026-11-09 to 10)

`smooth_messages.patch` adds an optional C2 switch weight to the bond messages (4 to 5 A) and the angle messages (product of two switches, 3.0 to 3.5 A) of a copy of ALIGNN (`alignn_smooth` on the cluster; `alignn_6a20fb5` stays unpatched). A weight is needed because a new edge enters a message with a nonzero coefficient, so tapering features cannot remove the jump.

- **Checks (job 21742, `logs/stage12check_21742.out`, `results/matpes_r2scan_paper/stage12/`).** Flag off reproduces unpatched ALIGNN (0 in energy, 2e-14 in forces). Forces match finite differences within 1.5e-6 eV/A. With the flag on and AFF-r2SCAN's weights, which were not trained for this model, the force change across 3.5 A is 0.0017-0.0023 eV/A, against 0.47-0.51 before. Chunked against full cell in float64: 1.4e-14 eV/atom and 3.5e-12 eV/A on the 772-atom stack.
- **Speed probe (job 21743, `logs/stage12probe_21743.out`).** 4,000 training frames at batch 8 took 21.6 s per epoch on one GB10.
- **Training (job 21744, ended 2026-11-10).** AFF-r2SCAN's recorded settings on MatPES r2SCAN on one GB10: 100 epochs, batch 64, learning rate 1e-3, a random split of 347,889 / 19,327 / 19,328 frames. About 18 minutes per epoch. The lowest validation loss, 0.1834, was at epoch 99. That checkpoint is `models/aff_smooth` on the cluster (`best_model.pt` md5 8e4976dd2cd536de84044c7470726809). Its accuracy and energy conservation are in stage 13.

Swaps at the 52-neighbor cap remain a separate possible discontinuity where the cap binds (water, STMV). Stage 13 found no effect of the cap on energy conservation in water.

## Stage 13: the retrained model against AFF-r2SCAN (job 21860, 2026-11-10)

`compare_models.py` on one GB10, both models through the patched ALIGNN copy (switch weights off for AFF-r2SCAN, on for the retrained model). Results in `results/matpes_r2scan_paper/stage13/test_21860.json` and `nve_21860.json`, log in `logs/stage13compare_21860.out`. Per-frame errors and the energy traces (`.npz`) stay on the cluster.

Errors against MatPES r2SCAN, single precision. "Test" is the 19,328 test frames of the retraining run, "train" a sample of 2,000 of its training frames. No frame failed.

| Frames | Model | E MAE (meV/atom) | E RMSE (meV/atom) | F MAE (eV/A) | F RMSE (eV/A) | Largest F error (eV/A) |
|---|---|---:|---:|---:|---:|---:|
| Test | AFF-r2SCAN | 39.78 | 172.15 | 0.1925 | 1.168 | 421.6 |
| Test | retrained | 42.32 | 211.74 | 0.1383 | 0.526 | 176.5 |
| Train | AFF-r2SCAN | 40.60 | 115.83 | 0.1775 | 0.725 | 15.7 |
| Train | retrained | 36.09 | 109.22 | 0.1188 | 0.210 | 3.2 |

On the test frames the retrained model's energy error is 6% higher and its force error 28% lower. The retraining run's split is ALIGNN's own split of this data with seed 123. AFF-r2SCAN's split file was not found, so it is assumed, not confirmed, that AFF-r2SCAN never trained on these test frames; if it did, its test errors here are too low.

Energy conservation, float64, velocity Verlet, 2,000 steps from the same start for both models. Drift is the change of total energy from the first to the last step.

| System | Atoms | Start | Step (fs) | Model | Drift (meV/atom) | Largest deviation (meV/atom) | Final T (K) |
|---|---:|---|---:|---|---:|---:|---:|
| Si, rattled 0.05 A | 512 | 1500 K | 1.0 | AFF-r2SCAN | +24.55 | 25.11 | 1,032 |
| | | | | retrained | +0.084 | 0.117 | 927 |
| Water | 1,536 | 300 K | 0.5 | AFF-r2SCAN | +1,404 | 1,404 | 7,724 |
| | | | | retrained | +0.91 | 0.92 | 698 |
| Water, cap lifted to 10,000 | 1,536 | 300 K | 0.5 | AFF-r2SCAN | +1,461 | 1,464 | 8,173 |
| | | | | retrained | +0.57 | 0.59 | 682 |

- AFF-r2SCAN does not conserve energy: Si gains 24.5 meV/atom in 2 ps, and water heats to thousands of kelvin within 1 ps. The retrained model's drift is about 290 times smaller in Si and 1,500 to 2,600 times smaller in water.
- Lifting the 52-neighbor cap changes neither model's behavior in water, so the cap is not what breaks energy conservation there. Without the cap the models see more bonds than in training, so only the conservation of those two runs means anything.
- With the retrained model the water goes from 300 K to about 670 K within 500 steps and stays near 690 K, while the total energy is conserved to within 1 meV/atom. The water cell is used as built, not equilibrated.
- Not explained: the retrained model's energy in water rises steadily, and 3 of the 2,000 steps change the cell's energy by more than 20 times the median step (at most 141 meV for the cell, against a median of 6.1 meV), with and without the cap. Si has no such step.

### Reruns with the retrained model (job 21861, 2026-11-11)

`stage13_rerun.sbatch` on one GB10, 2 h 19 min in all. Results in `results/matpes_r2scan_paper/stage13/` (`relax_poly_si_150k_smooth.json`, `relax_check_smooth_trained_21861.json`, `stmv_forces_smooth_21861.json`), log in `logs/stage13rerun_21861.out`. Structures, forces and per-atom energies stay on the cluster.

- **Relaxation.** The stage-7 run again: the same 155,281-atom polycrystal as built, one graph, FIRE at fixed cell, target 0.05 eV/A. It converged in 658 steps (7.4 s each, 1.35 h), where AFF-r2SCAN stopped at 0.496 eV/A after 2,000. Energy -8.612310 to -8.664916 eV/atom (a drop of 52.6 meV/atom), largest force 21.3 to 0.0479 eV/A, RMS per component 0.00022 eV/A, no atom above 0.05 eV/A.
- **Bond-crossing test** (`relax_check.py`, trained weights). Ratio of the force change across a cutoff to the change on one side: 0.22 for three bonds at 3.5 A, 0.23 for one at 5 A and 0.22 for the control bond, the value of a smooth function. AFF-r2SCAN gave 16 to 38 at 3.5 A.
- **STMV.** 1,066,628 atoms in 729 cubes (largest piece 22,263 atoms), GPU 58.3 GB, host 2.4 GB, 3,320 s (0.92 GB10-hours, 321 atoms/s; AFF-r2SCAN 3,084 s). E -5.449249 eV/atom and largest force component 8.57 eV/A, against -5.544803 and 20.84 with AFF-r2SCAN. As before this is a test of size and cost only, and the neighbor cap binds in every piece.

The retrained model took 8% to 9% longer than AFF-r2SCAN in these two runs (7.4 against 6.8 s per relaxation step, 3,320 against 3,084 s for STMV). Stage 14 with the retrained model is in the stage 14 section.

## Stage 15: the billion-atom runs with the retrained model (written 2026-11-11, not yet run)

The force runs of a billion atoms or more again, with the stage-12 model (`models/aff_smooth`), the same cells and the same piece sizes, results in `results/aff_smooth/stage15/`. `bash stage15_submit.sh forces` submits the 1,021,299,267-atom polycrystal (`ff_any.py`, 10 tasks, with per-atom energies), 10^9 atoms of rattled Si (`ff_stream.py`, 36^3 cubes as in stage 4, 12 tasks) and 2,000,376,000 atoms of rattled Si (`ff_stream.py`, as in stage 9, 20 tasks): about 163 GB10-hours, AFF-r2SCAN's 149.5 times the 1.09 slowdown of stage 13. Modes `energy` (the two energy-only Si runs, about 29) and `stack` (the gate stack, about 11) are separate. Every array waits for `stage15_check.sbatch`, which compares chunked with full-cell results in double precision through both programs with the trained weights and fails above 1e-9 eV/A or 1e-11 eV/atom.

Stage 4 ran the 10^9-atom Si cell with `ff_max_atoms.py`, which cannot take the retrained model, so this run uses `ff_stream.py` and its random displacements differ from stage 4's (same lattice and amplitude).

## Stage 14: stacking-fault energies of Al (job 21783, 2026-11-09)

A check before any dislocation or strain simulation: can these models describe slip in aluminum? `stacking_fault.py` (`stage14_gsf.sbatch`) follows the test of Choudhary et al., J. Phys.: Condens. Matter 27, 015003 (2015), in double precision on one GB10 (3 h 6 min for the four models). For each model it finds the lattice constant from the energy of FCC Al against a, builds a 576-atom (111) slab at that lattice constant (4 x 3 in plane, 24 layers, 15 A of vacuum), moves the upper half rigidly along [11-2] and [1-10], and records the energy change per area as built and with every atom relaxed along z (FIRE, 0.005 eV/A; every point converged within 77 steps). The (111) surface energy comes from the same slab. A 36-layer slab gives the same intrinsic fault energy to 0.001 mJ/m^2, and the energy returns to zero after one full period.

Energies in mJ/m^2, relaxed with the as-built value in parentheses. The first two columns are from Table 1 of the paper.

| Quantity | Exp. or DFT | COMB3 | AFF-r2SCAN | `alignn_ff_matpes_r2scan` | `alignn_ff_matpes_pbe` | `alignn_ff_mptrj` |
|---|---|---|---|---|---|---|
| Lattice constant (A) | 4.05 | 4.05 | 3.982 | 3.989 | 4.041 | 4.044 |
| Bulk modulus (GPa) | 79 | 83 | 73 | 77 | not reliable | not reliable |
| (111) surface energy | 750 to 980 | 720 | 469 (473) | 442 (442) | 461 (461) | 83 (85) |
| Intrinsic stacking fault | 116 to 164 | 146 | 24 (24) | 29 (29) | 43 (44) | 28 (28) |
| Unstable stacking fault | 154 to 291 | 194 | 171 (182) | 199 (226) | 150 (175) | 43 (48) |
| Maximum along [1-10] | 250, 663 | 552 | 384 (595) | 365 (619) | 289 (520) | 102 (207) |
| Atoms on top, along [11-2] | | | 438 (1272) | 341 (1166) | 310 (1128) | 118 (542) |

- **The intrinsic fault is 3 to 7 times too low in every model** (24 to 43 against 116 to 164), and the surface energy is about half (far less for `alignn_ff_mptrj`, whose unstable fault is also far too low). The unstable fault and the bulk modulus of the two r2SCAN models are reasonable. With a fault this cheap, a strained Al polycrystal would fill with wide stacking faults and twins, so a dislocation simulation with these models would not be aluminum.
- **The 3.5 A cutoff shows in the curves.** The as-built curve along [11-2] steps by 25 to 42 mJ/m^2 (5 for `alignn_ff_mptrj`), 13 to 23 times its typical bend, at 0.59 A for AFF-r2SCAN and 0.95 to 1.00 A for the others. A bond across the fault passes 3.5 A at 0.61 and 1.01 A of displacement when a = 3.98 A, and at 0.69 and 0.96 A when a = 4.04 A.
- **The energy also jumps at the 5 A bond cutoff.** In the lattice-constant scan the energy rises by 6.9 (AFF-r2SCAN), 14.0, 11.0 and 5.1 meV/atom above the trend between a = 4.080 and 4.091 A, where the third neighbor shell (24 per atom, at a sqrt(3/2)) passes 5 A. That is about 0.3 meV per bond for AFF-r2SCAN, which fits with the one bond tested in stage 7 showing no visible change, but a whole shell passes at once in an FCC metal, 2.5% above the lattice constant of the r2SCAN models and 1% above that of the other two. The bulk moduli first fitted for `alignn_ff_matpes_pbe` and `alignn_ff_mptrj` (122 and 74 GPa) spanned this jump and are not reliable; `stacking_fault.py` now lists such jumps and keeps them out of the fit.

Results are in `results/<model>/stage14/gsf_al_21783.json` (every curve, the lattice-constant scan and a 12 x 21 as-built gamma surface) with a quick-look `.png` each.

### Copper (job 21849, 2026-11-10)

The same test with `ELEMENT=Cu` (2 h 51 min; every relaxed point converged within 73 steps, the 36-layer check agrees to 2e-7 mJ/m^2, the energy returns to zero after one period). Reference fault energies are DFT-PBE values with the range of other calculations and experiment from Muzyk and Kurzydlowski, J. Min. Metall. Sect. B 55, 271 (2019), Table 1.

| Quantity | Reference | AFF-r2SCAN | `alignn_ff_matpes_pbe` | `alignn_ff_mptrj` | `alignn_ff_matpes_r2scan` (not at its own lattice constant) |
|---|---|---|---|---|---|
| Lattice constant (A) | 3.615 (experiment) | 3.581 | 3.640 | 3.610 | 3.615 (fallback) |
| Bulk modulus (GPa) | | 125 | 122 | 101 | |
| (111) surface energy | | 1170 (1174) | 1158 (1162) | 90 (91) | 1165 (1293) |
| Intrinsic stacking fault | 45 (38 to 58 calculated, 40 to 45 measured) | 25 (25) | 66 (66) | 34 (34) | 97 (91) |
| Unstable stacking fault | 161 (158 to 184 calculated) | 173 (189) | 155 (162) | 169 (184) | 192 (188) |
| Maximum along [1-10] | | 562 (798) | 492 (675) | 487 (694) | 564 (716) |
| Atoms on top, along [11-2] | | 883 (1850) | 627 (1701) | 834 (1480) | 959 (1714) |

- **Copper is described far better than aluminum.** For AFF-r2SCAN the unstable fault is inside the range of DFT values and the intrinsic fault is low by about 45% (25 against 45), where for Al it was low by a factor of 5 to 7. AFF-r2SCAN gives about 25 mJ/m^2 for both metals, so it is close for Cu because Cu really has a low fault energy. `alignn_ff_mptrj` again has almost no surface energy.
- **Both cutoffs show in the lattice-constant scan.** Where the second neighbors pass 3.5 A (a = 3.50 A) the energy steps by 6.2, 13.6 and 28.8 meV/atom for the three 4+4-layer models, and by about 1.2 for AFF-r2SCAN, under the 2 meV at which the script reports a jump. Where the fourth shell passes 5 A (a = 3.536 A) it steps by 2.8 (`alignn_ff_matpes_pbe`), 5.7 (`alignn_ff_matpes_r2scan`) and 3.6 (AFF-r2SCAN). At AFF-r2SCAN's lattice constant that shell is at 5.06 A, 0.06 A outside the cutoff, and inside it each atom would have 54 neighbors, above the cap of 52.
- **The `alignn_ff_matpes_r2scan` column is not at that model's lattice constant.** Its lowest energy in the scan is at a = 3.534 A, on the edge of the 5 A jump and 0.6 meV/atom below the smooth minimum near 3.57 A. The fit of that run failed there and the slab was built at the reference 3.615 A, 1.2% too large, which is why its surface relaxes by 128 mJ/m^2 and some relaxed values exceed the as-built ones. `stacking_fault.py` now fits only the jump-free part of the scan around the reference lattice constant; this column needs a rerun.

### Retrained model (jobs 21862 for Al and 21863 for Cu, 2026-11-11)

The same tests on the stage-12 model (`SMOOTH=aff_smooth MODELS=""`, 15 and 12 minutes). Results in `results/aff_smooth/stage14/`, logs `logs/stage14gsf_21862.out` and `logs/stage14gsf_21863.out`. Every relaxed point converged to 0.005 eV/A, the 36-layer check agrees to 0.001 mJ/m^2, and the energy returns to zero after one period. Relaxed values, as built in parentheses; AFF-r2SCAN and the references are repeated from the tables above.

| Quantity | Al reference | Al, AFF-r2SCAN | Al, retrained | Cu reference | Cu, AFF-r2SCAN | Cu, retrained |
|---|---|---|---|---|---|---|
| Lattice constant (A) | 4.05 | 3.982 | 3.998 | 3.615 | 3.581 | 3.585 |
| Bulk modulus (GPa) | 79 | 73 | 58 | | 125 | 154 |
| (111) surface energy | 750 to 980 | 469 (473) | 673 (675) | | 1170 (1174) | 1247 (1251) |
| Intrinsic stacking fault | 116 to 164 | 24 (24) | 69 (70) | 45 (38 to 58) | 25 (25) | 44 (44) |
| Unstable stacking fault | 154 to 291 | 171 (182) | 209 (225) | 161 (158 to 184) | 173 (189) | 181 (200) |
| Maximum along [1-10] | 250, 663 | 384 (595) | 452 (720) | | 562 (798) | 560 (806) |
| Atoms on top, along [11-2] | | 438 (1272) | 638 (1645) | | 883 (1850) | 981 (1866) |

- **The kinks are gone.** The largest bend of the as-built curve along [11-2] is 4.6 mJ/m^2 for Al and 5.3 for Cu, about twice the median bend (2.0 and 2.5), where AFF-r2SCAN stepped by 29 mJ/m^2 in Al, 16 times its median of 1.8.
- **The energy jumps at the cutoffs are gone.** The script finds no jump in either lattice-constant scan, where AFF-r2SCAN rose by 6.9 meV/atom in Al and 3.6 in Cu as a neighbor shell passed 5 A.
- **Copper's fault energies now match.** The intrinsic fault is 44 against 45 from DFT, and the unstable fault 181 against 158 to 184.
- **Aluminum is closer but still low.** The intrinsic fault rose from 24 to 69 against 116 to 164, and the surface energy from 469 to 673 against 750 to 980. The bulk modulus fell to 58 GPa against 79, further off than AFF-r2SCAN's 73. An Al dislocation run with this model would still overstate stacking faults.

## Other structures (stage 6, job 19326)

AFF-r2SCAN on one GB10, 2026-11-07, with `cells.py` structures: FCC Al (a = 4.05 A) and Cu (3.615 A), water molecules on a cubic grid at 1.0 g/cm^3 with random orientations (the liquid's bond and angle counts per atom, not its structure), and Voronoi polycrystals of Si. None is relaxed: the 40-grain polycrystal has forces up to 58 eV/A at its grain boundaries, so these measure cost, not physics.

Chunked against full cell, float64, forces, checkpointed:

| Cell | Atoms | Cubes | Halo (A) | Cap binds | Max dE (eV/atom) | Max dF (eV/A) | Max F (eV/A) |
|---|---:|---|---:|---|---:|---:|---:|
| FCC Al, pristine | 2,048 | 2 x 2 x 2 | 20.01 | no | 1.8e-15 | 1.2e-14 | 0 (symmetry) |
| Si polycrystal, 4 grains | 3,957 | 2 x 3 x 1 | 20.01 | no | 1.8e-15 | 4.6e-13 | 58 |
| water | 1,536 | 2 x 2 x 2 | 20.01 | yes | 1.8e-11 | 8.2e-8 | 15 |
| water | 1,536 | 2 x 2 x 2 | 25 | yes | 0 | 4.8e-13 | 15 |
| water, cap lifted | 1,536 | 2 x 2 x 2 | 20.01 | no | 2.7e-15 | 4.1e-13 | 15 |

Largest single graph (pristine cells; checkpointed forces / energy only), and the typical rate below the memory limit (median over sizes from 25% of the largest up to it; the largest Al energy-only cell took 97.5 s against 5.9 s for the same Cu cell, from memory pressure):

| Structure | Bonds/atom | Angles/atom | Forces: atoms, MB/atom, atoms/s | Energy only: atoms, MB/atom, atoms/s |
|---|---:|---:|---|---|
| Si (stage 1) | 28 | 16 | 195,112, 0.526, 22,750 | 551,368, 0.195, 64,311 |
| FCC Al | 42 | 144 | 70,304, 1.544, 8,801 | 186,624, 0.649, 24,518 |
| FCC Cu | 42 | 144 | 70,304, 1.544, 8,738 | 157,216, 0.649, 24,192 |
| water | 51.7 | 258 | 41,472, 2.413, 5,598 | 98,304, 1.055, 14,648 |

Chunked throughput on cells of about 1M atoms (987,293 for the 40-grain polycrystal), from 20 pieces (all 8 for Si and the polycrystal without forces) sized at 60% of the largest single graph, 20.01 A halo. GB10-hours per 10^9 atoms extrapolate linearly:

| Structure | Forces | Atoms per piece | Evaluated per owned atom | Owned atoms/s | GB10-hours per 10^9 atoms |
|---|---|---:|---:|---:|---:|
| Si | yes | 105,306 | 2.83 | 8,028 | 35 |
| Si polycrystal | yes | 103,615 | 2.83 | 8,066 | 34 |
| FCC Al | yes | 40,485 | 4.89 | 1,847 | 150 |
| FCC Cu | yes | 34,820 | 7.45 | 1,212 | 229 |
| water | yes | 24,569 | 12.11 | 481 | 577 |
| Si | no | 261,032 | 2.09 | 29,389 | 9 |
| Si polycrystal | no | 259,468 | 2.10 | 28,954 | 10 |
| FCC Al | no | 108,266 | 2.92 | 8,508 | 33 |
| FCC Cu | no | 70,261 | 4.44 | 5,545 | 50 |
| water | no | 51,364 | 6.24 | 2,419 | 115 |

The extrapolation matches the measured billion-atom Si runs: 9 against 9.0 GB10-hours without forces, and 35 against 40.2 with forces, where the billion-atom run used smaller pieces (3.35 atoms evaluated per owned atom against 2.83 here). Per atom, close-packed metals cost 4 to 7 times as much as Si with forces and water 17 times. Most of the gap comes from the halo: a denser structure fits fewer atoms per piece into the same volume while the halo stays 20 A wide, so a water piece owns only one atom in twelve. Grain boundaries add no cost.
- `logs/`: Slurm output of every job above (stage1_, chunk_<job>_<task>, reduce_, stage5_), with start and end times,
  one line per cube, and exit status. atomgptlab04 prints times in UTC, the other nodes in US Eastern time.
- `results/<model>/`: result JSONs. The per-task force arrays of the 64M and 1B runs (768 MB and 12 GB each) are
  not kept, and the `*_reduced.json` files hold their force statistics.
