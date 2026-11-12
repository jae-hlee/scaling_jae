# Other force fields on NVIDIA GB10: largest cell before running out of memory

How many atoms the model families of the large-scale records (DP, SNAP, NEP, Allegro) can run on
one GB10 (121.7 GiB unified memory), for comparison with ALIGNN-FF in `../alignn_ff_max_atoms/`.
Started 2026-11-11. All four have results (below); the DP compressed limit is bracketed, not pinned.

Each model is trained on a GB10 on the pure-Si frames of the MatPES r2SCAN data (the data
AFF-r2SCAN was trained on), with the size settings of its record paper where published. The test is
the largest pristine diamond-Si cell (8 i^3 atoms, a = 5.43 A) for which one static evaluation and a
few more steps finish with the right energy.

Memory at run time depends on a model's size settings and on the number of neighbors, not on its
trained weights, so each ceiling describes the settings used.

| Model | Code | Settings taken from the record | How the sweep runs it |
|---|---|---|---|
| NEP | GPUMD v5.9.1 | trillion-atom Cu NEP: cutoffs 6 and 4 A, `n_max 4 4`, `basis_size 8 8`, 30 neurons, ZBL below 2 A | `gpumd`, static evaluation then 10 NVE steps |
| SNAP | LAMMPS stable 30Sep2026, KOKKOS on CUDA; FitSNAP | billion-atom C SNAP: quadratic, 2J = 8. Cutoff 4.8 A as in the published quadratic Si SNAP (the carbon cutoff of 2.7 A reaches one shell of Si) | `lmp` with `snap/kk`, `run 0` then 5 NVE steps, skin 2 A, 0.5 fs |
| Allegro | nequip 0.19.1, nequip-allegro 0.8.3 | SC23 model: 2 layers, `l_max` 2, 64 tensor features, 4.0 A cutoff, latent width 1024, float32. The MLP shapes are approximated, because the current model class takes one width per MLP | NequIP ASE calculator on a model compiled with AOT Inductor, matscipy neighbor list, 1 + 5 evaluations |
| DP | DeePMD-kit 3.2.0, PyTorch backend | Cu records: `se_e2_a`, cutoff 8 A, embedding 32-64-128, fitting 240-240-240, double precision, compressed (2022). `sel` comes from the training data, not the records' 512 or 500 | `DeepPot` with the vesin neighbor list, 1 + 5 evaluations, uncompressed and compressed |

NEP and SNAP run in the engines their records used. Allegro and DP run through their Python
interfaces, with neighbor lists whose memory is linear in the number of atoms, so the model sets the
limit. The reasons: the Allegro LAMMPS pair style needs MPI, which the nodes lack, and has open
reports of crashes on GPUs (pair_nequip_allegro issues 86 and 87); the DP LAMMPS plugin needs two
more builds. The records themselves ran in LAMMPS, so these two rows are not the engines of Table 1.

## Training data (job 21865, 2026-11-11)

The MatPES r2SCAN set has 155 pure-Si frames with 482 atoms in all: 116 frames of 2 atoms, the rest
1 to 24 atoms. Energies run from -8.775 to -6.681 eV/atom, volumes from 13.9 to 127.7 A^3/atom
(diamond Si is 20.0), the largest force component is 15.0 eV/A, and no cell was left-handed. The
split is 123 / 16 / 16 frames (398 / 42 / 42 atoms).

This is enough to produce working models for the memory test and too little for potentials one
would trust. The training frames give 123 energies and 1,194 force components, fewer than the 1,597
coefficients of a quadratic SNAP with 2J = 8, so that fit is underdetermined.

The four model jobs were submitted on 2026-11-11 as 21866 (NEP), 21867 (SNAP), 21868 (Allegro) and
21869 (DP).

## Results so far (2026-11-11)

Largest pristine diamond-Si cell (8 i^3 atoms) that ran a static evaluation and the following steps with the energy per atom of the 512-atom cell, on one GB10. Memory is the largest drop of the node's available memory during the run (CPU and GPU together), so it is comparable between these rows but only roughly with the ALIGNN-FF numbers, which are GPU allocations. "Test" errors are on the 16 held-out frames.

| Model | Job | Largest cell that ran | First that failed | Memory per atom | Speed (atom-steps/s) | Test errors (energy, forces) |
|---|---|---:|---:|---:|---:|---|
| NEP, GPUMD v5.9.1 | 21866 | 56,623,104 (i 192, 126.6 GB) | 59,319,000 (i 195), killed by the host for memory | about 2.3 kB | 1.7e7 | 24.0 meV/atom, 0.104 eV/A |
| Allegro, nequip 0.19.1 (ASE calculator, AOT Inductor) | 21868 | 74,088 (i 21, 108.0 GB) | 85,184 (i 22), killed by the host for memory | about 1.46 MB | 1.0e4 | 59.7 meV/atom, 0.155 eV/A |
| SNAP, LAMMPS KOKKOS | 21887 | 157,464,000 (i 270, 115.5 GB) | 164,566,592 (i 274), ended by SIGTERM at 114.5 GB | about 0.73 kB | 2.3e5 | 235 meV/atom, 2.67 eV/A (fit underdetermined) |
| DP, DeePMD-kit 3.2.0, uncompressed (`DeepPot`, vesin) | 21888 | 32,768 (i 16, 97.8 GB) | 39,304 (i 17), killed by the host for memory | about 3.0 MB | 1.0e4 | 232 meV/atom, 0.223 eV/A |
| DP, compressed | 21888 | 32,768 (i 16, 98.2 GB) | 46,656 (i 18) killed; 39,304 not finished (the job itself was killed) | about 3.0 MB | 2.7e4 | same model |
| ALIGNN-FF (AFF-r2SCAN), one graph with checkpointing, `../alignn_ff_max_atoms/` | | 195,112 | | 0.526 MB (GPU) | | |

- **NEP.** `make` built `gpumd` and `nep` in 17 s with the two flag changes; training took 105 s for 10,000 generations. The training data put 70 radial and 20 angular neighbors in `nep.txt`, so gpumd's 32-bit limit is 85,899,345 atoms, above the size that ran out of memory: memory is the limit. Energy per atom agreed with the 512-atom cell to 1e-6 eV at every size. The bisection stopped at 2% of i, so the true limit lies between 56.6 and 59.3 million atoms. The size just below the limit ran at 1.46e7 atom-steps/s instead of about 1.7e7, a sign of memory pressure.
- **Allegro.** 12.5 million parameters (SC23 sizes), trained for 100 epochs (validation at the end: 46.8 meV/atom, 0.112 eV/A). The job started at 18:38:57 UTC although it was submitted about 18:25, and its log begins with the installation, training and compilation already done and seven sizes already measured, so (my reading) Slurm requeued it during the 74,088-atom size, after the 85,184- and 125,000-atom sizes had been killed for memory, and the requeue overwrote the log of the first run. The requeued run repeated that size, which then fit; `sacct -j 21868 --duplicates` would show both runs. Every evaluation includes stress, which the ASE target of a compiled model always computes. ALIGNN-FF through its own ASE calculator stops at 54,872 atoms.
- **SNAP, sweep (job 21887, 1 MD step per size, 3-hour limit per size).** 157,464,000 atoms ran in 36 minutes at 2.3e5 atom-steps/s with the energy per atom of the 512-atom cell. The next sizes (164.6, 171.9 and 238.3 million) ended with signal 15 (SIGTERM) after drawing 114 to 124 GB, with nothing on stderr. `sweep.py` records that as "error"; my reading is that the node's memory protection ended them (the other models' failures at the limit were SIGKILL), so the ceiling is between 157.5 and 164.6 million atoms.
- **SNAP, build and fit.** LAMMPS built with KOKKOS 5.2.1 for BLACKWELL121 in 281 s. The quadratic fit with 2J = 8 has 1,596 coefficients against 1,317 equations, so FitSNAP reproduces the training frames exactly (errors of 1e-9) and the test frames badly (R^2 below zero). Memory and speed do not depend on the coefficients, so the sweep stands for any quadratic 2J = 8 SNAP with a 4.8 A cutoff. With 5 steps per size, sizes above about 100 million atoms would pass the 1-hour limit per size, so the sweep was restarted with 1 step and a 3-hour limit.
- **DP, rerun (job 21888).** Both fixes below worked: the build with CUDA kernels succeeded (`TORCH_CUDA_ARCH_LIST="9.0+PTX"`, 167 s) and vesin ran on the GPU once CUDA 13's NVRTC was loaded first (the process also maps a CUDA 12.8 copy later). The uncompressed model stops between 32,768 and 39,304 atoms at about 3 MB per atom, close to the roughly 3.4 MB per atom of the 2020 Cu record (16 GB per V100 / 4,656 atoms). The compressed model is 2.6 times faster but uses the same memory at 32,768 atoms; the 2022 record's 0.13 MB per atom came from LAMMPS with further kernel work, not from this Python path. Why compression saves no memory here was not investigated. Test errors through DeePMD's own neighbor list: 232 meV/atom and 0.223 eV/A (overfitted; validation RMSE 274 meV/atom).
- **DP, first run (job 21869).** The build with DeePMD's CUDA kernels failed (reason in `logs/dp_pip_cuda_21869.log` on the cluster), so the build without them was used and the compressed model cannot run on the GPU. Training ran (1,887 s, 100,000 steps; at most 168 neighbors within 8 A in the training data, so `sel` came out near 188 rather than the records' 512; validation RMSE 274 meV/atom and 0.375 eV/A against 0.7 meV/atom and 0.012 eV/A on the training frames, so it overfits). `dp test` and the sweep both stopped in the vesin neighbor list on the GPU: `nvrtc: error: invalid value for --gpu-architecture (-arch)`. Two causes, found by reading the sources, and the changes made for the next run:
  - The CUDA build: PyTorch's CMake detects compute capability 12.1 and its text replacement for capability 2.1 (`string(REPLACE "2.1" "2.1(2.0)" ...)` in `select_compute_arch.cmake`) turns it into `12.1(2.0)`, so nvcc was asked for `compute_20` (`nvcc fatal : Unsupported gpu architecture 'compute_20'`). `dp.sbatch` now sets `TORCH_CUDA_ARCH_LIST="9.0+PTX"`, which skips the detection; the driver compiles the PTX for the GB10 when the library loads. A two-digit entry such as `12.1` is not accepted by the pattern in PyTorch 2.2's version of that file (`^([0-9]\.[0-9]...)$`), and the 2.13 version was not checked.
  - The neighbor list: vesin compiles its kernels through the gpu-lite header with `--gpu-architecture=sm_<major*10+minor>` and uses the first `libnvrtc` already loaded in the process. CUDA 12.9 added `sm_121`, so the NVRTC in use must have been a CUDA 12 copy. `py_eval.py` now loads `/usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so.13` before torch and records the NVRTC files that are mapped. DeePMD's other linear-memory list, `nv`, needs `nvalchemi-toolkit-ops`, which requires Python 3.11 or newer.
  - `dp test` is replaced by `py_eval.py errors` with DeePMD's own (all-pairs) neighbor list, which is fine for test cells of at most 24 atoms.

## Files

- `probe.sbatch`: what a GB10 node has for building these codes. Ran as a job on 2026-11-11
  (atomgptlab05): GCC 13.3, CMake 3.28.3, CUDA 13.0 compiler in `/usr/local/cuda/bin` (not on the
  path), Python 3.10 with torch 2.13.0+cu130, internet access; no MPI, Fortran or ninja. `nvcc`
  rejects `-arch=sm_60` and accepts `sm_75`, `sm_120` and `sm_121`. `nvidia-smi` reports no memory
  use on a GB10.
- `si_data.py`, `data.sbatch`: the pure-Si frames of the MatPES r2SCAN training data as extended
  XYZ (total energy, forces, no stress) and as DeePMD-kit systems, split 80/10/10 with a fixed seed.
  The test frames are kept away from every training.
- `sweep.py`: the size sweep. One engine process per size, doubling the atom count until a run
  fails, then bisection. Memory is the largest drop of `MemAvailable` on the node during the run.
  A size counts only if the engine finishes and its energy per atom matches the 512-atom cell,
  because 32-bit index overflow can give wrong numbers instead of an error. Results are written
  after every size, so a killed job loses nothing and a resubmitted job continues. Engines:
  `gpumd`, `lammps`, `python`.
- `py_eval.py`: one size, or test-set errors, through the NequIP ASE calculator or DeePMD-kit's
  `DeepPot`. Used by `sweep.py` for Allegro and DP.
- `nep.sbatch`: builds `gpumd` and `nep` (the makefile's `sm_60` and `c++14` become `sm_121` and
  `c++17` for CUDA 13), trains for 10,000 generations, reports test errors, sweeps.
- `snap.sbatch`, `fitsnap_json.py`: builds LAMMPS with KOKKOS and ML-SNAP without MPI, fits the SNAP
  with FitSNAP (a second, CPU build of LAMMPS as a shared library), sweeps. If the fit fails, the
  sweep uses `Si_Zuo_JPCA2020.quadratic` from LAMMPS, which has the same 2J, cutoff and form.
- `allegro.sbatch`: installs nequip and allegro into a venv on top of the `alignn` env, trains,
  compiles, reports test errors, sweeps. If the compile fails, the checkpoint is evaluated
  uncompiled and the result file says so.
- `dp.sbatch`: builds DeePMD-kit from source against the env's torch with its CUDA kernels (without
  them if that fails), trains, freezes, compresses, tests, sweeps both models.

Each job skips a stage whose result exists. The venvs (`venv_snap`, `venv_allegro`, `venv_dp`) see
the `alignn` env's packages and add their own, so that env is not changed.

On the cluster everything is in `/data/jlee859/ffother/` (`src/`, `data/`, `nep/`, `snap/`,
`allegro/`, `dp/`, `work/`, `results/`, `logs/`).

## Checked against source before running (2026-11-11)

Each job was compared, by reading, with the source of the version it uses (GPUMD v5.9.1, LAMMPS
stable_30Sep2026_update1, FitSNAP commit 1a9b0507, nequip v0.19.1, allegro v0.8.3, DeePMD-kit
v3.2.0): input keywords, file formats, command lines, output lines that the scripts parse. What
that found and what was changed:

- NEP: with `nep.restart` present, `generation` is the number of further generations and the first
  column of `loss.out` starts again at 100, so the job now counts lines and asks for the remainder.
  `nep` compiles model-specific kernels with nvcc when it starts; if they fail to start it exits,
  so the job tries once more with `nep_compile off`.
- SNAP: the FitSNAP input needs an `[EXTRAS]` section (it is read without a check after the fit).
  FitSNAP refuses left-handed cells.
- DP: the PyTorch backend imports e3nn, which only an extra declares. A compressed model on the GPU
  needs the CUDA build, so the compressed sweep is skipped without it. On the GPU vesin reserves
  512 pairs per atom at an 8 A cutoff; the sweep sets `VESIN_CUDA_MAX_PAIRS_PER_POINT=128`.
- Allegro: every config key, command and call matched. Only stage logic changed.
- All: `si_data.py` makes left-handed cells right-handed (DeePMD-kit and FitSNAP need that).
  `sweep.py` continues an earlier result only if the model, the engine binary and the settings are
  the same (a fingerprint), fails a size whose energy is not a number, and tries a size once more
  if the job itself ended during it.

Not checked by this: that anything compiles or runs on a GB10.

## Known before running

- Nobody has reported any of these codes on a GB10, on aarch64 with CUDA 13, or on a
  compute-capability-12.1 GPU. Every build here is a first attempt.
- gpumd sizes its neighbor lists from two integers in `nep.txt`, the largest neighbor counts in the
  training data, times 1.25. Dense training frames therefore raise the memory per atom of every
  later run. DP's `sel` behaves the same way.
- gpumd sizes its angular neighbor list with a 32-bit product, atoms times 1.25 times the largest
  angular neighbor count of the training data. For 16 neighbors (diamond Si within 4 A) that allows
  107,374,182 atoms, fewer if the training data has denser frames. gpumd reports exceeding it as
  "out of memory", so the sweep result records whether the first failing size was above it.
- The ASE target of a compiled NequIP model also computes stress.
- LAMMPS with KOKKOS keeps host and device copies of atom data, and on a GB10 both come out of the
  same memory. KOKKOS has an internal build option for unified memory
  (`Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY`) that was not used.
