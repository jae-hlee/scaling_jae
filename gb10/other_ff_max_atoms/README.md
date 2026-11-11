# Other force fields on NVIDIA GB10: largest cell before running out of memory

How many atoms the model families of the large-scale records (DP, SNAP, NEP, Allegro) can run on
one GB10 (121.7 GiB unified memory), for comparison with ALIGNN-FF in `../alignn_ff_max_atoms/`.
Started 2026-11-11. **There are no results yet. Of the scripts here only `probe.sbatch` has run.**

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
