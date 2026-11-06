# scaling_jae

HPC GPU scaling benchmarks, organized by GPU. The main study is on the **NVIDIA Blackwell B200**, with two independent tracks under `b200/`:

- **`b200/alignn_ff/`** — ALIGNN-FF (graph neural network force field) inference scaling on Cu FCC supercells, via PyTorch + DGL + matscipy on a single B200. System sizes from N = 4 to ~780k atoms.
- **`b200/vasp_dft/`** — VASP plane-wave DFT single-point SCF scaling on Si diamond supercells (2·n³ atoms), across 1/2/4/8 B200 GPUs.

Each track has its own regenerable analysis under `analysis/` with an analyze script, a metrics JSON, plots, and a summary (`v*_`-prefixed under `b200/alignn_ff/analysis/v*/`). For detailed findings, read the summary files — this README is a map.

ALIGNN-FF was also run on an **NVIDIA GB10** (Grace-Blackwell, 121.7 GiB unified memory; `gb10/`) and an **NVIDIA H200 NVL** (139.8 GiB; `h200/`), and the top-level `analysis/` compares all GPUs in one plot and report.

## Highlights

### ALIGNN-FF (Cu FCC)

- Iterated from a quadratic DGL `line_graph` bottleneck (v4) to a sparse hand-rolled construction (v5) that reduced per-iter cost by orders of magnitude and made N ~ 10⁵–10⁶ atom inference tractable on one B200.
- Discovered a **float32 energy-drift cliff at N ≈ 470k atoms** — the same pristine Cu FCC crystal returns an energy that grows by up to +40% past the threshold, while float64 stays at 0.604 eV at all four sizes it could reach (up to 364,500 atoms). Diagnosed as a precision issue upstream of the graph-level readout (the `_Float64Pool` wrapper in v6 fixes a smaller pre-cliff drift but does not close the cliff; instrumentation in `probe_pool.py` shows the pool input is already corrupted).
- **Validity window in f32: N ≤ 442,368 atoms.** See `b200/alignn_ff/analysis/v6/v6_summary.md`.

### VASP DFT (Si diamond)

- Strong scaling of SCF runs across 3³–6³ supercells × 1/2/4/8 GPUs, plus single-shot timing at 10³–16³.
- Best strong-scaling result: **1.27× on 4 GPUs at 6³ (432 atoms)** — 32% parallel efficiency. 8 GPUs is always slower than 4 GPUs for every size tested; the problems are simply too small to keep multiple B200s busy.
- Size-scaling fit on the 12³→14³ pair (3456 → 5488 atoms, 8 GPU, ALGO=Fast): **T ∝ N^1.80**.
- Two largest runs (15³ = 6750 atoms, 16³ = 8192 atoms) crashed before any SCF step completed with a segfault in MPI all-to-all communication on 8 GPUs, not an out-of-memory error. See `b200/vasp_dft/analysis/summary.md`.

### Cross-GPU comparison (GB10, H200 vs B200)

- **Same settings as the B200** (`*/alignn_ff_b200settings/`): Cu FCC, 5 Å cutoff, 12 neighbors, energy only, float32, checkpoint `v12.2.2024_dft_3d_307k`. DGL does not run on these machines, so the checkpoint is loaded into the pure-PyTorch ALIGNN model, which matches the B200 DGL energies within 1·10⁻⁵ eV/atom up to 500 atoms. Largest cell: **442,368 atoms** on the GB10 (112 GB; the next size is killed by the host) and **562,432 atoms** on the H200 (142.5 GB), against the B200's 780k. Model inference on the H200 is within 13% of the B200 (0.70 vs 0.62 s at 256k atoms) and ~8× faster than the GB10; graph construction is CPU-bound and depends on the host (1.1 s on the B200 and GB10 hosts, 1.5 s on the H200 host at 256k). Energies from these reruns show the pre-v6 float32 readout drift (0.604 → 0.600 eV/atom), and past ~470k atoms the H200 run reproduces the B200's float32 cliff (up to 0.658 eV/atom); timings are unaffected.
- **Molecular-dynamics workload** (`*/alignn_ff/`): energy + forces + stress single points on Si diamond supercells with the 52-neighbor AFF-r2SCAN force field (figshare file 67360764, then the default `matpes_r2scan`). Largest cell: **54,872 atoms** on the GB10 (107 GB) and **74,088 atoms** on the H200 (145 GB). Memory is **1.95 MB/atom** on both through the ASE calculator, and the H200 is about **4× faster** than the GB10 from about 4,000 atoms up (2-5× below that). Through ASE, the same model uses 10× the memory per atom of energy only (0.195 MB/atom, `gb10/alignn_ff_max_atoms/`). Times through ASE are three model passes, because the calculator reruns the model for energy, forces and stress.

- **Maximum atoms on the GB10** (`gb10/alignn_ff_max_atoms/`, added 2026-11-02, 1B atoms 2026-11-06): four force fields on Si, with and without forces. Largest single evaluation with AFF-r2SCAN: **551,368 atoms** energy only and **195,112 atoms** with forces and layer checkpointing (54,872 through ASE, reproduced). For the `alignn_ff_*` models: 238,328 (262,144 for `alignn_ff_matpes_r2scan`) and 97,336. Chunked with halos, forces reached **8M atoms on one GB10, 64M on five or six and 1B on four** without reaching a memory limit (slowest task 1,379 s for AFF-r2SCAN on five, 6,231 s for `alignn_ff_mptrj` on six, 14,522 s for AFF-r2SCAN at 1B). Without forces, 64M atoms on four took 351 s and 1,473 s per task, and 1B atoms on three took 3,235 s per task with AFF-r2SCAN. At 1B each task also held the whole cell, up to 73.9 GB of host memory in the GB10's shared CPU and GPU memory.

## Maximum atoms by GPU

Every maximum-atom run with data, largest first within each group.

| GPU | Forces | Max atoms | How | Model | Element |
|---|---|---:|---|---|---|
| **B200** | no | 780,448† | 1 GPU | `v12.2.2024_dft_3d_307k` | Cu |
|  | yes | not run | | | |
| **GB10** | no | 1,000,000,000‡ | 3 GPUs, chunked | AFF-r2SCAN | Si |
| | no | 64,000,000‡ | 4 GPUs, chunked | AFF-r2SCAN | Si |
| | no | 64,000,000‡ | 4 GPUs, chunked | `alignn_ff_mptrj` | Si |
| | no | 8,000,000‡ | 1 GPU, chunked | AFF-r2SCAN | Si |
|  | no | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_mptrj` | Si |
|  | no | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_matpes_pbe` | Si |
|  | no | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_matpes_r2scan` | Si |
|  | no | 551,368 | 1 GPU | AFF-r2SCAN | Si |
|  | no | 453,962 | 1 GPU | Untrained radius model | Si |
|  | no | 442,368† | 1 GPU | `v12.2.2024_dft_3d_307k` | Cu |
|  | no | 262,144 | 1 GPU | `alignn_ff_matpes_r2scan` | Si |
|  | no | 238,328 | 1 GPU | `alignn_ff_mptrj` | Si |
|  | no | 238,328 | 1 GPU | `alignn_ff_matpes_pbe` | Si |
|  | no | 101,306 | 1 GPU | Untrained kNN model | Si |
|  | yes | 1,000,000,000‡ | 4 GPUs, chunked | AFF-r2SCAN | Si |
|  | yes | 64,000,000‡ | 6 GPUs, chunked | `alignn_ff_mptrj` | Si |
|  | yes | 64,000,000‡ | 5 GPUs, chunked | AFF-r2SCAN | Si |
|  | yes | 8,000,000‡ | 1 GPU, chunked | AFF-r2SCAN | Si |
|  | yes | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_mptrj` | Si |
|  | yes | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_matpes_pbe` | Si |
|  | yes | 8,000,000‡ | 1 GPU, chunked | `alignn_ff_matpes_r2scan` | Si |
|  | yes | 195,112 | 1 GPU, checkpointed | AFF-r2SCAN | Si |
|  | yes | 140,608 | 1 GPU, `forward_tensors` | AFF-r2SCAN | Si |
|  | yes | 97,336 | 1 GPU, checkpointed | `alignn_ff_mptrj` | Si |
|  | yes | 97,336 | 1 GPU, checkpointed | `alignn_ff_matpes_pbe` | Si |
|  | yes | 97,336 | 1 GPU, checkpointed | `alignn_ff_matpes_r2scan` | Si |
|  | yes | 54,872 | 1 GPU, ASE | AFF-r2SCAN | Si |
|  | yes | 46,656 | 1 GPU, `forward_tensors` | `alignn_ff_matpes_r2scan` | Si |
|  | yes | 39,304 | 1 GPU, `forward_tensors` | `alignn_ff_mptrj` | Si |
|  | yes | 39,304 | 1 GPU, `forward_tensors` | `alignn_ff_matpes_pbe` | Si |
|  | yes | 13,824 | 1 GPU, ASE | `alignn_ff_mptrj` | Si |
|  | yes | 13,824 | 1 GPU, ASE | `alignn_ff_matpes_pbe` | Si |
|  | yes | 13,824 | 1 GPU, ASE | `alignn_ff_matpes_r2scan` | Si |
| **H200** | no | 562,432† | 1 GPU | `v12.2.2024_dft_3d_307k` | Cu |
|  | yes | 74,088 | 1 GPU, ASE | AFF-r2SCAN | Si |
| **GB10, other groups' models** | yes | 140,608§ | 1 GPU | UMA-S | Si |
|  | yes | 128,000‡ | 1 GPU | UMA-S | Si |
|  | yes | 32,768§ | 1 GPU | M3GNet-MatPES-PBE | Si |
|  | yes | 21,952§ | 1 GPU | MACE-MP-0 | Si |
|  | yes | 21,952§ | 1 GPU | CHGNet | Si |

† Energies go wrong past 442,368 atoms on the B200 and H200. The GB10 and H200 runs also drift by up to 0.004 eV/atom below that.

‡ Largest size tried, no memory limit reached.

§ From the ALIGNN 2.0 paper's scaling table (energy + force).

Forces include stress, except in chunked runs, which use layer checkpointing and rattled Si. Chunked `alignn_ff_*` models use a 20 Å halo, which changes forces by at most 2e-8 eV/Å on Si and 1.3e-7 eV/Å on SiGe in double precision. In single precision, roundoff alone makes chunked and full-cell forces differ by up to 1e-3 eV/Å (`gb10/alignn_ff_max_atoms/`). Differences among the `alignn_ff_*` models are where the host killed each run, as memory per atom is identical. 13,824 is the last size tried below the kill (17,576 was skipped). AFF-r2SCAN is the paper's force field (2+2 layers). The `alignn_ff_*` models have 4+4 layers. The untrained-architecture, UMA-S and paper-table rows come from the ALIGNN 2.0 project, and their raw data is not in this repo. VASP DFT, for reference: the largest converged run is 432 Si atoms, 5,488 on 8 B200s was a 2-step timing run, and 6,750 crashed.

## Repo layout

```
b200/
├── alignn_ff/                ALIGNN-FF scaling on Cu FCC
│   ├── scale5b_v{4,5,6}.py     versioned scaling scripts; v6 is the active one
│   ├── diagnose_drift.py       f32 vs f64 + atom-permutation drift diagnosis
│   ├── probe_pool.py           instrumented pool to isolate drift source
│   ├── job*.sh                 SLURM wrappers (b200 partition)
│   ├── scaling_alignn_v*.npz   per-size checkpoint data
│   └── analysis/v{4,5,6}/      per-version study: v*_analyze.py + v*_summary.md + plots
└── vasp_dft/                 VASP DFT SCF scaling on Si diamond supercells
    ├── {3,4,5,6}x*/{1,2,4,8}/    strong-scaling sweep (all INCAR/OSZICAR/OUTCAR)
    ├── {10,12,14,15,16}x*/{4,8}/ single-shot timing at larger sizes
    └── analysis/                 analyze.py + summary.md + plots + metrics.json
gb10/, h200/                  ALIGNN-FF on one NVIDIA GB10 / H200 NVL
├── alignn_ff/                  energy+forces+stress sweep: bench_ff_scaling.py, job.sh, result JSON, README
├── alignn_ff_b200settings/     B200 settings on pure-PyTorch ALIGNN: scale_pure_b200settings.py, job.sh, result .npz
└── alignn_ff_max_atoms/        GB10 only: four force fields, all evaluation paths, chunked to 1B atoms, README
analysis/                     cross-GPU comparison
├── scaling_overview.py         -> scaling_overview.png (VASP + energy-only ALIGNN-FF on B200/GB10/H200)
├── scaling_forces.py           -> scaling_forces.png (energy+forces+stress ALIGNN-FF on GB10/H200)
├── scaling_all.py              -> scaling_all.png (all seven runs on one plot)
├── cross_gpu_figure.py         -> cross_gpu.png (two panels: energy-only B200/GB10/H200; with forces GB10/H200)
└── results_report.py           -> results_report.pdf (summary, plot, full result tables)
```

## Reproducing the analysis

The raw data (`.npz` for ALIGNN-FF, `OUTCAR/OSZICAR` for VASP) is committed. Each `analyze.py` re-derives every plot and `metrics.json` from that data:

```bash
# ALIGNN-FF
cd b200/alignn_ff/analysis/v6 && python v6_analyze.py

# VASP DFT
cd b200/vasp_dft/analysis && python analyze.py

# cross-GPU plots and report (plots first)
cd analysis && python scaling_overview.py && python scaling_forces.py && python scaling_all.py && python cross_gpu_figure.py && python results_report.py
```

Requires `numpy` and `matplotlib` only. The actual ALIGNN-FF sweeps (scripts in `b200/alignn_ff/`) additionally need `torch`, `dgl`, `ase`, `matscipy`, `alignn`, and `jarvis-tools`, and were run on a Blackwell B200 via SLURM. The GB10 and H200 sweeps use the pure-PyTorch ALIGNN 2.0 (`torch`, `ase`, `matscipy`, `jarvis-tools`; no DGL) and were run via SLURM on atomgptlab (GB10) and Skipjack (H200). The VASP data was produced by the GPU build of VASP on the B200 partition; `INCAR` files are committed per run, but VASP binaries, `POSCAR`, and `POTCAR` are not.

## License

MIT — see `LICENSE`.
