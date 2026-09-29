# scaling_jae

HPC GPU scaling benchmarks, organised by GPU. The main study is on the **NVIDIA Blackwell B200**, with two independent tracks under `b200/`:

- **`b200/alignn_ff/`** — ALIGNN-FF (graph neural network force field) inference scaling on Cu FCC supercells, via PyTorch + DGL + matscipy on a single B200. System sizes from N = 4 to ~780k atoms.
- **`b200/vasp_dft/`** — VASP plane-wave DFT single-point SCF scaling on Si diamond supercells (2·n³ atoms), across 1/2/4/8 B200 GPUs.

Each track has its own regenerable analysis under `analysis/` with an analyze script, a metrics JSON, plots, and a summary (`v*_`-prefixed under `b200/alignn_ff/analysis/v*/`). For detailed findings, read the summary files — this README is a map.

ALIGNN-FF was also run on an **NVIDIA GB10** (Grace-Blackwell, 121.7 GiB unified memory; `gb10/`) and an **NVIDIA H200 NVL** (139.8 GiB; `h200/`), and the top-level `analysis/` compares all GPUs in one plot and report.

## Highlights

### ALIGNN-FF (Cu FCC)

- Iterated from a quadratic DGL `line_graph` bottleneck (v4) to a sparse hand-rolled construction (v5) that reduced per-iter cost by orders of magnitude and made N ~ 10⁵–10⁶ atom inference tractable on one B200.
- Discovered a **float32 energy-drift cliff at N ≈ 470k atoms** — the same pristine Cu FCC crystal returns an energy that grows by up to +39% past the threshold, while a float64 run is flat at 0.604 eV across every size. Diagnosed as a precision issue upstream of the graph-level readout (the `_Float64Pool` wrapper in v6 fixes a smaller pre-cliff drift but does not close the cliff; instrumentation in `probe_pool.py` shows the pool input is already corrupted).
- **Validity window in f32: N ≤ 442,368 atoms.** See `b200/alignn_ff/analysis/v6/v6_summary.md`.

### VASP DFT (Si diamond)

- Strong scaling of SCF runs across 3³–6³ supercells × 1/2/4/8 GPUs, plus single-shot timing at 10³–16³.
- Best strong-scaling result: **1.27× on 4 GPUs at 6³ (432 atoms)** — 32% parallel efficiency. 8 GPUs is always slower than 4 GPUs for every size tested; the problems are simply too small to keep multiple B200s busy.
- Size-scaling fit on the 12³→14³ pair (3456 → 5488 atoms, 8 GPU, ALGO=Fast): **T ∝ N^1.80**.
- Two largest runs (15³ = 6750 atoms, 16³ = 8192 atoms) crashed before any SCF step completed — almost certainly OOM at the charge-mixer allocation on 8 GPUs. See `b200/vasp_dft/analysis/summary.md`.

### Cross-GPU comparison (GB10, H200 vs B200)

- **Same settings as the B200** (`*/alignn_ff_b200settings/`): Cu FCC, 5 Å cutoff, 12 neighbours, energy only, float32, checkpoint `v12.2.2024_dft_3d_307k`. DGL does not run on these machines, so the checkpoint is loaded into the pure-PyTorch ALIGNN model, which reproduces the B200 DGL energies to 4·10⁻⁶ eV/atom. Largest cell: **442,368 atoms** on the GB10 (112 GB; the next size is killed by the host) and **562,432 atoms** on the H200 (142.5 GB), against the B200's 780k. Model inference on the H200 is within ~15% of the B200 (0.70 vs 0.62 s at 256k atoms) and ~9× faster than the GB10; graph construction is CPU-bound and depends on the host (1.1 s on the B200 and GB10 hosts, 1.5 s on the H200 host at 256k). Energies from these reruns show the pre-v6 float32 readout drift (0.604 → 0.600 eV/atom), and past ~470k atoms the H200 run reproduces the B200's float32 cliff (up to 0.658 eV/atom); timings are unaffected.
- **Molecular-dynamics workload** (`*/alignn_ff/`): energy + forces + stress single points on Si diamond supercells with the smooth 52-neighbour `matpes_r2scan` force field. Largest cell: **54,872 atoms** on the GB10 (107 GB) and **74,088 atoms** on the H200 (145 GB). Memory is **1.95 MB/atom** on both, so any GPU's ceiling is its memory divided by that; the H200 is **4× faster** than the GB10 at every size. With forces, memory per atom is ~8× that of the energy-only runs (0.25 MB/atom on the GB10).

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
└── alignn_ff_b200settings/     B200 settings on pure-PyTorch ALIGNN: scale_pure_b200settings.py, job.sh, result .npz
analysis/                     cross-GPU comparison
├── scaling_overview.py         -> scaling_overview.png (VASP + energy-only ALIGNN-FF on B200/GB10/H200)
├── scaling_forces.py           -> scaling_forces.png (energy+forces+stress ALIGNN-FF on GB10/H200)
├── scaling_all.py              -> scaling_all.png (all seven runs on one plot)
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
cd analysis && python scaling_overview.py && python scaling_forces.py && python scaling_all.py && python results_report.py
```

Requires `numpy` and `matplotlib` only. The actual ALIGNN-FF sweeps (scripts in `b200/alignn_ff/`) additionally need `torch`, `dgl`, `ase`, `matscipy`, `alignn`, and `jarvis-tools`, and were run on a Blackwell B200 via SLURM. The GB10 and H200 sweeps use the pure-PyTorch ALIGNN 2.0 (`torch`, `ase`, `matscipy`, `jarvis-tools`; no DGL) and were run via SLURM on atomgptlab (GB10) and Skipjack (H200). The VASP data was produced by the GPU build of VASP on the B200 partition; `INCAR` files are committed per run, but VASP binaries, `POSCAR`, and `POTCAR` are not.

## License

MIT — see `LICENSE`.
