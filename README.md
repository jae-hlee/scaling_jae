# scaling_jae

HPC GPU scaling benchmarks on **NVIDIA Blackwell B200**. Two independent tracks, both under `b200/`:

- **`b200/alignn_ff/`** — ALIGNN-FF (graph neural network force field) inference scaling on Cu FCC supercells, via PyTorch + DGL + matscipy on a single B200. System sizes from N = 4 to ~780k atoms.
- **`b200/vasp_dft/`** — VASP plane-wave DFT single-point SCF scaling on Si diamond supercells (2·n³ atoms), across 1/2/4/8 B200 GPUs.

Each track has its own regenerable analysis under `analysis/` with `analyze.py`, `metrics.json`, plots, and a `summary.md`. For detailed findings, read the summary files — this README is a map.

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

## Repo layout

```
b200/
├── alignn_ff/                ALIGNN-FF scaling on Cu FCC
│   ├── scale5b_v{4,5,6}.py     versioned scaling scripts; v6 is the active one
│   ├── diagnose_drift.py       f32 vs f64 + atom-permutation drift diagnosis
│   ├── probe_pool.py           instrumented pool to isolate drift source
│   ├── job*.sh                 SLURM wrappers (b200 partition)
│   ├── scaling_alignn_v*.npz   per-size checkpoint data
│   ├── output/                 SLURM stdout/stderr from the runs
│   └── analysis/v{4,5,6}/      per-version study: analyze.py + summary.md + plots
└── vasp_dft/                 VASP DFT SCF scaling on Si diamond supercells
    ├── {3,4,5,6}x*/{1,2,4,8}/    strong-scaling sweep (all INCAR/OSZICAR/OUTCAR)
    ├── {10,12,14,15,16}x*/{4,8}/ single-shot timing at larger sizes
    └── analysis/                 analyze.py + summary.md + plots + metrics.json
```

## Reproducing the analysis

The raw data (`.npz` for ALIGNN-FF, `OUTCAR/OSZICAR` for VASP) is committed. Each `analyze.py` re-derives every plot and `metrics.json` from that data:

```bash
# ALIGNN-FF
cd b200/alignn_ff/analysis/v6 && python v6_analyze.py

# VASP DFT
cd b200/vasp_dft/analysis && python analyze.py
```

Requires `numpy` and `matplotlib` only. The actual ALIGNN-FF sweeps (scripts in `b200/alignn_ff/`) additionally need `torch`, `dgl`, `ase`, `matscipy`, `alignn`, and `jarvis-tools`, and were run on a Blackwell B200 via SLURM. The VASP data was produced by the GPU build of VASP on the same partition; `INCAR` files are committed per run, but VASP binaries, `POSCAR`, and `POTCAR` are not.

## License

MIT — see `LICENSE`.
