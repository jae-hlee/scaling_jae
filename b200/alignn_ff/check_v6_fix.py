"""
check_v6_fix.py
===============

Historical sanity check for the v6 _Float64Pool wrapper, written when the
hypothesis was that the wrapper would close the N >= 500k energy drift. It
did not. Running this now is informative but no longer a pass/fail gate --
expect roughly:

    i=20  N=32,000    E ~= 0.604013   |ΔE vs f64| < 2e-6   (wrapper working)
    i=50  N=500,000   E ~= 0.639      |ΔE vs f64| ~= 0.035 (upstream cliff)
    i=58  N=780,448   E ~= 0.845      |ΔE vs f64| ~= 0.24  (upstream cliff)

The first row confirms the wrapper is active and fixing the pre-cliff pool
drift; the other two are expected to still drift because the cliff is
upstream of the pool (see `analysis/v6/summary.md` and `probe_pool.py`).

Usage (B200):

    sbatch -J v6_check --partition=b200 --qos=blackwell_test --gres=gpu:1 \
        --ntasks=1 --cpus-per-task=8 --time=00:30:00 \
        --wrap "source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh && \
                conda activate b200 && \
                export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
                python check_v6_fix.py"
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.build.supercells import make_supercell
from jarvis.io.vasp.inputs import Poscar

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scale5b_v6 import (  # noqa: E402
    _load_alignn_ff, fast_graph, _CU_POSCAR,
)
from alignn.ff.ff import default_path  # noqa: E402
from jarvis.db.jsonutils import loadjson  # noqa: E402

REFERENCE_F64 = 0.604015  # from diag_1332079.out, flat across all sizes tested


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model, config = _load_alignn_ff(device)
    cutoff = float(config["cutoff"])
    max_neighbors = int(config["max_neighbors"])
    print(f"cutoff={cutoff}  max_neighbors={max_neighbors}", flush=True)
    print(f"model.readout type: {type(model.readout).__name__}", flush=True)

    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()

    # Pick a baseline (f32 matches f64) and two drift-region sizes.
    sizes = [20, 50, 58]
    print(f"\n  {'i':>3}  {'N':>9}  {'E (eV)':>10}  "
          f"{'|ΔE vs f64|':>12}  {'t (s)':>6}", flush=True)
    print("  " + "-" * 55, flush=True)

    for i in sizes:
        n = 4 * i ** 3
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])
        t0 = time.perf_counter()
        g, lg, lat = fast_graph(sc, device, cutoff, max_neighbors)
        with torch.no_grad():
            out = model((g, lg, lat.clone()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        e = float(out["out"].detach().cpu().item())
        print(f"  {i:>3}  {n:>9}  {e:>10.6f}  "
              f"{abs(e - REFERENCE_F64):>12.2e}  {dt:>6.2f}", flush=True)
        del g, lg, lat, sc

    print("\nPre-cliff sizes (i <= 48) should be <2e-6 eV from the f64 ref.",
          flush=True)
    print("Post-cliff sizes (i >= 49) still drift -- pool fix is not enough.",
          flush=True)


if __name__ == "__main__":
    main()
