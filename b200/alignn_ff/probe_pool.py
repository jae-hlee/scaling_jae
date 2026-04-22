"""
probe_pool.py
=============

Instrument the v6 readout wrapper to answer three questions cleanly:

  1. Is _Float64Pool.forward actually being called during model((g, lg, lat))?
     If no print lines appear below, the wrapper is installed but never
     invoked -- meaning the drift fix can't possibly land and we need to
     figure out why ALIGNN's forward bypasses it.

  2. Given the f32 input `feat`, how different is its mean in f32 vs f64?
     If |mean_f32 - mean_f64| is tiny (O(1e-6)), the pool is NOT the source
     of the drift and we should look upstream (message-passing layers).
     If it's large (O(1e-2)), the pool IS the source.

  3. How different is the *input* to the pool from its value at baseline N?
     If the message passing has already baked in the drift, per-element
     statistics of feat at drift sizes will differ from baseline in ways
     f64 can't undo after the fact.

Expects to be run under SLURM on a B200 (needs GPU).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from ase.build.supercells import make_supercell
from jarvis.io.vasp.inputs import Poscar

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scale5b_v6 import (  # noqa: E402
    _load_alignn_ff, fast_graph, _CU_POSCAR,
)


class _InstrumentedPool(nn.Module):
    """Same math as _Float64Pool, but emits a one-line per-call report so we
    can see whether it runs and what it produces."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.call_idx = 0

    def forward(self, graph, feat):
        bnn = graph.batch_num_nodes()
        n = int(bnn.item()) if bnn.numel() == 1 else int(bnn.sum().item())
        feat64 = feat.double()
        pooled64 = feat64.sum(dim=0, keepdim=True) / float(n)
        pooled32 = feat.sum(dim=0, keepdim=True) / float(n)
        diff = (pooled64.float() - pooled32).abs()
        feat_stats = (
            f"|feat| mean={feat.abs().mean().item():.4f} "
            f"max={feat.abs().max().item():.4f}"
        )
        print(
            f"  [pool call {self.call_idx}] N={n}  "
            f"|mean_f64 - mean_f32|_inf={diff.max().item():.3e}  "
            f"|.|_2={diff.norm().item():.3e}  "
            f"{feat_stats}",
            flush=True,
        )
        self.call_idx += 1
        return pooled64.to(feat.dtype)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type != "cuda":
        print("ABORT: need GPU for this probe (N=500k on CPU is infeasible)",
              flush=True)
        sys.exit(1)

    model, config = _load_alignn_ff(device)
    # Replace the v6 wrapper with the instrumented one.
    model.readout = _InstrumentedPool(getattr(model.readout, "base",
                                               model.readout))
    print(f"model.readout type: {type(model.readout).__name__}", flush=True)

    cutoff = float(config["cutoff"])
    max_neighbors = int(config["max_neighbors"])
    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()

    # Baseline + two drift-region sizes. If the pool is the culprit, the
    # |mean_f64 - mean_f32| diff should explode at the drift sizes. If not,
    # the diff stays tiny and we've ruled the pool out.
    for i in [20, 50, 55]:
        n_atoms = 4 * i ** 3
        sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])
        g, lg, lat = fast_graph(sc, device, cutoff, max_neighbors)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model((g, lg, lat.clone()))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        e = float(out["out"].detach().cpu().item())
        print(f"  >>> i={i}  N={n_atoms}  E={e:.6f}  t={dt:.2f}s\n",
              flush=True)
        del g, lg, lat, sc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
