"""B200 scaling settings, rerun with the pure-PyTorch ALIGNN (no DGL).

Reproduces b200/alignn_ff/scale5b_v6.py on GPUs where DGL is unavailable
(the GB10 is aarch64 + CUDA 13; the Skipjack env has no DGL):
  - same checkpoint: v12.2.2024_dft_3d_307k (the B200 run's default_path())
  - same structure: Cu FCC, 4-atom cubic cell a = 3.6 A, i x i x i (4 i^3 atoms)
  - same graph: 5.0 A cutoff, 12-neighbour cap, full line graph, float32
  - same quantity: energy only, torch.no_grad, out["out"] (eV/atom)
  - same timing: one call per size; graph build and inference timed
    separately with CUDA syncs; the first size includes GPU warm-up
Loading the checkpoint into ALIGNNAtomWisePure reproduces the B200 DGL energies
to 4e-6 eV/atom (0.604015 vs 0.604014 at N = 4..256), checked in-run below.
Writes scaling_alignn_pure_<tag>.npz with the B200 file's keys
(natoms, times_nl, times_line, times_graph, times_inference, energies) plus
peak_mem_gb; times_nl / times_line are NaN (the pure builder does both in one
call; times_graph is their sum, as in the B200 file). Stops at the first OOM.
"""
import alignn.train_alignn  # noqa: F401  (macOS import order)
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from ase.build import bulk
from jarvis.core.atoms import ase_to_atoms

from alignn.ff.ff import get_figshare_model_ff
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig)
from alignn.torch_graph_builder import build_pure_torch_graph

B200_REF = {4: 0.604015, 32: 0.604014, 108: 0.604014, 256: 0.604014}


def load_model(device):
    """v12.2.2024_dft_3d_307k weights in the pure-torch model, energy only."""
    d = get_figshare_model_ff("v12.2.2024_dft_3d_307k")
    cfg = json.load(open(os.path.join(d, "config.json")))
    m = dict(cfg["model"])
    m.update(name="alignn_atomwise_pure", calculate_gradient=False,
             stresswise_weight=0, graphwise_weight=1.0, atomwise_weight=0,
             gradwise_weight=0)
    F = ALIGNNAtomWisePureConfig
    keep = set(F.model_fields if hasattr(F, "model_fields") else F.__fields__)
    net = ALIGNNAtomWisePure(F(**{k: v for k, v in m.items() if k in keep}))
    sd = torch.load(os.path.join(d, "best_model.pt"), map_location="cpu",
                    weights_only=False)
    res = net.load_state_dict(sd.get("model", sd), strict=False)
    print("checkpoint", d, "| missing", res.missing_keys,
          "| unexpected", len(res.unexpected_keys), flush=True)
    return net.to(device).eval(), cfg.get("atom_features", "cgcnn")


def is_oom(err):
    """CUDA / host out-of-memory, however torch raises it."""
    return isinstance(err, (torch.cuda.OutOfMemoryError, MemoryError)) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def main():
    """Size sweep until OOM."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max_size", type=int, default=99)
    p.add_argument("--tag", default="gpu")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    print("device", dev, "|", torch.cuda.get_device_name(0)
          if dev.type == "cuda" else "cpu", flush=True)
    net, feats = load_model(dev)
    base = bulk("Cu", "fcc", a=3.6, cubic=True)
    out_file = "scaling_alignn_pure_%s.npz" % args.tag
    rec = {k: [] for k in ("natoms", "times_nl", "times_line", "times_graph",
                           "times_inference", "energies", "peak_mem_gb")}
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    print("%4s %9s %10s %10s %10s %9s" % ("i", "n_atom", "graph", "infer", "E/atom", "mem GB"))
    for i in range(1, args.max_size + 1):
        n = 4 * i ** 3
        try:
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            a = base.repeat((i, i, i))
            sync(); t0 = time.perf_counter()
            g, lg = build_pure_torch_graph(
                ase_to_atoms(a), two_body_cutoff=5.0, max_neighbors=12,
                atom_features=feats, device=dev)
            lat = torch.tensor(np.asarray(a.cell), dtype=torch.float32, device=dev)
            sync(); t_graph = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.no_grad():
                e = float(net((g, lg, lat))["out"].reshape(-1)[0])
            sync(); t_inf = time.perf_counter() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
            del g, lg, lat
        except Exception as err:  # noqa: BLE001
            if not is_oom(err):
                raise
            print("  ! stopped at size %d (n=%d): OOM %s" % (i, n, str(err)[:100]), flush=True)
            break
        finally:
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        for k, v in zip(rec, (n, np.nan, np.nan, t_graph, t_inf, e, mem)):
            rec[k].append(v)
        chk = ("  (B200 %.6f)" % B200_REF[n]) if n in B200_REF else ""
        print("%4d %9d %9.3fs %9.3fs %10.6f %9.1f%s" % (i, n, t_graph, t_inf, e, mem, chk), flush=True)
        np.savez(out_file, **{k: np.array(v) for k, v in rec.items()})
    print("done ->", out_file, flush=True)


if __name__ == "__main__":
    main()
