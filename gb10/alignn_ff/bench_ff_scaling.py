"""Largest cell an ALIGNN force field can evaluate on one GPU, and how fast.

Protocol of the manuscript's scaling table (tab:scaling): the 8-atom
conventional silicon cell repeated i x i x i (8 i^3 atoms), one full
energy + forces + stress single point per size through the ASE
calculator, stepping i up until the GPU runs out of memory. Per size it
records the median wall time over --repeats calls, the peak GPU memory
(allocated and reserved) and the memory per atom, and rewrites the JSON
after every size so an OOM or a walltime kill loses nothing.

    python bench_ff_scaling.py --out bench_h200          # default FF
    python bench_ff_scaling.py --model_dir /path/to/dir  # any checkpoint

Sizes: 2,4,...,14 (14 = 21,952 atoms, the table's timing column), then
every integer up to --max_i.
"""
import alignn.train_alignn  # noqa: F401  macOS: before the data imports
import argparse
import gc
import hashlib
import json
import os
import statistics
import time

import numpy as np
import torch
from ase.build import bulk

from alignn.ff.calculators import (
    AlignnAtomwiseCalculator, get_figshare_model_ff)


def single_point(base, i, calc, device):
    """Build the i x i x i supercell and time one E+F+S evaluation."""
    atoms = base.repeat((i, i, i))
    atoms.calc = calc
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    energy = atoms.get_potential_energy()  # one calculate(): E, F and S
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, float(energy) / len(atoms), float(np.abs(forces).max()), \
        float(np.abs(stress).max()), len(atoms)


def is_oom(err):
    """True for CUDA/host out-of-memory errors however torch raises them."""
    return isinstance(err, (torch.cuda.OutOfMemoryError, MemoryError)) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="matpes_r2scan",
                   help="registry name for get_figshare_model_ff")
    p.add_argument("--model_dir", default=None,
                   help="checkpoint dir (config.json + best_model.pt); "
                        "overrides --model")
    p.add_argument("--max_i", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="bench_ff_scaling")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if device.type == "cuda" else None)
    model_dir = args.model_dir or get_figshare_model_ff(args.model)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    ckpt = os.path.join(model_dir, "best_model.pt")
    ckpt_md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()
    print("device:", device, "|", gpu,
          "| GPU memory %.1f GB" % (total_gb or 0), flush=True)
    print("checkpoint:", model_dir, "| md5", ckpt_md5, "| model",
          cfg["model"]["name"], "hidden", cfg["model"]["hidden_features"],
          "| cutoff", cfg["cutoff"], "nbrs", cfg["max_neighbors"],
          "| smooth", cfg["model"].get("use_cutoff_function"), flush=True)
    calc = AlignnAtomwiseCalculator(path=model_dir, device=str(device))

    base = bulk("Si", "diamond", a=5.43, cubic=True)  # 8-atom conventional
    print("base cell atoms:", len(base), flush=True)
    for _ in range(2):  # warm-up: kernels, caches
        single_point(base, 2, calc, device)

    sizes = [i for i in list(range(2, 15, 2)) + list(range(15, 41))
             if i <= args.max_i]
    rows = []
    payload = {"model": args.model, "model_dir": model_dir,
               "checkpoint_md5": ckpt_md5, "gpu": gpu,
               "gpu_memory_gb": total_gb,
               "structure": "Si diamond conventional cell (8 atoms), i^3",
               "quantity": "energy + forces + stress single point",
               "repeats": args.repeats, "results": rows, "oom": None}
    for i in sizes:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reps = []
        try:
            for _ in range(args.repeats):
                reps.append(single_point(base, i, calc, device))
        except Exception as err:  # noqa: BLE001  (we must record OOMs)
            if not is_oom(err):
                raise
            n = 8 * i**3
            payload["oom"] = {"i": i, "natoms": n, "error": str(err)[:300]}
            print(f"[OOM] i={i}, N={n}: {str(err)[:120]}", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            json.dump(payload, open(args.out + ".json", "w"), indent=1)
            break
        times = [r[0] for r in reps]
        n = reps[0][4]
        alloc = (torch.cuda.max_memory_allocated() / 1e9
                 if device.type == "cuda" else None)
        reserved = (torch.cuda.max_memory_reserved() / 1e9
                    if device.type == "cuda" else None)
        row = {"i": i, "natoms": n, "t_median_s": statistics.median(times),
               "t_all_s": times, "mem_alloc_gb": alloc,
               "mem_reserved_gb": reserved,
               "mb_per_atom": (alloc * 1e3 / n) if alloc else None,
               "e_per_atom_eV": reps[0][1], "max_abs_force": reps[0][2],
               "max_abs_stress": reps[0][3]}
        rows.append(row)
        print("i=%2d N=%7d  t=%.3f s  mem %.1f GB alloc / %.1f GB reserved"
              "  (%.2f MB/atom)  E/atom %.4f" % (
                  i, n, row["t_median_s"], alloc or 0, reserved or 0,
                  row["mb_per_atom"] or 0, row["e_per_atom_eV"]), flush=True)
        json.dump(payload, open(args.out + ".json", "w"), indent=1)
    print("done ->", args.out + ".json", flush=True)


if __name__ == "__main__":
    main()
