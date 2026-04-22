"""
diagnose_drift.py
=================

Tests the hypothesis that the large-N energy drift seen in scaling_alignn_v5
is float32 accumulation error in ALIGNN's graph-level reduction.

For each requested size, runs a single forward pass in float32 and in float64
and reports both values. If they agree in the regime where f32 is already
showing drift, the hypothesis is falsified -- the problem is algorithmic, not
precision. If they disagree (f64 flat, f32 drifting), the hypothesis is
confirmed and the fix space is "use higher precision in the relevant ops."

Memory caveat
-------------
The v5 float32 run used ~157 GiB of 178 GiB HBM at N=780,448. float64 roughly
doubles activation memory, so the float64 path will OOM well before the
drift-onset size (N=500k). The useful diagnostic within memory is still:

  1. Confirm f32 == f64 in the baseline regime (e.g., N ≤ 250k). If so, the
     f32 path is numerically fine there and the machinery is trustworthy.
  2. Watch how f64 approaches OOM -- the largest size where f64 fits gives a
     trusted reference for that N.
  3. Combined with the linear (per-N) drift pattern in f32 above N=500k, this
     narrows the cause to accumulation/precision rather than a kernel
     algorithm switch.

Usage
-----
    python diagnose_drift.py                                # default sizes
    python diagnose_drift.py --sizes 20,30,40,45            # custom list
    python diagnose_drift.py --permute-sizes 50,55          # order-dependence
                                                            # check at drift N

The --permute-sizes flag runs each size twice in float32 with the atoms
shuffled the second time, using the same physics (same positions, different
order). A nonzero |ΔE| in float32 is direct evidence of order-dependent
summation -- i.e. precision loss.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import dgl
from ase.build.supercells import make_supercell
from alignn.graphs import compute_bond_cosines
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.ff.ff import default_path
from jarvis.db.jsonutils import loadjson
from jarvis.io.vasp.inputs import Poscar

# Reuse v5's graph primitives (these already accept a dtype).
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from scale5b_v5 import (  # noqa: E402
    _neighbor_list_matscipy,
    _fast_line_graph,
    _species_features,
    _CU_POSCAR,
)


def _load_model(device: torch.device, dtype: torch.dtype):
    path = default_path()
    config = loadjson(os.path.join(path, "config.json"))
    config["model"]["calculate_gradient"] = False
    config["model"]["stresswise_weight"] = 0
    config["model"]["graphwise_weight"] = 1.0
    config["model"]["atomwise_weight"] = 0
    config["model"]["gradwise_weight"] = 0
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
    state = torch.load(
        os.path.join(path, "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(state)
    model.to(device).to(dtype).eval()
    return model, config


def _build_graph(sc, device, cutoff, max_neighbors, dtype, permute_seed=None):
    """Build (g, lg, lattice). If permute_seed is set, shuffle the atom
    ordering first -- same physics, different memory layout."""
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        perm = rng.permutation(len(sc))
        sc = sc[perm]

    n = len(sc)
    src, dst, cart, dist, _ = _neighbor_list_matscipy(
        sc, cutoff=cutoff, device=device, max_neighbors=max_neighbors,
        float_dtype=dtype,
    )
    unit = cart / dist[:, None].clamp(min=1e-12)
    feats = _species_features(sc.get_chemical_symbols(), "atomic_number",
                              device, dtype=dtype)
    frac = torch.from_numpy(sc.get_scaled_positions()).to(device=device, dtype=dtype)
    lattice = torch.from_numpy(np.asarray(sc.cell)).to(device=device, dtype=dtype)

    g = dgl.graph((src, dst), num_nodes=n, device=device)
    g.ndata["atom_features"] = feats
    g.ndata["frac_coords"] = frac
    g.edata["r"] = cart
    g.edata["dist"] = dist
    g.edata["unit"] = unit
    vol = torch.abs(torch.det(lattice))
    g.ndata["V"] = vol.repeat(n)

    lg = _fast_line_graph(g)
    lg.apply_edges(compute_bond_cosines)
    return g, lg, lattice


def _forward(model, g, lg, lattice):
    with torch.no_grad():
        out = model((g, lg, lattice.clone()))
    return float(out["out"].detach().cpu().item())


def _run(device, dtype, base, cutoff, max_neighbors, sizes, permute_seed=None,
         label=""):
    tag = "f64" if dtype == torch.float64 else "f32"
    if permute_seed is not None:
        tag = f"{tag}-perm{permute_seed}"
    if label:
        tag = f"{tag}-{label}"
    print(f"\n=== {tag} ===", flush=True)

    try:
        model, _ = _load_model(device, dtype)
    except Exception as e:
        print(f"  model load FAILED ({type(e).__name__}): {e}", flush=True)
        return [{"tag": tag, "N": None, "status": f"load-fail: {type(e).__name__}"}]

    results = []
    for i in sizes:
        n = 4 * i ** 3
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])
            t0 = time.perf_counter()
            g, lg, lattice = _build_graph(
                sc, device, cutoff, max_neighbors, dtype, permute_seed=permute_seed,
            )
            energy = _forward(model, g, lg, lattice)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                           if device.type == "cuda" else 0.0)
            print(f"  i={i:3d}  N={n:>9}  E={energy: .6f}  "
                  f"t={dt:6.2f}s  peak={peak_mem_gb:6.1f} GB", flush=True)
            results.append({
                "tag": tag, "i": i, "N": n, "E": energy,
                "time_s": dt, "peak_mem_gb": peak_mem_gb, "status": "ok",
            })
            del g, lg, lattice, sc
            gc.collect()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            err = type(e).__name__
            msg = str(e).split("\n")[0][:160]
            print(f"  i={i:3d}  N={n:>9}  FAILED: {err}: {msg}", flush=True)
            results.append({
                "tag": tag, "i": i, "N": n, "E": None,
                "status": f"{err}: {msg}",
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Past the first OOM, everything larger will also OOM -- stop.
            if "OutOfMemoryError" in err or "out of memory" in str(e).lower():
                break

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20,30,40,45",
                    help="comma-separated multipliers to test in both dtypes")
    ap.add_argument("--permute-sizes", default="50,54,58",
                    help="comma-separated multipliers for the float32 atom-"
                         "permutation test (order-dependence check)")
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--output", default="drift_diag.json")
    ap.add_argument("--skip-permute", action="store_true")
    ap.add_argument("--skip-f64", action="store_true")
    args = ap.parse_args()

    if args.cuda_device.lower() not in ("", "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()
    config = loadjson(os.path.join(default_path(), "config.json"))
    cutoff = float(config["cutoff"])
    max_neighbors = int(config["max_neighbors"])
    print(f"cutoff={cutoff}  max_neighbors={max_neighbors}", flush=True)

    all_results = []
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    # f32 reference at the dtype-comparison sizes
    all_results += _run(device, torch.float32, base, cutoff, max_neighbors, sizes)

    # f64 at the same sizes (will OOM past a threshold; that's expected)
    if not args.skip_f64:
        all_results += _run(device, torch.float64, base, cutoff, max_neighbors, sizes)

    # Permutation-order test in f32 at drift-region sizes
    if not args.skip_permute:
        psizes = [int(s) for s in args.permute_sizes.split(",") if s.strip()]
        all_results += _run(device, torch.float32, base, cutoff, max_neighbors,
                            psizes, permute_seed=None, label="orig")
        all_results += _run(device, torch.float32, base, cutoff, max_neighbors,
                            psizes, permute_seed=1, label="permuted")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # -----------------------------------------------------------------------
    # Comparison tables
    # -----------------------------------------------------------------------
    def pick(tag, i):
        for r in all_results:
            if r["tag"] == tag and r.get("i") == i:
                return r
        return None

    print("\n=== f32 vs f64 (same size) ===", flush=True)
    print(f"{'i':>3}  {'N':>9}  {'E_f32':>11}  {'E_f64':>11}  {'|ΔE|':>10}", flush=True)
    print("-" * 55, flush=True)
    for i in sizes:
        r32 = pick("f32", i)
        r64 = pick("f64", i)
        e32 = r32.get("E") if r32 else None
        e64 = r64.get("E") if r64 else None
        if e32 is not None and e64 is not None:
            print(f"  {i:>3}  {4*i**3:>9}  {e32:>11.6f}  {e64:>11.6f}  {abs(e32-e64):>10.2e}", flush=True)
        else:
            e32s = f"{e32:11.6f}" if e32 is not None else "       ---"
            e64s = f"{e64:11.6f}" if e64 is not None else "       ---"
            print(f"  {i:>3}  {4*i**3:>9}  {e32s}  {e64s}  {'---':>10}", flush=True)

    if not args.skip_permute:
        psizes = [int(s) for s in args.permute_sizes.split(",") if s.strip()]
        print("\n=== f32 atom-permutation test (same physics, shuffled order) ===", flush=True)
        print(f"{'i':>3}  {'N':>9}  {'E_orig':>11}  {'E_perm':>11}  {'|ΔE|':>10}", flush=True)
        print("-" * 55, flush=True)
        for i in psizes:
            r0 = pick("f32-orig", i)
            r1 = pick("f32-perm1-permuted", i)
            e0 = r0.get("E") if r0 else None
            e1 = r1.get("E") if r1 else None
            if e0 is not None and e1 is not None:
                print(f"  {i:>3}  {4*i**3:>9}  {e0:>11.6f}  {e1:>11.6f}  {abs(e0-e1):>10.2e}", flush=True)
            else:
                e0s = f"{e0:11.6f}" if e0 is not None else "       ---"
                e1s = f"{e1:11.6f}" if e1 is not None else "       ---"
                print(f"  {i:>3}  {4*i**3:>9}  {e0s}  {e1s}  {'---':>10}", flush=True)

    print(f"\nRaw results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
