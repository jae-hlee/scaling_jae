"""
scale5b_v5.py
=============

Delta from v4:

1. Replaces `dgl.line_graph(g, shared=True)` with a hand-rolled sparse
   construction `_fast_line_graph(g)`. The v4 benchmark (see
   `analysis/summary.md`) showed `dgl.line_graph` scaled as N^1.995 and was
   99.98% of total runtime at N=389,344 atoms. With a bounded
   `max_neighbors`, the combinatorics of the line graph are O(N), so the
   quadratic scaling was a DGL library path, not an algorithmic limit. The
   replacement is O(E log E) (dominated by a single argsort by destination
   node) plus O(L) to emit line-graph edges.

2. Adds checkpoint resume: if the `.npz` output already exists on disk, its
   contents are loaded and the main loop skips sizes that were already
   completed. Useful when a SLURM wall kills the job mid-sweep -- the next
   submission continues from the next unfinished size instead of redoing
   everything from N=4.

Inherits v4's matscipy CPU neighbor list and all other behavior.

Usage
-----
    python scale5b_v5.py              # scaling sweep (resumes if .npz exists)
    python scale5b_v5.py --verify     # ASE + line-graph equivalence check
    python scale5b_v5.py --max-size N # cap size (default 99)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch

import dgl
from ase import Atoms
from ase.build.supercells import make_supercell
from ase.neighborlist import neighbor_list as ase_neighbor_list
from matscipy.neighbours import neighbour_list as matscipy_neighbours
from alignn.graphs import compute_bond_cosines
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.ff.ff import default_path
from jarvis.core.specie import get_node_attributes
from jarvis.db.jsonutils import loadjson
from jarvis.io.vasp.inputs import Poscar


# =============================================================================
# Feature cache
# =============================================================================

_FEATURE_CACHE: dict = {}


def _species_features(symbols, atom_features, device, dtype=torch.float32):
    for s in set(symbols):
        k = (s, atom_features)
        if k not in _FEATURE_CACHE:
            _FEATURE_CACHE[k] = np.asarray(
                get_node_attributes(s, atom_features=atom_features),
                dtype=np.float32,
            )
    arr = np.stack([_FEATURE_CACHE[(s, atom_features)] for s in symbols], 0)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


# =============================================================================
# Fast neighbor list: matscipy on CPU -> GPU tensors
# =============================================================================

def _neighbor_list_matscipy(
    atoms: Atoms,
    cutoff: float,
    device: torch.device,
    max_neighbors: Optional[int] = 12,
    float_dtype: torch.dtype = torch.float32,
):
    i_np, j_np, D_np, S_np = matscipy_neighbours("ijDS", atoms, cutoff)

    if len(i_np) == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty3 = torch.empty((0, 3), dtype=float_dtype, device=device)
        return empty_long, empty_long, empty3, torch.empty(0, dtype=float_dtype, device=device), empty3

    src = torch.from_numpy(i_np).to(device=device, dtype=torch.long, non_blocking=True)
    dst = torch.from_numpy(j_np).to(device=device, dtype=torch.long, non_blocking=True)
    cart = torch.from_numpy(D_np).to(device=device, dtype=float_dtype, non_blocking=True)
    shift = torch.from_numpy(S_np).to(device=device, dtype=float_dtype, non_blocking=True)
    dist = cart.norm(dim=1)

    if max_neighbors is not None and max_neighbors > 0 and src.shape[0] > 0:
        N = int(atoms.positions.shape[0])
        E = src.shape[0]
        rank = torch.empty(E, dtype=torch.long, device=device)
        rank[torch.argsort(dist)] = torch.arange(E, device=device)
        key = src * E + rank
        order = torch.argsort(key)
        src, dst = src[order], dst[order]
        cart, dist, shift = cart[order], dist[order], shift[order]
        src_start = torch.searchsorted(src, torch.arange(N, device=device))
        within = torch.arange(E, device=device) - src_start[src]
        sel = within < max_neighbors
        src, dst = src[sel], dst[sel]
        cart, dist, shift = cart[sel], dist[sel], shift[sel]

    return src, dst, cart, dist, shift


# =============================================================================
# Fast sparse line graph  (the v5 bottleneck fix)
# =============================================================================

def _fast_line_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Equivalent to `dgl.line_graph(g, shared=True)` (with DGL's default
    backtracking=True), but built with sparse segment ops instead of whatever
    DGL is doing that scales as N^2.

    Semantics: for every ordered pair of directed edges (e1, e2) in g where
    dst(e1) == src(e2), the line graph has a directed edge e1 -> e2. Node
    features on lg are set to reference g's edge features (shared with v4).

    Complexity: O(E log E) for the argsort + O(L) to emit lg edges, where
    L = sum over nodes v of in_deg(v) * out_deg(v). With bounded
    max_neighbors, L = O(N).
    """
    device = g.device
    src_e, dst_e = g.edges()
    E = int(src_e.shape[0])
    N = int(g.num_nodes())

    if E == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        lg = dgl.graph((empty, empty), num_nodes=0, device=device)
        for k, v in g.edata.items():
            lg.ndata[k] = v
        return lg

    # Group edges by their dst node. sort_order[k] is the original index of
    # the edge whose dst is the k-th in sorted order.
    sort_order = torch.argsort(dst_e)
    counts = torch.bincount(dst_e, minlength=N)
    offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)
    # Incoming edges to node v are: sort_order[offsets[v] : offsets[v+1]].

    # For each edge e2, the "middle atom" of the potential angle is src_e[e2].
    # We need every e1 whose dst equals that middle atom.
    n_e1_per_e2 = counts[src_e]               # shape (E,)
    total = int(n_e1_per_e2.sum().item())

    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        lg = dgl.graph((empty, empty), num_nodes=E, device=device)
    else:
        # lg edge destination: e2 repeated n_e1_per_e2[e2] times.
        lg_dst = torch.repeat_interleave(torch.arange(E, device=device), n_e1_per_e2)
        # lg edge source: for each e2, the block sort_order[offsets[v] :
        # offsets[v] + n_e1_per_e2[e2]] where v = src_e[e2]. Expanded with
        # segment offsets so every e2 indexes its own block.
        starts_per_e2 = offsets[src_e]
        starts_rep = torch.repeat_interleave(starts_per_e2, n_e1_per_e2)
        cumn = n_e1_per_e2.cumsum(0)
        block_starts = cumn - n_e1_per_e2
        block_starts_rep = torch.repeat_interleave(block_starts, n_e1_per_e2)
        inner = torch.arange(total, device=device) - block_starts_rep
        lg_src = sort_order[starts_rep + inner]
        lg = dgl.graph((lg_src, lg_dst), num_nodes=E, device=device)

    # `shared=True` equivalent: lg.ndata entries reference the same tensors
    # as g.edata (DGL stores by reference on assignment).
    for k, v in g.edata.items():
        lg.ndata[k] = v

    return lg


# =============================================================================
# Public fast_graph
# =============================================================================

def fast_graph(
    atoms: Atoms,
    device: torch.device,
    cutoff: float = 5.0,
    max_neighbors: Optional[int] = 12,
    atom_features: str = "atomic_number",
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]:
    n_atoms = len(atoms)

    src, dst, cart, dist, _shift = _neighbor_list_matscipy(
        atoms, cutoff=cutoff, device=device, max_neighbors=max_neighbors,
    )
    unit = cart / dist[:, None].clamp(min=1e-12)

    feats = _species_features(
        atoms.get_chemical_symbols(), atom_features, device,
    )
    frac = torch.from_numpy(atoms.get_scaled_positions()).to(
        device=device, dtype=torch.float32,
    )
    lattice = torch.from_numpy(np.asarray(atoms.cell)).to(
        device=device, dtype=torch.float32,
    )

    g = dgl.graph((src, dst), num_nodes=n_atoms, device=device)
    g.ndata["atom_features"] = feats
    g.ndata["frac_coords"] = frac
    g.edata["r"] = cart
    g.edata["dist"] = dist
    g.edata["unit"] = unit
    vol = torch.abs(torch.det(lattice))
    g.ndata["V"] = vol.repeat(n_atoms)

    lg = _fast_line_graph(g)
    lg.apply_edges(compute_bond_cosines)

    return g, lg, lattice


# =============================================================================
# Sanity checks
# =============================================================================

_CU_POSCAR = """System
1.0
3.6 0.0 0.0
0.0 3.6 0.0
0.0 0.0 3.6
Cu
4
direct
0.0 0.0 0.0 Cu
0.0 0.5 0.5 Cu
0.5 0.0 0.5 Cu
0.5 0.5 0.0 Cu
"""


def verify_against_ase(cutoff: float, max_neighbors: int, sizes=(1, 2, 3, 4)) -> bool:
    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()
    device = torch.device("cpu")
    all_ok = True
    print("  ASE sanity check:")
    for i in sizes:
        sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])
        n = len(sc)
        i_ase = ase_neighbor_list("i", sc, cutoff)
        deg = np.bincount(i_ase, minlength=n)
        expected = int(np.minimum(deg, max_neighbors).sum())

        src, *_ = _neighbor_list_matscipy(
            sc, cutoff=cutoff, device=device, max_neighbors=max_neighbors,
        )
        got = int(src.shape[0])
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"    size {i:>2}x  n_atom={n:<5d}  ase_expected={expected:<7d}  "
              f"got={got:<7d}  {status}")
    return all_ok


def verify_line_graph_equivalence(cutoff: float, max_neighbors: int,
                                  sizes=(2, 3, 4)) -> bool:
    """
    For a few supercell sizes, build the line graph two ways -- the old
    `dgl.line_graph(g, shared=True)` path and the new `_fast_line_graph(g)`
    path -- and confirm they produce identical edge sets and identical
    `compute_bond_cosines` output. This is the correctness gate for the
    v5 bottleneck fix.
    """
    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()
    device = torch.device("cpu")
    all_ok = True
    print("  Line-graph equivalence check (old DGL vs sparse v5):")
    for i in sizes:
        sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])
        n = len(sc)
        src, dst, cart, dist, _ = _neighbor_list_matscipy(
            sc, cutoff=cutoff, device=device, max_neighbors=max_neighbors,
        )
        g = dgl.graph((src, dst), num_nodes=n, device=device)
        g.edata["r"] = cart

        lg_old = dgl.line_graph(g, shared=True)
        lg_old.apply_edges(compute_bond_cosines)

        lg_new = _fast_line_graph(g)
        lg_new.apply_edges(compute_bond_cosines)

        e_old = lg_old.num_edges()
        e_new = lg_new.num_edges()
        if e_old != e_new:
            print(f"    size {i:>2}x  n_atom={n:<5d}  n_lg_edges_old={e_old} "
                  f"new={e_new}  FAIL (edge count)")
            all_ok = False
            continue

        def row_key(u, v, E):
            return u.to(torch.long) * (E + 1) + v.to(torch.long)

        u_o, v_o = lg_old.edges()
        u_n, v_n = lg_new.edges()
        E = max(int(u_o.max().item()), int(u_n.max().item())) + 1
        idx_o = torch.argsort(row_key(u_o, v_o, E))
        idx_n = torch.argsort(row_key(u_n, v_n, E))
        same_edges = torch.equal(u_o[idx_o], u_n[idx_n]) and torch.equal(v_o[idx_o], v_n[idx_n])
        if not same_edges:
            print(f"    size {i:>2}x  n_atom={n:<5d}  n_lg_edges={e_old}  "
                  f"FAIL (edge sets differ)")
            all_ok = False
            continue

        h_diff = (lg_old.edata["h"][idx_o] - lg_new.edata["h"][idx_n]).abs().max().item()
        status = "OK" if h_diff < 1e-6 else "FAIL"
        if h_diff >= 1e-6:
            all_ok = False
        print(f"    size {i:>2}x  n_atom={n:<5d}  n_lg_edges={e_old:<8d}  "
              f"max |Δcos(θ)|={h_diff:.2e}  {status}")
    return all_ok


# =============================================================================
# Scaling driver
# =============================================================================

def _load_alignn_ff(device: torch.device):
    path = default_path()
    config = loadjson(os.path.join(path, "config.json"))
    config["model"]["calculate_gradient"] = False
    config["model"]["stresswise_weight"] = 0
    config["model"]["graphwise_weight"] = 1.0
    config["model"]["atomwise_weight"] = 0
    config["model"]["gradwise_weight"] = 0
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
    model.load_state_dict(torch.load(
        os.path.join(path, "best_model.pt"),
        map_location=device, weights_only=False,
    ))
    model.to(device).eval()
    return model, config


def _load_checkpoint(path: str):
    """Resume helper. Returns six lists (natoms, t_nl, t_line, t_graph, t_inf,
    energies) plus a set of already-completed N values. Empty lists if no
    file."""
    if not os.path.exists(path):
        return [], [], [], [], [], [], set()
    prior = np.load(path)
    natoms = list(map(int, prior["natoms"]))
    t_nl = list(map(float, prior["times_nl"]))
    t_line = list(map(float, prior["times_line"]))
    t_graph = list(map(float, prior["times_graph"]))
    t_inf = list(map(float, prior["times_inference"]))
    energies = list(map(float, prior["energies"]))
    done = set(natoms)
    return natoms, t_nl, t_line, t_graph, t_inf, energies, done


def run_scaling(
    max_size: int = 99,
    cuda_device: Optional[str] = "2",
    verify: bool = True,
    checkpoint: str = "scaling_alignn_v5.npz",
    plot: str = "scaling_alignn_v5.png",
) -> None:
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = _load_alignn_ff(device)
    cutoff = float(config["cutoff"])
    max_neighbors = int(config["max_neighbors"])
    print(f"max_neighbors: {max_neighbors}  cutoff: {cutoff}")

    if verify:
        if not verify_against_ase(cutoff, max_neighbors, sizes=(1, 2, 3, 4)):
            print("  ! ASE sanity check FAILED -- aborting.")
            sys.exit(1)
        if not verify_line_graph_equivalence(cutoff, max_neighbors):
            print("  ! Line-graph equivalence check FAILED -- aborting.")
            sys.exit(1)
        print("  All verification passed.")

    natoms, t_nl, t_line, t_graph_total, t_inf, energies, done = \
        _load_checkpoint(checkpoint)
    if done:
        max_prior = max(done)
        print(f"Resuming from {checkpoint}: {len(done)} sizes already done "
              f"(up to N={max_prior}).")

    base = Poscar.from_string(_CU_POSCAR).atoms.ase_converter()

    print()
    print(f"{'size':>4}  {'n_atom':>9}  {'nl':>8}  {'line':>8}  "
          f"{'graph':>10}  {'inference':>10}  {'E (eV)':>10}")
    print("-" * 75)

    for i in range(1, max_size + 1):
        n = 4 * i ** 3
        if n in done:
            continue
        try:
            sc = make_supercell(base, [[i, 0, 0], [0, i, 0], [0, 0, i]])

            # --- (a) matscipy CPU neighbor search + transfer ---
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            src, dst, cart, dist, _ = _neighbor_list_matscipy(
                sc, cutoff=cutoff, device=device, max_neighbors=max_neighbors,
            )
            if device.type == "cuda": torch.cuda.synchronize()
            t_nl_step = time.perf_counter() - t0

            # --- (b) DGL graph + (sparse) line graph ---
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            unit = cart / dist[:, None].clamp(min=1e-12)
            feats = _species_features(sc.get_chemical_symbols(), "atomic_number", device)
            frac = torch.from_numpy(sc.get_scaled_positions()).to(device=device, dtype=torch.float32)
            lattice = torch.from_numpy(np.asarray(sc.cell)).to(device=device, dtype=torch.float32)

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
            if device.type == "cuda": torch.cuda.synchronize()
            t_line_step = time.perf_counter() - t0

            t_graph_step = t_nl_step + t_line_step

            # --- (c) inference ---
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model((g, lg, lattice.clone()))
                energy = out["out"].detach().cpu().item()
            if device.type == "cuda": torch.cuda.synchronize()
            t_inf_step = time.perf_counter() - t0

            del g, lg, out, sc, frac, lattice, cart, dist, unit, src, dst
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()

            natoms.append(n)
            t_nl.append(t_nl_step)
            t_line.append(t_line_step)
            t_graph_total.append(t_graph_step)
            t_inf.append(t_inf_step)
            energies.append(energy)
            done.add(n)

            print(f"{i:>4}  {n:>9}  {t_nl_step:>6.3f}s  "
                  f"{t_line_step:>6.3f}s  {t_graph_step:>8.3f}s  "
                  f"{t_inf_step:>8.3f}s  {energy:>10.3f}")

            np.savez(
                checkpoint,
                natoms=np.array(natoms),
                times_nl=np.array(t_nl),
                times_line=np.array(t_line),
                times_graph=np.array(t_graph_total),
                times_inference=np.array(t_inf),
                energies=np.array(energies),
            )

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  ! stopped at size {i} (n={n}): {type(e).__name__}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break

    if len(natoms) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(np.array(natoms))
        n_arr = np.array(natoms)[order]
        t_nl_arr = np.array(t_nl)[order]
        t_line_arr = np.array(t_line)[order]
        t_inf_arr = np.array(t_inf)[order]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(n_arr, t_nl_arr, "-o", label="Neighbor list (matscipy CPU)")
        ax1.plot(n_arr, t_line_arr, "-s", label="DGL graph + sparse line graph")
        ax1.plot(n_arr, t_inf_arr, "-^", label="ALIGNN-FF inference")
        ax1.set(xlabel="Number of atoms", ylabel="Time (s)",
                title="ALIGNN-FF scaling (breakdown)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3); ax1.legend()

        total = np.array(t_graph_total)[order] + np.array(t_inf)[order]
        ax2.plot(n_arr, total, "-o", label="Total")
        ax2.set(xlabel="Number of atoms", ylabel="Time (s)", title="Total")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.grid(True, which="both", alpha=0.3); ax2.legend()

        plt.tight_layout()
        plt.savefig(plot, dpi=200)
        print(f"\nSaved plot: {plot}")
        print(f"Saved data: {checkpoint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--max-size", type=int, default=99)
    ap.add_argument("--cuda-device", default="2")
    args = ap.parse_args()
    cuda = None if args.cuda_device.lower() in ("", "cpu") else args.cuda_device
    if args.verify:
        path = default_path()
        config = loadjson(os.path.join(path, "config.json"))
        cutoff = float(config["cutoff"])
        max_nbr = int(config["max_neighbors"])
        ok_ase = verify_against_ase(cutoff, max_nbr)
        ok_lg = verify_line_graph_equivalence(cutoff, max_nbr)
        sys.exit(0 if (ok_ase and ok_lg) else 1)
    run_scaling(max_size=args.max_size, cuda_device=cuda)


if __name__ == "__main__":
    main()
