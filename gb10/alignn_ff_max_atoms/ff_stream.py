"""Chunked ALIGNN-FF on a crystal too large to hold: every piece is generated
from the lattice when it is evaluated (cells.crystal_piece), so no task
builds the cell. Without forces a task keeps no per-atom arrays at all. With
forces, rank r of W takes the x slab of cubes [r*kx/W, (r+1)*kx/W) and keeps
forces only for the atoms its pieces touch, in a dense array over the x
cells of its slab plus halo (cells.local_rows); reduce adds the slabs block
by block, never holding a full-size array. Rattled positions use a hash of
the atom index (cells.hash_normal), so every piece displaces an atom the same
way. Model, graph and energy code come from ff_max_atoms.py, as in ff_any.py.

chunk   --kind Si|SiGe|Al|Cu|SiO2 --rep R (R^3 conventional cells), options as
        ff_any chunk; --validate (one task, small cells) compares with a full-
        cell evaluation of cells.crystal_hash
reduce  energy per atom; with forces, largest force, net force and rms force,
        slab by slab; --save_dense writes the summed forces (small cells only)

    python ff_stream.py chunk --model_dir M --kind Si --rep 630 --rattle 0.05 --cluster_atoms 330820 --rank 0 --world 10 --out R/s9_2B_energy
    python ff_stream.py chunk --model_dir M --kind Si --rep 630 --rattle 0.05 --forces --ckpt --cluster_atoms 106000 --rank 0 --world 20 --out R/s9_2B_forces
    python ff_stream.py reduce --out R/s9_2B_forces
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import cells
import ff_max_atoms as ffm


def chunk(args, device):
    """Evaluate this rank's x slab of cubes, generating each piece."""
    model, cfg = ffm.load_model(args.model_dir, device, ckpt=args.ckpt)
    h = args.halo or ffm.reach(cfg) + args.halo_margin
    if h < cfg["cutoff"]:
        raise SystemExit(f"halo {h} A is shorter than the cutoff")
    basis, a0, _ = cells.LATTICES[args.kind]
    a, nb, rep = args.a or a0, len(basis), args.rep
    n = rep**3 * nb
    L = np.full(3, rep * a)
    k = (np.full(3, args.k, dtype=np.int64) if args.k else
         cells.choose_k(L, n, h, args.cluster_atoms, cfg["cutoff"]))
    if args.world > k[0]:
        raise SystemExit(f"{args.world} ranks but only {k[0]} x slabs of cubes")
    edges = cells.cube_edges(L, k)
    cx0 = args.rank * int(k[0]) // args.world
    cx1 = (args.rank + 1) * int(k[0]) // args.world
    mine = [(cx, cy, cz) for cx in range(cx0, cx1) for cy in range(k[1])
            for cz in range(k[2])][:args.max_chunks or None]
    ixa, ixb = cells.slab_cells(args.kind, edges[0], cx0, cx1, h, a,
                                args.rattle)
    plane = rep * rep * nb
    forces = (torch.zeros(((ixb - ixa) * plane, 3), dtype=torch.float64)
              if args.forces else None)
    payload = ffm.header(args, cfg, device)
    payload.update(
        natoms=n, kind=args.kind, rep=rep, a_A=a, nb=nb, rattle=args.rattle,
        seed=args.seed, box_A=L.tolist(), k=k.tolist(),
        core_A=(L / k).tolist(), halo_A=h, halo_exact=h >= ffm.reach(cfg),
        forces=args.forces, ckpt=args.ckpt, rank=args.rank, world=args.world,
        slab_cubes_x=[cx0, cx1], ixa=ixa, ixb=ixb,
        local_force_rows=(ixb - ixa) * plane if args.forces else 0,
        n_cubes=int(np.prod(k)), cubes_done=0, owned_done=0, energy=0.0,
        pieces=[])
    ffm.log(f"N={n} box {L[0]:.1f} A, {k.tolist()} cubes, halo {h:.2f} A, "
            f"rank {args.rank}/{args.world}: x cubes {cx0}-{cx1 - 1}, "
            f"{len(mine)} pieces, x cells {ixa}..{ixb - 1}")
    t_start = time.perf_counter()
    for i, cube in enumerate(mine):
        lo = np.array([edges[d][cube[d]] for d in range(3)])
        hi = np.array([edges[d][cube[d] + 1] for d in range(3)])
        t0 = time.perf_counter()
        gid, cpos, owned, zz, ix_u = cells.crystal_piece(
            args.kind, rep, lo, hi, h, a, args.rattle, args.seed)
        if not owned.any():
            payload["cubes_done"] += 1
            payload["pieces"].append({"cube": list(cube), "atoms": len(gid),
                                      "owned": 0, "skipped": True})
            continue
        cpos = cpos - cpos.min(0) + 1.0
        box = np.diag(cpos.max(0) + cfg["cutoff"] + 2.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        g = ffm.build_graph(cpos, box, zz, cfg, device)
        assert g["max_shift"] == 0, "piece touches its own image"
        p_t = torch.tensor(cpos, dtype=ffm.DTYPE, device=device)
        lat = torch.tensor(box, dtype=ffm.DTYPE, device=device)
        own_t = torch.from_numpy(owned).to(device)
        ffm.sync(device)
        t1 = time.perf_counter()
        if args.forces:
            p_t.requires_grad_(True)
            e_atom, e_bond = ffm.atom_energies(model, p_t, lat, g)
            e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
            (grad,) = torch.autograd.grad(e_own, p_t)
            rows = cells.local_rows(gid, ix_u, ixa, rep, nb)
            forces.index_add_(0, torch.from_numpy(rows),
                              -grad.detach().double().cpu())
            del grad
        else:
            with torch.inference_mode():
                e_atom, e_bond = ffm.atom_energies(model, p_t, lat, g)
                e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
        e_own = float(e_own)
        del e_atom, e_bond, g, p_t
        ffm.sync(device)
        t2 = time.perf_counter()
        payload["energy"] += e_own
        payload["cubes_done"] += 1
        payload["owned_done"] += int(owned.sum())
        payload["pieces"].append({
            "cube": list(cube), "atoms": len(gid), "owned": int(owned.sum()),
            "t_build_s": t1 - t0, "t_eval_s": t2 - t1,
            **ffm.gpu_peak_mem(device)})
        if (i + 1) % 200 == 0:  # progress, and evidence if the task dies
            payload["wall_s"] = time.perf_counter() - t_start
            ffm.dump(payload, args.out)
            ffm.log(f"{i + 1}/{len(mine)} pieces, {payload['owned_done']:,} "
                    f"atoms, {payload['wall_s']:.0f} s")
    wall = time.perf_counter() - t_start
    payload.update(wall_s=wall, host_peak_gb=ffm.host_peak_gb(),
                   owned_atoms_per_s=payload["owned_done"] / max(wall, 1e-9))
    whole = args.world == 1 and payload["cubes_done"] == payload["n_cubes"]
    if whole:
        payload["e_per_atom"] = payload["energy"] / n
    dense = None
    if args.forces:
        local = forces.numpy()
        if args.world > 1:
            np.save(f"{args.out}_forces.npy", local.astype(
                np.float64 if ffm.DTYPE == torch.float64 else np.float32))
        elif whole:
            dense = cells.fold_planes(local, ixa, ixb, rep, nb,
                                      np.zeros((n, 3)))
            payload.update(ffm.force_stats(dense))
            if args.save_dense:
                np.save(f"{args.out}_dense_forces.npy", dense)
    ffm.dump(payload, args.out)
    if args.validate:
        if not whole or n > 200_000:
            raise SystemExit("--validate needs one task over a small cell")
        pos, z, Lf = cells.crystal_hash(args.kind, rep, args.rattle,
                                        args.seed, a)
        payload["validation"] = ffm.validate(
            model, cfg, pos, z, Lf,
            None if dense is None else torch.from_numpy(dense), payload,
            device)
        ffm.dump(payload, args.out)
    ffm.log({k_: v for k_, v in payload.items() if k_ != "pieces"})


def reduce(args):
    """Total energy, and force statistics summed slab by slab."""
    parts = sorted(glob.glob(f"{args.out}_rank*.json"))
    ps = [json.load(open(p)) for p in parts]
    n, world = ps[0]["natoms"], ps[0]["world"]
    assert len(ps) == world, f"{len(ps)} of {world} parts"
    assert sum(p["cubes_done"] for p in ps) == ps[0]["n_cubes"], "cubes missing"
    assert sum(p["owned_done"] for p in ps) == n, "atoms missing"
    out = {k: ps[0][k] for k in ("model_dir", "natoms", "kind", "rep", "a_A",
                                 "rattle", "box_A", "k", "halo_A",
                                 "halo_exact", "forces", "gpu")}
    out["e_per_atom"] = sum(p["energy"] for p in ps) / n
    out["wall_s_slowest_rank"] = max(p["wall_s"] for p in ps)
    out["gb10_hours"] = sum(p["wall_s"] for p in ps) / 3600
    out["piece_atoms_max"] = max(q["atoms"] for p in ps for q in p["pieces"])
    out["mem_alloc_gb_max"] = max(q.get("mem_alloc_gb") or 0
                                  for p in ps for q in p["pieces"])
    out["host_peak_gb_max"] = max(p["host_peak_gb"] for p in ps)
    if ps[0]["forces"]:
        rep, nb = ps[0]["rep"], ps[0]["nb"]
        plane = rep * rep * nb
        slabs = [(np.load(p.replace(".json", "_forces.npy"), mmap_mode="r"),
                  q["ixa"], q["ixb"]) for p, q in zip(parts, ps)]
        dense = np.zeros((n, 3)) if args.save_dense else None
        fmax, net, sumsq = 0.0, np.zeros(3), 0.0
        for g0 in range(0, rep, args.block):
            g1 = min(rep, g0 + args.block)
            acc = np.zeros(((g1 - g0) * plane, 3))
            for arr, ixa, ixb in slabs:
                for j in range(ixb - ixa):
                    g = (ixa + j) % rep
                    if g0 <= g < g1:
                        acc[(g - g0) * plane:(g - g0 + 1) * plane] += \
                            arr[j * plane:(j + 1) * plane]
            fmax = max(fmax, float(np.abs(acc).max()))
            net += acc.sum(0)
            sumsq += float((acc * acc).sum())
            if dense is not None:
                dense[g0 * plane:g1 * plane] = acc
        out.update(max_abs_f=fmax, net_force=np.abs(net).tolist(),
                   rms_f=float(np.sqrt(sumsq / (3 * n))))
        if dense is not None:
            np.save(f"{args.out}_dense_forces.npy", dense)
    json.dump(out, open(f"{args.out}_reduced.json", "w"), indent=1)
    ffm.log(json.dumps(out))


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["chunk", "reduce"])
    p.add_argument("--model_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--kind", choices=list(cells.LATTICES), default="Si")
    p.add_argument("--rep", type=int, default=10,
                   help="conventional cells per side")
    p.add_argument("--a", type=float, default=None,
                   help="lattice constant override, A")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cluster_atoms", type=int, default=50000,
                   help="target atoms per piece (cube + halo)")
    p.add_argument("--k", type=int, default=0,
                   help="cubes per side (overrides --cluster_atoms)")
    p.add_argument("--halo", type=float, default=0,
                   help="halo in A (default: the model's reach, exact)")
    p.add_argument("--halo_margin", type=float, default=0.01)
    p.add_argument("--forces", action="store_true")
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--max_chunks", type=int, default=0,
                   help="stop after this many cubes (timing sample)")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--save_dense", action="store_true",
                   help="write summed forces as an (N, 3) array (small cells)")
    p.add_argument("--block", type=int, default=16,
                   help="x planes per block in reduce")
    p.add_argument("--float64", action="store_true",
                   help="double precision (exactness tests)")
    args = p.parse_args()
    if args.float64:
        ffm.DTYPE = torch.float64
        torch.set_default_dtype(torch.float64)
    if args.mode == "reduce":
        return reduce(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.world > 1:
        args.out = f"{args.out}_rank{args.rank:04d}"
    ffm.log("device", device, "| alignn", os.path.dirname(ffm.alignn.__file__))
    chunk(args, device)


if __name__ == "__main__":
    main()
