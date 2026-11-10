"""ALIGNN-FF on structures other than the built-in Si cells: the largest
single graph (sweep) and exact chunked evaluation (chunk) of any periodic
orthorhombic cell written by cells.py.

Model loading, graph building and the energy and force evaluation are
imported from ff_max_atoms.py, which is left unchanged because the stage 4
and 5 job scripts check its md5. The cube-and-halo geometry is in cells.py,
where it is checked without a GPU (python cells.py test).

sweep   one structure family (--kind Si|SiGe|Al|Cu|water, see cells.py)
        stepped up in size until out of memory, one evaluation path per
        process: e (energy only), ef (E+F+S through forward_tensors) or ckpt
        (ef with layer checkpointing). Every row also records bonds and
        line-graph edges (bond angles) per atom, which set the cost.
chunk   a cells.py structure (--structure PREFIX) cut into kx*ky*kz cubes,
        each evaluated with every atom within the model's reach, as in
        ff_max_atoms chunk. Every piece also records its bonds and
        line-graph edges. --max_chunks N evaluates only N pieces, to
        measure throughput on a cell too large to finish. The halo is exact
        only while no atom has more neighbors within the cutoff than the
        model's cap (52); Si, Al and Cu stay below it, liquid water and
        other dense, hydrogen-rich structures do not, and then a halo of
        the model's reach leaves a small truncation error (recorded as
        neighbor_cap_binds; measure it with --validate, --halo and
        --max_neighbors).
reduce  sums the parts of a multi-task chunk run (ff_max_atoms reduce).

    python ff_any.py sweep --model_dir M --kind water --path ckpt --out OUT
    python ff_any.py chunk --model_dir M --structure S/al_8k --k 2 --forces --ckpt --validate --out OUT
    python ff_any.py chunk --model_dir M --structure S/water_1m --cluster_atoms 12000 --forces --ckpt --max_chunks 20 --out OUT
"""
import argparse
import itertools
import os
import statistics
import time

import numpy as np
import torch

import cells
import ff_max_atoms as ffm


def make(kind, i):
    """Size i of a structure family: i conventional cells (crystals) or
    i water molecules per side."""
    return cells.water(i) if kind == "water" else cells.crystal(kind, i)


def graph_stats(model, pos, lat, g):
    """Bonds, and line-graph edges counted the way atom_energies builds
    them: for every bond i->j within the three-body cutoff, one edge per
    bond j->k within the cutoff."""
    with torch.no_grad():
        src, dst = g["src"], g["dst"]
        r = pos.index_select(0, dst) - pos.index_select(0, src) + g["shift"] @ lat
        ok = torch.linalg.vector_norm(r, dim=1) <= model.three_body_cutoff
        out_deg = torch.bincount(src[ok], minlength=pos.shape[0])
        lg = int(out_deg.index_select(0, dst[ok]).sum())
        degree = torch.bincount(src, minlength=pos.shape[0])
    # max_degree at the neighbor cap means the cap may have removed bonds
    # (an atom with exactly that many neighbors also reaches it)
    return {"edges": int(src.numel()), "lg_edges": lg,
            "max_degree": int(degree.max()) if src.numel() else 0}


def owned_energies(e_atom, e_bond, src, own_t):
    """Energy of each owned atom, with each bond term (penalty, ZBL)
    counted on the bond's first atom, as a float32 numpy array."""
    ea = e_atom.detach().clone()
    ea.index_add_(0, src, e_bond.detach())
    return ea[own_t].float().cpu().numpy()


def evaluate(path, pos, L, z, model, cfg, device):
    """One evaluation of a periodic cell, with timings and graph sizes."""
    cell = np.diag(L)
    t0 = time.perf_counter()
    g = ffm.build_graph(pos, cell, z, cfg, device)
    p = torch.tensor(pos, dtype=ffm.DTYPE, device=device)
    lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
    ffm.sync(device)
    t1 = time.perf_counter()
    if path == "e":
        with torch.inference_mode():
            e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
            e = float(e_atom.sum() + e_bond.sum())
        fmax = None
    else:
        p.requires_grad_(True)
        res = model.forward_tensors(p, lat, g["feats"], g["src"], g["dst"],
                                    g["shift"], True, g["Z"])
        e = float(res["energy"])
        fmax = float(res["forces"].abs().max())
        del res
    ffm.sync(device)
    t2 = time.perf_counter()
    out = dict(t_graph_s=t1 - t0, t_eval_s=t2 - t1, t_total_s=t2 - t0,
               e=e, fmax=fmax, **ffm.gpu_peak_mem(device))
    out.update(graph_stats(model, p.detach(), lat, g))
    return out


def sweep(args, device):
    """Step one structure family up until out of memory, one path."""
    model, cfg = ffm.load_model(args.model_dir, device,
                                ckpt=args.path == "ckpt")
    if args.max_neighbors:
        cfg["max_neighbors"] = args.max_neighbors
    payload = ffm.header(args, cfg, device)
    payload.update(path=args.path, kind=args.kind, results=[], oom=None,
                   error=None, structure=f"cells.py {args.kind}, size i")
    pos, z, L = make(args.kind, 2)
    for _ in range(2):  # warm-up
        evaluate(args.path, pos, L, z, model, cfg, device)
    sizes = [i for i in list(range(2, 15, 2)) + list(range(15, 1000))
             if i <= args.max_i]
    for i in sizes:
        pos, z, L = make(args.kind, i)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reps = []
        try:
            while len(reps) < args.repeats:
                reps.append(evaluate(args.path, pos, L, z, model, cfg, device))
                if reps[-1]["t_total_s"] > 30:  # big cells: one call
                    break
        except Exception as err:  # noqa: BLE001  (record why it stopped)
            kind = "oom" if ffm.is_oom(err) else "error"
            payload[kind] = {"i": i, "natoms": len(pos),
                             "error": f"{type(err).__name__}: {err}"[:300]}
            ffm.log(f"[{kind.upper()}] i={i} N={len(pos)}: {str(err)[:150]}")
            ffm.dump(payload, args.out)
            break
        row = {"i": i, "natoms": len(pos)}
        for key in reps[0]:
            vals = [r[key] for r in reps]
            row[key] = (statistics.median(vals) if key.startswith("t_")
                        else max(vals) if key.startswith("mem_") else vals[0])
        row["t_all_s"] = [r["t_total_s"] for r in reps]
        row["e_per_atom"] = row.pop("e") / len(pos)
        row["edges_per_atom"] = row["edges"] / len(pos)
        row["lg_edges_per_atom"] = row["lg_edges"] / len(pos)
        row["host_peak_gb"] = ffm.host_peak_gb()
        if row.get("mem_alloc_gb"):
            row["mb_per_atom"] = row["mem_alloc_gb"] * 1e3 / len(pos)
        payload["results"].append(row)
        ffm.log(f"i={i:3d} N={len(pos):9d}  t={row['t_total_s']:8.3f} s"
                f"  alloc {row.get('mem_alloc_gb', 0):6.1f} GB"
                f"  ({row.get('mb_per_atom', 0):.3f} MB/atom)"
                f"  bonds/atom {row['edges_per_atom']:.1f}"
                f"  angles/atom {row['lg_edges_per_atom']:.1f}"
                f"  E/atom {row['e_per_atom']:.5f}")
        ffm.dump(payload, args.out)
    ffm.log("done ->", args.out + ".json")


def chunk(args, device):
    """Evaluate one cells.py structure piece by piece (this rank's pieces)."""
    model, cfg = ffm.load_model(args.model_dir, device, ckpt=args.ckpt)
    if args.max_neighbors:
        cfg["max_neighbors"] = args.max_neighbors
    h = args.halo or ffm.reach(cfg) + args.halo_margin
    if h < cfg["cutoff"]:
        raise SystemExit(f"halo {h} A is shorter than the cutoff")
    pos, z, L, meta = cells.read_structure(args.structure)
    if args.rattle:
        pos = np.mod(np.asarray(pos) + np.random.default_rng(args.seed).normal(
            0, args.rattle, pos.shape), L)
    n = len(pos)
    if args.kxyz:
        k = np.array(args.kxyz, dtype=np.int64)
    elif args.k:
        k = np.full(3, args.k, dtype=np.int64)
    else:
        k = cells.choose_k(L, n, h, args.cluster_atoms, cfg["cutoff"])
    c, order, starts = cells.assign_cubes(pos, L, k)
    cubes = list(itertools.product(*[range(int(x)) for x in k]))
    mine = cubes[args.rank::args.world][:args.max_chunks or None]
    payload = ffm.header(args, cfg, device)
    payload.update(
        natoms=n, structure=os.path.abspath(args.structure),
        structure_meta=meta, box_A=L.tolist(), k=k.tolist(),
        core_A=c.tolist(), halo_A=h, halo_exact=h >= ffm.reach(cfg),
        forces=args.forces, ckpt=args.ckpt, rattle=args.rattle,
        rank=args.rank, world=args.world, n_cubes=len(cubes),
        cubes_done=0, owned_done=0, energy=0.0, pieces=[])
    ffm.log(f"N={n} box {np.round(L, 2).tolist()} A, {k.tolist()} cubes of "
            f"{np.round(c, 2).tolist()} A, halo {h:.2f} A, {len(mine)} cubes on "
            f"rank {args.rank}/{args.world}")
    forces = torch.zeros((n, 3), dtype=torch.float64) if args.forces else None
    en_idx, en_val = [], []
    t_start = time.perf_counter()
    for cube in mine:
        gidx, cpos, owned = cells.gather_piece(cube, pos, L, k, c, order,
                                               starts, h)
        if not owned.any():  # e.g. a cube in the vacuum above a slab
            payload["cubes_done"] += 1
            payload["pieces"].append({
                "cube": [int(x) for x in cube], "atoms": len(cpos),
                "owned": 0, "edges": 0, "lg_edges": 0, "max_degree": 0,
                "skipped": True})
            continue
        # non-periodic piece in a box with more than one cutoff of vacuum
        cpos = cpos - cpos.min(0) + 1.0
        box = np.diag(cpos.max(0) + cfg["cutoff"] + 2.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        g = ffm.build_graph(cpos, box, np.asarray(z[gidx]), cfg, device)
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
            forces.index_add_(0, torch.from_numpy(gidx),
                              -grad.detach().double().cpu())
            del grad
            if args.save_energies:
                with torch.no_grad():
                    en_val.append(owned_energies(e_atom, e_bond, g["src"],
                                                 own_t))
        else:
            with torch.inference_mode():
                e_atom, e_bond = ffm.atom_energies(model, p_t, lat, g)
                e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
                if args.save_energies:  # inference tensors stay inside
                    en_val.append(owned_energies(e_atom, e_bond, g["src"],
                                                 own_t))
        if args.save_energies:
            en_idx.append(gidx[owned])
        e_own = float(e_own)
        del e_atom, e_bond
        ffm.sync(device)
        t2 = time.perf_counter()
        mem = ffm.gpu_peak_mem(device)
        stats = graph_stats(model, p_t.detach(), lat, g)
        del g, p_t
        payload["energy"] += e_own
        payload["cubes_done"] += 1
        payload["owned_done"] += int(owned.sum())
        piece = {"cube": [int(x) for x in cube], "atoms": len(cpos),
                 "owned": int(owned.sum()), "t_graph_s": t1 - t0,
                 "t_eval_s": t2 - t1, **stats, **mem}
        payload["pieces"].append(piece)
        ffm.log(f"cube {cube}: {len(cpos)} atoms ({piece['owned']} own)"
                f"  {stats['lg_edges']} angles  graph {t1 - t0:.2f} s"
                f"  eval {t2 - t1:.2f} s  peak {mem.get('mem_alloc_gb', 0):.1f} GB")
    wall = time.perf_counter() - t_start
    payload.update(wall_s=wall, host_peak_gb=ffm.host_peak_gb(),
                   owned_atoms_per_s=payload["owned_done"] / max(wall, 1e-9))
    # the halo is exact only if no atom loses bonds to the neighbor cap:
    # with the cap binding, whether bond j-i exists depends on j's other
    # neighbors, which reach beyond ff_max_atoms.reach
    payload["neighbor_cap_binds"] = any(
        q["max_degree"] >= (cfg.get("max_neighbors") or 10**9)
        for q in payload["pieces"])
    if args.world == 1 and payload["cubes_done"] == len(cubes):
        payload["e_per_atom"] = payload["energy"] / n
        if args.forces:
            payload.update(ffm.force_stats(forces.numpy()))
    if args.forces and args.world > 1:
        np.save(f"{args.out}_forces.npy", forces.numpy().astype(
            np.float64 if ffm.DTYPE == torch.float64 else np.float32))
    if args.save_energies:
        np.savez(f"{args.out}_energies.npz",
                 idx=np.concatenate(en_idx) if en_idx else np.zeros(0, int),
                 e=np.concatenate(en_val) if en_val else np.zeros(0, np.float32))
    ffm.dump(payload, args.out)
    if args.validate:  # pieces are saved first in case this fails
        payload["validation"] = ffm.validate(
            model, cfg, np.asarray(pos), np.asarray(z), L, forces, payload,
            device)
        ffm.dump(payload, args.out)
    ffm.log({k_: v for k_, v in payload.items()
             if k_ not in ("pieces", "structure_meta")})


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["sweep", "chunk", "reduce"])
    p.add_argument("--model_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--kind", choices=["Si", "SiGe", "Al", "Cu", "water"])
    p.add_argument("--path", choices=["ef", "ckpt", "e"])
    p.add_argument("--max_i", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--structure", help="cells.py output prefix")
    p.add_argument("--cluster_atoms", type=int, default=50000,
                   help="target atoms per piece (cube + halo)")
    p.add_argument("--k", type=int, default=0,
                   help="cubes per side (overrides --cluster_atoms)")
    p.add_argument("--kxyz", type=int, nargs=3, default=None,
                   help="cubes per axis (overrides --k)")
    p.add_argument("--halo", type=float, default=0,
                   help="halo in A (default: the model's reach, exact)")
    p.add_argument("--halo_margin", type=float, default=0.01)
    p.add_argument("--forces", action="store_true")
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--max_chunks", type=int, default=0,
                   help="stop after this many cubes (throughput sample)")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--float64", action="store_true",
                   help="double precision (exactness tests)")
    p.add_argument("--save_energies", action="store_true",
                   help="write each owned atom's energy to OUT_energies.npz "
                   "(idx, e) for energy maps")
    p.add_argument("--max_neighbors", type=int, default=0,
                   help="override the model's neighbor cap (e.g. 10000 to "
                   "lift it, to test its effect on the halo)")
    args = p.parse_args()
    if args.float64:
        ffm.DTYPE = torch.float64
        torch.set_default_dtype(torch.float64)
    if args.mode == "reduce":
        return ffm.reduce(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.mode == "chunk" and args.world > 1:
        args.out = f"{args.out}_rank{args.rank:04d}"
    ffm.log("device", device, "| neighbor list",
            "matscipy" if ffm._matscipy_nl else "ase",
            "| alignn", os.path.dirname(ffm.alignn.__file__))
    (sweep if args.mode == "sweep" else chunk)(args, device)


if __name__ == "__main__":
    main()
