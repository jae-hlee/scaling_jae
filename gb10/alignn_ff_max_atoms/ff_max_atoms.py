"""Largest Si cell an ALIGNN-FF evaluates on one GPU, with and without
forces, and exact chunked evaluation of larger cells.

sweep   Si diamond, 8 i^3 atoms, i stepped up until the GPU runs out of
        memory, one evaluation path per process:
          ase   E+F+S through AlignnAtomwiseCalculator (the paper's table)
          ef    E+F+S through model.forward_tensors (the LAMMPS path)
          ckpt  ef, with each layer's activations recomputed during the
                backward pass instead of kept from the forward pass
          e     energy only, no autograd graph
chunk   one cell cut into k^3 cubes. Each cube is evaluated together with
        every atom within the model's reach (its halo, or a shorter
        --halo, which is no longer exact). The energies of
        the cube's own atoms are summed and their gradient is added to
        the forces of every atom in the piece, so the summed result is the
        full-cell result. --rank/--world split the cubes over Slurm array
        tasks, and reduce adds the parts up.
reduce  sums the parts written by chunk --world W.

    python ff_max_atoms.py sweep --model_dir M --path ef --out OUT
    python ff_max_atoms.py chunk --model_dir M --rep 10 --forces --validate
    python ff_max_atoms.py chunk --model_dir M --rep 200 --rank 0 --world 8
    python ff_max_atoms.py reduce --out OUT
"""
import argparse
import glob
import itertools
import json
import math
import os
import resource
import statistics
import time

import torch  # before ase (macOS)
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.data import chemical_symbols
from jarvis.core.specie import get_node_attributes
from torch.utils.checkpoint import checkpoint

import alignn
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
    cutoff_function_based_edges,
    zbl_edge_energies,
)
from alignn.torch_graph_builder import _topk_per_source

A_SI = 5.43
# conventional diamond cell, fractional
BASIS = np.array([
    [0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])

DTYPE = torch.float32  # --float64 for exactness tests

try:
    from matscipy.neighbours import neighbour_list as _matscipy_nl
except ImportError:
    _matscipy_nl = None


def log(*a):
    """Print with flush."""
    print(*a, flush=True)


def sync(device):
    """Wait for the GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def host_peak_gb():
    """Peak resident memory of this process (Linux reports kB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if os.uname().sysname == "Linux" else peak / 1e9


def is_oom(err):
    """True for out-of-memory errors however torch raises them."""
    text = str(err).lower()
    return isinstance(err, (torch.cuda.OutOfMemoryError, MemoryError)) or (
        isinstance(err, RuntimeError)
        and ("out of memory" in text or "alloc_failed" in text))


def load_model(model_dir, device, ckpt=False):
    """Pure-torch ALIGNN-FF from config.json + best_model.pt."""
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**cfg["model"]))
    sd = torch.load(os.path.join(model_dir, "best_model.pt"),
                    map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "_species_table"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, "
                           f"unexpected {unexpected}")
    if not (cfg["model"].get("energy_mult_natoms", True)
            and isinstance(model.fc, torch.nn.Linear)
            and model.fc.out_features == 1):
        raise RuntimeError("energy is not a sum of atom energies")
    model.to(device=device, dtype=DTYPE).eval()
    if ckpt:
        for layer in list(model.alignn_layers) + list(model.gcn_layers):
            fwd = layer.forward_tensors
            layer.forward_tensors = (
                lambda *a, _f=fwd: checkpoint(_f, *a, use_reentrant=False)
                if torch.is_grad_enabled() else _f(*a))
    return model, cfg


def reach(cfg):
    """Farthest distance (A) at which one atom can change another atom's
    energy. An ALIGNN layer updates atoms and bonds over the bond graph
    (one cutoff) and then bonds over the line graph (one three-body
    cutoff), a GCN layer updates atoms and bonds over the bond graph.
    dx: reach of atom features, dy: reach of bond features, measured from
    either end of the bond."""
    m = cfg["model"]
    rc = cfg["cutoff"]
    tb = max(cfg.get("three_body_cutoff") or rc,
             m.get("three_body_cutoff", 3.5))
    dx = dy = 0.0
    for _ in range(m["alignn_layers"]):
        dx, dy = max(dx, dy) + rc, max(dx, dy)
        dy += tb
    for _ in range(m["gcn_layers"]):
        dx, dy = max(dx, dy) + rc, max(dx, dy)
    return dx


_FEATS = {}


def build_graph(pos, cell, numbers, cfg, device, pbc=True):
    """Bond graph of one cell: src, dst, integer image shifts, atom
    features and atomic numbers, as forward_tensors takes them."""
    rc = float(cfg["cutoff"])
    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)
    if _matscipy_nl is not None:  # same call form as alignn's builder
        i, j, S = _matscipy_nl("ijS", atoms, rc)
    else:
        from ase.neighborlist import neighbor_list
        i, j, S = neighbor_list("ijS", atoms, rc)
    del atoms
    src = torch.from_numpy(np.ascontiguousarray(i)).long()
    dst = torch.from_numpy(np.ascontiguousarray(j)).long()
    shift = torch.from_numpy(np.ascontiguousarray(S)).double()
    k = cfg.get("max_neighbors")
    if k and src.numel():
        if torch.bincount(src, minlength=len(pos)).max() > k:
            p = torch.from_numpy(pos)
            d = (p[dst] - p[src] + shift @ torch.from_numpy(cell)).norm(dim=1)
            keep = _topk_per_source(src, d, int(k), len(pos))
            src, dst, shift = src[keep], dst[keep], shift[keep]
    zs, inv = np.unique(numbers, return_inverse=True)
    key = cfg["atom_features"]
    table = []
    for z in zs:
        if (key, z) not in _FEATS:
            _FEATS[key, z] = get_node_attributes(
                chemical_symbols[int(z)], atom_features=key)
        table.append(_FEATS[key, z])
    table = torch.tensor(np.array(table), dtype=DTYPE)
    return {
        "src": src.to(device), "dst": dst.to(device),
        "shift": shift.to(device=device, dtype=DTYPE),
        "feats": table[torch.from_numpy(inv)].to(device),
        "Z": torch.from_numpy(np.asarray(numbers)).long().to(device),
        "max_shift": int(np.abs(S).max()) if len(S) else 0,
    }


def atom_energies(model, pos, lat, g):
    """Energy of each atom and of each bond (penalty and ZBL terms), with
    the same arithmetic as forward_tensors: their sum is its energy."""
    src, dst, shift = g["src"], g["dst"], g["shift"]
    n = pos.shape[0]
    r = pos.index_select(0, dst) - pos.index_select(0, src) + shift @ lat
    bl = torch.linalg.vector_norm(r, dim=1)

    # line graph exactly as forward_tensors builds it
    E = src.shape[0]
    allowed_ids = torch.nonzero(bl <= model.three_body_cutoff).squeeze(-1)
    sub_src = src.index_select(0, allowed_ids)
    order = torch.argsort(sub_src, stable=True)
    sorted_edge_ids = allowed_ids.index_select(0, order)
    sorted_src = sub_src.index_select(0, order)
    node_range = torch.arange(n, device=src.device)
    bucket_start = torch.searchsorted(sorted_src, node_range)
    bucket_end = torch.searchsorted(sorted_src, node_range, right=True)
    A_v = dst.index_select(0, allowed_ids)
    starts = bucket_start.index_select(0, A_v)
    counts = bucket_end.index_select(0, A_v) - starts
    total = int(counts.sum().item())
    lg_src = torch.repeat_interleave(allowed_ids, counts)
    cum = torch.cumsum(counts, dim=0)
    offsets = torch.arange(total, device=src.device) - torch.repeat_interleave(
        cum - counts, counts)
    pos_idx = torch.repeat_interleave(starts, counts) + offsets
    lg_dst = sorted_edge_ids.index_select(0, pos_idx)
    r_ij = r.index_select(0, lg_src)
    r_jk = r.index_select(0, lg_dst)
    h_cos = (-(r_ij * r_jk).sum(dim=-1) / (
        torch.linalg.vector_norm(r_ij, dim=-1).clamp_min(1e-12)
        * torch.linalg.vector_norm(r_jk, dim=-1).clamp_min(1e-12))
    ).clamp(-1.0, 1.0)

    x = model.atom_embedding(g["feats"])
    if model.use_element_embedding:
        x = x + model.element_embedding(g["Z"])
    if model.use_cutoff_function:
        c_off = cutoff_function_based_edges(
            bl, inner_cutoff=model.inner_cutoff, exponent=model.exponent)
        if model.multiply_cutoff:
            y = model.edge_embedding(bl) * c_off.unsqueeze(-1)
        else:
            y = model.edge_embedding(c_off)
    else:
        y = model.edge_embedding(bl)
    z = model.angle_embedding(h_cos)
    for layer in model.alignn_layers:
        x, y, z = layer.forward_tensors(
            src, dst, n, lg_src, lg_dst, E, x, y, z)
    for layer in model.gcn_layers:
        x, y = layer.forward_tensors(src, dst, n, x, y)
    # N * fc(mean x) = sum of fc(x_i), since fc is linear
    e_atom = model.fc(x).squeeze(-1)
    e_bond = torch.zeros_like(bl)
    if model.use_penalty:
        e_bond = e_bond + torch.where(
            bl < model.penalty_threshold,
            model.penalty_factor * (model.penalty_threshold - bl),
            torch.zeros_like(bl))
    if model.use_zbl:
        zf = g["Z"].to(bl.dtype)
        e_bond = e_bond + zbl_edge_energies(
            zf.index_select(0, src), zf.index_select(0, dst), bl,
            model.zbl_cutoff)
    return e_atom, e_bond


def si_cell(i):
    """ase Si diamond supercell of the paper's protocol (8 i^3 atoms)."""
    return bulk("Si", "diamond", a=A_SI, cubic=True).repeat((i, i, i))


# ------------------------------------------------------------------ sweep


def evaluate(path, atoms, model, cfg, calc, device):
    """One evaluation, with its timings and results."""
    out = {}
    if path == "ase":
        atoms.calc = calc
        calc.reset()  # time a real evaluation, not cached results
        sync(device)
        t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        atoms.get_stress()
        sync(device)
        out["t_total_s"] = time.perf_counter() - t0
        out.update(e=float(e), fmax=float(np.abs(f).max()))
        return out
    t0 = time.perf_counter()
    cell = np.array(atoms.cell)
    g = build_graph(atoms.positions, cell, atoms.numbers, cfg, device)
    pos = torch.tensor(atoms.positions, dtype=DTYPE, device=device)
    lat = torch.tensor(cell, dtype=DTYPE, device=device)
    sync(device)
    t1 = time.perf_counter()
    if path == "e":
        with torch.inference_mode():
            e_atom, e_bond = atom_energies(model, pos, lat, g)
            e = float(e_atom.sum() + e_bond.sum())
        fmax = None
    else:
        pos.requires_grad_(True)
        res = model.forward_tensors(pos, lat, g["feats"], g["src"], g["dst"],
                                    g["shift"], True, g["Z"])
        e = float(res["energy"])
        fmax = float(res["forces"].abs().max())
        del res
    sync(device)
    t2 = time.perf_counter()
    out.update(t_graph_s=t1 - t0, t_eval_s=t2 - t1, t_total_s=t2 - t0,
               e=e, fmax=fmax, edges=int(g["src"].numel()))
    return out


def sweep(args, device):
    """Step the Si cell up until out of memory, one path."""
    model, cfg = load_model(args.model_dir, device, ckpt=args.path == "ckpt")
    calc = None
    if args.path == "ase":
        from alignn.ff.calculators import AlignnAtomwiseCalculator
        calc = AlignnAtomwiseCalculator(path=args.model_dir,
                                        device=str(device))
    payload = header(args, cfg, device)
    payload.update(path=args.path, results=[], oom=None, error=None,
                   structure="Si diamond conventional cell (8 atoms), i^3")
    for _ in range(2):  # warm-up
        evaluate(args.path, si_cell(2), model, cfg, calc, device)
    sizes = [i for i in list(range(2, 15, 2)) + list(range(15, 1000))
             if i <= args.max_i]
    for i in sizes:
        atoms = si_cell(i)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        reps = []
        try:
            while len(reps) < args.repeats:
                reps.append(evaluate(args.path, atoms, model, cfg, calc,
                                     device))
                if reps[-1]["t_total_s"] > 30:  # big cells: one call
                    break
        except Exception as err:  # noqa: BLE001  (record why it stopped)
            kind = "oom" if is_oom(err) else "error"
            payload[kind] = {"i": i, "natoms": len(atoms),
                             "error": f"{type(err).__name__}: {err}"[:300]}
            log(f"[{kind.upper()}] i={i} N={len(atoms)}: {str(err)[:150]}")
            dump(payload, args.out)
            break
        row = {"i": i, "natoms": len(atoms)}
        for key in reps[0]:
            vals = [r[key] for r in reps]
            row[key] = (statistics.median(vals) if key.startswith("t_")
                        else vals[0])
        row["t_all_s"] = [r["t_total_s"] for r in reps]
        row["e_per_atom"] = row.pop("e") / len(atoms)
        row.update(gpu_peak_mem(device), host_peak_gb=host_peak_gb())
        if row.get("mem_alloc_gb"):
            row["mb_per_atom"] = row["mem_alloc_gb"] * 1e3 / len(atoms)
        payload["results"].append(row)
        log(f"i={i:3d} N={len(atoms):9d}  t={row['t_total_s']:8.3f} s"
            f"  alloc {row.get('mem_alloc_gb', 0):6.1f} GB"
            f"  ({row.get('mb_per_atom', 0):.3f} MB/atom)"
            f"  E/atom {row['e_per_atom']:.5f}")
        dump(payload, args.out)
    log("done ->", args.out + ".json")


# ------------------------------------------------------------------ chunk


def species(name, rep):
    """Atomic numbers in si_positions order: Si diamond, or SiGe
    zincblende on the same sites (a test case where neighbors differ)."""
    basis = {"Si": [14] * 8, "SiGe": [14] * 4 + [32] * 4}[name]
    return np.tile(np.array(basis, dtype=np.uint8), rep**3)


def si_positions(rep, rattle, seed):
    """Si diamond rep^3 supercell positions (float64), wrapped into the box."""
    grid = np.stack(np.meshgrid(*[np.arange(rep)] * 3, indexing="ij"), -1)
    pos = ((grid.reshape(-1, 1, 3) + BASIS[None]) * A_SI).reshape(-1, 3)
    if rattle:
        pos += np.random.default_rng(seed).normal(0, rattle, pos.shape)
    return np.mod(pos, rep * A_SI)


def core_side(cluster_volume, h):
    """Side of the cube whose halo-padded volume (a box rounded by h) is
    cluster_volume."""
    lo, hi = 0.0, cluster_volume ** (1 / 3)
    for _ in range(60):
        c = (lo + hi) / 2
        v = (c**3 + 6 * c * c * h + 3 * math.pi * c * h * h
             + 4 / 3 * math.pi * h**3)
        lo, hi = (c, hi) if v < cluster_volume else (lo, c)
    return lo


def chunk(args, device):
    """Evaluate one cell piece by piece (this rank's pieces)."""
    model, cfg = load_model(args.model_dir, device, ckpt=args.ckpt)
    h = args.halo or reach(cfg) + args.halo_margin
    if h < cfg["cutoff"]:
        raise SystemExit(f"halo {h} A is shorter than the cutoff")
    rep, L = args.rep, args.rep * A_SI
    pos = si_positions(rep, args.rattle, args.seed)
    numbers = species(args.species, rep)
    n = len(pos)
    if args.k:
        k = args.k
    else:
        c_max = core_side(args.cluster_atoms * L**3 / n, h)
        if c_max < cfg["cutoff"]:
            raise SystemExit(
                f"--cluster_atoms {args.cluster_atoms} leaves a core under "
                f"{cfg['cutoff']} A inside the {h:.1f} A halo, use more")
        k = max(1, math.ceil(L / c_max))
    c = L / k
    b3 = np.minimum((pos // c).astype(np.int64), k - 1)
    bid = (b3[:, 0] * k + b3[:, 1]) * k + b3[:, 2]
    del b3
    order = np.argsort(bid, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(bid, minlength=k**3))])
    del bid
    m = math.ceil(h / c)
    cubes = list(itertools.product(range(k), repeat=3))
    mine = cubes[args.rank::args.world][:args.max_chunks or None]
    payload = header(args, cfg, device)
    payload.update(
        natoms=n, rep=rep, box_A=L, k=k, core_A=c, halo_A=h,
        halo_exact=h >= reach(cfg),
        forces=args.forces, ckpt=args.ckpt, rattle=args.rattle,
        species=args.species,
        rank=args.rank, world=args.world, n_cubes=k**3,
        cubes_done=0, owned_done=0, energy=0.0, pieces=[])
    log(f"N={n} box {L:.1f} A, {k}^3 cubes of {c:.2f} A, halo {h:.2f} A,"
        f" {len(mine)} cubes on rank {args.rank}/{args.world}")
    forces = torch.zeros((n, 3), dtype=torch.float64) if args.forces else None
    t_start = time.perf_counter()
    for cx, cy, cz in mine:
        lo = np.array([cx, cy, cz]) * c
        hi = lo + c
        idx_parts, pos_parts, own_parts = [], [], []
        for d in itertools.product(range(-m, m + 1), repeat=3):
            nb = np.array([cx, cy, cz]) + d
            img = np.floor_divide(nb, k)
            nb = nb - img * k
            b = (nb[0] * k + nb[1]) * k + nb[2]
            idx = order[starts[b]:starts[b + 1]]
            p = pos[idx] + img * L
            if d != (0, 0, 0):
                gap = np.maximum(np.maximum(lo - p, p - hi), 0)
                keep = (gap * gap).sum(1) <= h * h
                idx, p = idx[keep], p[keep]
            idx_parts.append(idx)
            pos_parts.append(p)
            own_parts.append(np.full(len(idx), d == (0, 0, 0)))
        gidx = np.concatenate(idx_parts)
        cpos = np.concatenate(pos_parts)
        owned = np.concatenate(own_parts)
        # non-periodic piece in a box with more than one cutoff of vacuum
        cpos = cpos - cpos.min(0) + 1.0
        box = np.diag(cpos.max(0) + cfg["cutoff"] + 2.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        g = build_graph(cpos, box, numbers[gidx], cfg, device)
        assert g["max_shift"] == 0, "piece touches its own image"
        p_t = torch.tensor(cpos, dtype=DTYPE, device=device)
        lat = torch.tensor(box, dtype=DTYPE, device=device)
        own_t = torch.from_numpy(owned).to(device)
        sync(device)
        t1 = time.perf_counter()
        if args.forces:
            p_t.requires_grad_(True)
            e_atom, e_bond = atom_energies(model, p_t, lat, g)
            e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
            (grad,) = torch.autograd.grad(e_own, p_t)
            forces.index_add_(0, torch.from_numpy(gidx),
                              -grad.detach().double().cpu())
            del grad
        else:
            with torch.inference_mode():
                e_atom, e_bond = atom_energies(model, p_t, lat, g)
                e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
        e_own = float(e_own)
        del e_atom, e_bond, g, p_t
        sync(device)
        t2 = time.perf_counter()
        payload["energy"] += e_own
        payload["cubes_done"] += 1
        payload["owned_done"] += int(owned.sum())
        piece = {"cube": [cx, cy, cz], "atoms": len(cpos),
                 "owned": int(owned.sum()), "t_graph_s": t1 - t0,
                 "t_eval_s": t2 - t1, **gpu_peak_mem(device)}
        payload["pieces"].append(piece)
        log(f"cube {cx},{cy},{cz}: {len(cpos)} atoms ({piece['owned']} own)"
            f"  graph {t1 - t0:.2f} s  eval {t2 - t1:.2f} s"
            f"  peak {piece.get('mem_alloc_gb', 0):.1f} GB")
    wall = time.perf_counter() - t_start
    payload.update(wall_s=wall, host_peak_gb=host_peak_gb(),
                   owned_atoms_per_s=payload["owned_done"] / max(wall, 1e-9))
    if args.world == 1 and payload["cubes_done"] == k**3:
        payload["e_per_atom"] = payload["energy"] / n
        if args.forces:
            payload.update(force_stats(forces.numpy()))
    if args.forces and args.world > 1:
        np.save(f"{args.out}_forces.npy", forces.numpy().astype(
            np.float64 if DTYPE == torch.float64 else np.float32))
    dump(payload, args.out)
    if args.validate:  # pieces are saved first in case this fails
        payload["validation"] = validate(model, cfg, pos, numbers, L, forces,
                                         payload, device)
        dump(payload, args.out)
    log(json.dumps({k_: v for k_, v in payload.items() if k_ != "pieces"}))


def validate(model, cfg, pos, numbers, L, forces, payload, device):
    """Full-cell evaluation of the same cell: energy (forward_tensors and
    this script's sum of atom energies) and forces, against the pieces."""
    cell = np.eye(3) * L
    g = build_graph(pos, cell, numbers, cfg, device)
    p_t = torch.tensor(pos, dtype=DTYPE, device=device,
                       requires_grad=True)
    lat = torch.tensor(cell, dtype=DTYPE, device=device)
    res = model.forward_tensors(p_t, lat, g["feats"], g["src"], g["dst"],
                                g["shift"], False, g["Z"])
    with torch.inference_mode():
        e_atom, e_bond = atom_energies(model, p_t.detach(), lat, g)
        e_sum = float(e_atom.sum() + e_bond.sum())
    n = len(pos)
    out = {"e_full_per_atom": float(res["energy"]) / n,
           "e_atom_sum_per_atom": e_sum / n,
           "e_pieces_per_atom": payload["energy"] / n}
    out["max_abs_de_per_atom"] = abs(out["e_pieces_per_atom"]
                                     - out["e_full_per_atom"])
    if forces is not None:
        f_full = res["forces"].detach().double().cpu().numpy()
        out["max_abs_f_full"] = float(np.abs(f_full).max())
        out["max_abs_df"] = float(np.abs(forces.numpy() - f_full).max())
    log("validation:", json.dumps(out))
    return out


def force_stats(f):
    """Largest force and the net force (zero for exact forces)."""
    return {"max_abs_f": float(np.abs(f).max()),
            "net_force": np.abs(f.sum(0)).tolist()}


def reduce(args):
    """Sum the parts of one chunked run."""
    parts = sorted(glob.glob(f"{args.out}_rank*.json"))
    ps = [json.load(open(p)) for p in parts]
    n, world = ps[0]["natoms"], ps[0]["world"]
    assert len(ps) == world, f"{len(ps)} of {world} parts"
    assert sum(p["cubes_done"] for p in ps) == ps[0]["n_cubes"]
    assert sum(p["owned_done"] for p in ps) == n
    out = {k: ps[0][k] for k in ("model_dir", "natoms", "box_A", "k",
                                 "core_A", "halo_A", "halo_exact", "forces",
                                 "gpu")}
    out["e_per_atom"] = sum(p["energy"] for p in ps) / n
    out["wall_s_slowest_rank"] = max(p["wall_s"] for p in ps)
    out["piece_atoms_max"] = max(q["atoms"] for p in ps for q in p["pieces"])
    out["mem_alloc_gb_max"] = max(q.get("mem_alloc_gb") or 0
                                  for p in ps for q in p["pieces"])
    if ps[0]["forces"]:
        f = sum(np.load(p.replace(".json", "_forces.npy")).astype(np.float64)
                for p in parts)
        out.update(force_stats(f))
    json.dump(out, open(f"{args.out}_reduced.json", "w"), indent=1)
    log(json.dumps(out))


# ------------------------------------------------------------------ common


def gpu_peak_mem(device):
    """Peak allocated and reserved GPU memory since the last reset."""
    if device.type != "cuda":
        return {}
    return {"mem_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            "mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}


def header(args, cfg, device):
    """What ran where."""
    import hashlib
    ckpt = os.path.join(args.model_dir, "best_model.pt")
    m = cfg["model"]
    return {
        "model_dir": os.path.abspath(args.model_dir),
        "checkpoint_md5": hashlib.md5(open(ckpt, "rb").read()).hexdigest(),
        "layers": [m["alignn_layers"], m["gcn_layers"]],
        "hidden": m["hidden_features"],
        "element_embedding": bool(m.get("element_embedding")),
        "cutoff": cfg["cutoff"], "max_neighbors": cfg["max_neighbors"],
        "three_body_cutoff": cfg.get("three_body_cutoff"),
        "reach_A": reach(cfg),
        "gpu": (torch.cuda.get_device_name(0) if device.type == "cuda"
                else "cpu"),
        "gpu_memory_gb": (torch.cuda.get_device_properties(0).total_memory
                          / 1e9 if device.type == "cuda" else None),
        "neighbor_list": "matscipy" if _matscipy_nl else "ase",
        "torch": torch.__version__,
        "alignn": os.path.dirname(alignn.__file__),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
    }


def dump(payload, out):
    """Rewrite the result file (atomic)."""
    tmp = out + ".json.tmp"
    json.dump(payload, open(tmp, "w"), indent=1)
    os.replace(tmp, out + ".json")


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["sweep", "chunk", "reduce"])
    p.add_argument("--model_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--path", choices=["ase", "ef", "ckpt", "e"])
    p.add_argument("--max_i", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--cluster_atoms", type=int, default=50000,
                   help="target atoms per piece (cube + halo)")
    p.add_argument("--k", type=int, default=0,
                   help="cubes per side (overrides --cluster_atoms)")
    p.add_argument("--halo", type=float, default=0,
                   help="halo in A (default: the model's reach, exact)")
    p.add_argument("--halo_margin", type=float, default=0.01)
    p.add_argument("--forces", action="store_true")
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--species", choices=["Si", "SiGe"], default="Si")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--max_chunks", type=int, default=0,
                   help="stop after this many cubes (timing sample)")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--float64", action="store_true",
                   help="double precision (exactness tests)")
    args = p.parse_args()
    if args.float64:
        global DTYPE
        DTYPE = torch.float64
        torch.set_default_dtype(torch.float64)
    if args.mode == "reduce":
        return reduce(args)
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    if args.mode == "chunk" and args.world > 1:
        args.out = f"{args.out}_rank{args.rank:04d}"
    log("device", device, "| neighbor list",
        "matscipy" if _matscipy_nl else "ase",
        "| alignn", os.path.dirname(alignn.__file__))
    (sweep if args.mode == "sweep" else chunk)(args, device)


if __name__ == "__main__":
    main()
