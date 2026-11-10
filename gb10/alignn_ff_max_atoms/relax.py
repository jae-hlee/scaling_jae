"""FIRE relaxation at fixed cell of a cells.py structure with ALIGNN-FF on one
GPU: as one graph (--single, for cells that fit) or with the exact
cube-and-halo decomposition of ff_any.py (default). Positions, velocities
and the FIRE state are saved every --save_every steps and a rerun with the
same --out resumes from them, so a relaxation longer than a Slurm time limit
can simply be resubmitted.

FIRE follows ASE's implementation (dt 0.1, dtmax 1.0, Nmin 5, finc 1.1,
fdec 0.5, alpha 0.1, falpha 0.99, unit masses), except that each atom's step
is capped at --maxmove (ASE caps the norm over all atoms, which for 10^5 or
more atoms makes steps vanishingly small).

    python relax.py --model_dir M --structure S/poly_si_150k --single --out R/relax_poly_si_150k
    python relax.py --model_dir M --structure S/poly_si_1m --cluster_atoms 110000 --out R/relax_poly_si_1m

Writes OUT.json (energy, largest force and time per step), OUT_state.npz,
and at the end OUT_relaxed.{pos.npy,Z.npy,json} (a cells.py structure) and
OUT_energies.npy (float32 energy of every atom, bond terms on their first
atom, for energy maps).
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

import cells
import ff_max_atoms as ffm

FIRE = dict(nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99)


def _piece_tensors(cpos, z, cfg, device):
    """Graph and tensors of one non-periodic piece in a vacuum box."""
    cpos = cpos - cpos.min(0) + 1.0
    box = np.diag(cpos.max(0) + cfg["cutoff"] + 2.0)
    g = ffm.build_graph(cpos, box, z, cfg, device)
    assert g["max_shift"] == 0, "piece touches its own image"
    lat = torch.tensor(box, dtype=ffm.DTYPE, device=device)
    return g, torch.tensor(cpos, dtype=ffm.DTYPE, device=device), lat


def energy_forces_single(model, cfg, pos, z, L, device):
    """Energy and forces of the periodic cell as one graph."""
    cell = np.diag(L)
    g = ffm.build_graph(pos, cell, z, cfg, device)
    p = torch.tensor(pos, dtype=ffm.DTYPE, device=device, requires_grad=True)
    lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
    e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
    e = e_atom.sum() + e_bond.sum()
    (grad,) = torch.autograd.grad(e, p)
    return float(e), -grad.detach().double().cpu().numpy()


def energy_forces_chunked(model, cfg, pos, z, L, k, h, device):
    """Energy and forces summed over cube-and-halo pieces (exact while the
    halo covers the receptive field and the neighbor cap does not bind)."""
    c, order, starts = cells.assign_cubes(pos, L, k)
    forces = torch.zeros((len(pos), 3), dtype=torch.float64)
    energy = 0.0
    for cube in itertools.product(*[range(int(x)) for x in k]):
        gidx, cpos, owned = cells.gather_piece(cube, pos, L, k, c, order,
                                               starts, h)
        if not owned.any():
            continue
        g, p_t, lat = _piece_tensors(cpos, np.asarray(z[gidx]), cfg, device)
        p_t.requires_grad_(True)
        own_t = torch.from_numpy(owned).to(device)
        e_atom, e_bond = ffm.atom_energies(model, p_t, lat, g)
        e_own = e_atom[own_t].sum() + e_bond[own_t[g["src"]]].sum()
        (grad,) = torch.autograd.grad(e_own, p_t)
        forces.index_add_(0, torch.from_numpy(gidx),
                          -grad.detach().double().cpu())
        energy += float(e_own)
        del g, p_t, e_atom, e_bond, e_own, grad
    return energy, forces.numpy()


def atom_energy_map(model, cfg, pos, z, L, device, single, k, h):
    """Energy of every atom (bond terms counted on their first atom)."""
    out = np.zeros(len(pos), dtype=np.float32)
    with torch.no_grad():
        if single:
            cell = np.diag(L)
            g = ffm.build_graph(pos, cell, z, cfg, device)
            p = torch.tensor(pos, dtype=ffm.DTYPE, device=device)
            lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
            e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
            e_atom.index_add_(0, g["src"], e_bond)
            return e_atom.float().cpu().numpy()
        c, order, starts = cells.assign_cubes(pos, L, k)
        for cube in itertools.product(*[range(int(x)) for x in k]):
            gidx, cpos, owned = cells.gather_piece(cube, pos, L, k, c, order,
                                                   starts, h)
            if not owned.any():
                continue
            g, p_t, lat = _piece_tensors(cpos, np.asarray(z[gidx]), cfg,
                                         device)
            e_atom, e_bond = ffm.atom_energies(model, p_t, lat, g)
            e_atom.index_add_(0, g["src"], e_bond)
            out[gidx[owned]] = e_atom[torch.from_numpy(owned).to(device)
                                      ].float().cpu().numpy()
    return out


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--structure", required=True, help="cells.py prefix")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--single", action="store_true",
                   help="one graph for the whole cell (must fit in memory)")
    p.add_argument("--cluster_atoms", type=int, default=100000,
                   help="target atoms per piece (chunked)")
    p.add_argument("--halo", type=float, default=0,
                   help="halo in A (default: the model's reach, exact)")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="stop when every atom's force is below this, eV/A")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--dtmax", type=float, default=1.0)
    p.add_argument("--maxmove", type=float, default=0.1,
                   help="largest step of any atom, A")
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = ffm.load_model(args.model_dir, device, ckpt=True)
    pos0, z, L, meta = cells.read_structure(args.structure)
    z = np.asarray(z)
    h = args.halo or ffm.reach(cfg) + 0.01
    k = None if args.single else cells.choose_k(
        L, len(pos0), h, args.cluster_atoms, cfg["cutoff"])
    state_file = args.out + "_state.npz"
    if os.path.exists(state_file):
        s = np.load(state_file)
        pos, v = s["pos"], s["v"]
        dt, a, nsteps, step = float(s["dt"]), float(s["a"]), int(s["nsteps"]), int(s["step"])
        log = json.load(open(args.out + ".json"))["steps"]
        ffm.log(f"resuming at step {step}")
    else:
        pos = np.array(pos0, dtype=np.float64)
        v = np.zeros_like(pos)
        dt, a, nsteps, step, log = args.dt, FIRE["astart"], 0, 0, []
    payload = ffm.header(args, cfg, device)
    payload.update(structure=os.path.abspath(args.structure),
                   structure_meta=meta, natoms=len(pos), box_A=L.tolist(),
                   single=args.single, k=None if k is None else k.tolist(),
                   halo_A=h, fmax_target=args.fmax, maxmove_A=args.maxmove)

    def ef(x):
        if args.single:
            return energy_forces_single(model, cfg, x, z, L, device)
        return energy_forces_chunked(model, cfg, x, z, L, k, h, device)

    def save():
        np.savez(state_file, pos=pos, v=v, dt=dt, a=a, nsteps=nsteps,
                 step=step)
        payload["steps"] = log
        ffm.dump(payload, args.out)

    converged = False
    while step < args.max_steps:
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        e, f = ef(pos)
        fnorm = np.sqrt((f * f).sum(1))
        row = {"step": step, "e_per_atom": e / len(pos),
               "fmax": float(fnorm.max()), "frms": float(np.sqrt((f * f).mean())),
               "dt": dt, "t_s": time.perf_counter() - t0,
               **ffm.gpu_peak_mem(device)}
        log.append(row)
        ffm.log(f"step {step:5d}  E/atom {row['e_per_atom']:.6f}  "
                f"fmax {row['fmax']:.4f}  dt {dt:.3f}  {row['t_s']:.1f} s")
        if row["fmax"] < args.fmax:
            converged = True
            break
        if v.any():  # FIRE mixing (skipped on the first step and after a reset)
            vf = float(np.vdot(f, v))
            if vf > 0:
                v = (1 - a) * v + a * f * (np.linalg.norm(v) / np.linalg.norm(f))
                if nsteps > FIRE["nmin"]:
                    dt = min(dt * FIRE["finc"], args.dtmax)
                    a *= FIRE["fa"]
                nsteps += 1
            else:
                v[:] = 0
                a = FIRE["astart"]
                dt *= FIRE["fdec"]
                nsteps = 0
        v += dt * f
        dr = dt * v
        norm = np.sqrt((dr * dr).sum(1))
        dr *= np.minimum(1.0, args.maxmove / np.maximum(norm, 1e-30))[:, None]
        pos = np.mod(pos + dr, L)
        step += 1
        if step % args.save_every == 0:
            save()
    payload["converged"] = converged
    save()
    cells.write_structure(
        args.out + "_relaxed", pos, z, L,
        dict(meta, relaxed_from=os.path.abspath(args.structure),
             relax_steps=step, relax_fmax=log[-1]["fmax"],
             relax_converged=converged))
    np.save(args.out + "_energies.npy",
            atom_energy_map(model, cfg, pos, z, L, device, args.single, k, h))
    ffm.log(f"done: {step} steps, converged {converged}, "
            f"fmax {log[-1]['fmax']:.4f}, E/atom {log[-1]['e_per_atom']:.6f}")


if __name__ == "__main__":
    main()
