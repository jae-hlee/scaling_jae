"""Why a FIRE relaxation stalls: which atoms keep large forces, whether they
sit at a bond-length cutoff of the model's graph, and whether the forces jump
when a bond crosses that cutoff.

The line graph keeps a bond only if it is at most 3.5 A long, with no
smoothing of the angle terms, so the forces may be discontinuous there. For
a few bonds, one atom is moved along the bond so that the length becomes
target - 0.002, target - 0.0002, target + 0.0002 and target + 0.002 A, and
the forces of the whole cell are compared:
    across  largest change of any atom's force across the 0.0004 A step that
            straddles the target
    same    largest change over a 0.0018 A step on one side of it
Smooth forces change in proportion to the distance moved, so across/same is
about 0.22. A jump at the target makes it much larger. Bonds near 5 A and a
control bond far from both cutoffs get the same test.

    python relax_check.py --model_dir M --relaxed R/relax_poly_si_150k_relaxed --out R/relax_check
"""
import argparse
import os

import numpy as np
import torch

import cells
import ff_max_atoms as ffm

THRESHOLDS = (0.05, 0.1, 0.2, 0.4)
EPS = (-2e-3, -2e-4, 2e-4, 2e-3)


def evaluate(model, cfg, pos, z, L, device):
    """Forces of the periodic cell as one graph (as relax.py) and its energy,
    summed in double precision so that small differences can be seen."""
    cell = np.diag(L)
    g = ffm.build_graph(pos, cell, z, cfg, device)
    p = torch.tensor(pos, dtype=ffm.DTYPE, device=device, requires_grad=True)
    lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
    e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
    (grad,) = torch.autograd.grad(e_atom.sum() + e_bond.sum(), p)
    e = float(e_atom.detach().double().sum() + e_bond.detach().double().sum())
    return e, -grad.detach().double().cpu().numpy()


def bonds(pos, L, z, cfg, device):
    """Directed bonds of the graph: first atom, unit vector, length."""
    cell = np.diag(L)
    g = ffm.build_graph(pos, cell, z, cfg, device)
    src = g["src"].cpu().numpy()
    dst = g["dst"].cpu().numpy()
    r = pos[dst] - pos[src] + g["shift"].double().cpu().numpy() @ cell
    bl = np.sqrt((r * r).sum(1))
    return src, r / bl[:, None], bl


def gap_to(src, bl, n, cut):
    """Distance from cut of each atom's closest bond length."""
    gap = np.full(n, np.inf)
    np.minimum.at(gap, src, np.abs(bl - cut))
    return gap


def closest_bond(src, bl, atom, target):
    """Index of the bond of this atom with the length closest to target."""
    idx = np.flatnonzero(src == atom)
    return int(idx[np.argmin(np.abs(bl[idx] - target))])


def largest_change(a, c):
    """Largest change of any atom's force between two force arrays."""
    return float(np.sqrt(((a - c) ** 2).sum(1)).max())


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--relaxed", required=True,
                   help="cells.py prefix of the relaxed structure")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = ffm.load_model(args.model_dir, device, ckpt=True)
    pos, z, L, meta = cells.read_structure(args.relaxed, mmap=False)
    pos, z = np.array(pos, dtype=np.float64), np.asarray(z)
    n = len(pos)
    payload = ffm.header(args, cfg, device)
    flags = {k: getattr(model, k, None) for k in (
        "three_body_cutoff", "use_cutoff_function", "inner_cutoff",
        "multiply_cutoff")}
    flags = {k: v.item() if torch.is_tensor(v) else v for k, v in flags.items()}
    payload.update(relaxed=os.path.abspath(args.relaxed), natoms=n,
                   relax_steps=meta.get("relax_steps"),
                   relax_converged=meta.get("relax_converged"), flags=flags)
    ffm.log(f"{n:,} atoms, {meta.get('relax_steps')} FIRE steps, "
            f"converged {meta.get('relax_converged')}")
    ffm.log("model:", flags)

    e0, f0 = evaluate(model, cfg, pos, z, L, device)
    fnorm = np.sqrt((f0 * f0).sum(1))
    order = np.argsort(-fnorm)
    payload.update(
        e_per_atom=e0 / n, fmax=float(fnorm.max()),
        frms_component=float(np.sqrt((f0 * f0).mean())),
        force_percentiles={str(q): float(np.percentile(fnorm, q))
                           for q in (50, 90, 99, 99.9, 99.99)},
        atoms_above={str(t): int((fnorm > t).sum()) for t in THRESHOLDS})
    ffm.log(f"E/atom {e0 / n:.6f}  largest force {payload['fmax']:.4f}  "
            f"rms per component {payload['frms_component']:.5f} eV/A")
    ffm.log("force percentiles (eV/A):", {
        k: round(v, 4) for k, v in payload["force_percentiles"].items()})
    for t in THRESHOLDS:
        c = payload["atoms_above"][str(t)]
        ffm.log(f"atoms with force above {t} eV/A: {c:,} ({100 * c / n:.3f}%)")

    src, unit, bl = bonds(pos, L, z, cfg, device)
    degree = np.bincount(src, minlength=n)
    short = np.bincount(src[bl <= 3.5], minlength=n)
    gap35, gap50 = gap_to(src, bl, n, 3.5), gap_to(src, bl, n, 5.0)
    payload["max_degree"] = int(degree.max())
    ffm.log(f"largest number of bonds on an atom {degree.max()} "
            f"(cap {cfg['max_neighbors']})")

    groups = {"all atoms": np.ones(n, bool),
              "force > 0.05": fnorm > 0.05, "force > 0.2": fnorm > 0.2,
              "top 20": np.isin(np.arange(n), order[:20])}
    payload["near_cutoff"] = {}
    ffm.log("\nshare of atoms with a bond within d of a cutoff")
    ffm.log(f"{'group':14s} {'atoms':>8s} {'3.5 A, 0.01':>12s} "
            f"{'3.5 A, 0.05':>12s} {'5 A, 0.01':>10s} {'5 A, 0.05':>10s}")
    for name, m in groups.items():
        row = [float((g[m] <= d).mean()) if m.any() else float("nan")
               for g in (gap35, gap50) for d in (0.01, 0.05)]
        payload["near_cutoff"][name] = dict(
            atoms=int(m.sum()), gap35_01=row[0], gap35_05=row[1],
            gap50_01=row[2], gap50_05=row[3])
        ffm.log(f"{name:14s} {int(m.sum()):8,d} {row[0]:12.3f} {row[1]:12.3f} "
                f"{row[2]:10.3f} {row[3]:10.3f}")

    ffm.log("\nlargest forces")
    ffm.log("  atom     force  bonds<=3.5  shortest bond  gap to 3.5  gap to 5")
    for i in order[:20]:
        ffm.log(f"{i:6d}  {fnorm[i]:8.4f}  {short[i]:10d}  "
                f"{bl[src == i].min():13.4f}  {gap35[i]:10.4f}  {gap50[i]:8.4f}")

    picks = []
    for i in order:  # the largest-force atoms with a bond close to 3.5 A
        if gap35[i] <= 0.1:
            picks.append((f"3.5 A, force {fnorm[i]:.3f}", int(i), 3.5))
            if len(picks) == 3:
                break
    picks.append((f"5 A, force {fnorm[order[0]]:.3f}", int(order[0]), 5.0))
    picks.append(("control", int(order[n // 2]), None))

    payload["crossing"] = []
    ffm.log("\nforces across a cutoff (eV/A); smooth forces give across/same "
            "of about 0.22")
    ffm.log(f"{'bond':22s} {'length':>8s} {'across':>9s} {'same':>9s} "
            f"{'ratio':>7s} {'dE across':>10s} {'dE same':>10s}")
    for label, atom, target in picks:
        b = closest_bond(src, bl, atom, 4.25 if target is None else target)
        cut = float(bl[b]) if target is None else target
        res = []
        for eps in EPS:
            x = pos.copy()
            x[atom] = np.mod(x[atom] + (bl[b] - (cut + eps)) * unit[b], L)
            res.append(evaluate(model, cfg, x, z, L, device))
        (em, fm), (e1, f1), (e2, f2), (ep, fp) = res
        across = largest_change(f1, f2)
        same = max(largest_change(fm, f1), largest_change(fp, f2))
        row = dict(label=label, atom=atom, bond_A=float(bl[b]),
                   target_A=cut, across=across, same=same,
                   ratio=across / max(same, 1e-30), de_across=e2 - e1,
                   de_same=max(abs(e1 - em), abs(ep - e2)))
        payload["crossing"].append(row)
        ffm.log(f"{label:22s} {bl[b]:8.4f} {across:9.5f} {same:9.5f} "
                f"{row['ratio']:7.2f} {row['de_across']:10.5f} "
                f"{row['de_same']:10.5f}")
        ffm.dump(payload, args.out)
    ffm.dump(payload, args.out)


if __name__ == "__main__":
    main()
