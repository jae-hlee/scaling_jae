"""Checks of smooth_messages.patch on small cells, on one GPU.

ref      (unpatched ALIGNN) energies and forces of the test cells, float64
compare  (patched ALIGNN)
           1. smooth_messages off reproduces ref exactly
           2. smooth_messages on: the training path forward(g) agrees with
              forward_tensors, float32
           3. smooth_messages on: forces equal -dE/dx by central differences
              on atoms of the test cells, float64

    PYTHONPATH=alignn_6a20fb5 python smooth_check.py ref --model_dir M --out R.npz
    PYTHONPATH=alignn_smooth python smooth_check.py compare --model_dir M \
        --smooth_dir M_SMOOTH --ref R.npz

Exits 1 if any check fails.
"""
import argparse
import sys

import numpy as np
import torch
from ase import Atoms

import cells
import ff_max_atoms as ffm


def test_cells():
    """Rattled Si and water, small enough for double precision."""
    return {"si": cells.crystal("Si", 3, rattle=0.05, seed=1),
            "water": cells.water(4, seed=1)}


def energy_forces(model, cfg, pos, z, L, device):
    """Energy and forces from forward_tensors."""
    cell = np.diag(L)
    g = ffm.build_graph(pos, cell, z, cfg, device)
    p = torch.tensor(pos, dtype=ffm.DTYPE, device=device, requires_grad=True)
    lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
    res = model.forward_tensors(p, lat, g["feats"], g["src"], g["dst"],
                                g["shift"], False, g["Z"])
    return float(res["energy"]), res["forces"].detach().double().cpu().numpy()


def ref(args, device):
    """Write the reference energies and forces."""
    ffm.DTYPE = torch.float64
    model, cfg = ffm.load_model(args.model_dir, device)
    out = {}
    for name, (pos, z, L) in test_cells().items():
        e, f = energy_forces(model, cfg, pos, z, L, device)
        out[f"{name}_e"], out[f"{name}_f"] = e, f
        print(f"{name}: {len(pos)} atoms, E/atom {e / len(pos):.6f}")
    np.savez(args.out, **out)


def forward_path(model, pos, z, L, device):
    """Energy and forces from forward((g, lg, lat)), as in training."""
    from jarvis.core.atoms import ase_to_atoms
    from alignn.torch_graph_builder import build_pure_torch_graph
    atoms = Atoms(numbers=z, positions=pos, cell=np.diag(L), pbc=True)
    g, lg = build_pure_torch_graph(
        ase_to_atoms(atoms), two_body_cutoff=5.0, three_body_cutoff=3.5,
        max_neighbors=52, atom_features="cgcnn", device=device)
    lat = torch.tensor(np.diag(L), dtype=torch.float32, device=device)
    res = model((g, lg, lat))
    # "out" leaves out the short-bond penalty that forward_tensors includes
    # in its energy (water has O-H bonds under the 1 A threshold)
    energy = float(res["out"].squeeze()) * len(pos)
    if model.config.use_penalty:
        bl = g.edata["r"].norm(dim=1)
        pen = model.config.penalty_factor * (
            model.config.penalty_threshold - bl).clamp_min(0.0)
        energy += float(pen.sum())
    return energy, res["grad"].detach().double().cpu().numpy()


def compare(args, device):
    """Run the three checks."""
    failed = 0

    def report(name, value, tol, unit):
        nonlocal failed
        ok = value <= tol
        failed += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value:.2e} {unit} "
              f"(limit {tol:.0e})", flush=True)

    data = np.load(args.ref)
    cellset = test_cells()
    ffm.DTYPE = torch.float64
    model, cfg = ffm.load_model(args.model_dir, device)
    for name, (pos, z, L) in cellset.items():
        e, f = energy_forces(model, cfg, pos, z, L, device)
        report(f"flag off vs unpatched, {name} energy",
               abs(e - float(data[f"{name}_e"])) / len(pos), 1e-12, "eV/atom")
        report(f"flag off vs unpatched, {name} forces",
               float(np.abs(f - data[f"{name}_f"]).max()), 1e-10, "eV/A")

    ffm.DTYPE = torch.float32
    smooth, cfg = ffm.load_model(args.smooth_dir, device)
    smooth.config.stresswise_weight = 0.0  # no cell volume on these graphs
    for name, (pos, z, L) in cellset.items():
        e1, f1 = energy_forces(smooth, cfg, pos, z, L, device)
        e2, f2 = forward_path(smooth, pos, z, L, device)
        report(f"forward vs forward_tensors, {name} energy",
               abs(e1 - e2) / len(pos), 2e-5, "eV/atom")
        report(f"forward vs forward_tensors, {name} forces",
               float(np.abs(f1 - f2).max()), 5e-3, "eV/A")

    ffm.DTYPE = torch.float64
    smooth, cfg = ffm.load_model(args.smooth_dir, device)
    h = 1e-4
    for name, (pos, z, L) in cellset.items():
        _, f = energy_forces(smooth, cfg, pos, z, L, device)
        worst = 0.0
        for atom in range(3):
            for axis in range(3):
                x = pos.copy()
                x[atom, axis] += h
                ep, _ = energy_forces(smooth, cfg, np.mod(x, L), z, L, device)
                x[atom, axis] -= 2 * h
                em, _ = energy_forces(smooth, cfg, np.mod(x, L), z, L, device)
                worst = max(worst, abs(f[atom, axis] + (ep - em) / (2 * h)))
        report(f"forces vs finite differences, {name}", worst, 1e-5, "eV/A")
    print("FAILED" if failed else "ALL PASS", flush=True)
    return 1 if failed else 0


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["ref", "compare"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--smooth_dir")
    p.add_argument("--ref")
    p.add_argument("--out")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "ref":
        ref(args, device)
        return 0
    return compare(args, device)


if __name__ == "__main__":
    sys.exit(main())
