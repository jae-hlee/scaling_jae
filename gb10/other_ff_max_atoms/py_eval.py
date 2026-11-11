"""One model, one job at a time, through its Python interface: Allegro (NequIP
ASE calculator) or DP (DeePMD-kit DeepPot). Used by sweep.py (engine python)
and for test-set errors.

size    pristine diamond Si of 8 i^3 atoms (a = 5.43 A): one static energy and
        force evaluation, then --steps more evaluations of the same cell with
        the atoms moved by up to 0.01 A (so nothing is cached). Prints one
        line "RESULT {json}" with the energy per atom, times, and the peak
        GPU memory torch allocated.
errors  energy and force errors on the frames of an extended XYZ file.

Both use neighbor lists whose memory grows linearly with the number of atoms
(matscipy for the NequIP calculator, vesin for DeepPot), so the size at which
they stop is set by the model.

    python py_eval.py size --kind allegro --model si_ase.nequip.pt2 --i 20
    python py_eval.py size --kind dp --model model.pth --i 20
    python py_eval.py errors --kind allegro --model si_ase.nequip.pt2 --xyz test.extxyz
"""
import argparse
import json
import sys
import time

import numpy as np

A_SI = 5.43
DIAMOND = np.array([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
                    (.25, .25, .25), (.25, .75, .75), (.75, .25, .75),
                    (.75, .75, .25)])


def diamond(i):
    """Positions (n, 3) and cell (3, 3) of i x i x i cubic diamond cells."""
    g = np.arange(i) * A_SI
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pos = (off[:, None, :] + DIAMOND[None] * A_SI).reshape(-1, 3)
    return pos, np.eye(3) * A_SI * i


class Allegro:
    """NequIP ASE calculator on a compiled model, or on a checkpoint."""

    def __init__(self, model, device, eager):
        from nequip.integrations.ase import NequIPCalculator
        if eager:
            self.calc = NequIPCalculator._from_saved_model(
                model, device=device, chemical_species_to_atom_type_map=True)
        else:
            self.calc = NequIPCalculator.from_compiled_model(
                compile_path=model, device=device,
                chemical_species_to_atom_type_map=True)
        self.how = "checkpoint, eager" if eager else "compiled"

    def __call__(self, pos, cell):
        from ase import Atoms
        atoms = Atoms(numbers=np.full(len(pos), 14), positions=pos, cell=cell,
                      pbc=True)
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        return e, np.asarray(atoms.get_forces())


class DP:
    """DeePMD-kit DeepPot with a linear-memory neighbor list."""

    def __init__(self, model, nlist):
        from deepmd.infer import DeepPot
        self.dp = DeepPot(model, nlist_backend=nlist)
        self.how = f"DeepPot, nlist_backend {nlist}"

    def __call__(self, pos, cell):
        n = len(pos)
        out = self.dp.eval(pos.reshape(1, -1), cell.reshape(1, 9), [0] * n)
        return float(np.asarray(out[0]).reshape(-1)[0]), \
            np.asarray(out[1]).reshape(n, 3)


def load(args):
    """The evaluator for --kind."""
    if args.kind == "allegro":
        return Allegro(args.model, args.device, args.eager)
    return DP(args.model, args.nlist)


def gpu_peak_gb():
    """Peak memory torch allocated on the GPU, in GB."""
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # torch absent or no GPU: the number is optional
        return None


def size(args):
    """One static evaluation and --steps more of an i^3 cell."""
    t0 = time.perf_counter()
    model = load(args)
    t_load = time.perf_counter() - t0
    pos, cell = diamond(args.i)
    n = len(pos)
    t0 = time.perf_counter()
    e, f = model(pos, cell)
    t_static = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        model(pos + rng.uniform(-0.01, 0.01, pos.shape), cell)
    t_steps = time.perf_counter() - t0
    res = {"atoms": n, "e_static_per_atom": e / n,
           "f_static_max": float(np.abs(f).max()),
           "t_load_s": t_load, "t_static_s": t_static,
           "steps_evaluated": args.steps, "t_steps_s": t_steps,
           "torch_peak_gb": gpu_peak_gb(), "how": model.how}
    if args.steps and t_steps > 0:
        res["atom_steps_per_s"] = args.steps * n / t_steps
        res["s_per_atom_step"] = t_steps / (args.steps * n)
    print("RESULT " + json.dumps(res), flush=True)


def read_extxyz(path):
    """Frames of the extended XYZ files si_data.py writes: cell, positions,
    forces, total energy."""
    frames = []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line.strip():
                break
            n = int(line)
            head = fh.readline()
            lat = head.split('Lattice="')[1].split('"')[0].split()
            e = float(head.split("energy=")[1].split()[0])
            rows = [fh.readline().split() for _ in range(n)]
            a = np.array([[float(x) for x in r[1:7]] for r in rows])
            frames.append((np.array(lat, dtype=float).reshape(3, 3),
                           a[:, :3], a[:, 3:], e))
    return frames


def errors(args):
    """Energy and force errors against the reference of an extended XYZ."""
    model = load(args)
    de, df, skipped = [], [], 0
    for k, (cell, pos, f_ref, e_ref) in enumerate(read_extxyz(args.xyz)):
        try:  # a compiled model can refuse a frame with one atom or no neighbor
            e, f = model(pos, cell)
        except Exception as err:
            skipped += 1
            print(f"frame {k} ({len(pos)} atoms) skipped: "
                  f"{type(err).__name__}: {str(err)[:200]}")
            continue
        de.append((e - e_ref) / len(pos))
        df.append((f - f_ref).reshape(-1))
    if not de:
        raise SystemExit("no frame could be evaluated")
    de, df = np.array(de), np.concatenate(df)
    res = {"frames": len(de), "frames_skipped": skipped,
           "force_components": int(df.size),
           "energy_mae_meV_atom": 1e3 * float(np.abs(de).mean()),
           "energy_rmse_meV_atom": 1e3 * float(np.sqrt((de ** 2).mean())),
           "force_mae_eV_A": float(np.abs(df).mean()),
           "force_rmse_eV_A": float(np.sqrt((df ** 2).mean())),
           "how": model.how, "xyz": args.xyz}
    print(f"energy: MAE {res['energy_mae_meV_atom']:.2f}, RMSE "
          f"{res['energy_rmse_meV_atom']:.2f} meV/atom, {res['frames']} frames"
          f" ({skipped} skipped)")
    print(f"forces: MAE {res['force_mae_eV_A']:.4f}, RMSE "
          f"{res['force_rmse_eV_A']:.4f} eV/A, {res['force_components']} components")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.out)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("size", "errors"):
        s = sub.add_parser(name)
        s.add_argument("--kind", choices=["allegro", "dp"], required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--device", default="cuda", help="allegro")
        s.add_argument("--eager", action="store_true",
                       help="allegro: --model is a checkpoint, not a compiled model")
        s.add_argument("--nlist", default="vesin", help="dp: nlist_backend")
        if name == "size":
            s.add_argument("--i", type=int, required=True)
            s.add_argument("--steps", type=int, default=5)
        else:
            s.add_argument("--xyz", required=True)
            s.add_argument("--out")
    args = p.parse_args()
    return size(args) if args.mode == "size" else errors(args)


if __name__ == "__main__":
    sys.exit(main())
