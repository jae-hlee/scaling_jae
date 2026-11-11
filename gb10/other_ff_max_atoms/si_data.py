"""Single-element frames of the MatPES r2SCAN training data, for training the
comparison models (NEP, DP, SNAP, Allegro) on the same DFT data as AFF-r2SCAN.

Reads the id_prop.json that ALIGNN trained on (entries with "atoms" as a
jarvis dict, "energy_per_atom" and "forces"), keeps the frames that contain
only --element, splits them at random (seed fixed) and writes

    OUT/all.extxyz, train.extxyz, val.extxyz, test.extxyz
        extended XYZ as ASE writes it: total energy in eV in the comment
        line, positions in A and forces in eV/A per atom. No stress.
    OUT/deepmd/{train,val,test}/n<atoms>/
        DeePMD-kit systems, one per atom count: type.raw, type_map.raw,
        set.000/{coord,box,energy,force}.npy
    OUT/summary.json
        counts, sizes, energy and force ranges, the ids of each split

Needs numpy only.

    python si_data.py --data DIR_WITH_ID_PROP --element Si --out data/si
"""
import argparse
import collections
import json
import os
import random
import zipfile

import numpy as np


def load_id_prop(data_dir):
    """Entries of id_prop.json or id_prop.json.zip."""
    zp = os.path.join(data_dir, "id_prop.json.zip")
    if os.path.exists(zp):
        return json.loads(zipfile.ZipFile(zp).read("id_prop.json"))
    return json.load(open(os.path.join(data_dir, "id_prop.json")))


def frame(entry, id_tag, energy_key, force_key):
    """Cell (3x3, rows), Cartesian positions, forces, total energy, id."""
    a = entry["atoms"]
    cell = np.array(a["lattice_mat"], dtype=float)
    pos = np.array(a["coords"], dtype=float)
    if not a.get("cartesian", False):
        pos = pos @ cell
    flipped = False
    if np.linalg.det(cell) < 0:
        # DeePMD-kit and FitSNAP need a right-handed cell. Swapping two lattice
        # vectors keeps the same crystal, since the positions are Cartesian
        cell = cell[[0, 2, 1]]
        flipped = True
    f = np.array(entry[force_key], dtype=float).reshape(-1, 3)
    n = len(pos)
    if f.shape != (n, 3):
        raise ValueError(f"{entry.get(id_tag)}: {n} atoms, forces {f.shape}")
    # the id goes into the comment line of the XYZ files as key=value
    fid = "".join(c if c.isalnum() or c in "-_." else "_"
                  for c in str(entry.get(id_tag)))
    return {"id": fid, "cell": cell, "pos": pos, "f": f,
            "e": float(entry[energy_key]) * n, "n": n, "flipped": flipped}


def write_extxyz(path, frames, element, source):
    """One file, frames in the given order."""
    with open(path, "w") as fh:
        for fr in frames:
            lat = " ".join(f"{x:.10f}" for x in fr["cell"].reshape(-1))
            fh.write(f"{fr['n']}\n")
            fh.write(f'Lattice="{lat}" '
                     "Properties=species:S:1:pos:R:3:forces:R:3 "
                     f"energy={fr['e']:.10f} pbc=\"T T T\" "
                     f"config_type={source} frame_id={fr['id']}\n")
            for p, f in zip(fr["pos"], fr["f"]):
                fh.write(f"{element} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} "
                         f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")


def write_deepmd(root, frames, element):
    """One DeePMD-kit system per atom count (a system holds frames of one
    composition). Returns the number of systems."""
    groups = collections.defaultdict(list)
    for fr in frames:
        groups[fr["n"]].append(fr)
    for n, frs in sorted(groups.items()):
        d = os.path.join(root, f"n{n:04d}")
        os.makedirs(os.path.join(d, "set.000"), exist_ok=True)
        with open(os.path.join(d, "type.raw"), "w") as fh:
            fh.write("\n".join(["0"] * n) + "\n")
        with open(os.path.join(d, "type_map.raw"), "w") as fh:
            fh.write(element + "\n")
        s = os.path.join(d, "set.000")
        np.save(os.path.join(s, "coord.npy"),
                np.array([fr["pos"].reshape(-1) for fr in frs]))
        np.save(os.path.join(s, "box.npy"),
                np.array([fr["cell"].reshape(-1) for fr in frs]))
        np.save(os.path.join(s, "energy.npy"), np.array([fr["e"] for fr in frs]))
        np.save(os.path.join(s, "force.npy"),
                np.array([fr["f"].reshape(-1) for fr in frs]))
    return len(groups)


def stats(frames):
    """Sizes and ranges of a list of frames."""
    if not frames:
        return {"frames": 0}
    n = np.array([fr["n"] for fr in frames])
    epa = np.array([fr["e"] / fr["n"] for fr in frames])
    fmax = np.array([np.abs(fr["f"]).max() for fr in frames])
    vol = np.array([abs(np.linalg.det(fr["cell"])) / fr["n"] for fr in frames])
    return {"frames": len(frames), "atoms_total": int(n.sum()),
            "atoms_min": int(n.min()), "atoms_median": float(np.median(n)),
            "atoms_max": int(n.max()),
            "energy_per_atom_min": float(epa.min()),
            "energy_per_atom_median": float(np.median(epa)),
            "energy_per_atom_max": float(epa.max()),
            "force_component_max": float(fmax.max()),
            "frames_with_force_above_10": int((fmax > 10).sum()),
            "left_handed_cells_made_right_handed": int(
                sum(fr["flipped"] for fr in frames)),
            "volume_per_atom_min": float(vol.min()),
            "volume_per_atom_median": float(np.median(vol)),
            "volume_per_atom_max": float(vol.max())}


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="folder with id_prop.json")
    p.add_argument("--element", default="Si")
    p.add_argument("--out", required=True)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--id_tag", default="jid")
    p.add_argument("--energy_key", default="energy_per_atom")
    p.add_argument("--force_key", default="forces")
    args = p.parse_args()

    entries = load_id_prop(args.data)
    print(f"{len(entries):,} entries; keys of the first: {sorted(entries[0])}")
    print("atoms keys:", sorted(entries[0]["atoms"]))

    # how many single-element frames each element has, for choosing a system
    single = collections.Counter()
    for e in entries:
        els = set(e["atoms"]["elements"])
        if len(els) == 1:
            single[next(iter(els))] += 1
    print("single-element frames, 15 most common:", single.most_common(15))
    print("requested:", args.element, single.get(args.element, 0))

    frames, skipped = [], 0
    for e in entries:
        if set(e["atoms"]["elements"]) != {args.element}:
            continue
        try:
            frames.append(frame(e, args.id_tag, args.energy_key, args.force_key))
        except (KeyError, ValueError, TypeError) as err:
            skipped += 1
            print("skipped:", err)
    del entries
    if not frames:
        raise SystemExit(f"no frames of pure {args.element}")

    order = list(range(len(frames)))
    random.Random(args.seed).shuffle(order)
    n_val = max(1, round(args.val_frac * len(frames)))
    n_test = max(1, round(args.test_frac * len(frames)))
    split = {"test": [frames[i] for i in order[:n_test]],
             "val": [frames[i] for i in order[n_test:n_test + n_val]],
             "train": [frames[i] for i in order[n_test + n_val:]]}

    os.makedirs(args.out, exist_ok=True)
    source = "matpes_r2scan"
    write_extxyz(os.path.join(args.out, "all.extxyz"), frames, args.element, source)
    summary = {"element": args.element, "source": os.path.abspath(args.data),
               "seed": args.seed, "skipped": skipped, "all": stats(frames),
               "sizes": dict(sorted(collections.Counter(
                   fr["n"] for fr in frames).items())),
               "splits": {}}
    for name, frs in split.items():
        write_extxyz(os.path.join(args.out, f"{name}.extxyz"), frs,
                     args.element, source)
        nsys = write_deepmd(os.path.join(args.out, "deepmd", name), frs,
                            args.element)
        summary["splits"][name] = {**stats(frs), "deepmd_systems": nsys,
                                   "ids": [fr["id"] for fr in frs]}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    show = {k: v for k, v in summary["all"].items()}
    print("all frames:", json.dumps(show))
    print("atoms per frame -> number of frames:", json.dumps(summary["sizes"]))
    for name in ("train", "val", "test"):
        s = summary["splits"][name]
        print(f"{name}: {s['frames']} frames, {s.get('atoms_total', 0):,} atoms, "
              f"{s['deepmd_systems']} DeePMD systems")
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
