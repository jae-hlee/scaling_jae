"""The extended XYZ files of si_data.py as FitSNAP JSON: one file per frame in
OUT/<group>/, with the keys of FitSNAP's own examples (total energy in eV,
positions and lattice in A with one lattice vector per row, forces in eV/A).
The data has no stress, so a zero stress is written and must get no weight.
Standard library only.

    python fitsnap_json.py --xyz train.extxyz --out JSON --group Si_train
"""
import argparse
import json
import os


def frames(path):
    """Lattice rows, positions, forces, total energy and element symbols."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line.strip():
                return
            n = int(line)
            head = fh.readline()
            lat = [float(x) for x in head.split('Lattice="')[1].split('"')[0].split()]
            e = float(head.split("energy=")[1].split()[0])
            rows = [fh.readline().split() for _ in range(n)]
            # FitSNAP refuses a left-handed cell. Swapping two lattice vectors
            # keeps the same crystal, since the positions are Cartesian
            a, b, c = lat[0:3], lat[3:6], lat[6:9]
            det = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
            if det < 0:
                b, c = c, b
            yield {"lattice": [a, b, c],
                   "symbols": [r[0] for r in rows],
                   "pos": [[float(x) for x in r[1:4]] for r in rows],
                   "f": [[float(x) for x in r[4:7]] for r in rows],
                   "e": e, "n": n}


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xyz", required=True)
    p.add_argument("--out", required=True, help="FitSNAP dataPath")
    p.add_argument("--group", required=True, help="subfolder and group name")
    args = p.parse_args()
    d = os.path.join(args.out, args.group)
    os.makedirs(d, exist_ok=True)
    count = atoms = 0
    for k, fr in enumerate(frames(args.xyz)):
        data = {"Dataset": {
            "Data": [{"Stress": [[0.0, 0.0, 0.0]] * 3, "Positions": fr["pos"],
                      "Energy": fr["e"], "AtomTypes": fr["symbols"],
                      "Lattice": fr["lattice"], "NumAtoms": fr["n"],
                      "Forces": fr["f"]}],
            "PositionsStyle": "angstrom", "AtomTypeStyle": "chemicalsymbol",
            "Label": f"{args.group}_{k:05d}", "StressStyle": "bar",
            "LatticeStyle": "angstrom", "EnergyStyle": "electronvolt",
            "ForcesStyle": "electronvoltperangstrom"}}
        with open(os.path.join(d, f"{args.group}_{k:05d}.json"), "w") as fh:
            json.dump(data, fh)
        count += 1
        atoms += fr["n"]
    print(f"{args.group}: {count} frames, {atoms} atoms -> {d}")


if __name__ == "__main__":
    main()
