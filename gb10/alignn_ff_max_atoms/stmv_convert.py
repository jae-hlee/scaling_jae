"""The NAMD STMV benchmark (satellite tobacco mosaic virus, complete all-atom
capsid and RNA with water and ions, 1,066,628 atoms) as a cells.py structure.
Elements come from the masses in the PSF, coordinates from the PDB and the
orthorhombic periodic cell from the NAMD config.

    python stmv_convert.py --src stmv/stmv --out structures/stmv
"""
import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

import cells

ELEMENTS = {1: (1, 1.008), 12: (6, 12.011), 14: (7, 14.007), 16: (8, 15.999),
            23: (11, 22.990), 24: (12, 24.305), 31: (15, 30.974),
            32: (16, 32.06), 35: (17, 35.45), 39: (19, 39.098), 40: (20, 40.08)}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 11: "Na", 12: "Mg", 15: "P", 16: "S",
          17: "Cl", 19: "K", 20: "Ca"}


def one_file(src, pattern):
    """The single file in src that matches pattern."""
    found = sorted(glob.glob(os.path.join(src, pattern)))
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {src}, found {found}")
    return found[0]


def read_psf(path):
    """Atomic numbers and residue names, in file order."""
    with open(path) as fh:
        for line in fh:
            if "!NATOM" in line:
                natom = int(line.split()[0])
                break
        else:
            raise SystemExit("no !NATOM in the PSF")
        z = np.zeros(natom, dtype=np.uint8)
        residues = Counter()
        for i in range(natom):
            f = fh.readline().split()
            mass = float(f[7])
            key = int(round(mass))
            if key not in ELEMENTS or abs(ELEMENTS[key][1] - mass) > 0.1:
                raise SystemExit(f"atom {i + 1}: unknown mass {mass}")
            z[i] = ELEMENTS[key][0]
            residues[f[3]] += 1
    return z, residues


def read_pdb(path, natom):
    """Coordinates from the fixed-width columns of a PDB file."""
    pos = np.zeros((natom, 3))
    i = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                if i == natom:
                    raise SystemExit("more atoms in the PDB than in the PSF")
                pos[i] = float(line[30:38]), float(line[38:46]), float(line[46:54])
                i += 1
    if i != natom:
        raise SystemExit(f"{i} atoms in the PDB, {natom} in the PSF")
    return pos


def read_cell(path):
    """Box lengths from cellBasisVector1/2/3; the cell must be orthorhombic."""
    text = open(path).read()
    rows = []
    for k in (1, 2, 3):
        m = re.search(rf"^\s*cellBasisVector{k}\s+(\S+)\s+(\S+)\s+(\S+)", text, re.M)
        if m is None:
            raise SystemExit(f"no cellBasisVector{k} in {path}")
        rows.append([float(x) for x in m.groups()])
    cell = np.array(rows)
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-6:
        raise SystemExit(f"cell is not orthorhombic: {rows}")
    return np.diag(cell)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True, help="unpacked STMV folder")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    z, residues = read_psf(one_file(args.src, "*.psf"))
    pos = read_pdb(one_file(args.src, "*.pdb"), len(z))
    L = read_cell(one_file(args.src, "*.namd"))
    pos = np.mod(pos, L)
    pos[pos >= L] = 0.0
    nn = cKDTree(pos, boxsize=L).query(pos, k=2)[0][:, 1]
    counts = {SYMBOL[int(k)]: int(v) for k, v in zip(*np.unique(z, return_counts=True))}
    meta = {"source": "NAMD STMV benchmark, ks.uiuc.edu/Research/namd/utilities",
            "elements_by_symbol": counts,
            "atoms_by_residue_top": dict(residues.most_common(8)),
            "min_distance_A": float(nn.min())}
    cells.write_structure(args.out, pos, z, L, meta)
    print(f"{len(z):,} atoms, box {L.tolist()} A, "
          f"{len(z) / L.prod():.4f} atoms/A^3, elements {counts}")
    print(f"closest pair {nn.min():.3f} A, atoms by residue {meta['atoms_by_residue_top']}")


if __name__ == "__main__":
    main()
