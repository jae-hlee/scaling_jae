"""Per-atom energy maps for figures, reduced from the stage-7 and stage-10
runs to small files: the atoms of a thin slab through the cell, with each
atom's energy relative to the median of its material in that slab, so grain
boundaries and interfaces stand out. Small slabs are kept atom by atom
(positions and energies), large ones are binned into an image of the mean
relative energy per pixel. numpy only.

relaxed  energies from relax.py (OUT_energies.npy, one per atom of the
         OUT_relaxed structure)
chunked  energies from ff_any.py --save_energies (OUT_rank*_energies.npz,
         idx and e of each task's owned atoms), assembled into one array

    python energy_maps.py relaxed --structure R/relax_poly_si_150k_relaxed --energies R/relax_poly_si_150k_energies.npy --axis 2 --out maps/poly_si_150k
    python energy_maps.py chunked --structure S/poly_si_1b --parts 'R/s7_1B_poly_forces_rank*_energies.npz' --axis 2 --pixel 2 --out maps/poly_si_1b
    python energy_maps.py chunked --structure S/stack_1e8 --parts 'R/s10_stack_1e8_forces_rank*_energies.npz' --groups S/stack_1e8.layer.npy --axis 1 --pixel 1 --out maps/stack_1e8

Writes OUT.npz (kind "atoms" or "grid", the slab, the per-group medians,
and x, y, de, group or the image) and OUT.json (the same numbers without
the arrays).
"""
import argparse
import glob
import json

import numpy as np

import cells


def slab_atoms(pos, axis, center, width, block=50_000_000):
    """Indices of atoms with |pos[:, axis] - center| < width / 2, reading the
    (memory-mapped) positions in blocks."""
    out = []
    for s in range(0, len(pos), block):
        x = np.asarray(pos[s:s + block, axis])
        out.append(s + np.flatnonzero(np.abs(x - center) < width / 2))
    return np.concatenate(out)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["relaxed", "chunked"])
    p.add_argument("--structure", required=True, help="cells.py prefix")
    p.add_argument("--energies", help="relax.py OUT_energies.npy (relaxed)")
    p.add_argument("--parts", help="glob of ff_any OUT_rank*_energies.npz "
                   "(chunked), or one OUT_energies.npz")
    p.add_argument("--groups", help="per-atom group file (e.g. stack .layer.npy); "
                   "medians are taken per group")
    p.add_argument("--by_species", action="store_true",
                   help="take medians per group and element, so that the Si "
                   "and O of an oxide are not compared with one median")
    p.add_argument("--axis", type=int, default=2,
                   help="slab normal: 0 x, 1 y, 2 z")
    p.add_argument("--center", type=float, default=None,
                   help="slab center along the axis, A (default: middle)")
    p.add_argument("--width", type=float, default=5.0, help="slab width, A")
    p.add_argument("--pixel", type=float, default=2.0,
                   help="pixel size of the image, A (large slabs)")
    p.add_argument("--max_atoms", type=int, default=200_000,
                   help="keep slabs up to this size atom by atom")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    pos, z, L, meta = cells.read_structure(a.structure)
    n = len(pos)
    center = L[a.axis] / 2 if a.center is None else a.center
    sel = slab_atoms(pos, a.axis, center, a.width)
    if a.mode == "relaxed":
        e_all = np.load(a.energies, mmap_mode="r")
        assert len(e_all) == n, f"{len(e_all)} energies for {n} atoms"
        e = np.asarray(e_all[sel], dtype=np.float64)
        missing = 0
    else:
        files = sorted(glob.glob(a.parts))
        if not files:
            raise SystemExit(f"no files match {a.parts}")
        if not len(sel):
            raise SystemExit("no atoms in the slab")
        # sel is sorted (slab_atoms reads in order), so each task's owned
        # atoms are matched to slab atoms by binary search, without a
        # full-size array
        e = np.full(len(sel), np.nan)
        for f in files:
            d = np.load(f)
            idx, ev = d["idx"], d["e"]
            k = np.searchsorted(sel, idx)
            k[k == len(sel)] = 0
            ok = sel[k] == idx
            e[k[ok]] = ev[ok]
        missing = int(np.isnan(e).sum())
    grp = (np.asarray(np.load(a.groups, mmap_mode="r")[sel], dtype=np.int64)
           if a.groups else np.zeros(len(sel), dtype=np.int64))
    if a.by_species:
        grp = grp * 1000 + np.asarray(z[sel], dtype=np.int64)
    ref = {}
    de = np.full(len(sel), np.nan)
    for gi in np.unique(grp):
        m = (grp == gi) & ~np.isnan(e)
        if m.any():
            ref[int(gi)] = float(np.median(e[m]))
            de[m] = e[m] - ref[int(gi)]
    a1, a2 = [d for d in range(3) if d != a.axis]
    x = np.asarray(pos[sel, a1], dtype=np.float64)
    y = np.asarray(pos[sel, a2], dtype=np.float64)
    good = ~np.isnan(de)
    info = {
        "structure": a.structure, "mode": a.mode, "natoms": int(n),
        "axis": a.axis, "plane_axes": [a1, a2], "center_A": float(center),
        "width_A": a.width, "box_A": L.tolist(), "slab_atoms": int(len(sel)),
        "missing_energies": missing, "median_by_group_eV": ref,
        "de_percentiles_eV": dict(zip(
            ["1", "50", "90", "99", "99.9"],
            [float(v) for v in np.percentile(de[good], [1, 50, 90, 99, 99.9])]))
        if good.any() else {},
        "layers": meta.get("layers"),
    }
    if len(sel) <= a.max_atoms:
        info["kind"] = "atoms"
        np.savez(a.out + ".npz", kind="atoms", x=x, y=y, de=de, group=grp,
                 extent=[0, L[a1], 0, L[a2]], info=json.dumps(info))
    else:
        nx = int(np.ceil(L[a1] / a.pixel))
        ny = int(np.ceil(L[a2] / a.pixel))
        rng = [[0, L[a1]], [0, L[a2]]]
        s, _, _ = np.histogram2d(x[good], y[good], bins=[nx, ny], range=rng,
                                 weights=de[good])
        c, _, _ = np.histogram2d(x[good], y[good], bins=[nx, ny], range=rng)
        img = np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)
        info.update(kind="grid", pixel_A=a.pixel, shape=[nx, ny])
        np.savez(a.out + ".npz", kind="grid", image=img, counts=c.astype(np.int32),
                 extent=[0, L[a1], 0, L[a2]], info=json.dumps(info))
    json.dump(info, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
