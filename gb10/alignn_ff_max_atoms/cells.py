"""Structures for ff_any.py, and the cube-and-halo geometry, with numpy and
scipy only (no torch, no model), so both can be checked on any machine.

build   write one structure as <out>.pos.npy (float64 positions, wrapped into
        the box), <out>.Z.npy (uint8 atomic numbers) and <out>.json (box
        lengths and how it was built). Boxes are orthorhombic and periodic.
          --kind Si|SiGe|Al|Cu|SiO2  crystal of --rep conventional cells per
                                 side (diamond Si/SiGe, FCC Al/Cu, ideal
                                 beta-cristobalite SiO2); --a overrides the
                                 lattice constant (Cu at 3.6 A for the B200
                                 settings)
          --kind water           --rep^3 water molecules on a cubic grid at
                                 1.0 g/cm^3, random orientations (not
                                 equilibrated: for cost, not physics)
          --kind poly            Voronoi polycrystal of --element in a cube of
                                 side --box (A) with --grains random grains
          --kind stack           crystal slabs stacked along z in a square
                                 periodic x-y box (--layers Si:400,SiO2:30,Al:200
                                 thicknesses in A, --lateral A, --gap, --vacuum),
                                 a planar gate stack for device-scale tests
test    self-checks of the builders and of the decomposition geometry,
        including the on-demand crystal pieces used by ff_stream.py

    python cells.py build --kind Al --rep 63 --out structures/al_1m
    python cells.py build --kind poly --element Si --box 271.5 --grains 40 --out structures/poly_si_1m
    python cells.py build --kind stack --layers Si:400,SiO2:30,Al:200 --lateral 1737.6 --out structures/stack_1e8
    python cells.py test
"""
import argparse
import itertools
import json
import math
import time

import numpy as np

FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
DIAMOND = np.array([
    [0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
# ideal beta-cristobalite: Si on the diamond sites, O midway along every
# Si-Si bond (linear Si-O-Si, Si-O = a*sqrt(3)/8 = 1.55 A at a = 7.16 A)
_O_DIRS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 8.0
CRISTOBALITE = np.concatenate(
    [DIAMOND, np.mod((FCC[:, None, :] + _O_DIRS[None]).reshape(-1, 3), 1.0)])
# name: (basis, lattice constant in A, atomic numbers of the basis sites)
LATTICES = {
    "Si": (DIAMOND, 5.43, [14] * 8),
    "SiGe": (DIAMOND, 5.43, [14] * 4 + [32] * 4),  # zincblende on Si sites
    "Al": (FCC, 4.05, [13] * 4),
    "Cu": (FCC, 3.615, [29] * 4),
    "SiO2": (CRISTOBALITE, 7.16, [14] * 8 + [8] * 16),
}
# nearest-neighbor distance in units of the lattice constant
_NN = {"Si": math.sqrt(3) / 4, "SiGe": math.sqrt(3) / 4,
       "Al": 1 / math.sqrt(2), "Cu": 1 / math.sqrt(2),
       "SiO2": math.sqrt(3) / 8}
WATER_PER_A3 = 0.03343  # molecules per A^3 at 1.0 g/cm^3
OH, HOH = 0.9572, math.radians(104.52)


def nearest_neighbor(name, a=None):
    """Nearest-neighbor distance (A) of a crystal in LATTICES."""
    return (a or LATTICES[name][1]) * _NN[name]


# ------------------------------------------------------------------ builders


def crystal(name, rep, rattle=0.0, seed=0, a=None):
    """rep^3 conventional cells. Same atom order as ff_max_atoms.si_positions
    and .species, so Si and SiGe cells are identical to the earlier runs."""
    basis, a0, zs = LATTICES[name]
    a = a or a0
    grid = np.stack(np.meshgrid(*[np.arange(rep)] * 3, indexing="ij"), -1)
    pos = ((grid.reshape(-1, 1, 3) + basis[None]) * a).reshape(-1, 3)
    if rattle:
        pos += np.random.default_rng(seed).normal(0, rattle, pos.shape)
    L = np.full(3, rep * a)
    z = np.tile(np.array(zs, dtype=np.uint8), rep**3)
    return np.mod(pos, L), z, L


def random_rotations(n, rng):
    """n rotation matrices, uniform over SO(3) (normalized Gaussian
    quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def water(nmol, seed=0):
    """nmol^3 water molecules on a cubic grid at 1.0 g/cm^3, each randomly
    oriented. Bond and angle counts per atom are set by the density, so
    this is a fair cost test of liquid water; it is not an equilibrated
    liquid (neighboring molecules can sit closer than in a real one)."""
    spacing = WATER_PER_A3 ** (-1 / 3)
    grid = np.stack(np.meshgrid(*[np.arange(nmol)] * 3, indexing="ij"),
                    -1).reshape(-1, 3) * spacing
    local = np.array([[0, 0, 0],
                      [OH * math.sin(HOH / 2), OH * math.cos(HOH / 2), 0],
                      [-OH * math.sin(HOH / 2), OH * math.cos(HOH / 2), 0]])
    rot = random_rotations(len(grid), np.random.default_rng(seed))
    pos = (grid[:, None, :] + np.einsum("mij,aj->mai", rot, local)).reshape(-1, 3)
    L = np.full(3, nmol * spacing)
    z = np.tile(np.array([8, 1, 1], dtype=np.uint8), len(grid))
    return np.mod(pos, L), z, L


def _block(name, reps, a):
    """Unwrapped positions and atomic numbers of reps[0] x reps[1] x reps[2]
    conventional cells, in crystal() order."""
    basis, _, zs = LATTICES[name]
    grid = np.stack(np.meshgrid(*[np.arange(r) for r in reps], indexing="ij"),
                    -1).reshape(-1, 1, 3)
    pos = ((grid + basis[None]) * a).reshape(-1, 3)
    return pos, np.tile(np.array(zs, dtype=np.uint8), int(np.prod(reps)))


def stack(layers, lateral, gap=2.0, vacuum=10.0):
    """Crystal slabs stacked along z in a square, periodic x-y box: a planar
    gate stack (e.g. Si / SiO2 / Al) for device-scale tests. layers is a
    list of (name, thickness in A). The first layer sets the in-plane period
    (the multiple of its lattice constant closest to `lateral`); every other
    layer uses the closest whole number of its own cells and is strained in
    plane to fit (reported). Thicknesses are rounded to whole cells, `gap`
    (A) separates the top atomic plane of one layer from the bottom plane of
    the next, and `vacuum` (A, more than the cutoff) sits above the last, so
    the periodic image of the top layer does not touch the bottom one.
    Returns positions, atomic numbers, box lengths, layer index per atom and
    a per-layer summary. Interfaces are as built, not relaxed."""
    a0 = LATTICES[layers[0][0]][1]
    lxy = max(1, round(lateral / a0)) * a0
    pos_parts, z_parts, lid_parts, info = [], [], [], []
    z0 = 0.0
    for li, (name, thick) in enumerate(layers):
        a = LATTICES[name][1]
        m, n = max(1, round(lxy / a)), max(1, round(thick / a))
        p, z = _block(name, (m, m, n), a)
        s = lxy / (m * a)
        p[:, :2] *= s
        p[:, 2] += z0 - p[:, 2].min()
        top = float(p[:, 2].max())
        info.append({"layer": name, "cells": [m, m, n], "atoms": len(p),
                     "in_plane_strain": s - 1, "z_bottom_A": z0,
                     "z_top_A": top})
        pos_parts.append(p)
        z_parts.append(z)
        lid_parts.append(np.full(len(p), li, dtype=np.uint8))
        z0 = top + gap
    L = np.array([lxy, lxy, z0 - gap + vacuum])
    pos = np.concatenate(pos_parts)
    np.mod(pos, L, out=pos)
    return pos, np.concatenate(z_parts), L, np.concatenate(lid_parts), info


# ------------------------------------------- crystals generated piece by piece

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(x):
    """splitmix64 finalizer, elementwise on uint64 (wraps modulo 2**64)."""
    x = x ^ (x >> np.uint64(30))
    x = x * _M1
    x = x ^ (x >> np.uint64(27))
    x = x * _M2
    return x ^ (x >> np.uint64(31))


def hash_normal(ids, seed, comp):
    """Standard normal numbers that depend only on (atom index, seed,
    component), so every piece that contains an atom displaces it the same
    way without any shared random state."""
    salt = np.uint64((seed * 0x9E3779B97F4A7C15) % 2**64)
    key = np.asarray(ids).astype(np.uint64) * np.uint64(6) \
        + np.uint64(2 * comp) + salt
    u1 = ((_mix(key) >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    u2 = ((_mix(key + np.uint64(1)) >> np.uint64(11)).astype(np.float64)
          + 0.5) * 2.0**-53
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


def crystal_hash(name, rep, rattle=0.0, seed=0, a=None):
    """crystal(), but rattled with hash_normal, so that crystal_piece can
    generate any part of it on its own. For checks on small cells."""
    basis, a0, zs = LATTICES[name]
    a = a or a0
    pos, z = _block(name, (rep, rep, rep), a)
    if rattle:
        ids = np.arange(len(pos))
        for comp in range(3):
            pos[:, comp] += rattle * hash_normal(ids, seed, comp)
    L = np.full(3, rep * a)
    return np.mod(pos, L), z, L


def cube_edges(L, k):
    """Cube boundaries per axis: k[d] + 1 values from 0 to exactly L[d], so
    that neighboring cubes share boundaries exactly and the cubes cover the
    box."""
    return [np.append(np.arange(k[d]) * (L[d] / k[d]), L[d]) for d in range(3)]


def crystal_piece(name, rep, lo, hi, h, a=None, rattle=0.0, seed=0):
    """The atoms of crystal_hash(name, rep, rattle, seed, a) within distance
    h of the box [lo, hi), periodic images included, generated from the
    lattice without building the cell. Returns global atom indices
    (crystal() order), unwrapped positions, an owned flag (position inside
    [lo, hi)), atomic numbers and each atom's unwrapped cell index along x
    (for slab-local force rows, see local_rows)."""
    basis, a0, zs = LATTICES[name]
    a = a or a0
    nb = len(basis)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    margin = 8.0 * rattle
    i0 = np.floor((lo - h - margin) / a).astype(np.int64) - 1
    i1 = np.floor((hi + h + margin) / a).astype(np.int64) + 1
    cells_ = np.stack(np.meshgrid(*[np.arange(i0[d], i1[d] + 1)
                                    for d in range(3)], indexing="ij"),
                      -1).reshape(-1, 3)
    iw = np.mod(cells_, rep)
    cid = (iw[:, 0] * rep + iw[:, 1]) * rep + iw[:, 2]
    gid = (cid[:, None] * nb + np.arange(nb)[None]).reshape(-1)
    pos = ((cells_[:, None, :] + basis[None]) * a).reshape(-1, 3)
    if rattle:
        for comp in range(3):
            pos[:, comp] += rattle * hash_normal(gid, seed, comp)
    gap = np.maximum(np.maximum(lo - pos, pos - hi), 0)
    keep = (gap * gap).sum(1) <= h * h
    owned = ((pos >= lo) & (pos < hi)).all(1)
    ix_u = np.repeat(cells_[:, 0], nb)
    gid, pos, owned, ix_u = gid[keep], pos[keep], owned[keep], ix_u[keep]
    z = np.array(zs, dtype=np.uint8)[gid % nb]
    return gid, pos, owned, z, ix_u


def slab_cells(name, edges_x, cx0, cx1, h, a=None, rattle=0.0):
    """Unwrapped x cell range [ixa, ixb) that holds every atom of the pieces
    of cubes cx0 <= cx < cx1 (crystal_piece's range plus its margin)."""
    a = a or LATTICES[name][1]
    margin = 8.0 * rattle
    ixa = int(np.floor((edges_x[cx0] - h - margin) / a)) - 1
    ixb = int(np.floor((edges_x[cx1] + h + margin) / a)) + 2
    return ixa, ixb


def local_rows(gid, ix_u, ixa, rep, nb):
    """Rows of a slab-local array for atoms with global index gid and
    unwrapped x cell ix_u: the global row with the x cell replaced by its
    offset from ixa. Two images of one atom get different rows."""
    plane = rep * rep * nb
    return gid + (ix_u - ixa - gid // plane) * plane


def fold_planes(local, ixa, ixb, rep, nb, out):
    """Add a slab-local array (planes ixa..ixb-1 of rep*rep*nb rows) into
    the global array `out`, wrapping x cells periodically."""
    plane = rep * rep * nb
    for j in range(ixb - ixa):
        g = (ixa + j) % rep
        out[g * plane:(g + 1) * plane] += local[j * plane:(j + 1) * plane]
    return out


def _voronoi_cells(seeds, L):
    """For each seed: the neighboring seeds (periodic images included) that
    bound its Voronoi cell, and the cell's largest distance from the seed."""
    from scipy.spatial import Voronoi
    n = len(seeds)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) * L
    ext = (shifts[:, None, :] + seeds[None]).reshape(-1, 3)
    center = 13  # index of shift (0, 0, 0)
    vor = Voronoi(ext)
    nbrs = [[] for _ in range(n)]
    for p, q in vor.ridge_points:
        for a, b in ((p, q), (q, p)):
            if center * n <= a < (center + 1) * n:
                nbrs[a - center * n].append(b)
    radius = np.empty(n)
    for g in range(n):
        region = vor.regions[vor.point_region[center * n + g]]
        if -1 in region or not region:
            raise RuntimeError("unbounded Voronoi cell: use more grains")
        radius[g] = np.linalg.norm(vor.vertices[region] - seeds[g], axis=1).max()
    return [ext[np.array(v)] for v in nbrs], radius


def polycrystal(name, box, grains, seed=0, d_min=None, log=print):
    """Voronoi polycrystal of `grains` randomly oriented grains of crystal
    `name` filling a periodic cube of side `box` (A). Each grain is its
    lattice, rotated about its seed, cut to the seed's Voronoi cell. Atoms
    of different grains closer than d_min (default 0.75 nearest-neighbor
    distances) are thinned by removing one of each pair. Returns
    positions, atomic numbers, box lengths and grain index per atom."""
    basis, a, zs = LATTICES[name]
    zs = np.array(zs, dtype=np.uint8)
    d_min = d_min or 0.75 * nearest_neighbor(name)
    L = np.full(3, float(box))
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, box, (grains, 3))
    rots = random_rotations(grains, rng)
    nbrs, radius = _voronoi_cells(seeds, L)
    pos_parts, z_parts, g_parts, edge_parts = [], [], [], []
    t0 = time.perf_counter()
    for g in range(grains):
        s, nb = seeds[g], nbrs[g]
        normal = nb - s                        # facet normals (unnormalized)
        offset = 0.5 * ((nb * nb).sum(1) - s @ s)
        norm = np.linalg.norm(normal, axis=1)
        m = int(math.ceil(radius[g] / a)) + 1
        rng_ = np.arange(-m, m + 1)
        for i0 in range(0, len(rng_), 8):      # slabs of 8 cells, bounded memory
            grid = np.stack(np.meshgrid(rng_[i0:i0 + 8], rng_, rng_,
                                        indexing="ij"), -1).reshape(-1, 1, 3)
            local = ((grid + basis[None]) * a).reshape(-1, 3)
            site = np.tile(np.arange(len(basis)), len(local) // len(basis))
            keep = (local * local).sum(1) <= radius[g] ** 2
            local, site = local[keep], site[keep]
            p = s + local @ rots[g].T
            margin = (offset[None] - p @ normal.T) / norm[None]
            inside = (margin > 0).all(1)
            pos_parts.append(p[inside])
            z_parts.append(zs[site[inside]])
            g_parts.append(np.full(int(inside.sum()), g, dtype=np.int32))
            edge_parts.append(margin[inside].min(1) < d_min)
        if log and (g + 1) % max(1, grains // 10) == 0:
            log(f"  grain {g + 1}/{grains}  {sum(map(len, z_parts)):,} atoms"
                f"  {time.perf_counter() - t0:.0f} s")
    pos = np.concatenate(pos_parts)
    del pos_parts
    np.mod(pos, L, out=pos)  # in place: at 10^9 atoms each copy is 24 GB
    z = np.concatenate(z_parts)
    gid = np.concatenate(g_parts)
    near = np.flatnonzero(np.concatenate(edge_parts))
    del z_parts, g_parts, edge_parts
    # thin atoms of different grains closer than d_min (only atoms within
    # d_min of a facet can be)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos[near], boxsize=L)
    pairs = tree.query_pairs(d_min, output_type="ndarray")
    pairs = near[pairs]
    pairs = pairs[gid[pairs[:, 0]] != gid[pairs[:, 1]]]
    drop = set()
    for i, j in pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]:
        if i not in drop and j not in drop:
            drop.add(max(i, j))
    keep = np.ones(len(pos), bool)
    keep[list(drop)] = False
    return pos[keep], z[keep], L, gid[keep], len(drop)


# --------------------------------------------------------------------- I/O


def write_structure(out, pos, z, L, meta):
    """<out>.pos.npy, <out>.Z.npy and <out>.json."""
    np.save(out + ".pos.npy", np.ascontiguousarray(pos, dtype=np.float64))
    np.save(out + ".Z.npy", np.ascontiguousarray(z, dtype=np.uint8))
    meta = dict(meta, natoms=int(len(pos)), box_A=[float(x) for x in L],
                elements={int(k): int(v) for k, v in
                          zip(*np.unique(z, return_counts=True))})
    json.dump(meta, open(out + ".json", "w"), indent=1)
    return meta


def read_structure(prefix, mmap=True):
    """Positions and atomic numbers (memory-mapped by default, so a task
    does not need a private copy), box lengths and metadata."""
    meta = json.load(open(prefix + ".json"))
    mode = "r" if mmap else None
    pos = np.load(prefix + ".pos.npy", mmap_mode=mode)
    z = np.load(prefix + ".Z.npy", mmap_mode=mode)
    return pos, z, np.array(meta["box_A"], dtype=np.float64), meta


# -------------------------------------------------------------- decomposition


def core_side(cluster_volume, h):
    """Side of the cube whose halo-padded volume (a box rounded by h) is
    cluster_volume (same as ff_max_atoms.core_side)."""
    lo, hi = 0.0, cluster_volume ** (1 / 3)
    for _ in range(60):
        c = (lo + hi) / 2
        v = (c**3 + 6 * c * c * h + 3 * math.pi * c * h * h
             + 4 / 3 * math.pi * h**3)
        lo, hi = (c, hi) if v < cluster_volume else (lo, c)
    return lo


def choose_k(L, natoms, h, cluster_atoms, cutoff):
    """Cubes per axis so that a piece (cube plus halo) holds about
    cluster_atoms atoms at the cell's mean density."""
    c_max = core_side(cluster_atoms * float(np.prod(L)) / natoms, h)
    if c_max < cutoff:
        raise ValueError(f"{cluster_atoms} atoms per piece leaves a core under "
                         f"{cutoff} A inside the {h:.1f} A halo, use more")
    return np.maximum(1, np.ceil(L / c_max)).astype(np.int64)


def assign_cubes(pos, L, k, block=50_000_000):
    """Cube side per axis, atom indices sorted by cube, and where each
    cube's atoms start in that order. Positions must lie in [0, L)."""
    c = L / k
    bid = np.empty(len(pos), dtype=np.int64)
    for s in range(0, len(pos), block):  # bounded temporaries for huge cells
        b3 = np.minimum((np.asarray(pos[s:s + block]) // c).astype(np.int64),
                        k - 1)
        bid[s:s + block] = (b3[:, 0] * k[1] + b3[:, 1]) * k[2] + b3[:, 2]
    order = np.argsort(bid, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(
        bid, minlength=int(np.prod(k))))])
    return c, order, starts


def gather_piece(cube, pos, L, k, c, order, starts, h):
    """One piece: the cube's own atoms and every atom (periodic images
    included) within distance h of the cube. Returns global indices,
    positions (unwrapped, images shifted by whole boxes) and an owned flag."""
    cube = np.asarray(cube)
    lo, hi = cube * c, (cube + 1) * c
    m = np.ceil(h / c).astype(np.int64)
    idx_parts, pos_parts, own_parts = [], [], []
    for d in itertools.product(*[range(-mi, mi + 1) for mi in m]):
        nb = cube + d
        img = np.floor_divide(nb, k)
        nb = nb - img * k
        b = (nb[0] * k[1] + nb[1]) * k[2] + nb[2]
        idx = order[starts[b]:starts[b + 1]]
        p = np.asarray(pos[idx]) + img * L
        own = not any(d)
        if not own:
            gap = np.maximum(np.maximum(lo - p, p - hi), 0)
            keep = (gap * gap).sum(1) <= h * h
            idx, p = idx[keep], p[keep]
        idx_parts.append(idx)
        pos_parts.append(p)
        own_parts.append(np.full(len(idx), own))
    return (np.concatenate(idx_parts), np.concatenate(pos_parts),
            np.concatenate(own_parts))


# --------------------------------------------------------------------- tests


def _toy_energies(pos, rc, L=None):
    """Per-atom energies of a two-hop toy model with reach 2*rc:
    rho_j = sum_k w(r_jk), E_i = sum_j w(r_ij) * sqrt(1 + rho_j)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos, boxsize=L) if L is not None else cKDTree(pos)
    pairs = tree.query_pairs(rc, output_type="ndarray")
    d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    if L is not None:
        d -= np.round(d / L) * L
    r = np.linalg.norm(d, axis=1)
    w = (1 - r / rc) ** 2
    n = len(pos)
    rho = np.bincount(pairs[:, 0], w, n) + np.bincount(pairs[:, 1], w, n)
    f = np.sqrt(1 + rho)
    return (np.bincount(pairs[:, 0], w * f[pairs[:, 1]], n)
            + np.bincount(pairs[:, 1], w * f[pairs[:, 0]], n))


def test():
    """Checks that need no GPU and no model."""
    rng = np.random.default_rng(1)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # 1. Si/SiGe crystals reproduce ff_max_atoms.si_positions / species
    rep = 4
    g = np.stack(np.meshgrid(*[np.arange(rep)] * 3, indexing="ij"), -1)
    ref = np.mod(((g.reshape(-1, 1, 3) + DIAMOND[None]) * 5.43).reshape(-1, 3),
                 rep * 5.43)
    pos, z, L = crystal("Si", rep)
    check("Si crystal matches ff_max_atoms", np.array_equal(pos, ref))
    _, z2, _ = crystal("SiGe", rep)
    check("SiGe species match", np.array_equal(
        z2, np.tile(np.array([14] * 4 + [32] * 4, np.uint8), rep**3)))

    # 2-3. every atom owned once; each piece holds exactly the atoms and
    # images within h of its cube (brute force), in an orthorhombic box
    L = np.array([23.1, 31.7, 27.4])
    pos = rng.uniform(0, 1, (1500, 3)) * L
    for k, h in ((np.array([2, 3, 2]), 7.5), (np.array([1, 1, 1]), 9.0),
                 (np.array([4, 2, 3]), 12.0)):
        c, order, starts = assign_cubes(pos, L, k)
        owned_count = np.zeros(len(pos), int)
        exact = True
        m = np.ceil(h / L).astype(int) + 1
        shifts = np.array(list(itertools.product(*[range(-x, x + 1) for x in m])))
        for cube in itertools.product(*[range(x) for x in k]):
            gidx, cpos, own = gather_piece(cube, pos, L, k, c, order, starts, h)
            owned_count[gidx[own]] += 1
            lo, hi = np.array(cube) * c, (np.array(cube) + 1) * c
            img = np.rint((cpos - pos[gidx]) / L).astype(int)
            got = {(int(i),) + tuple(s) for i, s in zip(gidx, img)}
            allp = (pos[None] + (shifts * L)[:, None]).reshape(-1, 3)
            alli = np.tile(np.arange(len(pos)), len(shifts))
            alls = np.repeat(shifts, len(pos), axis=0)
            gap = np.maximum(np.maximum(lo - allp, allp - hi), 0)
            sel = (gap * gap).sum(1) <= h * h
            want = {(int(i),) + tuple(s) for i, s in zip(alli[sel], alls[sel])}
            exact &= got == want
            inside = ((pos >= lo) & (pos < hi)).all(1)
            exact &= set(gidx[own]) == set(np.flatnonzero(inside))
        check(f"pieces exact, k={k.tolist()} h={h}", exact)
        check(f"each atom owned once, k={k.tolist()}", (owned_count == 1).all())

    # 4. toy model with reach 2*rc: summing owned-atom energies over pieces
    # with h >= 2*rc gives the periodic full-cell energies exactly
    rc = 3.0
    pos = rng.uniform(0, 1, (2000, 3)) * L
    e_full = _toy_energies(pos, rc, L)
    k = np.array([2, 3, 2])
    c, order, starts = assign_cubes(pos, L, k)
    e_pieces = np.zeros(len(pos))
    for cube in itertools.product(*[range(x) for x in k]):
        gidx, cpos, own = gather_piece(cube, pos, L, k, c, order, starts, 2 * rc + 0.01)
        e_pieces[gidx[own]] = _toy_energies(cpos, rc)[own]
    check("toy model, halo = reach: per-atom energies exact",
          np.allclose(e_pieces, e_full, rtol=0, atol=1e-12),
          f"(max diff {np.abs(e_pieces - e_full).max():.1e})")
    e_short = np.zeros(len(pos))
    for cube in itertools.product(*[range(x) for x in k]):
        gidx, cpos, own = gather_piece(cube, pos, L, k, c, order, starts, rc)
        e_short[gidx[own]] = _toy_energies(cpos, rc)[own]
    check("toy model, halo < reach: differs (control)",
          np.abs(e_short - e_full).max() > 1e-6)

    # 5. water: composition, O-H bond length, density
    pos, z, L = water(6)
    o = pos[z == 8]
    hyd = pos[z == 1].reshape(-1, 2, 3)
    d = hyd - o[:, None]
    d -= np.round(d / L) * L
    check("water O-H 0.9572 A", np.allclose(np.linalg.norm(d, axis=2), OH))
    check("water density", abs(len(o) / np.prod(L) / WATER_PER_A3 - 1) < 1e-9)

    # 6. polycrystal: fills the box near bulk density, no close pairs
    # across grains, all three crystal types build
    from scipy.spatial import cKDTree
    for name, box, grains in (("Si", 54.3, 6), ("Al", 48.6, 5), ("Cu", 43.4, 5)):
        pos, z, L, gid, ndrop = polycrystal(name, box, grains, seed=3, log=None)
        basis, a, _ = LATTICES[name]
        bulk = len(basis) / a**3
        filled = (len(pos) + ndrop) / np.prod(L) / bulk  # before thinning
        dens = len(pos) / np.prod(L) / bulk
        tree = cKDTree(pos, boxsize=L)
        pr = tree.query_pairs(0.75 * nearest_neighbor(name) - 1e-9,
                              output_type="ndarray")
        cross = int((gid[pr[:, 0]] != gid[pr[:, 1]]).sum()) if len(pr) else 0
        # these grains are tiny (~1,000 atoms), so boundaries are a large
        # share of the volume and thinning removes several percent
        check(f"polycrystal {name}: {len(pos)} atoms, {grains} grains",
              abs(filled - 1) < 0.03 and dens > 0.85 and cross == 0
              and len(np.unique(gid)) == grains,
              f"(filled {filled:.3f} of bulk before thinning, {dens:.3f} after, "
              f"{ndrop} removed, {cross} close pairs)")

    # 7. SiO2 (ideal beta-cristobalite): 4 O around every Si and 2 Si around
    # every O, all at a*sqrt(3)/8
    pos, z, L = crystal("SiO2", 2)
    tree = cKDTree(pos, boxsize=L)
    d_sio = nearest_neighbor("SiO2")
    pr = tree.query_pairs(d_sio + 0.05, output_type="ndarray")
    dd = pos[pr[:, 0]] - pos[pr[:, 1]]
    dd -= np.round(dd / L) * L
    cnt = np.bincount(pr.ravel(), minlength=len(pos))
    check("SiO2: Si-O 1.55 A, 4 O per Si, 2 Si per O",
          np.allclose(np.linalg.norm(dd, axis=1), d_sio)
          and (np.sort(z[pr], axis=1) == [8, 14]).all()
          and (cnt[z == 14] == 4).all() and (cnt[z == 8] == 2).all(),
          f"(Si-O {d_sio:.3f} A, {int((z == 14).sum())} Si, "
          f"{int((z == 8).sum())} O)")

    # 8. gate stack: layers in order along z, nothing closer than a Si-O
    # bond, all atoms in the box (lateral 21.72 A keeps SiO2 strain at 1%)
    pos, z, L, lid, info = stack([("Si", 10.86), ("SiO2", 7.16),
                                  ("Al", 12.15)], 21.72)
    dmin = cKDTree(pos, boxsize=L).query(pos, k=2)[0][:, 1].min()
    order_ok = all(info[i]["z_top_A"] < info[i + 1]["z_bottom_A"]
                   for i in range(len(info) - 1))
    check(f"stack Si/SiO2/Al: {len(pos)} atoms, box {np.round(L, 2).tolist()}",
          dmin > 1.5 and order_ok and sum(x["atoms"] for x in info) == len(pos)
          and ((pos >= 0) & (pos < L)).all(),
          f"(closest pair {dmin:.3f} A, in-plane strains "
          f"{[round(x['in_plane_strain'], 4) for x in info]})")

    # 9. crystal pieces generated on demand are exactly the pieces
    # gather_piece cuts from the full cell: same atoms, same images, same
    # positions to roundoff, same species, each atom owned once
    rep, h, rat = 6, 7.0, 0.05
    pos, z, L = crystal_hash("Si", rep, rat, seed=7)
    k = np.array([3, 2, 2])
    c, order, starts = assign_cubes(pos, L, k)
    edges = cube_edges(L, k)
    same, worst = True, 0.0
    owned_n = np.zeros(len(pos), int)
    dense = np.zeros(len(pos))
    for cube in itertools.product(*[range(x) for x in k]):
        g1, p1, o1 = gather_piece(cube, pos, L, k, c, order, starts, h)
        np.add.at(dense, g1, 1)
        lo = np.array([edges[d][cube[d]] for d in range(3)])
        hi = np.array([edges[d][cube[d] + 1] for d in range(3)])
        g2, p2, o2, z2, _ = crystal_piece("Si", rep, lo, hi, h, rattle=rat,
                                          seed=7)
        im1 = np.rint((p1 - pos[g1]) / L).astype(int)
        im2 = np.rint((p2 - pos[g2]) / L).astype(int)
        worst = max(worst, float(np.abs(p2 - pos[g2] - im2 * L).max()))
        same &= ({(int(g),) + tuple(i) for g, i in zip(g1, im1)}
                 == {(int(g),) + tuple(i) for g, i in zip(g2, im2)})
        same &= set(g1[o1]) == set(g2[o2]) and np.array_equal(z2, z[g2])
        owned_n[g2[o2]] += 1
    check("on-demand crystal pieces match full-cell pieces",
          same and worst < 1e-9 and (owned_n == 1).all(),
          f"(largest position difference {worst:.1e} A)")

    # 10. slab-local force rows: two "ranks" take x slabs of cubes, add one
    # count per atom copy into local arrays and fold them; the result is the
    # full-cell count of copies per atom
    nb = len(DIAMOND)
    folded = np.zeros(len(pos))
    rows_ok = True
    for cx0, cx1 in ((0, 1), (1, 3)):
        ixa, ixb = slab_cells("Si", edges[0], cx0, cx1, h, rattle=rat)
        local = np.zeros((ixb - ixa) * rep * rep * nb)
        for cube in itertools.product(range(cx0, cx1), range(k[1]),
                                      range(k[2])):
            lo = np.array([edges[d][cube[d]] for d in range(3)])
            hi = np.array([edges[d][cube[d] + 1] for d in range(3)])
            g2, _, _, _, ix_u = crystal_piece("Si", rep, lo, hi, h,
                                              rattle=rat, seed=7)
            rows = local_rows(g2, ix_u, ixa, rep, nb)
            rows_ok &= bool(rows.min() >= 0 and rows.max() < len(local))
            np.add.at(local, rows, 1)
        fold_planes(local, ixa, ixb, rep, nb, folded)
    check("slab-local rows fold back to the full-cell result",
          rows_ok and np.array_equal(dense, folded))

    # 11. pristine crystal, atoms exactly on cube faces: every atom owned once
    rep = 5
    Lp = np.full(3, rep * LATTICES["Al"][1])
    k = np.array([2, 3, 2])
    edges = cube_edges(Lp, k)
    owned_n = np.zeros(4 * rep**3, int)
    for cube in itertools.product(*[range(x) for x in k]):
        lo = np.array([edges[d][cube[d]] for d in range(3)])
        hi = np.array([edges[d][cube[d] + 1] for d in range(3)])
        g2, _, o2, _, _ = crystal_piece("Al", rep, lo, hi, 6.0)
        owned_n[g2[o2]] += 1
    check("pristine Al pieces: every atom owned once", (owned_n == 1).all())
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["build", "test"])
    p.add_argument("--kind", choices=list(LATTICES) + ["water", "poly",
                                                       "stack"])
    p.add_argument("--rep", type=int, help="cells (crystal) or molecules "
                   "(water) per side")
    p.add_argument("--a", type=float, default=None,
                   help="lattice constant override, A (crystal)")
    p.add_argument("--layers", default="Si:400,SiO2:30,Al:200",
                   help="name:thickness_A,... bottom to top (stack)")
    p.add_argument("--lateral", type=float, default=1737.6,
                   help="target x-y period, A (stack)")
    p.add_argument("--gap", type=float, default=2.0,
                   help="spacing between layers, A (stack)")
    p.add_argument("--vacuum", type=float, default=10.0,
                   help="vacuum above the top layer, A (stack)")
    p.add_argument("--element", choices=list(LATTICES), default="Si",
                   help="crystal of each grain (poly)")
    p.add_argument("--box", type=float, help="cube side in A (poly)")
    p.add_argument("--grains", type=int, help="number of grains (poly)")
    p.add_argument("--d_min", type=float, default=None,
                   help="closest allowed pair across grains, A (poly)")
    p.add_argument("--rattle", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()
    if args.mode == "test":
        raise SystemExit(0 if test() else 1)
    t0 = time.perf_counter()
    meta = {"kind": args.kind, "seed": args.seed}
    if args.kind == "water":
        pos, z, L = water(args.rep, args.seed)
        meta.update(molecules_per_side=args.rep)
    elif args.kind == "poly":
        pos, z, L, gid, ndrop = polycrystal(args.element, args.box, args.grains,
                                            args.seed, args.d_min)
        meta.update(element=args.element, grains=args.grains,
                    removed_close_pairs=ndrop,
                    d_min_A=args.d_min or 0.75 * nearest_neighbor(args.element))
        np.save(args.out + ".grain.npy", gid)
    elif args.kind == "stack":
        layers = [(s.split(":")[0], float(s.split(":")[1]))
                  for s in args.layers.split(",")]
        pos, z, L, lid, info = stack(layers, args.lateral, args.gap,
                                     args.vacuum)
        meta.update(layers=info, gap_A=args.gap, vacuum_A=args.vacuum)
        np.save(args.out + ".layer.npy", lid)
    else:
        pos, z, L = crystal(args.kind, args.rep, args.rattle, args.seed,
                            args.a)
        meta.update(rep=args.rep, rattle=args.rattle,
                    a_A=args.a or LATTICES[args.kind][1])
    meta = write_structure(args.out, pos, z, L, meta)
    print(json.dumps(meta), f"built in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
