"""Stacking-fault energies of an FCC metal on its (111) plane with ALIGNN-FF,
in double precision on one GPU: the test Choudhary et al. (J. Phys.: Condens.
Matter 27, 015003, 2015) used to judge an Al potential for dislocations.

run      1. lattice constant and bulk modulus, from the energy of the FCC
            cell against its lattice constant. Steps where the energy
            jumps (a neighbor shell passing a cutoff of the graph) are
            listed and kept out of the fit
         2. a (111) slab with vacuum above it, x along [1-10], y along
            [11-2], z along [111]. The upper half is moved rigidly by u in
            the plane, and the energy change per area is the fault energy:
              along [11-2]  the first maximum is the unstable stacking
                            fault, the minimum at a/sqrt(6) the intrinsic
                            stacking fault, and at 2a/sqrt(6) atoms sit on
                            top of each other
              along [1-10]  one maximum, at half the nearest-neighbor
                            distance
            each as built and with every atom free to move along z only
            (the usual convention), starting from the relaxed slab
         3. the (111) surface energy, from the same slab
         4. the as-built fault energy on a grid of displacements (the gamma
            surface)
         5. the as-built intrinsic fault energy in a thicker slab, which
            must agree if the surfaces do not matter
summary  table of the runs matching --glob, next to the paper's values

    python stacking_fault.py run --model_dir M --out R/gsf_al
    python smooth_eval.py stacking_fault.py run --model_dir M_SMOOTH --out R/gsf_al
    python stacking_fault.py summary --glob 'results/*/stage14/gsf_al_*.json'

Writes OUT.json after every part and, if matplotlib is installed, OUT.png.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

import cells

torch = ffm = None  # imported by main() for run; summary needs numpy only

MJ_M2 = 16021.766  # 1 eV/A^2 in mJ/m^2
GPA = 160.21766    # 1 eV/A^3 in GPa
FIRE = dict(dt=0.1, dtmax=1.0, nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99)
# Table 1 of the paper: experiment or DFT, EAM (Mishin et al.), COMB3.
# Energies in mJ/m^2.
PAPER = {"Al": {
    "a": ("4.05", "4.05", "4.05"),
    "B": ("79.0", "77.0", "83.0"),
    "surf": ("750 to 980", "870", "720"),
    "isf": ("116 to 164", "146", "146"),
    "usf": ("154 to 291", "168", "194"),
    "max110": ("250, 663", "-", "552"),
}}
# the same ranges as numbers, for the figure
PAPER_RANGE = {"Al": {"isf": (116, 164), "usf": (154, 291),
                      "max110": (250, 663)}}


def energy(model, cfg, pos, z, L, device, forces=False):
    """Energy (eV) of the periodic cell as one graph, and its forces (eV/A)
    if asked. Positions are wrapped into the box first."""
    cell = np.diag(L)
    pos = np.ascontiguousarray(np.mod(pos, L), dtype=np.float64)
    g = ffm.build_graph(pos, cell, z, cfg, device)
    lat = torch.tensor(cell, dtype=ffm.DTYPE, device=device)
    p = torch.tensor(pos, dtype=ffm.DTYPE, device=device, requires_grad=forces)
    if not forces:
        with torch.no_grad():
            e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
        return float(e_atom.sum() + e_bond.sum()), None
    e_atom, e_bond = ffm.atom_energies(model, p, lat, g)
    e = e_atom.sum() + e_bond.sum()
    (grad,) = torch.autograd.grad(e, p)
    return float(e), -grad.detach().double().cpu().numpy()


def slab(a, nx, ny, nlayers, vacuum):
    """FCC (111) slab: positions, box and the layer of every atom. A layer is
    a triangular lattice with two atoms per a1 x a2 rectangle, and each layer
    is displaced by a2/3 along y from the one below (ABC stacking)."""
    a1, a2, d = a / math.sqrt(2), a * math.sqrt(1.5), a / math.sqrt(3)
    ix, iy, k, b = (v.ravel() for v in np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nlayers), np.arange(2),
        indexing="ij"))
    pos = np.empty((ix.size, 3))
    pos[:, 0] = (ix + 0.5 * b) * a1
    pos[:, 1] = (iy + 0.5 * b + k / 3.0) * a2
    pos[:, 2] = 0.5 * vacuum + k * d
    L = np.array([nx * a1, ny * a2, (nlayers - 1) * d + vacuum])
    pos[:, :2] = np.mod(pos[:, :2], L[:2])
    return pos, L, k


def shifted(pos, upper, ux, uy):
    """Copy with the upper half moved by (ux, uy) in the plane."""
    out = pos.copy()
    out[upper, 0] += ux
    out[upper, 1] += uy
    return out


def pair_stats(pos, L, nn):
    """Closest pair distance and each atom's number of neighbors within
    1.05 nn, by minimum image (box sides must exceed 2.1 nn)."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(r, np.inf)
    return float(r.min()), (r < 1.05 * nn).sum(1)


def check_geometry(a, pos, L, lay, upper):
    """Stop unless the slab is FCC and the fault displacements do what the
    docstring says. Returns +1 or -1: the sense along y in which a/sqrt(6)
    gives the intrinsic fault (the other sense puts atoms on top)."""
    nn, a1, a2, d = (a / math.sqrt(2), a / math.sqrt(2), a * math.sqrt(1.5),
                     a / math.sqrt(3))
    inside = (lay > 0) & (lay < lay.max())
    tol = 1e-6 * a

    def stats(ux, uy):
        return pair_stats(shifted(pos, upper, ux, uy), L, nn)

    rmin, coord = stats(0.0, 0.0)
    if abs(rmin - nn) > tol or not (coord[inside] == 12).all():
        raise SystemExit("slab is not FCC")
    for ux, uy in ((a1, 0.0), (0.0, a2)):  # one full period
        rmin, coord = stats(ux, uy)
        if abs(rmin - nn) > tol or not (coord[inside] == 12).all():
            raise SystemExit("slab does not repeat after one period")
    r1, c1 = stats(0.0, a2 / 3)
    r2, c2 = stats(0.0, 2 * a2 / 3)
    if abs(r1 - nn) < tol and abs(r2 - d) < tol and (c1[inside] == 12).all():
        return 1
    if abs(r2 - nn) < tol and abs(r1 - d) < tol and (c2[inside] == 12).all():
        return -1
    raise SystemExit("no intrinsic fault at a/sqrt(6) along y")


def energy_jumps(a_vals, e_vals, floor=2e-3):
    """Steps of the lattice-constant scan where the energy per atom leaves
    the trend of its neighbors by more than `floor` eV: the change over the
    step minus the median change over the two steps on either side. A shell
    of neighbors passing a cutoff of the model's graph shows up this way."""
    d = np.diff(e_vals)
    out = []
    for i in range(2, len(d) - 2):
        excess = float(d[i]) - float(np.median([d[i - 2], d[i - 1], d[i + 1],
                                                d[i + 2]]))
        if abs(excess) > floor:
            out.append({"index": i, "a_lo_A": float(a_vals[i]),
                        "a_hi_A": float(a_vals[i + 1]),
                        "jump_meV_per_atom": 1000 * excess})
    return out


def fit_minimum(a_vals, e_vals, breaks=(), half=6):
    """Lattice constant, energy per atom and bulk modulus from a cubic in
    the volume per atom. Only the stretch of the scan that contains its
    middle (the reference lattice constant) and no step in `breaks` is used
    (step i lies between points i and i + 1), so that a jump next to the
    minimum cannot move it: the fit is to the points around the lowest
    energy of that stretch. None if that lowest energy is at an end of the
    stretch or fewer than five points are left."""
    a_vals, e_vals = np.asarray(a_vals), np.asarray(e_vals)
    mid = len(a_vals) // 2
    seg_lo, seg_hi = 0, len(a_vals)
    for i in breaks:
        if i < mid:
            seg_lo = max(seg_lo, i + 1)
        else:
            seg_hi = min(seg_hi, i + 1)
    i0 = seg_lo + int(np.argmin(e_vals[seg_lo:seg_hi]))
    if i0 == seg_lo or i0 == seg_hi - 1:
        return None
    lo, hi = max(seg_lo, i0 - half), min(seg_hi, i0 + half + 1)
    if hi - lo < 5:
        return None
    v = a_vals[lo:hi] ** 3 / 4.0
    vc = v[i0 - lo]
    c = np.polyfit(v - vc, e_vals[lo:hi], 3)
    d1, d2 = np.polyder(c), np.polyder(c, 2)
    best = None
    for r in np.roots(d1):
        if (abs(r.imag) < 1e-9 and v[0] - vc <= r.real <= v[-1] - vc
                and np.polyval(d2, r.real) > 0
                and (best is None or abs(r.real) < abs(best))):
            best = float(r.real)
    if best is None:
        return None
    v0 = vc + best
    return {"a0_A": float((4 * v0) ** (1 / 3)),
            "e0_eV": float(np.polyval(c, best)),
            "B_GPa": float(v0 * np.polyval(d2, best) * GPA)}


def relax_z(ef, pos, z, L, fmax, max_steps, maxmove):
    """FIRE (as relax.py) on the z coordinates only. Returns the energy, the
    positions, the largest |F_z| left (eV/A) and the steps taken."""
    pos = pos.copy()
    v = np.zeros(len(pos))
    dt, a, nsteps = FIRE["dt"], FIRE["astart"], 0
    for step in range(max_steps + 1):
        e, f = ef(pos, z, L, True)
        fz = f[:, 2]
        left = float(np.abs(fz).max())
        if left < fmax or step == max_steps:
            return e, pos, left, step
        if v.any():
            if float(np.vdot(fz, v)) > 0:
                v = (1 - a) * v + a * fz * (np.linalg.norm(v)
                                            / np.linalg.norm(fz))
                if nsteps > FIRE["nmin"]:
                    dt = min(dt * FIRE["finc"], FIRE["dtmax"])
                    a *= FIRE["fa"]
                nsteps += 1
            else:
                v[:] = 0
                a = FIRE["astart"]
                dt *= FIRE["fdec"]
                nsteps = 0
        v += dt * fz
        pos[:, 2] += np.clip(dt * v, -maxmove, maxmove)


def key_values(rows_y, rows_x):
    """The numbers the paper tabulates, from the two lines. The second
    differences between 0 and the intrinsic fault are small for a smooth
    curve; a jump in the energy shows up as a large one."""
    gy = np.array([r["gamma_mJ_m2"] for r in rows_y])
    gx = np.array([r["gamma_mJ_m2"] for r in rows_x])
    n = len(gy) - 1
    i_isf = n // 3
    i_usf = int(np.argmax(gy[:i_isf + 1]))
    i_110 = int(np.argmax(gx))
    bend = np.abs(gy[2:i_isf + 1] - 2 * gy[1:i_isf] + gy[:i_isf - 1])
    return {
        "isf_mJ_m2": float(gy[i_isf]), "isf_u_A": rows_y[i_isf]["u_A"],
        "isf_is_minimum": bool(gy[i_isf] < gy[i_isf - 1]
                               and gy[i_isf] < gy[i_isf + 1]),
        "usf_mJ_m2": float(gy[i_usf]), "usf_u_A": rows_y[i_usf]["u_A"],
        "on_top_mJ_m2": float(gy[2 * n // 3]),
        "max_110_mJ_m2": float(gx[i_110]), "max_110_u_A": rows_x[i_110]["u_A"],
        "after_one_period_mJ_m2": [float(gy[-1]), float(gx[-1])],
        "largest_bend_mJ_m2": float(bend.max()),
        "largest_bend_u_A": rows_y[1 + int(np.argmax(bend))]["u_A"],
        "median_bend_mJ_m2": float(np.median(bend)),
    }


def intervals(period, step, multiple):
    """Number of steps over one period: close to period/step and a multiple
    of `multiple`, so that the special displacements are scan points."""
    return multiple * max(2, round(period / (multiple * step)))


def run(args, device):
    """Lattice constant, fault-energy lines and the checks, for one model."""
    ffm.DTYPE = torch.float64
    torch.set_default_dtype(torch.float64)
    model, cfg = ffm.load_model(args.model_dir, device)
    basis, a_ref, zs = cells.LATTICES[args.element]
    if basis is not cells.FCC:
        raise SystemExit(f"{args.element} is not an FCC metal of cells.py")
    t_start = time.perf_counter()
    count = [0]

    def ef(pos, z, L, forces=False):
        count[0] += 1
        return energy(model, cfg, pos, z, L, device, forces)

    payload = ffm.header(args, cfg, device)
    payload.update(element=args.element, dtype="float64")

    def save():
        payload.update(evaluations=count[0],
                       wall_s=time.perf_counter() - t_start)
        ffm.dump(payload, args.out)

    # 1. lattice constant and bulk modulus
    a_vals = np.linspace(a_ref * (1 - args.span), a_ref * (1 + args.span),
                         args.npts)
    e_vals = []
    for a in a_vals:
        pos, z, L = cells.crystal(args.element, args.bulk_rep, a=float(a))
        e_vals.append(ef(pos, z, L)[0] / len(pos))
    e_vals = np.array(e_vals)
    found = energy_jumps(a_vals, e_vals)
    for j in found:
        ffm.log(f"energy jump of {j['jump_meV_per_atom']:+.2f} meV/atom "
                f"between a = {j['a_lo_A']:.3f} and {j['a_hi_A']:.3f} A")
    fit = fit_minimum(a_vals, e_vals, [j["index"] for j in found])
    if args.a:
        a0, source = args.a, "given"
    elif fit:
        a0, source = fit["a0_A"], "fit"
    else:
        a0, source = a_ref, ("reference (no minimum inside the jump-free "
                             "part of the scan)")
    pos, z, L = cells.crystal(args.element, args.bulk_rep, a=a0)
    e_bulk = ef(pos, z, L)[0] / len(pos)
    payload.update(
        eos={"a_A": a_vals.tolist(), "e_per_atom_eV": e_vals.tolist(),
             "jumps": found, "fit": fit},
        a0_A=float(a0), a0_source=source, e_bulk_eV=e_bulk,
        B_GPa=fit["B_GPa"] if fit else None)
    ffm.log(f"lattice constant {a0:.4f} A ({source}), bulk modulus "
            f"{payload['B_GPa']} GPa, E/atom {e_bulk:.6f} eV")
    save()

    # 2. the slab
    rc = float(cfg["cutoff"])
    pos0, L, lay = slab(a0, args.nx, args.ny, args.layers, args.vacuum)
    if min(L[0], L[1]) <= 2 * rc or args.vacuum <= rc + 2 or args.layers % 2:
        raise SystemExit("need in-plane sides above twice the cutoff, vacuum "
                         "above the cutoff + 2 A and an even number of layers")
    z = np.full(len(pos0), zs[0], dtype=np.uint8)
    upper = lay >= args.layers // 2
    a1, a2 = a0 / math.sqrt(2), a0 * math.sqrt(1.5)
    area = float(L[0] * L[1])
    sign = check_geometry(a0, pos0, L, lay, upper)
    n = len(pos0)
    e0u = ef(pos0, z, L)[0]
    e0r, pos0r, left0, steps0 = relax_z(ef, pos0, z, L, args.fmax,
                                        args.max_steps, args.maxmove)
    payload.update(
        slab={"nx": args.nx, "ny": args.ny, "layers": args.layers,
              "natoms": n, "box_A": L.tolist(), "area_A2": area,
              "vacuum_A": args.vacuum, "isf_sense_along_y": sign,
              "relax_fz_left_eV_A": left0, "relax_steps": steps0,
              "fmax_target_eV_A": args.fmax, "max_steps": args.max_steps},
        surface_energy_mJ_m2={
            "unrelaxed": (e0u - n * e_bulk) / (2 * area) * MJ_M2,
            "relaxed": (e0r - n * e_bulk) / (2 * area) * MJ_M2})
    ffm.log(f"slab: {n} atoms, {args.layers} layers, box "
            f"{L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} A, relaxed in {steps0} "
            f"steps to {left0:.4f} eV/A | (111) surface energy "
            f"{payload['surface_energy_mJ_m2']['relaxed']:.0f} mJ/m^2 "
            f"({payload['surface_energy_mJ_m2']['unrelaxed']:.0f} as built)")
    save()

    def line(axis, period, npts, relaxed):
        """Fault energy at npts + 1 displacements over one period."""
        rows = []
        for t in np.linspace(0.0, period, npts + 1):
            ux, uy = (t, 0.0) if axis == "x" else (0.0, sign * t)
            if relaxed:
                e, _, left, steps = relax_z(
                    ef, shifted(pos0r, upper, ux, uy), z, L, args.fmax,
                    args.max_steps, args.maxmove)
                rows.append({"u_A": float(t),
                             "gamma_mJ_m2": (e - e0r) / area * MJ_M2,
                             "fz_left_eV_A": left, "steps": steps})
                ffm.log(f"  {axis} {t:6.3f} A  {rows[-1]['gamma_mJ_m2']:9.2f}"
                        f" mJ/m^2  {steps:4d} steps  {left:.4f} eV/A left")
            else:
                e = ef(shifted(pos0, upper, ux, uy), z, L)[0]
                rows.append({"u_A": float(t),
                             "gamma_mJ_m2": (e - e0u) / area * MJ_M2})
        return rows

    payload.update(unrelaxed={}, relaxed={}, key={})
    for name, relaxed, step in (("unrelaxed", False, args.step),
                                ("relaxed", True, args.relaxed_step)):
        ffm.log(f"{name} lines, step about {step} A")
        payload[name]["y"] = line("y", a2, intervals(a2, step, 3), relaxed)
        save()
        payload[name]["x"] = line("x", a1, intervals(a1, step, 2), relaxed)
        payload["key"][name] = key_values(payload[name]["y"],
                                          payload[name]["x"])
        k = payload["key"][name]
        ffm.log(f"{name}: intrinsic fault {k['isf_mJ_m2']:.1f}, unstable "
                f"fault {k['usf_mJ_m2']:.1f} at {k['usf_u_A']:.2f} A, atoms "
                f"on top {k['on_top_mJ_m2']:.1f}, maximum along [1-10] "
                f"{k['max_110_mJ_m2']:.1f} mJ/m^2 | largest bend "
                f"{k['largest_bend_mJ_m2']:.2f} at {k['largest_bend_u_A']:.2f}"
                f" A (median {k['median_bend_mJ_m2']:.2f})")
        save()

    # 4. gamma surface, as built
    payload["gamma_surface"] = None
    gx_n, gy_n = args.surface
    if gx_n and gy_n:
        ux = np.arange(gx_n) * a1 / gx_n
        uy = np.arange(gy_n) * a2 / gy_n
        grid = [[(ef(shifted(pos0, upper, x, sign * y), z, L)[0] - e0u)
                 / area * MJ_M2 for y in uy] for x in ux]
        payload["gamma_surface"] = {"ux_A": ux.tolist(), "uy_A": uy.tolist(),
                                    "gamma_mJ_m2": grid}
        save()

    # 5. thicker slab
    payload["thickness_check"] = None
    if args.check_layers:
        nl = args.layers + args.check_layers
        pos2, L2, lay2 = slab(a0, args.nx, args.ny, nl, args.vacuum)
        z2 = np.full(len(pos2), zs[0], dtype=np.uint8)
        up2 = lay2 >= nl // 2
        g2 = (ef(shifted(pos2, up2, 0.0, sign * a2 / 3), z2, L2)[0]
              - ef(pos2, z2, L2)[0]) / area * MJ_M2
        ref = payload["key"]["unrelaxed"]["isf_mJ_m2"]
        payload["thickness_check"] = {
            "layers": nl, "isf_unrelaxed_mJ_m2": g2,
            "difference_mJ_m2": g2 - ref}
        ffm.log(f"intrinsic fault, as built, with {nl} layers: {g2:.3f} "
                f"mJ/m^2 ({g2 - ref:+.3f} from {args.layers} layers)")
    save()
    ffm.log(f"done: {count[0]} evaluations in {payload['wall_s']:.0f} s")
    try:
        plot(payload, args.out)
    except Exception as err:  # the numbers are already saved
        ffm.log("no figure:", repr(err))


def plot(p, out):
    """Quick-look figure of the two lines, with the paper's ranges for Al."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid, surface = ("#0b0b0b", "#52514e", "#898781",
                                       "#e1e0d9", "#fcfcfb")
    c_built, c_relaxed = "#2a78d6", "#eb6834"
    u_isf = p["a0_A"] / math.sqrt(6)
    rng = PAPER_RANGE.get(p["element"], {})
    k = p["key"]["relaxed"]
    panels = [
        ("y", "Along [11-2], up to the intrinsic fault", 1.3 * u_isf,
         [("usf", k["usf_u_A"]), ("isf", u_isf)]),
        ("y", "Along [11-2], one full period", None, []),
        ("x", "Along [1-10], one full period", None,
         [("max110", k["max_110_u_A"])]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor=surface)
    for ax, (axis, title, xmax, refs) in zip(axes, panels):
        built, relaxed = p["unrelaxed"][axis], p["relaxed"][axis]
        if xmax:
            built = [r for r in built if r["u_A"] <= xmax]
            relaxed = [r for r in relaxed if r["u_A"] <= xmax]
        ax.plot([r["u_A"] for r in built], [r["gamma_mJ_m2"] for r in built],
                color=c_built, lw=2, label="as built")
        ax.plot([r["u_A"] for r in relaxed],
                [r["gamma_mJ_m2"] for r in relaxed], color=c_relaxed, lw=2,
                marker="o", ms=5, label="relaxed along z")
        for i, (name, u) in enumerate(refs):
            if name not in rng:
                continue
            lo, hi = rng[name]
            ax.plot([u, u], [lo, hi], color=muted, lw=7, alpha=0.45,
                    solid_capstyle="butt", zorder=1,
                    label="experiment or DFT (paper)" if i == 0 else None)
            ax.text(u, hi, f"{lo} to {hi}", fontsize=8, color=ink2,
                    va="bottom", ha="center")
        ax.margins(y=0.1)
        ax.set_facecolor(surface)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("displacement of the upper half (A)", fontsize=9,
                      color=ink2)
        ax.set_xlim(0, xmax or built[-1]["u_A"])
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=ink2, labelsize=9)
    axes[0].set_ylabel("fault energy (mJ/m$^2$)", fontsize=9, color=ink2)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper right", ncol=len(labels),
                        frameon=False, fontsize=9,
                        bbox_to_anchor=(0.995, 0.995))
    for text in legend.get_texts():
        text.set_color(ink2)
    fig.suptitle(
        f"{p['element']} (111) stacking-fault energy, "
        f"{os.path.basename(p['model_dir'].rstrip('/'))}, "
        f"a = {p['a0_A']:.3f} A", x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out + ".png", dpi=150, facecolor=surface)
    plt.close(fig)


def summary(args):
    """Markdown table of the runs next to the paper's Table 1."""
    runs = [json.load(open(f)) for f in sorted(glob.glob(args.glob))]
    runs = [r for r in runs if "relaxed" in r.get("key", {})]
    if not runs:
        raise SystemExit(f"no finished runs match {args.glob}")

    def both(r, name):
        return (f"{r['key']['relaxed'][name]:.0f} "
                f"({r['key']['unrelaxed'][name]:.0f})")

    # jumps and the fit are redone from the stored scan, so that runs made
    # before the fit avoided jumps are reported the same way
    for r in runs:
        a, e = r["eos"]["a_A"], r["eos"]["e_per_atom_eV"]
        r["jumps"] = energy_jumps(a, e)
        r["refit"] = fit_minimum(a, e, [j["index"] for j in r["jumps"]])

    rows = [
        ("Lattice constant (A)", "a", lambda r: f"{r['a0_A']:.3f}"),
        ("Bulk modulus (GPa)", "B",
         lambda r: f"{r['refit']['B_GPa']:.0f}" if r["refit"] else "-"),
        ("(111) surface energy", "surf",
         lambda r: (f"{r['surface_energy_mJ_m2']['relaxed']:.0f} "
                    f"({r['surface_energy_mJ_m2']['unrelaxed']:.0f})")),
        ("Intrinsic stacking fault", "isf", lambda r: both(r, "isf_mJ_m2")),
        ("Unstable stacking fault", "usf", lambda r: both(r, "usf_mJ_m2")),
        ("Maximum along [1-10]", "max110",
         lambda r: both(r, "max_110_mJ_m2")),
        ("Atoms on top, along [11-2]", None,
         lambda r: both(r, "on_top_mJ_m2")),
    ]
    paper = PAPER.get(runs[0]["element"], {})
    head = (["Quantity", "Exp. or DFT (paper)", "EAM (paper)",
             "COMB3 (paper)"]
            + [os.path.basename(r["model_dir"].rstrip("/")) for r in runs])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for label, name, get in rows:
        ref = paper.get(name, ("-", "-", "-"))
        print("| " + " | ".join([label, *ref] + [get(r) for r in runs]) + " |")
    print("Energies in mJ/m^2: relaxed along z, with the as-built value in "
          "parentheses. Paper: Choudhary et al., J. Phys.: Condens. Matter "
          "27, 015003 (2015), Table 1.")
    for r in runs:
        k = r["key"]["unrelaxed"]
        worst = max(row["fz_left_eV_A"] for axis in ("x", "y")
                    for row in r["relaxed"][axis])
        print(f"{os.path.basename(r['model_dir'].rstrip('/'))}: largest bend "
              f"of the as-built line {k['largest_bend_mJ_m2']:.2f} mJ/m^2 at "
              f"{k['largest_bend_u_A']:.2f} A (median "
              f"{k['median_bend_mJ_m2']:.2f}); intrinsic fault is a minimum: "
              f"{r['key']['relaxed']['isf_is_minimum']}; largest force left "
              f"after relaxing {worst:.4f} eV/A")
        for j in r["jumps"]:
            print(f"  energy jump of {j['jump_meV_per_atom']:+.2f} meV/atom "
                  f"between a = {j['a_lo_A']:.3f} and {j['a_hi_A']:.3f} A")
        if r["refit"] and abs(r["refit"]["a0_A"] - r["a0_A"]) > 0.002:
            print(f"  the fit without the jumps gives a = "
                  f"{r['refit']['a0_A']:.4f} A; the slab used "
                  f"{r['a0_A']:.4f} A")


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["run", "summary"])
    p.add_argument("--model_dir")
    p.add_argument("--out")
    p.add_argument("--glob")
    p.add_argument("--device", default="cuda")
    p.add_argument("--element", default="Al", help="FCC metal of cells.py")
    p.add_argument("--a", type=float, default=0,
                   help="lattice constant in A (default: the model's own)")
    p.add_argument("--bulk_rep", type=int, default=3)
    p.add_argument("--span", type=float, default=0.05,
                   help="lattice-constant scan, as a fraction either way")
    p.add_argument("--npts", type=int, default=41)
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--ny", type=int, default=3)
    p.add_argument("--layers", type=int, default=24,
                   help="(111) layers; the fault should be farther from "
                        "both surfaces than the model's reach")
    p.add_argument("--vacuum", type=float, default=15.0)
    p.add_argument("--step", type=float, default=0.05,
                   help="displacement step of the as-built lines, A")
    p.add_argument("--relaxed_step", type=float, default=0.1)
    p.add_argument("--fmax", type=float, default=0.005,
                   help="stop relaxing when every |F_z| is below this, eV/A")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--maxmove", type=float, default=0.05,
                   help="largest step of any atom, A")
    p.add_argument("--surface", type=int, nargs=2, default=[12, 21],
                   help="gamma-surface grid along x and y (0 0 to skip)")
    p.add_argument("--check_layers", type=int, default=12,
                   help="extra layers of the thicker slab (0 to skip)")
    args = p.parse_args()
    if args.mode == "summary":
        if not args.glob:
            p.error("summary needs --glob")
        return summary(args)
    if not (args.model_dir and args.out):
        p.error("run needs --model_dir and --out")
    global torch, ffm
    import torch
    import ff_max_atoms as ffm
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ffm.log("device", device, "| alignn", os.path.dirname(ffm.alignn.__file__))
    run(args, device)


if __name__ == "__main__":
    main()
