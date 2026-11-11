"""Per-atom energy maps of the polycrystals and the gate stack, from the
small files that gb10/alignn_ff_max_atoms/energy_maps.py writes on the
cluster (copied into gb10/alignn_ff_max_atoms/results/matpes_r2scan_paper/maps/).

    gb10_energy_maps.png  (a) relaxed 8-grain Si polycrystal, atom by atom
                          (b) 10^9-atom Si polycrystal, as built
                          (c) Si / SiO2 / Al gate stack, x-z cross-section, zoomed on the oxide
Each atom's energy is shown relative to the median of its material in the
slab, so grain boundaries and interfaces appear dark. Panels whose data do
not exist yet are left out.

Run from inside analysis/. Needs numpy and matplotlib only.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
MAPS = Path(os.environ.get("MAPS_DIR") or (
    HERE.parent / "gb10" / "alignn_ff_max_atoms" / "results"
    / "matpes_r2scan_paper" / "maps"))
INK, INK2 = "#1f1f1e", "#5f5e58"
# one-hue sequential ramp (light = bulk-like, dark = high energy); missing
# pixels stay the surface color
RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
RAMP.set_bad("#ffffff")

STACK_ZOOM_X = (0.0, 175.0)  # A; interfaces are a few A wide in a 1,738 A cell
STACK_ZOOM_Z = (385.0, 450.0)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(name):
    """A maps/*.npz file as a dict, or None if it does not exist yet."""
    f = MAPS / f"{name}.npz"
    if not f.exists():
        print(f"missing {f}, panel left out")
        return None
    d = np.load(f)
    out = {k: d[k] for k in d.files}
    out["kind"] = str(out["kind"])
    out["info"] = json.loads(str(out["info"]))
    return out


def vmax_of(values):
    """Top of the color scale: 99th percentile of the positive values."""
    v = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(v, 99)) if len(v) else 1.0


def panel(ax, d, title, xlim=None, ylim=None):
    """Draw one map, optionally zoomed; returns the image or scatter for the
    colorbar."""
    x0, x1, y0, y1 = d["extent"]
    if d["kind"] == "atoms":
        de = np.clip(d["de"], 0, None)
        art = ax.scatter(d["x"], d["y"], c=de, s=4, cmap=RAMP, vmin=0,
                         vmax=vmax_of(d["de"]), linewidths=0)
    else:
        img = np.clip(d["image"], 0, None)
        art = ax.imshow(img.T, origin="lower", extent=[x0, x1, y0, y1],
                        cmap=RAMP, vmin=0, vmax=vmax_of(d["image"]),
                        interpolation="nearest")
    ax.set_xlim(*(xlim or (x0, x1)))
    ax.set_ylim(*(ylim or (y0, y1)))
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    names = "xyz"
    a1, a2 = d["info"]["plane_axes"]
    ax.set_xlabel(f"{names[a1]} (Å)")
    ax.set_ylabel(f"{names[a2]} (Å)")
    return art


def main():
    poly = load("poly_si_150k")
    big = load("poly_si_1b")
    stack = load("stack_1e8")
    top = [d for d in (poly, big) if d is not None]
    if not top and stack is None:
        raise SystemExit(f"no maps in {MAPS} yet")
    rows = (1 if top else 0) + (1 if stack is not None else 0)
    ratios = ([1] if top else []) + ([0.75] if stack is not None else [])
    fig = plt.figure(figsize=(7.2, 3.1 * rows), layout="constrained")
    gs = fig.add_gridspec(rows, 2, height_ratios=ratios)
    label = "Energy above median (eV/atom)"
    for i, d in enumerate(top):
        ax = fig.add_subplot(gs[0, i] if len(top) == 2 else gs[0, :])
        n = d["info"]["natoms"]
        if d is poly:
            title = f"(a) Relaxed, {n:,} atoms"
        else:
            title = f"(b) As built, {n:,} atoms"
        art = panel(ax, d, title)
        # inset axes follow the equal-aspect box, so the bar matches its height
        fig.colorbar(art, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]), label=label)
    if stack is not None:
        ax = fig.add_subplot(gs[rows - 1, :])
        n = stack["info"]["natoms"]
        z0, z1 = STACK_ZOOM_Z
        art = panel(ax, stack, f"(c) Gate stack, {n:,} atoms, zoom on the oxide",
                    xlim=STACK_ZOOM_X, ylim=STACK_ZOOM_Z)
        for layer in stack["info"].get("layers") or []:
            lo, hi = max(layer["z_bottom_A"], z0), min(layer["z_top_A"], z1)
            ax.text(1.01, (lo + hi) / 2,
                    layer["layer"].replace("SiO2", "SiO$_2$"), va="center",
                    fontsize=8, color=INK2,
                    transform=ax.get_yaxis_transform())
        fig.colorbar(art, cax=ax.inset_axes([1.085, 0.0, 0.018, 1.0]), label=label)
    fig.savefig(HERE / "gb10_energy_maps.png")
    print("wrote gb10_energy_maps.png")


if __name__ == "__main__":
    main()
