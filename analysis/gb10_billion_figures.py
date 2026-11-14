"""Figures for the GB10 billion-atom manuscript, from the committed JSONs in
../gb10/alignn_ff_max_atoms/results/.

    gb10_decomposition.png  schematic of the cube + halo decomposition
    gb10_scaling.png        (a) GB10-seconds per evaluation and (b) GPU memory
                            against atoms, AFF-r2SCAN, single graph and chunked
    gb10_halo.png           largest force change against halo width, float64,
                            8,000 atoms of rattled Si, all four models

Run from inside analysis/. Needs numpy and matplotlib only. GB10_RESULTS
overrides the results folder and FIG_OUT the output folder (for running on
the cluster with a copy of the repo's results). GB10_MODEL=smooth draws the
scaling and halo figures for AFF-r2SCAN-smooth instead (stage 16 sweeps, 8M
and 64M runs, stage 15 billion-atom runs, stage 16 halo scan) and writes
them as gb10_scaling_smooth.png and gb10_halo_smooth.png.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("GB10_RESULTS")
           or HERE.parent / "gb10" / "alignn_ff_max_atoms" / "results")
OUT = Path(os.environ.get("FIG_OUT") or HERE)
SMOOTH = os.environ.get("GB10_MODEL", "") == "smooth"
SUFFIX = "_smooth" if SMOOTH else ""
# 2+2 model of the scaling figure: results folder of its chunked runs, folder
# of its single-graph sweeps, chunked runs with forces and energy only
# (8M, 64M, 1B), and its label in the halo figure
if SMOOTH:
    MAIN = ("aff_smooth", "aff_smooth/stage16",
            ("stage16/s16_8M_forces", "stage16/s16_64M_forces",
             "stage15/s15_1B_si_forces"),
            ("stage16/s16_8M_energy", "stage16/s16_64M_energy",
             "stage15/s15_1B_si_energy"),
            "AFF-r2SCAN-smooth (2+2, reach 20 Å)")
else:
    MAIN = ("matpes_r2scan_paper", "matpes_r2scan_paper",
            ("s2_8M_forces", "s2_64M_forces", "s4_1B_forces"),
            ("s2_8M_energy", "s3_64M_energy", "s4_1B_energy"),
            "AFF-r2SCAN (2+2, reach 20 Å)")

# categorical slots 1-4 (validated: lightness, chroma, CVD and normal-vision
# separation pass on a white surface); text stays in ink colors
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d3"
# behind labels inside an axes, so grid lines and points never cross the text
LABEL_BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(path):
    with open(path) as f:
        return json.load(f)


def sweep(model, path):
    (f,) = glob.glob(str(RES / model / f"sweep_{path}_*.json"))
    r = load(f)["results"]
    n = np.array([x["natoms"] for x in r], float)
    t = np.array([x["t_total_s"] for x in r])
    mem = np.array([x["mem_alloc_gb"] for x in r])
    return n, t, mem


def chunked(model, stem):
    """Atoms, total GB10-seconds over all tasks, largest piece GPU memory."""
    single = RES / model / f"{stem}.json"
    files = [single] if single.exists() else sorted(
        glob.glob(str(RES / model / f"{stem}_rank*.json")))
    ds = [load(f) for f in files]
    gpu_s = sum(d["wall_s"] for d in ds)
    mem = max(p["mem_alloc_gb"] for d in ds for p in d["pieces"])
    return ds[0]["natoms"], gpu_s, mem, len(ds)


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- scaling
def fig_scaling():
    m, msweep, stems_f, stems_e, _ = MAIN
    paths = [  # (sweep path, label, color, marker)
        ("ase", "E+F+S, ASE calculator", YELLOW, "v"),
        ("ef", "E+F+S, forward_tensors", BLUE, "o"),
        ("ckpt", "E+F+S, layer checkpointing", ORANGE, "s"),
        ("e", "energy only", AQUA, "^"),
    ]
    chunks_f = [chunked(m, s) for s in stems_f]
    chunks_e = [chunked(m, s) for s in stems_e]
    cap = load(glob.glob(str(RES / msweep / "sweep_e_*.json"))[0])["gpu_memory_gb"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for path, label, color, mk in paths:
        n, t, mem = sweep(msweep, path)
        a.plot(n, t, "-", color=color, marker=mk, ms=3.5, label=label)
        b.plot(n, mem, "-", color=color, marker=mk, ms=3.5, label=label)
    for rows, color, mk, label in (
            (chunks_f, ORANGE, "D", "forces, chunked"),
            (chunks_e, AQUA, "D", "energy only, chunked")):
        n = [r[0] for r in rows]
        a.plot(n, [r[1] for r in rows], "--", color=color, marker=mk, ms=5,
               mec="white", mew=0.8, label=label)
        b.plot(n, [r[2] for r in rows], "--", color=color, marker=mk, ms=5,
               mec="white", mew=0.8, label=label)
    n1b, s1b, _, tasks = chunks_f[-1]
    # below the chunked lines and right of the single-graph curves, which end
    # near 5e5 atoms and 10 s; centered so the text stays inside the axes
    a.annotate(f"$10^9$ atoms with forces:\n{s1b / 3600:.0f} GB10-hours",
               xy=(n1b, s1b), xytext=(3e7, 4), fontsize=8, color=INK,
               ha="center", va="top", bbox=LABEL_BOX,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    a.set(xscale="log", yscale="log", xlabel="Atoms",
          ylabel="GB10-seconds per evaluation")
    a.set_xlim(30, 3e9)
    a.text(0.02, 0.97, "(a)", transform=a.transAxes, va="top",
           fontweight="bold")
    b.axhline(cap, color=INK2, ls=":", lw=1)
    b.text(2e5, cap * 1.03, f"GB10 memory, {cap:.0f} GB", fontsize=8,
           color=INK2, va="bottom")
    b.set(xscale="log", xlabel="Atoms", ylabel="Peak GPU memory (GB)")
    b.set_xlim(30, 3e9)
    b.set_ylim(0, 150)
    b.text(0.02, 0.97, "(b)", transform=b.transAxes, va="top",
           fontweight="bold")
    for ax in (a, b):
        style(ax)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / f"gb10_scaling{SUFFIX}.png")
    plt.close(fig)


# ------------------------------------------------------------------- halo
def fig_halo():
    models = [
        (MAIN[1], MAIN[4], BLUE, "o"),
        ("alignn_ff_mptrj", "alignn_ff_mptrj (4+4, reach 40 Å)", ORANGE,
         "s"),
        ("alignn_ff_matpes_pbe", "alignn_ff_matpes_pbe (4+4, reach 40 Å)",
         AQUA, "^"),
        ("alignn_ff_matpes_r2scan",
         "alignn_ff_matpes_r2scan (4+4, reach 40 Å)", YELLOW, "v"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    f32 = []
    for m, label, color, mk in models:
        rows = sorted((d["halo_A"], d["validation"]["max_abs_df"])
                      for d in map(load, glob.glob(str(RES / m / "halo_*.json"))))
        h, df = zip(*rows)
        ax.plot(h, np.maximum(df, 1e-14), "-", color=color, marker=mk, ms=4,
                label=label)
        (cf,) = glob.glob(str(RES / m / "check_forces_*.json"))
        f32.append(load(cf)["validation"]["max_abs_df"])
    ax.axhspan(min(f32), max(f32), color=GRID, alpha=0.6, lw=0)
    ax.text(30, max(f32) * 1.8, "float32 roundoff, exact halo", fontsize=8,
            color=INK2, ha="right", bbox=LABEL_BOX)
    ax.axvline(20, color=INK2, ls=":", lw=1)
    # left of the 20 A line and above the band: every model is below 1e-4
    # there, so no line crosses the text
    ax.text(19.6, 0.06, "20 Å halo, used\nfor chunked runs", fontsize=8,
            color=INK2, ha="right", va="top", bbox=LABEL_BOX)
    ax.set(yscale="log", xlabel="Halo width (Å)",
           ylabel="Largest force change (eV/Å)")
    ax.set_ylim(3e-14, 0.1)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"gb10_halo{SUFFIX}.png")
    plt.close(fig)


# ---------------------------------------------------------- decomposition
def fig_decomposition():
    rng = np.random.default_rng(0)
    L, k = 12.0, 4
    c = L / k
    h = 1.1
    pts = rng.uniform(0, L, size=(1400, 2))
    lo = np.array([1, 1]) * c
    hi = lo + c
    gap = np.maximum(np.maximum(lo - pts, pts - hi), 0)
    dist = np.sqrt((gap**2).sum(1))
    own = dist == 0
    halo = (dist > 0) & (dist <= h)
    rest = ~(own | halo)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i in range(k + 1):
        ax.plot([i * c, i * c], [0, L], color=GRID, lw=0.8, zorder=1)
        ax.plot([0, L], [i * c, i * c], color=GRID, lw=0.8, zorder=1)
    ax.add_patch(FancyBboxPatch(lo, c, c, boxstyle=f"round,pad={h}",
                                fc="#fbe3d8", ec=ORANGE, lw=1.2, zorder=0))
    ax.add_patch(Rectangle(lo, c, c, fc="#d7e6f8", ec=BLUE, lw=1.4, zorder=0))
    ax.scatter(*pts[rest].T, s=3, color="#b8b7b0", lw=0, zorder=2)
    ax.scatter(*pts[halo].T, s=7, color=ORANGE, lw=0, zorder=3,
               label="halo atoms: in the graph, energies not counted")
    ax.scatter(*pts[own].T, s=7, color=BLUE, lw=0, zorder=3,
               label="owned atoms: energies counted")
    ax.annotate("", xy=(hi[0] + h, hi[1] - 0.5), xytext=(hi[0], hi[1] - 0.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    dot_box = dict(boxstyle="round,pad=0.12", fc="white", ec="none")
    ax.text(hi[0] + h / 2, hi[1] - 0.35, "h", ha="center", va="bottom",
            fontsize=9, style="italic", bbox=dot_box, zorder=5)
    ax.annotate("", xy=(lo[0], lo[1] - 0.35), xytext=(hi[0], lo[1] - 0.35),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text((lo[0] + hi[0]) / 2, lo[1] - 0.5, "c", ha="center", va="top",
            fontsize=9, style="italic", bbox=dot_box, zorder=5)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False,
              fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "gb10_decomposition.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_scaling()
    fig_halo()
    fig_decomposition()
    print(f"wrote gb10_scaling{SUFFIX}.png, gb10_halo{SUFFIX}.png, "
          "gb10_decomposition.png")
