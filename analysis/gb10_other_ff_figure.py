"""ALIGNN-FF (AFF-r2SCAN-smooth) next to the force-field families of the
large-scale records (SNAP, NEP, Allegro, Deep Potential) on one NVIDIA GB10,
from gb10/other_ff_max_atoms/results (the other models' size sweeps) and
gb10/alignn_ff_max_atoms/results (stage 16 sweeps, stage 15 2B-atom run).

    gb10_other_ff.png  (a) largest pristine Si cell evaluated on one GB10; for
                       chunked ALIGNN-FF the largest run, which is not a limit
                       (b) atom-steps per second with forces on one GB10

Rates are "typical": the median over the sizes from 25% of the largest up
to, but not including, the largest, which memory pressure can slow (the same
rule as summarize_cost.py). The chunked rate is the 2B-atom run's atoms over
its summed task time. The values used are printed.

Run from inside analysis/. Needs matplotlib only. OTHER_RESULTS and
GB10_RESULTS override the two results folders and FIG_OUT the output folder
(for running on the cluster).
"""
import glob
import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OTHER = Path(os.environ.get("OTHER_RESULTS")
             or HERE.parent / "gb10" / "other_ff_max_atoms" / "results")
RES = Path(os.environ.get("GB10_RESULTS")
           or HERE.parent / "gb10" / "alignn_ff_max_atoms" / "results")
OUT = Path(os.environ.get("FIG_OUT") or HERE)

# slot 1 of the validated categorical set of gb10_billion_figures.py marks
# this work; every other model is one neutral gray (identity is in the labels)
BLUE, GRAY = "#2a78d6", "#a8a7a1"
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d3"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(path):
    with open(path) as f:
        return json.load(f)


def typical(pairs, largest):
    """Median rate over sizes from 25% of the largest up to the largest."""
    rates = [r for n, r in pairs if 0.25 * largest <= n < largest and r]
    return statistics.median(rates) if rates else None


def other(model):
    """Largest size that ran and its typical atom-steps/s, from sweep.py."""
    d = load(OTHER / model / "sweep.json")
    ok = [r for r in d["rows"] if r.get("ok")]
    largest = max(r["atoms"] for r in ok)
    rate = typical([(r["atoms"], r.get("atom_steps_per_s")) for r in ok],
                   largest)
    return largest, rate


def alignn_sweep(path):
    """Largest single graph and typical atoms/s of a stage-16 sweep."""
    (f,) = glob.glob(str(RES / "aff_smooth" / "stage16" / f"sweep_{path}_*.json"))
    r = load(f)["results"]
    largest = r[-1]["natoms"]
    return largest, typical([(x["natoms"], x["natoms"] / x["t_total_s"])
                             for x in r], largest)


def alignn_chunked():
    """The 2B-atom force run: atoms, and atoms over summed task time."""
    ds = [load(f) for f in sorted(glob.glob(str(
        RES / "aff_smooth" / "stage15" / "s15_2B_si_forces_rank*.json")))]
    n = ds[0]["natoms"]
    return n, n / sum(d["wall_s"] for d in ds)


def atoms_label(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.3g} million"
    return f"{n:,}"


def rate_label(r):
    if r >= 1e6:
        return f"{r / 1e6:.3g} million"
    return f"{float(f'{r:.3g}'):,.0f}"


def main():
    rows = [  # label, (atoms, rate), this work?, hatched (not a limit)?
        ("ALIGNN-FF, chunked", alignn_chunked(), True, True),
        ("SNAP (LAMMPS)", other("snap"), False, False),
        ("NEP (GPUMD)", other("nep"), False, False),
        ("ALIGNN-FF, one graph", alignn_sweep("ckpt"), True, False),
        ("Allegro (ASE)", other("allegro"), False, False),
        ("ALIGNN-FF (ASE)", alignn_sweep("ase"), True, False),
        ("Deep Potential (Python)", other("dp"), False, False),
    ]
    print(f"{'model':26s} {'atoms':>15s} {'atom-steps/s':>14s}")
    for label, (n, r), _, _ in rows:
        print(f"{label:26s} {n:15,d} {r:14,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, col, fmt, lim, xlabel, tag in (
            (axes[0], 0, atoms_label, (1e4, 3e11),
             "Largest Si cell on one GB10 (atoms)", "(a)"),
            (axes[1], 1, rate_label, (1e3, 3e8),
             "Atom-steps per second, with forces", "(b)")):
        for yi, (label, vals, ours, hatched) in zip(y, rows):
            v = vals[col]
            ax.barh(yi, v, height=0.62, color=BLUE if ours else GRAY,
                    hatch="////" if hatched else None,
                    edgecolor="white" if hatched else "none", linewidth=0)
            ax.text(v * 1.25, yi, fmt(v), va="center", fontsize=7.5,
                    color=INK)
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontweight="bold",
                va="bottom")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], color=INK)
    fig.legend(handles=[
        Patch(color=BLUE, label="ALIGNN-FF (AFF-r2SCAN-smooth), this work"),
        Patch(color=GRAY, label="Other potentials, record settings"),
        Patch(facecolor=BLUE, hatch="////", edgecolor="white",
              label="Chunked: largest run, not a memory limit"),
    ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "gb10_other_ff.png")
    plt.close(fig)
    print(f"wrote {OUT / 'gb10_other_ff.png'}")


if __name__ == "__main__":
    main()
