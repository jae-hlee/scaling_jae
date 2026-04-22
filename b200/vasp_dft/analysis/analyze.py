"""Parse VASP OUTCAR/OSZICAR files and produce scaling plots + metrics.json.

Run from this directory: `python analyze.py`. Writes:

- strong_scaling.png   — speedup and parallel efficiency vs ngpu per supercell
- time_vs_size.png     — elapsed/SCF vs N_atoms, all ngpu overlaid
- per_scf_bar.png      — per-SCF cost bar chart, small systems only
- large_systems.png    — per-SCF cost for the 10/12/14/15/16 runs
- metrics.json         — machine-readable dump of every run

Assumes the standard layout `b200/vasp_dft/<n>x<n>x<n>/<ngpu>/{INCAR,OSZICAR,OUTCAR}`.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent  # b200/vasp_dft/
OUT = Path(__file__).resolve().parent

RE_ELAPSED = re.compile(r"Elapsed time \(sec\):\s+([0-9.]+)")
RE_CPU = re.compile(r"Total CPU time used \(sec\):\s+([0-9.]+)")
RE_LOOP = re.compile(r"LOOP\+:\s+cpu time\s+([0-9.]+):\s+real time\s+([0-9.]+)")
RE_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
RE_NKPTS = re.compile(r"NKPTS\s*=\s*(\d+)")
RE_NBANDS = re.compile(r"NBANDS\s*=\s*(\d+)")
RE_NELM = re.compile(r"NELM\s*=\s*(\d+)")
RE_DAV = re.compile(r"^DAV:\s+(\d+)")


def parse_run(run_dir: Path) -> dict | None:
    outcar = run_dir / "OUTCAR"
    oszicar = run_dir / "OSZICAR"
    if not outcar.exists():
        return None

    text = outcar.read_text(errors="ignore")
    nions = int(m.group(1)) if (m := RE_NIONS.search(text)) else None
    nkpts = int(m.group(1)) if (m := RE_NKPTS.search(text)) else None
    nbands = int(m.group(1)) if (m := RE_NBANDS.search(text)) else None
    nelm = int(m.group(1)) if (m := RE_NELM.search(text)) else None

    elapsed = float(m.group(1)) if (m := RE_ELAPSED.search(text)) else None
    cpu = float(m.group(1)) if (m := RE_CPU.search(text)) else None
    loops = [(float(a), float(b)) for a, b in RE_LOOP.findall(text)]

    dav_steps = 0
    if oszicar.exists():
        for line in oszicar.read_text(errors="ignore").splitlines():
            if RE_DAV.match(line):
                dav_steps += 1

    return {
        "path": str(run_dir.relative_to(ROOT)),
        "nions": nions,
        "nkpts": nkpts,
        "nbands": nbands,
        "nelm_requested": nelm,
        "scf_done": dav_steps,
        "elapsed_s": elapsed,
        "cpu_s": cpu,
        "loop_real_s": [b for _, b in loops],
        "first_scf_s": loops[0][1] if loops else None,
        "completed": elapsed is not None,
    }


def collect_all() -> list[dict]:
    rows = []
    for size_dir in sorted(ROOT.iterdir()):
        if not size_dir.is_dir() or not re.match(r"\d+x\d+x\d+$", size_dir.name):
            continue
        n = int(size_dir.name.split("x")[0])
        for ngpu_dir in sorted(size_dir.iterdir(), key=lambda p: int(p.name)):
            if not ngpu_dir.is_dir():
                continue
            ngpu = int(ngpu_dir.name)
            row = parse_run(ngpu_dir)
            if row is None:
                continue
            row.update(n=n, ngpu=ngpu)
            rows.append(row)
    return rows


def plot_strong_scaling(rows: list[dict]) -> None:
    small = [r for r in rows if r["n"] in (3, 4, 5, 6) and r["completed"]]
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728"}
    for n in (3, 4, 5, 6):
        pts = sorted([r for r in small if r["n"] == n], key=lambda r: r["ngpu"])
        if not pts:
            continue
        t1 = next((r["elapsed_s"] for r in pts if r["ngpu"] == 1), None)
        if t1 is None:
            continue
        xs = [r["ngpu"] for r in pts]
        sp = [t1 / r["elapsed_s"] for r in pts]
        eff = [s / g for s, g in zip(sp, xs)]
        natoms = 2 * n**3
        nkpts = pts[0]["nkpts"]
        label = f"{n}³ ({natoms} atoms, NKPTS={nkpts})"
        ax_s.plot(xs, sp, "o-", color=colors[n], label=label)
        ax_e.plot(xs, eff, "o-", color=colors[n], label=label)
    for ax in (ax_s, ax_e):
        ax.set_xlabel("GPUs")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
    ax_s.plot([1, 8], [1, 8], "k--", alpha=0.4, label="ideal")
    ax_s.set_ylabel("speedup vs 1 GPU")
    ax_s.set_title("Strong scaling — speedup")
    ax_s.legend(fontsize=8)
    ax_e.axhline(1.0, color="k", linestyle="--", alpha=0.4)
    ax_e.set_ylabel("parallel efficiency")
    ax_e.set_ylim(0, 1.2)
    ax_e.set_title("Strong scaling — efficiency")
    fig.tight_layout()
    fig.savefig(OUT / "strong_scaling.png", dpi=150)
    plt.close(fig)


def plot_time_vs_size(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    by_ngpu: dict[int, list[dict]] = {}
    for r in rows:
        if not r["completed"] or r["scf_done"] == 0:
            continue
        by_ngpu.setdefault(r["ngpu"], []).append(r)
    markers = {1: "o", 2: "s", 4: "^", 8: "D"}
    for ngpu in sorted(by_ngpu):
        pts = sorted(by_ngpu[ngpu], key=lambda r: r["nions"])
        xs = np.array([r["nions"] for r in pts])
        ys = np.array([r["elapsed_s"] / r["scf_done"] for r in pts])
        ax.loglog(xs, ys, marker=markers[ngpu], label=f"{ngpu} GPU", linewidth=1.3)
    # annotate slopes on the large-system 8-GPU pair
    ax.set_xlabel("N_atoms")
    ax.set_ylabel("elapsed / SCF cycle (s)")
    ax.set_title("VASP SCF cost vs system size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "time_vs_size.png", dpi=150)
    plt.close(fig)


def plot_per_scf_bar(rows: list[dict]) -> None:
    small = [r for r in rows if r["n"] in (3, 4, 5, 6) and r["completed"]]
    sizes = [3, 4, 5, 6]
    ngpus = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    x = np.arange(len(sizes))
    for j, g in enumerate(ngpus):
        ys = []
        for n in sizes:
            r = next((r for r in small if r["n"] == n and r["ngpu"] == g), None)
            ys.append(r["elapsed_s"] / r["scf_done"] if r else 0)
        ax.bar(x + (j - 1.5) * width, ys, width, label=f"{g} GPU")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}³ ({2*n**3})" for n in sizes])
    ax.set_xlabel("supercell (atoms)")
    ax.set_ylabel("elapsed / SCF (s)")
    ax.set_title("Per-SCF cost, small-cell strong scaling")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "per_scf_bar.png", dpi=150)
    plt.close(fig)


def plot_large_systems(rows: list[dict]) -> None:
    large = [r for r in rows if r["n"] >= 10]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, ys, labels, colors = [], [], [], []
    for r in sorted(large, key=lambda r: r["nions"] or 0):
        if r["scf_done"] > 0 and r["elapsed_s"]:
            xs.append(r["nions"])
            ys.append(r["elapsed_s"] / r["scf_done"])
            labels.append(f"{r['n']}³ / {r['ngpu']}gpu\n(NELM={r['scf_done']})")
            colors.append("#1f77b4")
        else:
            xs.append((r["nions"] or 2 * r["n"] ** 3))
            ys.append(0)
            labels.append(f"{r['n']}³ / {r['ngpu']}gpu\nFAILED")
            colors.append("#d62728")
    ax.bar(range(len(xs)), ys, color=colors)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("elapsed / SCF (s, avg incl. init)")
    ax.set_title("Large-system runs (red = no SCF completed)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "large_systems.png", dpi=150)
    plt.close(fig)


def compute_metrics(rows: list[dict]) -> dict:
    small = [r for r in rows if r["n"] in (3, 4, 5, 6) and r["completed"]]
    strong = {}
    for n in (3, 4, 5, 6):
        pts = sorted([r for r in small if r["n"] == n], key=lambda r: r["ngpu"])
        t1 = next((r["elapsed_s"] for r in pts if r["ngpu"] == 1), None)
        if t1 is None:
            continue
        strong[f"{n}x{n}x{n}"] = {
            "natoms": 2 * n**3,
            "nkpts": pts[0]["nkpts"],
            "points": [
                {
                    "ngpu": r["ngpu"],
                    "elapsed_s": r["elapsed_s"],
                    "scf_done": r["scf_done"],
                    "speedup": t1 / r["elapsed_s"],
                    "efficiency": (t1 / r["elapsed_s"]) / r["ngpu"],
                }
                for r in pts
            ],
        }
    # size scaling on the 12³→14³ pair (8 GPU)
    r12 = next((r for r in rows if r["n"] == 12 and r["ngpu"] == 8), None)
    r14 = next((r for r in rows if r["n"] == 14 and r["ngpu"] == 8), None)
    large_exponent = None
    if r12 and r14 and r12["scf_done"] and r14["scf_done"]:
        t12 = r12["elapsed_s"] / r12["scf_done"]
        t14 = r14["elapsed_s"] / r14["scf_done"]
        large_exponent = {
            "pair": "12x12x12 -> 14x14x14 @ 8 GPU",
            "atoms": [r12["nions"], r14["nions"]],
            "s_per_scf": [t12, t14],
            "exponent_p": float(np.log(t14 / t12) / np.log(r14["nions"] / r12["nions"])),
            "caveat": "NELM=2; includes init. First-vs-second SCF costs differ; treat as rough.",
        }
    failed = [r["path"] for r in rows if r["scf_done"] == 0]
    return {
        "strong_scaling": strong,
        "large_size_exponent": large_exponent,
        "failed_runs": failed,
        "all_runs": rows,
    }


def main() -> None:
    rows = collect_all()
    metrics = compute_metrics(rows)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_strong_scaling(rows)
    plot_time_vs_size(rows)
    plot_per_scf_bar(rows)
    plot_large_systems(rows)
    print(f"wrote {OUT/'metrics.json'} and 4 PNGs")


if __name__ == "__main__":
    main()
