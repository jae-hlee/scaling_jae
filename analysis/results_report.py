"""Compile every run into one PDF: scaling plot + full result tables.

B200 runs (ALIGNN-FF energy-only, VASP) plus the ALIGNN-FF energy+forces+stress
sweeps on a GB10 and an H200 under ../{gb10,h200}/alignn_ff/ (2026-09-27).

Run from this directory: `python results_report.py` (run scaling_overview.py
first). Writes `results_report.pdf` from ../b200/alignn_ff/scaling_alignn_v6.npz and
../b200/vasp_dft/analysis/metrics.json.
"""

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root (analysis/ sits at the top level)
B200 = ROOT / "b200"
VALID_MAX_N = 442_368
PAGE = (8.5, 11)
ROWS_PER_PAGE = 30
HEADER_BG, STRIPE, FLAG, WARN = "#e8eef8", "#f6f6f4", "#fdecea", "#fdf5e1"


def new_page(pdf, title=None):
    fig = plt.figure(figsize=PAGE)
    if title:
        fig.text(0.07, 0.95, title, fontsize=14, fontweight="bold")
    return fig


def table_page(pdf, title, header, rows, flags, notes, widths):
    fig = new_page(pdf, title)
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.80])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, colWidths=widths,
                 loc="upper center", cellLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d0cfca")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(fontweight="bold")
        elif flags[r - 1]:
            cell.set_facecolor(flags[r - 1])
        elif r % 2 == 0:
            cell.set_facecolor(STRIPE)
    fig.text(0.07, 0.06, notes, fontsize=8, color="#52514e", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def last_scf(run):
    """Energy (eV) and energy change of the last electronic step in OSZICAR."""
    steps = [ln.split() for ln in open(run / "OSZICAR")
             if ln.startswith(("DAV:", "RMM:"))]
    return float(steps[-1][2]), float(steps[-1][3])


def ediff(run):
    for ln in open(run / "INCAR"):
        k, _, v = ln.partition("=")
        if k.strip().upper() == "EDIFF":
            return float(v.split()[0])
    return 1e-4  # VASP default


def main():
    d = np.load(B200 / "alignn_ff" / "scaling_alignn_v6.npz")
    n = d["natoms"]
    nl, line, inf, e = (d["times_nl"], d["times_line"],
                        d["times_inference"], d["energies"])
    total = d["times_graph"] + inf  # times_graph = nl + line
    a_rows, a_flags = [], []
    for k in range(len(n)):
        i = round((n[k] / 4) ** (1 / 3))
        a_rows.append([f"{i}×{i}×{i}", f"{int(n[k]):,}", f"{nl[k]:.4f}",
                       f"{line[k]:.4f}", f"{inf[k]:.4f}", f"{total[k]:.4f}",
                       f"{e[k]:.6f}"])
        a_flags.append(FLAG if n[k] > VALID_MAX_N else None)

    m = json.load(open(B200 / "vasp_dft" / "analysis" / "metrics.json"))
    runs = sorted(m["all_runs"], key=lambda r: (r["n"], r["ngpu"]))
    v_rows, v_flags = [], []
    for r in runs:
        ok = bool(r["scf_done"])
        e_atom, status, flag = "–", "crashed", FLAG
        if ok:
            e_tot, de = last_scf(B200 / "vasp_dft" / r["path"])
            e_atom = f"{e_tot / (2 * r['n'] ** 3):.4f}"
            if abs(de) <= ediff(B200 / "vasp_dft" / r["path"]):
                status, flag = "converged", None
            else:
                status, flag = "not converged", WARN
        v_rows.append([
            f"{r['n']}×{r['n']}×{r['n']}", f"{2 * r['n'] ** 3:,}",
            str(r["ngpu"]), str(r["nkpts"] or "–"), f"{r['nbands'] or 0:,}",
            f"{r['scf_done']}/{r['nelm_requested']}",
            f"{r['elapsed_s']:.1f}" if ok else "–",
            f"{r['elapsed_s'] / r['scf_done']:.2f}" if ok else "–",
            e_atom, status])
        v_flags.append(flag)

        ff_rows, ff_flags = [], []
    ff_meta = {}
    for tag, path in [("GB10", ROOT / "gb10" / "alignn_ff" / "bench_GB10_13474.json"),
                      ("H200", ROOT / "h200" / "alignn_ff" / "bench_NVIDIA_H200_NVL_913733.json")]:
        j = json.load(open(path))
        ff_meta[tag] = j
        for r in j["results"]:
            ff_rows.append([tag, f"{r['i']}×{r['i']}×{r['i']}", f"{r['natoms']:,}",
                            f"{r['t_median_s']:.3f}", f"{r['mem_alloc_gb']:.1f}",
                            f"{r['mb_per_atom']:.2f}", f"{r['e_per_atom_eV']:.4f}"])
            ff_flags.append(None)
        if j["oom"]:
            o = j["oom"]
            ff_rows.append([tag, f"{o['i']}×{o['i']}×{o['i']}", f"{o['natoms']:,}",
                            "OOM", "–", "–", "–"])
            ff_flags.append(FLAG)

    with PdfPages(HERE / "results_report.pdf") as pdf:
        # page 1: summary + scaling plot
        fig = new_page(pdf)
        fig.text(0.07, 0.94, "GPU scaling runs: ALIGNN-FF and VASP",
                 fontsize=17, fontweight="bold")
        n_ok = sum(row[-1] != "crashed" for row in v_rows)
        summary = (
            "B200 runs: NVIDIA B200 GPUs (SLURM partition b200, QOS blackwell_test).\n\n"
            f"ALIGNN-FF, energy only (no forces), Cu FCC, 1 B200: {len(n)} supercells, 1×1×1 to 58×58×58 "
            f"({int(n[0])} to {int(n[-1]):,} atoms).\n"
            f"   Energies correct up to {VALID_MAX_N:,} atoms (48×48×48); larger sizes "
            "are affected by\n   the float32 drift and are shaded red in the table.\n\n"
            f"VASP, Si diamond: {len(runs)} runs over {len({r['n'] for r in runs})} "
            f"supercell sizes (54 to 8,192 atoms); {n_ok} completed,\n"
            f"   {len(runs) - n_ok} crashed before the first SCF cycle (15×15×15 and 16×16×16).\n\n"
            "ALIGNN-FF with energy + forces + stress (added 2026-09-27; the MD workload), Si "
            "diamond, 1 GPU,\n   default matpes_r2scan checkpoint (hidden 128, smooth cutoff, "
            "52 neighbours), same script on both cards:\n"
            f"   GB10 (121.7 GiB): largest cell {ff_meta['GB10']['results'][-1]['natoms']:,} atoms "
            f"at {ff_meta['GB10']['results'][-1]['mem_alloc_gb']:.0f} GB, OOM at "
            f"{ff_meta['GB10']['oom']['natoms']:,};\n   H200 NVL (139.8 GiB): "
            f"{ff_meta['H200']['results'][-1]['natoms']:,} atoms at "
            f"{ff_meta['H200']['results'][-1]['mem_alloc_gb']:.0f} GB, OOM at "
            f"{ff_meta['H200']['oom']['natoms']:,}.\n"
            "   Memory 1.95 MB/atom on both; the H200 is 4× faster than the GB10 at every size.")
        fig.text(0.07, 0.90, summary, fontsize=10, va="top", linespacing=1.4)
        ax = fig.add_axes([0.07, 0.10, 0.86, 0.50])
        ax.imshow(mpimg.imread(HERE / "scaling_overview.png"))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        a_head = ["Supercell", "Atoms", "Neighbor list (s)", "Graph (s)",
                  "Inference (s)", "Total (s)", "Model output (eV/atom)"]
        a_note = ("ALIGNN-FF v6 (scale5b_v6.py), one B200, float32. Times are wall "
                  "time for one energy evaluation.\nThe first row includes one-time GPU "
                  "warm-up. Red rows (N > 442,368): the model output is wrong "
                  "because of the\nfloat32 drift; the float64 reference is 0.604015 eV/atom.")
        pages = range(0, len(a_rows), ROWS_PER_PAGE)
        for p, s in enumerate(pages, 1):
            table_page(pdf, f"ALIGNN-FF results ({p}/{len(pages)})", a_head,
                       a_rows[s:s + ROWS_PER_PAGE], a_flags[s:s + ROWS_PER_PAGE],
                       a_note, [0.13, 0.13, 0.15, 0.12, 0.13, 0.12, 0.17])

        f_head = ["GPU", "Supercell", "Atoms", "Time (s)", "Peak mem (GB)",
                  "MB / atom", "Model output (eV/atom)"]
        f_note = ("ALIGNN-FF energy + forces + stress single points (bench_ff_scaling.py, "
                  "default matpes_r2scan checkpoint, md5 92cfe295), Si diamond 8-atom\n"
                  "conventional cell repeated i×i×i, float32. Time = median of three "
                  "calls including the graph build; memory = peak allocated on the GPU.\n"
                  "Red row = the first size that did not fit. GB10 = atomgptlab job 13474; "
                  "H200 NVL = Skipjack job 913733 (node gh204). Both 2026-09-27.")
        table_page(pdf, "ALIGNN-FF with forces and stress: GB10 and H200", f_head,
                   ff_rows, ff_flags, f_note, [0.09, 0.13, 0.13, 0.12, 0.15, 0.12, 0.2])

        v_head = ["Supercell", "Atoms", "GPUs", "k-points", "Bands",
                  "SCF cycles", "Elapsed (s)", "s / SCF", "E (eV/atom)", "Status"]
        v_note = ("VASP GPU build, single-point SCF. 3×3×3 to 6×6×6: NELM=20, EDIFF=1E-6. "
                  "10×10×10: NELM=5, ALGO=Fast.\n12×12×12 and larger: NELM=2, ALGO=Fast. "
                  "SCF cycles = completed / maximum allowed (NELM).\n"
                  "E = energy of the last SCF cycle / atom count. Converged = last energy "
                  "change below the run's EDIFF.\n"
                  "Amber rows were stopped early for timing; their E is an unconverged "
                  "intermediate value, not a result.\n"
                  "Red rows crashed before the first SCF cycle (likely out of memory).")
        table_page(pdf, "VASP results", v_head, v_rows, v_flags, v_note,
                   [0.11, 0.08, 0.06, 0.08, 0.08, 0.10, 0.10, 0.09, 0.11, 0.14])

    print(f"wrote {HERE / 'results_report.pdf'}")


if __name__ == "__main__":
    main()
