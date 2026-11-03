"""Compile every run into one PDF: scaling plot + full result tables.

B200 runs (ALIGNN-FF energy-only, VASP) plus the ALIGNN-FF energy+forces+stress
sweeps on a GB10 and an H200 under ../{gb10,h200}/alignn_ff/ (2026-09-27), and the
GB10 maximum-atoms runs under ../gb10/alignn_ff_max_atoms/ (2026-11-01).

Run from this directory: `python results_report.py` (run scaling_overview.py and scaling_all.py
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
            if any("\n" in h for h in header):
                cell.set_height(cell.get_height() * 1.9)
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

    ff_meta, ff_tables = {}, {}
    for tag, path in [("GB10", ROOT / "gb10" / "alignn_ff" / "bench_GB10_13474.json"),
                      ("H200", ROOT / "h200" / "alignn_ff" / "bench_NVIDIA_H200_NVL_913733.json")]:
        j = json.load(open(path))
        ff_meta[tag] = j
        ff_rows, ff_flags = [], []
        for r in j["results"]:
            ff_rows.append([f"{r['i']}×{r['i']}×{r['i']}", f"{r['natoms']:,}",
                            f"{r['t_median_s']:.3f}", f"{r['mem_alloc_gb']:.1f}",
                            f"{r['mb_per_atom']:.2f}", f"{r['e_per_atom_eV']:.4f}",
                            f"{r['max_abs_force']:.1e}"])
            ff_flags.append(None)
        if j["oom"]:
            o = j["oom"]
            ff_rows.append([f"{o['i']}×{o['i']}×{o['i']}", f"{o['natoms']:,}",
                            "OOM", "–", "–", "–", "–"])
            ff_flags.append(FLAG)
        ff_tables[tag] = (ff_rows, ff_flags)

    with PdfPages(HERE / "results_report.pdf") as pdf:
        def at_size(z, n):
            i = list(z["natoms"]).index(n)
            return z["times_graph"][i], z["times_inference"][i]
        e_sets = {"B200": d}
        for tag in ("GB10", "H200"):
            fs = sorted((ROOT / tag.lower() / "alignn_ff_b200settings").glob(
                "scaling_alignn_pure_*.npz"))
            e_sets[tag] = np.load(fs[-1]) if fs else None
        def ff_t(tag, n):
            return next(r["t_median_s"] for r in ff_meta[tag]["results"] if r["natoms"] == n)
        def ff_max(tag):
            return ff_meta[tag]["results"][-1]["natoms"]
        def e_max(tag):
            z = e_sets[tag]
            return int(z["natoms"][-1]) if z is not None else None
        g256, h256, b256 = (at_size(e_sets[t], 256000) for t in ("GB10", "H200", "B200"))
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
            "ALIGNN-FF, energy only, B200 settings rerun on a GB10 and an H200 (Cu FCC,\n"
            "   same checkpoint, pure-PyTorch model; energies match the B200 within 1e-5 eV/atom up to 500 atoms):\n"
            f"   largest cell {e_max('GB10'):,} atoms (GB10) and {e_max('H200'):,} (H200). "
            f"At 256,000 atoms, inference takes\n"
            f"   {g256[1]:.2f} s (GB10), {h256[1]:.2f} s (H200) and {b256[1]:.2f} s (B200).\n\n"
            "ALIGNN-FF with energy + forces + stress (added 2026-09-27; the MD workload), Si "
            "diamond, 1 GPU,\n   then-default matpes_r2scan checkpoint (hidden 128, "
            "52 neighbors), same script on both cards:\n"
            f"   GB10 (121.7 GiB): largest cell {ff_meta['GB10']['results'][-1]['natoms']:,} atoms "
            f"at {ff_meta['GB10']['results'][-1]['mem_alloc_gb']:.0f} GB, OOM at "
            f"{ff_meta['GB10']['oom']['natoms']:,};\n   H200 NVL (139.8 GiB): "
            f"{ff_meta['H200']['results'][-1]['natoms']:,} atoms at "
            f"{ff_meta['H200']['results'][-1]['mem_alloc_gb']:.0f} GB, OOM at "
            f"{ff_meta['H200']['oom']['natoms']:,}.\n"
            "   Memory 1.95 MB/atom on both; the H200 is about 4× faster than the GB10 from ~4,000 atoms up.\n\n"
            "Added 2026-11-02, GB10 only, same model (pages 13-14): 551,368 atoms energy only, 195,112 with forces\n"
            "   and layer checkpointing, and forces on 64M atoms chunked across 5-6 GPUs.")
        fig.text(0.07, 0.90, summary, fontsize=10, va="top", linespacing=1.4)
        ax = fig.add_axes([0.07, 0.06, 0.86, 0.47])
        ax.imshow(mpimg.imread(HERE / "scaling_overview.png"))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: all seven runs on one plot
        fig = new_page(pdf, "All runs: VASP and ALIGNN-FF on B200, GB10 and H200")
        ax = fig.add_axes([0.07, 0.22, 0.86, 0.66])
        ax.imshow(mpimg.imread(HERE / "scaling_all.png"))
        ax.axis("off")
        fig.text(0.07, 0.18, "Energy only: filled markers (B200 run and its GB10/H200 reruns, "
                 "Cu, 12 neighbors). Energy + forces + stress:\nhollow markers, solid lines "
                 "(Si, 52-neighbor matpes_r2scan, the MD workload). VASP: time per SCF "
                 "cycle, Si.", fontsize=9, color="#52514e", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # page 3: which GPU for which job (numbers pulled from the data above)
        rows = [
            ["Energy only, 256,000 atoms: inference (s)",
             f"{g256[1]:.2f}", f"{h256[1]:.2f}", f"{b256[1]:.2f}"],
            ["Energy only, 256,000 atoms: graph build (s, CPU)",
             f"{g256[0]:.2f}", f"{h256[0]:.2f}", f"{b256[0]:.2f}"],
            ["Energy only, older FF on Cu: largest cell (atoms)",
             f"{e_max('GB10'):,}", f"{e_max('H200'):,}", f"{int(d['natoms'][-1]):,}"],
            ["Energy + forces + stress, 21,952 atoms (s)",
             f"{ff_t('GB10', 21952):.2f}", f"{ff_t('H200', 21952):.2f}", "not run"],
            ["E + F + S through ASE: largest cell (atoms)",
             f"{ff_max('GB10'):,}", f"{ff_max('H200'):,}", "not run"],
            ["GPU memory", "121.7 GiB (unified)", "139.8 GiB", "178.3 GiB"],
        ]
        fig = new_page(pdf, "Which GPU for which job")
        ax = fig.add_axes([0.07, 0.74, 0.86, 0.17])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=["", "GB10", "H200 NVL", "B200"],
                     colWidths=[0.46, 0.18, 0.18, 0.18], loc="upper center",
                     cellLoc="right")
        t.auto_set_font_size(False)
        t.set_fontsize(8.5)
        t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#d0cfca")
            cell.set_linewidth(0.4)
            if c == 0:
                cell._loc = "left"
                cell.get_text().set_horizontalalignment("left")
            if r == 0:
                cell.set_facecolor(HEADER_BG)
                cell.set_text_props(fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor(STRIPE)
        text = (
            "Reading the numbers\n"
            f"  - Model evaluation: the H200 is within {100 * (h256[1] / b256[1] - 1):.0f}% of the "
            f"B200 and {g256[1] / h256[1]:.0f}x faster than the GB10\n"
            "    (energy only, identical settings and checkpoint). With forces and stress the "
            f"H200 is {ff_t('GB10', 21952) / ff_t('H200', 21952):.1f}x the GB10.\n"
            "  - Graph construction runs on the host CPU, so it follows the machine, not the GPU.\n"
            "  - For one evaluation, the largest cell is set by memory: 1.95 MB per atom with forces "
            "through ASE, 0.25 MB/atom\n"
            "    energy only, the same on the GB10 and H200. Other paths and chunking go further "
            "(pages 13-14).\n\n"
            "Recommendation\n"
            "  - Long training runs: GB10. Slowest per GPU, but locally available at no cost; "
            "multi-node training works\n"
            "    (MPtrj trained on 8 GB10s).\n"
            "  - Fast turnaround: H200, ~4x the GB10 through ASE, but it is a shared, "
            "allocation-billed resource. Its 74k atoms\n"
            "    is the largest ASE cell. The GB10 reaches 195k with checkpointing and 64M "
            "chunked on 5-6 GPUs.\n"
            f"  - B200: fastest and largest on paper, but not available to this project; the "
            f"H200 comes within {100 * (h256[1] / b256[1] - 1):.0f}%\n"
            "    of it on model evaluation, so it is not worth pursuing.\n\n"
            "Caveats: single-point evaluations, not full MD or training steps, so the ratios "
            "will shift somewhat for\n"
            "real workloads. Times with forces through ASE are three model passes.")
        fig.text(0.07, 0.70, text, fontsize=9, va="top", linespacing=1.45)
        pdf.savefig(fig)
        plt.close(fig)

        # page 4: what one billion atoms would take (derived from the forces sweeps)
        big = [r for r in ff_meta["GB10"]["results"] + ff_meta["H200"]["results"]
               if r["natoms"] >= 4096]
        mb = float(np.median([r["mb_per_atom"] for r in big]))  # MB/atom with forces
        h_fit = ff_max("H200")
        usable = h_fit / (ff_meta["H200"]["gpu_memory_gb"] * 1e3 / mb)  # measured fraction
        def fit_est(gib):
            return gib * 1.073741824e3 / mb * usable
        si_density = 8 / 5.43 ** 3               # Si atoms per A^3
        halo = 20.0                              # A: ~4 message-passing hops x 5 A cutoff
        def halo_factor(n):
            side = (n / si_density) ** (1 / 3)
            return ((side + 2 * halo) / side) ** 3
        gpus = [("GB10", ff_max("GB10"), "measured"),
                ("H200 NVL", h_fit, "measured"),
                ("MI250X GCD (Frontier)", fit_est(64.0), "memory rule")]
        rows = []
        for name, n, how in gpus:
            f = halo_factor(n)
            n_naive = 1e9 / n
            n_halo = n_naive * f
            extra = f"  (= {n_halo / 8:,.0f} nodes)" if "GCD" in name else ""
            rows.append([name, f"{n:,.0f} ({how})", f"{n_naive:,.0f}", f"{f:.1f}x",
                         f"{n_halo:,.0f}{extra}"])
        fig = new_page(pdf, "Scaling to one billion atoms (with forces, through ASE)")
        ax = fig.add_axes([0.07, 0.76, 0.86, 0.14])
        ax.axis("off")
        t = ax.table(cellText=rows,
                     colLabels=["GPU", "Atoms per GPU", "GPUs, no halo",
                                "Halo factor", "GPUs with halo"],
                     colWidths=[0.22, 0.22, 0.14, 0.12, 0.30], loc="upper center",
                     cellLoc="right")
        t.auto_set_font_size(False)
        t.set_fontsize(8.5)
        t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#d0cfca")
            cell.set_linewidth(0.4)
            if c == 0:
                cell._loc = "left"
                cell.get_text().set_horizontalalignment("left")
            if r == 0:
                cell.set_facecolor(HEADER_BG)
                cell.set_text_props(fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor(STRIPE)
        t_step = ff_meta["H200"]["results"][-1]["t_median_s"]
        text = (
            "The governing number\n"
            f"  - With forces the force field needs {mb:.2f} MB of GPU memory per atom through ASE on "
            "both cards measured, so 10^9 atoms\n"
            f"    is ~{mb:.1f} PB before overheads. 'Memory rule' rows apply that figure, and the "
            f"{100 * usable:.0f}% of memory the H200\n"
            "    could use, to the card's memory; they are estimates, not measurements.\n"
            "    A LAMMPS run would use forward_tensors (0.75 MB/atom measured), so these GPU counts overstate it.\n"
            "  - ALIGNN is a message-passing network, so when the crystal is split across GPUs each "
            "GPU also holds a\n"
            f"    ghost shell as thick as the receptive field (~{halo:.0f} A here: 2+2 layers x 5 A). "
            "Around a ~70k-atom Si\n"
            "    block that multiplies the atoms each GPU holds by the halo factor. As-is, "
            "the GCD count exceeds\n"
            "    Frontier (9,408 nodes).\n\n"
            "Levers, largest first\n"
            "  - Evaluation path (measured 2026-11-01, same model): 0.195 MB/atom energy only, 0.75 with "
            "forces without the\n"
            "    second-order graph, 0.53 with layer checkpointing. The second-order graph dominates "
            "the 1.95 MB, not the line graph.\n"
            "  - A leaner force field and half precision could cut memory further (unmeasured, and "
            "accuracy must be re-checked).\n"
            "  - Fewer message-passing layers shrink the halo: 20 A -> 10 A takes the factor from "
            "~2.5x to ~1.6x.\n"
            "  - Memory per GPU matters more than speed when the whole cell is held at once. Chunking "
            "trades memory for time\n"
            "    (measured to 64M atoms, pages 13-14).\n\n"
            "Software\n"
            "  - A run this size needs LAMMPS domain decomposition with ghost-atom exchange. "
            "ALIGNN has a LAMMPS\n"
            "    interface (pair_alignn, TorchScript); multi-layer ghost handling and an AMD build "
            "are not yet verified.\n"
            "  - The path is a ladder: 1 GPU -> 1 node -> 64 nodes -> thousands, fixing scaling "
            "at each step.\n\n"
            "Time, if it fits\n"
            f"  - At ~{t_step:.1f} s per step per GPU (H200 at its largest cell, three model passes through ASE),\n    1 ps at 1 fs "
            f"steps is ~{1000 * t_step / 60:.0f} min of wall time\n"
            "    plus communication; at thousands of nodes, every hour costs thousands of "
            "node-hours.\n\n"
            "First steps (cheap, one GB10): memory per atom with forces for a leaner model "
            "(fewer neighbors, fewer\n"
            "layers, half precision), and pair_alignn on 2 GPUs with a real ghost region.")
        fig.text(0.07, 0.71, text, fontsize=8.8, va="top", linespacing=1.42)
        pdf.savefig(fig)
        plt.close(fig)

        a_head = ["Supercell", "Atoms", "Neighbor list (s)", "Graph (s)",
                  "Inference (s)", "Total (s)", "Model output\n(eV/atom)"]
        a_note = ("ALIGNN-FF v6 (scale5b_v6.py), one B200, float32. Times are wall "
                  "time for one energy evaluation.\nThe first row includes one-time GPU "
                  "warm-up. Red rows (N > 442,368): the model output is wrong "
                  "because of the\nfloat32 drift; the float64 reference is 0.604015 eV/atom.")
        pages = range(0, len(a_rows), ROWS_PER_PAGE)
        for p, s in enumerate(pages, 1):
            table_page(pdf, f"ALIGNN-FF results, energy only, B200 ({p}/{len(pages)})", a_head,
                       a_rows[s:s + ROWS_PER_PAGE], a_flags[s:s + ROWS_PER_PAGE],
                       a_note, [0.13, 0.13, 0.15, 0.12, 0.13, 0.12, 0.17])

        # GB10 and H200 reruns of the B200 settings (Cu FCC, energy only), in the
        # B200 table's format so the three GPUs can be read row against row
        c_head = ["Supercell", "Atoms", "Graph (s)", "Inference (s)", "Total (s)",
                  "Peak mem\n(GB)", "Model output\n(eV/atom)"]
        f_head = ["Supercell", "Atoms", "Time (s)", "Peak mem\n(GB)", "MB/atom",
                  "Energy\n(eV/atom)", "Max |F|\n(eV/A)"]
        ff_jobs = {"GB10": "atomgptlab job 13474", "H200": "Skipjack job 913733"}
        c_jobs = {"GB10": ("NVIDIA GB10", "atomgptlab job 13484"),
                  "H200": ("NVIDIA H200 NVL", "Skipjack job 920089")}
        for tag, (gpu_name, job) in c_jobs.items():
            z = e_sets[tag]
            if z is None:
                continue
            rows, flags = [], []
            for n, tg, ti, e, mem in zip(z["natoms"], z["times_graph"],
                                         z["times_inference"], z["energies"],
                                         z["peak_mem_gb"]):
                k = round((int(n) / 4) ** (1 / 3))
                rows.append([f"{k}×{k}×{k}", f"{int(n):,}", f"{tg:.4f}", f"{ti:.4f}",
                             f"{tg + ti:.4f}", f"{mem:.1f}", f"{e:.6f}"])
                flags.append(FLAG if n > VALID_MAX_N else None)
            last = int(z["natoms"][-1])
            k_next = round((last / 4) ** (1 / 3)) + 1
            c_note = (f"B200 settings rerun on one {gpu_name}: Cu FCC, 5 A cutoff, 12 neighbors, "
                      "energy only, float32, checkpoint v12.2.2024_dft_3d_307k,\n"
                      "loaded into the pure-PyTorch ALIGNN (matches the B200 DGL energies within "
                      "1e-5 eV/atom up to 500 atoms). One call per size; the first row includes\n"
                      "GPU warm-up. Largest size shown is the largest that completed; the next "
                      f"({k_next}×{k_next}×{k_next} = {4 * k_next ** 3:,} atoms) did not fit. "
                      f"Red rows (N > {VALID_MAX_N:,}):\n"
                      "float32 energy drift, as on the B200. The port has no float64 readout, so smaller cells "
                      "also drift slightly (to ~0.600).\nTimings are unaffected. "
                      f"{job}, 2026-09-28.")
            pages_c = range(0, len(rows), ROWS_PER_PAGE)
            for pno, st in enumerate(pages_c, 1):
                suffix = f" ({pno}/{len(pages_c)})" if len(pages_c) > 1 else ""
                table_page(pdf, f"ALIGNN-FF results, energy only, {tag}{suffix}", c_head,
                           rows[st:st + ROWS_PER_PAGE], flags[st:st + ROWS_PER_PAGE],
                           c_note, [0.13, 0.13, 0.12, 0.13, 0.12, 0.13, 0.17])

            # same GPU, energy + forces + stress (Si, matpes_r2scan)
            f_rows, f_flags = ff_tables[tag]
            j = ff_meta[tag]
            f_note = (f"Energy + forces + stress single point on one {gpu_name}: Si diamond "
                      "(8-atom conventional cell, i×i×i), then-default matpes_r2scan\n"
                      "checkpoint (hidden 128, 2+2 layers, 5 A cutoff, 52 neighbors), "
                      f"ASE calculator, float32. Time = median of {j['repeats']} calls, each three model passes.\n"
                      "Peak mem = peak allocated GPU memory. Max |F| should be ~0 for the "
                      "perfect crystal (checks the forces).\n"
                      "Red row: the first size that ran out of GPU memory. "
                      f"{ff_jobs[tag]}, 2026-09-27.")
            table_page(pdf, f"ALIGNN-FF results with forces, {tag}", f_head, f_rows,
                       f_flags, f_note, [0.13, 0.13, 0.12, 0.14, 0.12, 0.17, 0.17])

        # GB10 maximum atoms: every evaluation path, then chunked runs
        ma = ROOT / "gb10" / "alignn_ff_max_atoms" / "results"
        ma_name = {"matpes_r2scan_paper": "AFF-r2SCAN"}
        ma_models = ["matpes_r2scan_paper", "alignn_ff_mptrj", "alignn_ff_matpes_pbe",
                     "alignn_ff_matpes_r2scan"]
        ma_paths = [("e", "energy only"), ("ase", "E+F+S, ASE"), ("ef", "E+F+S, tensors"),
                    ("ckpt", "E+F+S, tensors + ckpt")]
        s_rows = []
        for m in ma_models:
            for key, label in ma_paths:
                res = json.loads(next(ma.glob(f"{m}/sweep_{key}_*.json")).read_text())["results"]
                ref = res[1]["e_per_atom"]
                dev = max(abs(r["e_per_atom"] - ref) for r in res)
                last = res[-1]
                s_rows.append([ma_name.get(m, m), label, f"{last['natoms']:,}", f"{last['mem_alloc_gb']:.1f}",
                               f"{last['mb_per_atom']:.3f}", f"{last['t_total_s']:.1f}", f"{dev:.1e}"])
        s_note = ("Largest pristine Si cell (8 i^3 atoms) evaluated on one GB10, one process per path (2026-11-01,\n"
                  "jobs 19126-19129). The next size was killed by the host (exit 137). Time = median of 3 calls\n"
                  "(through ASE, each call is three model passes). "
                  "Tensors = forward_tensors (the LAMMPS path, no second-order graph).\nCkpt = layer checkpointing. "
                  "Max dE = largest change in energy per atom from the small-cell value over the sweep.\n"
                  "Not comparable to the Cu energy-only runs (older model, 12 neighbors).")
        table_page(pdf, "ALIGNN-FF largest single evaluation, GB10",
                   ["Model", "Path", "Atoms", "Peak mem\n(GB)", "MB/atom", "Time (s)", "Max dE\n(eV/atom)"],
                   s_rows, [None] * len(s_rows), s_note, [0.27, 0.22, 0.11, 0.10, 0.10, 0.10, 0.10])
        m_rows = []
        for m, tag, atoms, gpus, what in [
                ("matpes_r2scan_paper", "s2_8M_energy", "8M", 1, "energy"),
                ("matpes_r2scan_paper", "s2_8M_forces", "8M", 1, "forces"),
                ("matpes_r2scan_paper", "s3_64M_energy_reduced", "64M", 4, "energy"),
                ("matpes_r2scan_paper", "s2_64M_forces_reduced", "64M", 5, "forces"),
                ("alignn_ff_mptrj", "s2_8M_energy_h20", "8M", 1, "energy"),
                ("alignn_ff_mptrj", "s2_8M_forces_h20", "8M", 1, "forces"),
                ("alignn_ff_mptrj", "s3_64M_energy_h20_reduced", "64M", 4, "energy"),
                ("alignn_ff_mptrj", "s2_64M_forces_h20_reduced", "64M", 6, "forces"),
                ("alignn_ff_matpes_pbe", "s2_8M_energy_h20", "8M", 1, "energy"),
                ("alignn_ff_matpes_pbe", "s2_8M_forces_h20", "8M", 1, "forces"),
                ("alignn_ff_matpes_r2scan", "s2_8M_energy_h20", "8M", 1, "energy"),
                ("alignn_ff_matpes_r2scan", "s2_8M_forces_h20", "8M", 1, "forces")]:
            j = json.loads((ma / m / f"{tag}.json").read_text())
            pieces = j.get("pieces") or []
            piece = j.get("piece_atoms_max") or max(p["atoms"] for p in pieces)
            mem = j.get("mem_alloc_gb_max") or max(p.get("mem_alloc_gb") or 0 for p in pieces)
            wall = j.get("wall_s_slowest_rank") or j["wall_s"]
            m_rows.append([ma_name.get(m, m), atoms, str(gpus), what, f"{j['k'] ** 3:,}", f"{piece:,}",
                           f"{mem:.1f}", f"{wall:,.0f}", f"{j['e_per_atom']:.5f}"])
        m_note = ("Rattled Si (0.05 A), energy and forces, no stress (2026-11-01/02, jobs 19140-19200). Each cube is\n"
                  "evaluated with a halo on one GPU. Force runs use layer checkpointing. Time = slowest task. The AFF-r2SCAN\n"
                  "64M force run was 6 tasks on 5 nodes (one node ran two in turn, 2,689 s), and the 64M energy runs 6 tasks on 4.\n"
                  "These are the largest sizes tried: none\n"
                  "reached a memory limit. AFF-r2SCAN: exact 20.01 A halo. alignn_ff_* models: 20 A halo instead of the\n"
                  "exact 40 A, maximum force error 2e-9 to 2e-8 eV/A in a float64 check on 8,000 atoms. Tasks started\n"
                  "as nodes freed up, so the 64M force runs took 45 min (AFF-r2SCAN) and 1 h 59 min (alignn_ff_mptrj) from first start to\n"
                  "last finish.")
        table_page(pdf, "ALIGNN-FF chunked runs, GB10",
                   ["Model", "Atoms", "GB10s", "Computes", "Cubes", "Largest\npiece", "Peak mem\n(GB)", "Time (s)", "E (eV/atom)"],
                   m_rows, [None] * len(m_rows), m_note, [0.27, 0.07, 0.07, 0.10, 0.08, 0.10, 0.10, 0.09, 0.12])

        v_head = ["Supercell", "Atoms", "GPUs", "k-points", "Bands",
                  "SCF cycles", "Elapsed (s)", "s / SCF", "E (eV/atom)", "Status"]
        v_note = ("VASP GPU build, single-point SCF. 3×3×3 to 6×6×6: NELM=20, EDIFF=1E-6. "
                  "10×10×10: NELM=5, ALGO=Fast.\n12×12×12 and larger: NELM=2, ALGO=Fast. "
                  "SCF cycles = completed / maximum allowed (NELM).\n"
                  "E = energy of the last SCF cycle / atom count. Converged = last energy "
                  "change below the run's EDIFF.\n"
                  "Amber rows were stopped early for timing; their E is an unconverged "
                  "intermediate value, not a result.\n"
                  "Red rows crashed before the first SCF cycle (MPI segfault, not out of memory).")
        table_page(pdf, "VASP results", v_head, v_rows, v_flags, v_note,
                   [0.11, 0.08, 0.06, 0.08, 0.08, 0.10, 0.10, 0.09, 0.11, 0.14])

    print(f"wrote {HERE / 'results_report.pdf'}")


if __name__ == "__main__":
    main()
