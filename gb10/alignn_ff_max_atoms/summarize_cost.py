"""Tables from one stage-6 run (results/<model>/stage6): chunked against
full cell, the largest single graph per structure, and chunked throughput
with the GB10-hours one billion atoms would take at that rate.

    python summarize_cost.py RESULTS_DIR --job JOB
"""
import argparse
import glob
import json
import os
import statistics


def load(path):
    with open(path) as fh:
        return json.load(fh)


def fmt(x):
    return "n/a" if x is None else f"{x:.1e}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir")
    p.add_argument("--job", required=True)
    a = p.parse_args()
    suffix = f"_{a.job}.json"

    def files(prefix):
        return sorted(glob.glob(os.path.join(a.dir, f"{prefix}*{suffix}")))

    def name(path, prefix):
        return os.path.basename(path)[len(prefix):-len(suffix)]

    print("\n## Chunked against full cell, float64\n")
    print("| run | atoms | halo (A) | cap binds | max dE (eV/atom) "
          "| max dF (eV/A) | max F (eV/A) |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for f in files("exact_"):
        d = load(f)
        v = d.get("validation") or {}
        print(f"| {name(f, 'exact_')} | {d['natoms']:,} | {d['halo_A']:.2f} "
              f"| {d.get('neighbor_cap_binds')} "
              f"| {fmt(v.get('max_abs_de_per_atom'))} | {fmt(v.get('max_abs_df'))} "
              f"| {fmt(v.get('max_abs_f_full'))} |")

    print("\n## Largest single graph on one GB10\n")
    print("| structure | path | largest cell (atoms) | MB/atom | s per call "
          "at largest | atoms/s, typical | bonds/atom | angles/atom "
          "| how it stopped |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for f in files("sweep_"):
        d = load(f)
        rows = d["results"]
        stop = ("out of memory" if d.get("oom") else
                f"error: {d['error']['error'][:40]}" if d.get("error") else
                "host killed it, or max size")
        if not rows:
            print(f"| {d['kind']} | {d['path']} | none fit | | | | | | {stop} |")
            continue
        r = rows[-1]
        # the largest size can be slowed by memory pressure, so the typical
        # rate is the median over sizes from 25% of the largest up to it
        typical = [x for x in rows[:-1] if x["natoms"] >= 0.25 * r["natoms"]]
        rate = statistics.median(x["natoms"] / x["t_total_s"]
                                 for x in (typical or rows))
        print(f"| {d['kind']} | {d['path']} | {r['natoms']:,} "
              f"| {r.get('mb_per_atom', float('nan')):.3f} | {r['t_total_s']:.2f} "
              f"| {rate:,.0f} | {r['edges_per_atom']:.1f} "
              f"| {r['lg_edges_per_atom']:.1f} | {stop} |")

    print("\n## Chunked throughput on one GB10, sampled pieces\n")
    print("| run | pieces | atoms per piece | evaluated per owned atom "
          "| angles per piece atom | peak GPU (GB) | owned atoms/s "
          "| GB10-hours per 10^9 atoms | cap binds |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for f in files("sample_"):
        d = load(f)
        # pieces with no atoms of their own (vacuum) were skipped, not run
        ps = [q for q in d["pieces"] if not q.get("skipped")]
        if not ps:
            continue
        atoms = sum(q["atoms"] for q in ps)
        owned = sum(q["owned"] for q in ps)
        angles = sum(q["lg_edges"] for q in ps)
        peak = max(q.get("mem_alloc_gb") or 0 for q in ps)
        rate = d["owned_atoms_per_s"]
        print(f"| {name(f, 'sample_')} | {len(ps)} of {d['n_cubes']} "
              f"| {atoms / len(ps):,.0f} | {atoms / owned:.2f} "
              f"| {angles / atoms:.1f} | {peak:.1f} | {rate:,.0f} "
              f"| {1e9 / rate / 3600:,.0f} | {d.get('neighbor_cap_binds')} |")
    print("\nTypical atoms/s: median over sweep sizes from 25% of the largest "
          "up to (not including) the largest, which memory pressure can slow."
          "\nGB10-hours per 10^9 atoms assumes the sampled pieces are typical "
          "and cost grows linearly with atoms (measured for Si from 8M to 1B).")


if __name__ == "__main__":
    main()
