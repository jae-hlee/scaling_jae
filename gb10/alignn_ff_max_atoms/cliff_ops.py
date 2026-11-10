"""Tests of the 2^26 line-graph threshold. On the B200 and H200, single-graph
energies of Cu FCC (a = 3.6 A, 12 neighbors, v12.2.2024_dft_3d_307k) went
wrong once the line graph passed 2^26 = 67,108,864 edges (144 per atom,
between 442,368 and 470,596 atoms), with the error proportional to the
fraction of edges past 2^26. These tests look for the cause on one GB10.

construct  builds the graph and line graph of the B200-settings Cu cell with
           alignn's build_pure_torch_graph on the GPU, as the H200 rerun did,
           and compares the line graph edge by edge with the same algorithm
           in int64 numpy on the CPU (run with the alignn the H200 rerun used)
ops        runs every tensor operation of an ALIGNN edge-gated layer on just
           below and just above 2^26 rows (and on 2^27 rows of width 32, past
           2^32 elements) and compares each output with the same operation
           applied to slices of fewer than 2^24 rows

    PYTHONPATH=/data/jlee859/bundles/alignn_12ae44e python cliff_ops.py construct --n 49 --out R/construct_49
    python cliff_ops.py ops --out R/ops
"""
import argparse
import json
import time

import numpy as np
import torch

THRESH = 2**26
SLICE = 2**24


def log(*a):
    """Print with flush."""
    print(*a, flush=True)


def construct(args):
    """GPU line graph of the B200-settings Cu cell against a CPU reference."""
    from ase.build import bulk
    from jarvis.core.atoms import ase_to_atoms
    from alignn.torch_graph_builder import build_pure_torch_graph
    dev = torch.device("cuda")
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((args.n,) * 3)
    n = len(atoms)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g, lg = build_pure_torch_graph(ase_to_atoms(atoms), two_body_cutoff=5.0,
                                   max_neighbors=12, atom_features="cgcnn",
                                   device=dev)
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    src, dst = g.src.cpu().numpy(), g.dst.cpu().numpy()
    bl = g.edata["r"].norm(dim=1).cpu().numpy()
    ls, ld = lg.src.cpu().numpy(), lg.dst.cpu().numpy()
    E, T = len(src), len(ls)
    out = {"n": args.n, "natoms": n, "edges": E, "lg_edges": T,
           "lg_edges_over_2^26": T / THRESH, "t_build_s": t_build,
           "out_degree_min_max": [int(np.bincount(src, minlength=n).min()),
                                  int(np.bincount(src, minlength=n).max())],
           "in_degree_min_max": [int(np.bincount(dst, minlength=n).min()),
                                 int(np.bincount(dst, minlength=n).max())],
           "bond_length_min_max": [float(bl.min()), float(bl.max())]}
    # every line-graph edge must join bond (u, v) to a bond (v, w)
    bad = np.flatnonzero(dst[ls] != src[ld])
    out["lg_edges_not_sharing_an_atom"] = int(len(bad))
    out["first_bad_lg_edge"] = int(bad[0]) if len(bad) else None
    # the same algorithm in int64 numpy
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    nodes = np.arange(n)
    b0 = np.searchsorted(sorted_src, nodes)
    b1 = np.searchsorted(sorted_src, nodes, side="right")
    starts, counts = b0[dst], b1[dst] - b0[dst]
    ref_s = np.repeat(np.arange(E), counts)
    cum = np.cumsum(counts)
    offs = np.arange(cum[-1]) - np.repeat(cum - counts, counts)
    ref_d = order[np.repeat(starts, counts) + offs]
    out["reference_lg_edges"] = int(len(ref_s))
    if len(ref_s) == T:
        diff = np.flatnonzero((ls != ref_s) | (ld != ref_d))
        out["mismatches"] = int(len(diff))
        out["mismatches_past_2^26"] = int((diff >= THRESH).sum())
        out["first_mismatch"] = int(diff[0]) if len(diff) else None
    else:
        out["mismatches"] = "different edge counts"
    json.dump(out, open(args.out + ".json", "w"), indent=1)
    log(json.dumps(out))


def _compare(full, ref_fn, total):
    """Largest |full - ref| over rows below and at or above 2^26, with ref
    computed slice by slice; also the largest |ref| for scale."""
    below = above = scale = 0.0
    for s in range(0, total, SLICE):
        e = min(total, s + SLICE)
        r = ref_fn(s, e)
        d = (full[s:e] - r).abs()
        if d.dim() > 1:
            d = d.amax(dim=1)
        scale = max(scale, float(r.abs().max()))
        cut = min(max(THRESH - s, 0), e - s)  # rows of this slice below 2^26
        if cut > 0:
            below = max(below, float(d[:cut].max()))
        if cut < e - s:
            above = max(above, float(d[cut:].max()))
    return below, above, scale


def ops(args):
    """Each layer operation on > 2^26 rows against small slices."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    cases = [(THRESH - 2**20, 64), (THRESH + 2**22, 64),
             (THRESH + 2**22, 128), (2 * THRESH + 2**20, 32)]
    rows = []

    def record(op, T, H, below, above, scale, tol):
        worst = max(below, above)
        rel = worst / scale if scale else worst
        r = {"op": op, "rows": T, "width": H, "rows_over_2^26": T / THRESH,
             "elements_over_2^32": T * H / 2**32,
             "max_diff_below_2^26": below, "max_diff_from_2^26": above,
             "scale": scale, "status": "FAIL" if rel > tol else "ok"}
        rows.append(r)
        log(f"[{r['status']:4s}] {op:22s} rows {T:>11,} width {H:4d}  "
            f"diff below {below:.2e}  from 2^26 {above:.2e}  (scale {scale:.2e})")
        json.dump(rows, open(args.out + ".json", "w"), indent=1)

    with torch.no_grad():
        for T, H in cases:
            E = T // 12 + 1
            idx = torch.randint(0, E, (T,), device=dev)
            # repeat_interleave, as both line-graph builders use it
            counts = torch.full((E,), 12, device=dev, dtype=torch.long)
            counts[: (E * 12 - T)] = 11 if E * 12 > T else 12
            rep = torch.repeat_interleave(torch.arange(E, device=dev), counts)
            ref = np.repeat(np.arange(E), counts.cpu().numpy())
            d = np.flatnonzero(rep.cpu().numpy() != ref) if len(ref) == len(rep) else np.array([-1])
            record("repeat_interleave", int(len(rep)), 1,
                   float((d < THRESH).sum() > 0), float((d >= THRESH).sum() > 0),
                   1.0, 0.5)
            del rep, ref, counts
            # gather rows of a bond tensor, as e_src[src] and Bh[src]
            x = torch.randn(E, H, device=dev)
            out = x[idx]
            record("gather x[idx]", T, H, *_compare(out, lambda s, e: x[idx[s:e]], T), 0.0)
            out = x.index_select(0, idx)
            record("index_select", T, H, *_compare(out, lambda s, e: x.index_select(0, idx[s:e]), T), 0.0)
            del out
            # dense ops on (T, H) line-graph features
            zt = torch.randn(T, H, device=dev)
            lin = torch.nn.Linear(H, H).to(dev)
            out = lin(zt)
            record("Linear", T, H, *_compare(out, lambda s, e: lin(zt[s:e]), T), 1e-3)
            del out
            ln = torch.nn.LayerNorm(H).to(dev)
            out = ln(zt)
            record("LayerNorm", T, H, *_compare(out, lambda s, e: ln(zt[s:e]), T), 1e-3)
            del out
            out = torch.sigmoid(zt)
            record("sigmoid", T, H, *_compare(out, lambda s, e: torch.sigmoid(zt[s:e]), T), 0.0)
            del out
            out = torch.nn.functional.silu(zt)
            record("silu", T, H, *_compare(out, lambda s, e: torch.nn.functional.silu(zt[s:e]), T), 0.0)
            del out
            # scatter-sum of (T, H) messages into E bonds, as scatter_sum
            out = torch.zeros(E, H, device=dev).scatter_add_(
                0, idx.unsqueeze(-1).expand(T, H), zt)
            ref = torch.zeros(E, H, device=dev)
            for s in range(0, T, SLICE):
                e = min(T, s + SLICE)
                ref.scatter_add_(0, idx[s:e].unsqueeze(-1).expand(e - s, H), zt[s:e])
            dd = float((out - ref).abs().max())
            sc = float(ref.abs().max())
            record("scatter_add_ (sum)", T, H, 0.0, dd, sc, 1e-3)
            out = torch.zeros(E, H, device=dev).index_add_(0, idx, zt)
            dd = float((out - ref).abs().max())
            record("index_add_ (sum)", T, H, 0.0, dd, sc, 1e-3)
            del out, ref, zt, x, idx, lin, ln
            torch.cuda.empty_cache()
    fails = [r for r in rows if r["status"] == "FAIL"]
    log(f"{len(fails)} of {len(rows)} checks failed")


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["construct", "ops"])
    p.add_argument("--n", type=int, default=49, help="Cu cells per side")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    (construct if args.mode == "construct" else ops)(args)


if __name__ == "__main__":
    main()
