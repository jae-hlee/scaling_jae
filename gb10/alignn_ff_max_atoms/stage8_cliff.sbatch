#!/bin/bash
#SBATCH --job-name=ffcliff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage8_%j.out

# Stage 8: the 2^26 line-graph threshold behind the B200/H200 energy cliff
# (Cu FCC a = 3.6 A, 5 A cutoff, 12 neighbors, v12.2.2024_dft_3d_307k; the
# line graph has 144 edges per atom and passes 2^26 between 442,368 and
# 470,596 atoms). On one GB10:
#   1. cliff_ops.py construct: graph and line graph built on the GPU with
#      build_pure_torch_graph from the alignn the H200 rerun used (12ae44e),
#      at 442,368 and 470,596 atoms, against an int64 CPU construction
#   2. cliff_ops.py ops: each ALIGNN layer operation on just over 2^26 rows
#      against the same operation on small slices
#   3. the same Cu cells evaluated in pieces with ff_any.py (each piece about
#      150,000 atoms, so about 2.2e7 line-graph edges, well below 2^26),
#      energy only, at 256, 442,368, 470,596, 562,432 and 780,448 atoms. If
#      the cliff is an artifact of single graphs past 2^26 edges, these stay
#      at the small-cell value (B200: 0.604014 eV/atom at 4 to 256 atoms)
#   cd /data/jlee859/ffmax && sbatch stage8_cliff.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
BUNDLE=/data/jlee859/bundles/alignn_12ae44e
NEW=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/cliff
S=$B/structures
MV=v12_dft_3d_307k_pure
mkdir -p $R $S && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
PYTHONPATH=$NEW $PY $B/cells.py test || { echo "cells.py self-checks failed"; exit 4; }

# 1. line-graph construction at the threshold, with the H200 rerun's alignn
for n in 48 49; do
    PYTHONPATH=$BUNDLE $PY $B/cliff_ops.py construct --n $n --out $R/construct_${n}_$J
    echo "construct n=$n exit $? $(date)"
done

# 2. layer operations past 2^26 rows
PYTHONPATH=$NEW $PY $B/cliff_ops.py ops --out $R/ops_$J
echo "ops exit $? $(date)"

# 3. the B200 model, chunked below 2^26 line-graph edges per piece
SRC=$(PYTHONPATH=$BUNDLE $PY -c "from alignn.ff.ff import get_figshare_model_ff; print(get_figshare_model_ff('v12.2.2024_dft_3d_307k'))" | tail -n 1)
echo "v12 checkpoint: $SRC"
PYTHONPATH=$NEW $PY $B/prep_v12.py --src "$SRC" --out $B/models/$MV || { echo "prep_v12 failed"; exit 5; }
for n in 4 48 49 52 58; do
    [ -s $S/cu36_$n.json ] || PYTHONPATH=$NEW $PY $B/cells.py build --kind Cu --a 3.6 --rep $n --out $S/cu36_$n
    PYTHONPATH=$NEW $PY $B/ff_any.py chunk --model_dir $B/models/$MV --structure $S/cu36_$n \
        --cluster_atoms 150000 --out $R/chunk_cu36_${n}_$J
    echo "chunk n=$n exit $? $(date)"
done

# summary
$PY - "$R" "$J" <<'EOF'
import glob, json, sys
R, J = sys.argv[1], sys.argv[2]
# B200 single-graph (v6) energies, eV/atom, to 4 decimals
b200 = {256: 0.6040, 442368: 0.6040, 470596: 0.6070, 562432: 0.7027, 780448: 0.8449}
for f in sorted(glob.glob(f"{R}/construct_*_{J}.json")):
    d = json.load(open(f))
    print(f"construct n={d['n']}: {d['natoms']:,} atoms, {d['lg_edges']:,} line-graph edges "
          f"({d['lg_edges_over_2^26']:.3f} x 2^26), not sharing an atom {d['lg_edges_not_sharing_an_atom']}, "
          f"mismatches {d.get('mismatches')} (past 2^26: {d.get('mismatches_past_2^26')})")
try:
    ops = json.load(open(f"{R}/ops_{J}.json"))
    print(f"ops: {sum(r['status'] == 'FAIL' for r in ops)} of {len(ops)} failed")
    for r in ops:
        if r["status"] == "FAIL":
            print("  FAIL", r)
except FileNotFoundError:
    print("ops: no result")
print("chunked v12 Cu energies (eV/atom) against the B200 single-graph run (v6):")
for f in sorted(glob.glob(f"{R}/chunk_cu36_*_{J}.json"), key=lambda p: json.load(open(p))["natoms"]):
    d = json.load(open(f))
    n = d["natoms"]
    print(f"  {n:>9,} atoms  chunked {d.get('e_per_atom', float('nan')):.6f}  "
          f"B200 {b200.get(n, float('nan')):.4f}  pieces {d['n_cubes']}  "
          f"largest piece {max(q['atoms'] for q in d['pieces']):,}")
EOF
echo "end $(date)"
