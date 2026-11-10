#!/bin/bash
#SBATCH --job-name=ffmax9c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage9check_%j.out

# Stage 9 check: ff_stream.py (pieces generated from the lattice, slab-local
# forces) on one GB10 before the multi-billion-atom runs, AFF-r2SCAN:
#   1. cells.py self-checks, including on-demand pieces against full-cell
#      pieces and slab-local rows against full-cell accumulation
#   2. 8,000 atoms of rattled Si in double precision, one task, 2 x 2 x 2
#      cubes: chunked energy and forces against the full cell
#   3. the same cell split into 4 x 4 x 4 cubes, as one task and as two tasks
#      (x slabs) plus reduce: the summed forces must agree
#   4. pilots on the 2,000,376,000-atom cell (rep 630): 3 pieces of rank 0,
#      without forces (10 ranks) and with forces (20 ranks), for time per
#      piece, GPU memory and host memory
#   cd /data/jlee859/ffmax && sbatch stage9_check.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M/stage9
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
$PY $B/cells.py test || { echo "cells.py self-checks failed"; exit 4; }
fs() { $PY $B/ff_stream.py "$@" --model_dir $B/models/$M; }
SMALL="--kind Si --rep 10 --rattle 0.05 --forces --ckpt --float64"

# 2. chunked against full cell
fs chunk $SMALL --k 2 --validate --out $R/s9_exact
echo "exact exit $? $(date)"

# 3. one task against two tasks
fs chunk $SMALL --k 4 --save_dense --out $R/s9_one
echo "one task exit $?"
for r in 0 1; do
    fs chunk $SMALL --k 4 --rank $r --world 2 --out $R/s9_two
    echo "two tasks, rank $r exit $?"
done
$PY $B/ff_stream.py reduce --out $R/s9_two --save_dense
echo "reduce exit $?"
$PY -c "
import json, numpy as np
a = np.load('$R/s9_one_dense_forces.npy'); b = np.load('$R/s9_two_dense_forces.npy')
e1 = json.load(open('$R/s9_one.json'))['e_per_atom']; e2 = json.load(open('$R/s9_two_reduced.json'))['e_per_atom']
print('two tasks against one: max force difference %.2e eV/A (largest force %.2f), energy difference %.2e eV/atom' % (abs(a - b).max(), abs(a).max(), abs(e1 - e2)))
v = json.load(open('$R/s9_exact.json'))['validation']
print('chunked against full cell:', v)
"

# 4. pilots on 2.0 billion atoms
BIG="--kind Si --rep 630 --rattle 0.05 --max_chunks 3"
fs chunk $BIG --cluster_atoms 330820 --rank 0 --world 10 --out $R/s9_pilot_energy
echo "pilot energy exit $? $(date)"
fs chunk $BIG --forces --ckpt --cluster_atoms 106000 --rank 0 --world 20 --out $R/s9_pilot_forces
echo "pilot forces exit $? $(date)"
rm -f $R/s9_pilot_forces_rank0000_forces.npy  # 3 GB slab of a 3-piece pilot
echo "end $(date)"
