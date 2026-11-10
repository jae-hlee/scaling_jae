#!/bin/bash
#SBATCH --job-name=ffmax10c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage10check_%j.out

# Stage 10 check: a planar Si / SiO2 / Al gate stack (cells.py stack;
# beta-cristobalite SiO2, interfaces as built) on one GB10, AFF-r2SCAN:
#   1. cells.py self-checks
#   2. a 772-atom stack (21.72 A lateral; Si 2 cells, SiO2 1, Al 3) in
#      double precision, 2 x 2 x 2 cubes: chunked energy and forces against
#      the full cell, and whether the neighbor cap binds at the interfaces
#   3. a stack of about 1M atoms (162.9 A lateral; Si 400 A, SiO2 30 A,
#      Al 200 A): 40 pieces of 35,000 atoms through the full height, with
#      forces and per-atom energies, for the cost of the 1e8-atom run
#   cd /data/jlee859/ffmax && sbatch stage10_check.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage10
S=$B/structures
mkdir -p $R $S && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
$PY $B/cells.py test || { echo "cells.py self-checks failed"; exit 4; }
build() {
    local name=$1; shift
    [ -s $S/$name.json ] || $PY $B/cells.py build "$@" --out $S/$name
    echo "structure $name: $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['natoms'], 'atoms, box', [round(x, 2) for x in d['box_A']], [(l['layer'], l['atoms'], round(l['in_plane_strain'], 4)) for l in d['layers']])" $S/$name.json)"
}
ff() { $PY $B/ff_any.py "$@" --model_dir $B/models/$M; }

build stack_small --kind stack --layers Si:10.86,SiO2:7.16,Al:12.15 --lateral 21.72
ff chunk --structure $S/stack_small --kxyz 2 2 2 --forces --ckpt --validate \
    --float64 --out $R/exact_stack_small_$J
echo "exact exit $? $(date)"

build stack_1m --kind stack --layers Si:400,SiO2:30,Al:200 --lateral 162.9
ff chunk --structure $S/stack_1m --cluster_atoms 35000 --forces --ckpt \
    --max_chunks 40 --save_energies --out $R/sample_forces_stack_1m_$J
echo "sample exit $? $(date)"
$PY $B/summarize_cost.py $R --job $J
echo "end $(date)"
