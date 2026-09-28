#!/bin/bash
#SBATCH --job-name=ffscale
#SBATCH --account=kchoudh2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh205
#SBATCH --output=ffscale_%j.out

# Largest-cell scaling test of the default ALIGNN force field on one GPU
# (paper tab:scaling protocol). Pick the GPU type at submit time:
#   sbatch bench_skipjack.sbatch                 # h200 (141 GB)
#   sbatch -p b200 bench_skipjack.sbatch         # b200 (180 GB), once allowed
# 8 cores / 120G keep the billing at the GPU rate. gh205's CUDA is broken.
# Graph building for 100k-atom cells happens on the CPU side; 120G of host
# memory is enough for the sizes an H200/B200 can hold.
set -euo pipefail
B=/weka/scratch/jhu/kchoudh2/jlee859/scaling
PY=$HOME/miniforge3/envs/alignn/bin/python
export PYTHONPATH=/weka/scratch/jhu/kchoudh2/jlee859/bundles/alignn_12ae44e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p $B && cd $B
[ -f $B/bench_ff_scaling.py ] || { echo "missing $B/bench_ff_scaling.py"; exit 1; }
$PY -c "import ase, alignn.ff.calculators" || { echo "env lacks ase or alignn.ff (pip install ase on the login node)"; exit 4; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, sys; torch.zeros(1).cuda(); sys.exit(0)" 2>/dev/null || {
    echo "NO USABLE CUDA DEVICE on $(hostname)"; exit 3; }
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '_' | sed 's/_$//')
echo "start $(date) | $(hostname) | $TAG"
$PY $B/bench_ff_scaling.py --max_i 40 --repeats 3 \
    --out $B/bench_${TAG}_$SLURM_JOB_ID
echo "end $(date)"
