#!/bin/bash
#SBATCH --job-name=ffscale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=atomgptlab08
#SBATCH --output=ffscale_%j.out

# GB10 baseline of the same scaling test, to tie the numbers to the paper's
# tab:scaling (AFF-r2SCAN: 54,872 atoms at 107 GB, 2.0 s at 21,952, OOM ~64k).
#   sbatch bench_atomgptlab.sbatch
set -euo pipefail
B=/data/jlee859/scaling
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=/data/jlee859/bundles/alignn_12ae44e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p $B && cd $B
[ -f $B/bench_ff_scaling.py ] || { echo "missing $B/bench_ff_scaling.py"; exit 1; }
$PY -c "import ase, alignn.ff.calculators" || { echo "env lacks ase or alignn.ff (pip install ase on the login node)"; exit 4; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, sys; torch.zeros(1).cuda(); sys.exit(0)" 2>/dev/null || {
    echo "NO USABLE CUDA DEVICE on $(hostname)"; exit 3; }
echo "start $(date) | $(hostname)"
$PY $B/bench_ff_scaling.py --max_i 40 --repeats 3 \
    --out $B/bench_GB10_$SLURM_JOB_ID
echo "end $(date)"
