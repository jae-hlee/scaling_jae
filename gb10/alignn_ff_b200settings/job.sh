#!/bin/bash
#SBATCH --job-name=b200set
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=b200set_%j.out

# B200 scaling settings (Cu FCC, 5 A / 12 neighbours, energy only,
# v12.2.2024_dft_3d_307k) on one GB10, pure-PyTorch ALIGNN (no DGL).
set -euo pipefail
B=/data/jlee859/scaling_b200s
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=/data/jlee859/bundles/alignn_12ae44e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd $B
[ -f scale_pure_b200settings.py ] || { echo "missing script"; exit 1; }
$PY -c "import torch, sys; torch.zeros(1).cuda(); sys.exit(0)" 2>/dev/null || {
    echo "NO USABLE CUDA DEVICE on $(hostname)"; exit 3; }
echo "start $(date) | $(hostname)"
$PY scale_pure_b200settings.py --tag GB10_$SLURM_JOB_ID
echo "end $(date)"
