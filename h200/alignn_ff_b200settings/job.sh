#!/bin/bash
#SBATCH --job-name=b200set
#SBATCH --account=kchoudh2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --comment=accept_cost
#SBATCH --exclude=gh205
#SBATCH --output=b200set_%j.out

# B200 scaling settings (Cu FCC, 5 A / 12 neighbours, energy only,
# v12.2.2024_dft_3d_307k) on one H200 NVL, pure-PyTorch ALIGNN (no DGL).
# 8 cores / 120G stay under the h200 GPU billing rate. gh205's CUDA is broken.
set -euo pipefail
B=/weka/scratch/jhu/kchoudh2/jlee859/scaling_b200s
PY=$HOME/miniforge3/envs/alignn/bin/python
export PYTHONPATH=/weka/scratch/jhu/kchoudh2/jlee859/bundles/alignn_12ae44e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd $B
[ -f scale_pure_b200settings.py ] || { echo "missing script"; exit 1; }
$PY -c "import torch, sys; torch.zeros(1).cuda(); sys.exit(0)" 2>/dev/null || {
    echo "NO USABLE CUDA DEVICE on $(hostname)"; exit 3; }
echo "start $(date) | $(hostname)"
$PY scale_pure_b200settings.py --tag H200_$SLURM_JOB_ID
echo "end $(date)"
