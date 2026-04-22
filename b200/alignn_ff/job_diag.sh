#!/bin/bash
#SBATCH --job-name=alignn_diag
#SBATCH --partition=b200
#SBATCH --qos=blackwell_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=diag_%j.out
#SBATCH --error=diag_%j.err

source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate b200
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Diagnostic for the v5 large-N energy drift. Two sub-tests:
#  (1) float32 vs float64 at sizes where f64 fits (~N up to ~250k on 178 GiB).
#  (2) float32 with shuffled atom order at drift-region sizes -- a nonzero
#      delta is a direct signature of float32 accumulation error.
python diagnose_drift.py --cuda-device 0
