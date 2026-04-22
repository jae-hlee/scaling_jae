#!/bin/bash
#SBATCH --job-name=alignn_scale
#SBATCH --partition=b200
#SBATCH --qos=blackwell_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=alignn_%j.out
#SBATCH --error=alignn_%j.err

# Activate conda environment
source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate b200
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# v6 script: v5 + _Float64Pool wrapper on ALIGNN's readout. The wrapper fixes
# a small f32 pool-accumulation drift (~4e-3 eV wander) for N <= 442,368, but
# does NOT close the separate upstream cliff at N >= 470,596 -- see
# analysis/v6/summary.md. run_scaling runs ASE + line-graph equivalence
# verification before the sweep by default. Sweep auto-resumes from
# scaling_alignn_v6.npz if it exists.
python scale5b_v6.py --cuda-device 0
