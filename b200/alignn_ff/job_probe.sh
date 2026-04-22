#!/bin/bash
#SBATCH --job-name=alignn_probe
#SBATCH --partition=b200
#SBATCH --qos=blackwell_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=probe_%j.out
#SBATCH --error=probe_%j.err

source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate b200
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Instruments the v6 _Float64Pool to print per-call pool stats at baseline
# (i=20) and drift-region sizes (i=50, 55). Tells us whether the wrapper is
# actually invoked during forward, and whether the f32-vs-f64 pool-output
# diff is large (pool is the culprit) or tiny (drift is upstream).
python probe_pool.py
