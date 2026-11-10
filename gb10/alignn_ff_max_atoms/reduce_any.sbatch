#!/bin/bash
#SBATCH --job-name=ffanyR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/reduceany_%j.out

# Adds up the tasks of a multi-task chunk_any.sbatch run (ff_any reduce).
#   sbatch --dependency=afterok:<array job id> reduce_any.sbatch <model> <tag>
set -uo pipefail
M=${1:?model}; TAG=${2:?tag}
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
$PY $B/ff_any.py reduce --out $B/results/$M/$TAG
