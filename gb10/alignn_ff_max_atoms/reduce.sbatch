#!/bin/bash
#SBATCH --job-name=ffmaxR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/reduce_%j.out

# Adds up the parts of a multi-task chunk.sbatch run.
#   sbatch --dependency=afterok:<array job id> reduce.sbatch <model> <tag>
set -uo pipefail
M=${1:?model}; TAG=${2:?tag}
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
$PY $B/ff_max_atoms.py reduce --out $B/results/$M/$TAG
