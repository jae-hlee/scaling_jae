#!/bin/bash
#SBATCH --job-name=ffstreamR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/streamreduce_%j.out

# Adds up the tasks of a stream.sbatch run (ff_stream reduce): energy, and
# force statistics slab by slab (about 1.2 GB of memory at a time).
#   sbatch --dependency=afterok:<array job id> stream_reduce.sbatch <model> <tag>
set -uo pipefail
M=${1:?model}; TAG=${2:?tag}
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
$PY $B/ff_stream.py reduce --out $B/results/$M/$TAG
