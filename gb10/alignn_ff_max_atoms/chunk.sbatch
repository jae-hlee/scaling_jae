#!/bin/bash
#SBATCH --job-name=ffmax2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/chunk_%A_%a.out

# Stage 2: one large Si cell evaluated piece by piece. Each array task is
# one GB10 and takes every W-th piece (W = array size). One task = the
# single-GB10 run, W tasks = W GB10 nodes, then reduce.sbatch adds up.
#   sbatch --array=0-W-1 chunk.sbatch <model> <rep> <forces 0|1> <piece atoms> <tag> [more options, e.g. --halo 25 --rattle 0.05]
set -uo pipefail
M=${1:?model}; REP=${2:?rep}; FORCES=${3:?0 or 1}; CA=${4:?piece atoms}; TAG=${5:?tag}
shift 5
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M
mkdir -p $R && cd $R
EXTRA=""
[ "$FORCES" = 1 ] && EXTRA="--forces --ckpt"
echo "start $(date) | $(hostname) | $M rep $REP forces $FORCES piece $CA $* | task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
$PY $B/ff_max_atoms.py chunk --model_dir $B/models/$M --rep $REP $EXTRA \
    --cluster_atoms $CA --rank $SLURM_ARRAY_TASK_ID \
    --world $SLURM_ARRAY_TASK_COUNT --out $R/$TAG "$@"
echo "exit $? end $(date)"
