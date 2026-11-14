#!/bin/bash
#SBATCH --job-name=ffmax16c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/s16chunk_%A_%a.out

# Stage 16: one array task of a chunked rattled-Si run with the retrained
# model (models/aff_smooth), as chunk.sbatch does for AFF-r2SCAN: same cell,
# displacements (rattle 0.05 A, seed 0) and piece size, every W-th piece
# (W = array size). reduce.sbatch aff_smooth stage16/<tag> adds up the tasks.
#   sbatch --array=0-5 stage16_chunk.sbatch <rep> <forces 0|1> <piece atoms> <tag>
set -uo pipefail
REP=${1:?rep}; FORCES=${2:?0 or 1}; CA=${3:?piece atoms}; TAG=${4:?tag}
M=aff_smooth
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M/stage16
mkdir -p $R && cd $B
EXTRA=""
[ "$FORCES" = 1 ] && EXTRA="--forces --ckpt"
echo "start $(date) | $(hostname) | $M rep $REP forces $FORCES piece $CA | task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
$PY $B/smooth_ffm.py chunk --model_dir $B/models/$M --rep $REP $EXTRA \
    --cluster_atoms $CA --rattle 0.05 --rank $SLURM_ARRAY_TASK_ID \
    --world $SLURM_ARRAY_TASK_COUNT --out $R/$TAG
echo "exit $? end $(date)"
