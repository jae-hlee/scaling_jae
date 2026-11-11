#!/bin/bash
#SBATCH --job-name=ffany
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/chunkany_%A_%a.out

# One cells.py structure evaluated piece by piece with ff_any.py. Each array
# task is one GB10 and takes every W-th piece (W = array size); reduce_any
# adds the tasks up. Results go to results/<model>/<tag>.
#   sbatch --array=0-W-1 chunk_any.sbatch <model> <structure name> <tag> [ff_any chunk options]
# With SMOOTH=1 in the environment the model is one with smooth_messages (the
# retrained model of stage 12): the patched ALIGNN and smooth_eval.py are used.
set -uo pipefail
M=${1:?model}; ST=${2:?structure name}; TAG=${3:?tag}
shift 3
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
WRAP=()
if [ "${SMOOTH:-0}" = 1 ]; then
    export PYTHONPATH=$B/alignn_smooth
    WRAP=($B/smooth_eval.py)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M
mkdir -p "$(dirname $R/$TAG)" && cd $R
echo "start $(date) | $(hostname) | $M $ST $TAG $* | task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT | smooth ${SMOOTH:-0}"
$PY "${WRAP[@]}" $B/ff_any.py chunk --model_dir $B/models/$M --structure $B/structures/$ST \
    --rank $SLURM_ARRAY_TASK_ID --world $SLURM_ARRAY_TASK_COUNT --out $R/$TAG "$@"
rc=$?
echo "exit $rc end $(date)"
exit $rc  # a failed task keeps the afterok reduce from running
