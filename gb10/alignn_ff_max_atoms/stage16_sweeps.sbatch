#!/bin/bash
#SBATCH --job-name=ffmax16w
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage16sweeps_%j.out

# Stage 16, part 3: the largest single graphs of FCC Al, FCC Cu and water with
# the retrained model (energy only and checkpointed forces), again. In job
# 22028 on atomgptlab05 every one of these sweeps was stopped with SIGTERM
# ("Terminated", exit 143) after its last size at 77 to 83 GB, while
# AFF-r2SCAN's reached 108 to 121 GB on atomgptlab03 (stage 6) and the
# retrained model's Si sweep 119 GB on atomgptlab04 (job 22027). So that node,
# not the model, set those limits. Submit with atomgptlab05 excluded; the
# results replace sweep_*_22028 in results/aff_smooth/stage16/cost/.
#   cd /data/jlee859/ffmax && sbatch --exclude=atomgptlab03,atomgptlab05 stage16_sweeps.sbatch
set -uo pipefail
M=aff_smooth
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage16/cost
mkdir -p $R && cd $B
echo "start $(date) | $(hostname) | model $M"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models/$M && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
free -g | head -2
grep -E "MemTotal|MemAvailable" /proc/meminfo

ff() { $PY $B/smooth_eval.py $B/ff_any.py "$@" --model_dir $B/models/$M; }
for kind in Al Cu water; do
    for p in e ckpt; do
        ff sweep --kind $kind --path $p --out $R/sweep_${kind}_${p}_$J
        echo "sweep $kind $p exit $? $(date)"
    done
done
echo "end $(date)"
