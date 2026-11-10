#!/bin/bash
#SBATCH --job-name=ffsmtrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage12train_%j.out

# Stage 12: train AFF-r2SCAN with smooth_messages on MatPES r2SCAN, on one
# GB10, with the settings recorded in AFF-r2SCAN's config: 100 epochs, batch
# 64, learning rate 1e-3, random split of 347,889 / 19,327 / 19,328 frames,
# energy and force weights 1, stress weight 0. The probe (batch 8) took 21.6 s
# per epoch for 4,000 frames, about 31 min per epoch for the full set.
# Resubmitting continues from the last finished epoch.
#   cd /data/jlee859/ffmax && sbatch stage12_train.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
W=/data/jlee859/ffsmooth
OUT=$W/aff_smooth_run
WK=$W/work
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
echo "start $(date) | $(hostname)"
[ -d $B/alignn_smooth ] || { echo "run stage12_check.sbatch first"; exit 5; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
ID=$(find $W/data -name 'id_prop.json*' -print -quit)
[ -n "$ID" ] || { echo "no training data in $W/data (run stage12_probe.sbatch)"; exit 6; }
D=$(dirname "$ID")
mkdir -p $OUT $WK

# train.py writes config.json only after all three graph caches are built, so
# without it any cache here is partial and is rebuilt
RE=()
if [ -f $OUT/config.json ]; then
    RE=(--read_existing)
else
    rm -rf $WK/matpestrain_data $WK/matpesval_data $WK/matpestest_data
fi
RESTART=()
if [ -f $OUT/current_state.pt ] && [ -f $OUT/current_model.pt ]; then
    RESTART=(--restart_model_path $OUT/current_model.pt)
    echo "RESUMING from $OUT, checkpoint of $(date -r $OUT/current_state.pt)"
    for h in history_train history_val; do
        [ -f $OUT/$h.json ] && cp $OUT/$h.json $OUT/${h}_before_$SLURM_JOB_ID.json
    done
else
    echo "FRESH START"
fi

$PY $B/smooth_train_config.py --base $B/smooth_train_base.json \
    --aff $B/models/$M/config.json --out $W/aff_smooth_config.json \
    --output_dir $OUT --epochs 100 --batch_size 64 --learning_rate 0.001 \
    --keep_data_order false --resume ${RE[@]+"${RE[@]}"} \
    || { echo "config failed"; exit 7; }

cd $WK
$PY -c "import alignn; print('alignn from', alignn.__file__)"
$PY -m alignn.train_alignn --root_dir "$D" \
    --config_name $W/aff_smooth_config.json --output_dir $OUT \
    --target_key energy_per_atom --stresswise_key stresses \
    ${RESTART[@]+"${RESTART[@]}"} 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done
rc=${PIPESTATUS[0]}
echo "train exit $rc end $(date)"
exit $rc
