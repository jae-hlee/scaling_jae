#!/bin/bash
#SBATCH --job-name=ffprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage12probe_%j.out

# Stage 12 probe: time per epoch of training AFF-r2SCAN with smooth_messages
# on one GB10. 4,000 training frames, 400 validation, 400 test, 2 epochs, so
# the graph caches are small; the full MatPES r2SCAN training set is about 87
# times larger. Timestamps on each line give the epoch time.
# Submit after stage12_check.sbatch has passed.
#   cd /data/jlee859/ffmax && sbatch stage12_probe.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
W=/data/jlee859/ffsmooth
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
TAR=/data/jlee859/figshare_upload_data/matpes_r2scan_ff_train_data.tar.gz
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
echo "start $(date) | $(hostname)"
[ -d $B/alignn_smooth ] || { echo "run stage12_check.sbatch first"; exit 5; }
df -h /data | tail -n 1
mkdir -p $W/data $W/probe_work_$J $W/probe_out_$J
if [ -z "$(find $W/data -name 'id_prop.json*' -print -quit)" ]; then
    tar xzf $TAR -C $W/data || { echo "could not unpack $TAR"; exit 6; }
fi
ID=$(find $W/data -name 'id_prop.json*' -print -quit)
[ -n "$ID" ] || { echo "no id_prop.json in $TAR"; exit 6; }
D=$(dirname "$ID")
ls -la $D | head -n 8
[ -f $D/split_sizes.json ] && cat $D/split_sizes.json

$PY $B/smooth_train_config.py --base $B/smooth_train_base.json \
    --aff $B/models/$M/config.json --out $W/probe_config_$J.json \
    --output_dir $W/probe_out_$J --epochs 2 --warmup_steps 50 \
    --n_train 4000 --n_val 400 --n_test 400 \
    || { echo "config failed"; exit 7; }

cd $W/probe_work_$J
$PY -c "import alignn; print('alignn from', alignn.__file__)"
$PY -m alignn.train_alignn --root_dir "$D" \
    --config_name $W/probe_config_$J.json --output_dir $W/probe_out_$J \
    --target_key energy_per_atom --stresswise_key stresses 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done
rc=${PIPESTATUS[0]}
echo "train exit $rc end $(date)"
exit $rc
