#!/bin/bash
#SBATCH --job-name=ffcompare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage13compare_%j.out

# Stage 13: is the retrained model (stage 12, smooth_messages) as accurate as
# AFF-r2SCAN, and does it conserve energy? Submit after training job 21744 has
# finished ("train exit 0" in its log).
#   1. copies the run's best checkpoint (lowest validation loss) and config
#      to models/aff_smooth, with md5, so later stages use a fixed copy
#   2. compare_models.py test: both models on the run's 19,328 test frames and
#      on 2,000 of its training frames (about an hour). Looks for AFF-r2SCAN's
#      own split file to confirm it never trained on these test frames
#   3. compare_models.py nve: float64, both models from the same start, rattled
#      Si at 1500 K and water at 300 K, water also with the 52-neighbor cap
#      lifted, 2,000 steps each (about two hours)
#   cd /data/jlee859/ffmax && sbatch stage13_compare.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
W=/data/jlee859/ffsmooth
RUN=$W/aff_smooth_run
NEW=$B/models/aff_smooth
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage13
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
[ -d $B/alignn_smooth ] || { echo "no $B/alignn_smooth (stage12_check.sbatch makes it)"; exit 5; }
grep -qs "^train exit 0" $B/logs/stage12train_*.out \
    || { echo "training has not finished: no 'train exit 0' in logs/stage12train_*.out"; exit 6; }

if [ ! -s $NEW/model.md5 ]; then
    mkdir -p $NEW
    IDS=$(ls $RUN/ids_train_val_test.json $W/work/ids_train_val_test.json 2>/dev/null | head -n 1)
    [ -n "$IDS" ] || { echo "no ids_train_val_test.json in $RUN or $W/work"; exit 7; }
    cp $RUN/config.json $RUN/best_model.pt "$IDS" $NEW/ \
        || { echo "could not copy the run from $RUN"; exit 7; }
    for f in history_train.json history_val.json; do
        [ -f $RUN/$f ] && cp $RUN/$f $NEW/
    done
    (cd $NEW && md5sum best_model.pt config.json > model.md5)
fi
(cd $NEW && md5sum -c --quiet model.md5) || { echo "$NEW differs from its model.md5"; exit 2; }
cat $NEW/model.md5

ID=$(find $W/data -name 'id_prop.json*' -print -quit)
[ -n "$ID" ] || { echo "no training data in $W/data"; exit 6; }
D=$(dirname "$ID")
# only AFF-r2SCAN's own folder counts; other split files are listed to check by hand
AFF_IDS=$(find $B/models/$M -maxdepth 2 -name 'ids_train_val_test.json' -print -quit 2>/dev/null)
echo "AFF-r2SCAN split file: ${AFF_IDS:-none in $B/models/$M}"
echo "other split files (not used):"
find /data/jlee859/figshare_upload_data -maxdepth 3 -name 'ids_train_val_test.json*' 2>/dev/null | head -n 10
$PY -c "import alignn; print('alignn from', alignn.__file__)"

rc=0
$PY compare_models.py test --models aff=$B/models/$M smooth=$NEW --data "$D" \
    --ids $NEW/ids_train_val_test.json --run_config $NEW/config.json \
    --history $NEW/history_val.json --pred_csv $RUN/prediction_results_test_set.csv \
    ${AFF_IDS:+--aff_ids "$AFF_IDS"} --out $R/test_$J || rc=1
echo "test done, rc $rc $(date)"
$PY compare_models.py nve --models aff=$B/models/$M smooth=$NEW \
    --out $R/nve_$J || rc=1
echo "end rc $rc $(date)"
exit $rc
