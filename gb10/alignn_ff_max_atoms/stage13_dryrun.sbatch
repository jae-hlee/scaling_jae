#!/bin/bash
#SBATCH --job-name=ffdry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage13dry_%j.out

# Dry run of compare_models.py on the 2-epoch model of the stage-12 probe
# (job 21743: 4,000 training, 400 test frames), to find errors in the script
# before training ends. The numbers mean nothing; the probe model is barely
# trained. Can run while training job 21744 occupies another GB10.
#   cd /data/jlee859/ffmax && sbatch stage13_dryrun.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
W=/data/jlee859/ffsmooth
P=$W/probe_out_21743
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage13
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
ls $P
IDS=$(ls $P/ids_train_val_test.json $W/probe_work_21743/ids_train_val_test.json 2>/dev/null | head -n 1)
[ -n "$IDS" ] || { echo "no ids_train_val_test.json for the probe"; exit 7; }
ID=$(find $W/data -name 'id_prop.json*' -print -quit)
D=$(dirname "$ID")

rc=0
$PY compare_models.py test --models aff=$B/models/$M probe=$P --data "$D" \
    --ids "$IDS" --run_config $P/config.json --history $P/history_val.json \
    --pred_csv $P/prediction_results_test_set.csv --csv_model probe \
    --n_train_sample 100 --out $R/dry_test_$J || rc=1
echo "test done, rc $rc $(date)"
$PY compare_models.py nve --models aff=$B/models/$M probe=$P --steps 20 \
    --out $R/dry_nve_$J || rc=1
echo "end rc $rc $(date)"
exit $rc
