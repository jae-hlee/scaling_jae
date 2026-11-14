#!/bin/bash
#SBATCH --job-name=ffotherfig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=/data/jlee859/ffmax/logs/otherfig_%j.out

# The SI figure that puts ALIGNN-FF (AFF-r2SCAN-smooth) next to SNAP, NEP,
# Allegro and Deep Potential on one GB10, from analysis/gb10_other_ff_figure.py,
# read from the cluster's own results (/data/jlee859/ffother/results and
# /data/jlee859/ffmax/results, which the repo copies). Written to
# /data/jlee859/ffmax/figs/gb10_other_ff.png; the log lists the values plotted.
# Copy gb10_other_ff_figure.py to /data/jlee859/ffmax first.
#   cd /data/jlee859/ffmax && sbatch --exclude=atomgptlab03 other_ff_figure.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export OTHER_RESULTS=/data/jlee859/ffother/results
export GB10_RESULTS=$B/results
export FIG_OUT=$B/figs
mkdir -p $FIG_OUT
cd $B
echo "start $(date) | $(hostname)"
$PY $B/gb10_other_ff_figure.py
rc=$?
ls -la $FIG_OUT/gb10_other_ff.png
echo "end rc $rc $(date)"
exit $rc
