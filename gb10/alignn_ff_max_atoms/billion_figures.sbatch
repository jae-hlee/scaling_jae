#!/bin/bash
#SBATCH --job-name=fffigs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/data/jlee859/ffmax/logs/figs_%j.out

# The manuscript's gb10_scaling.png, gb10_halo.png and gb10_decomposition.png
# from analysis/gb10_billion_figures.py, read from a copy of the repo's
# results folder (fig_inputs) so the data are those of the committed figures.
# Written to /data/jlee859/ffmax/figs. Copy gb10_billion_figures.py and the
# results folder first.
# With GB10_MODEL=smooth the scaling and halo figures of AFF-r2SCAN-smooth
# (gb10_scaling_smooth.png, gb10_halo_smooth.png) are drawn from the cluster's
# own results folder, which holds the stage 15 and 16 files the repo copies.
#   cd /data/jlee859/ffmax && sbatch billion_figures.sbatch
#   cd /data/jlee859/ffmax && GB10_MODEL=smooth sbatch billion_figures.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
cd $B
echo "start $(date) | $(hostname) | model ${GB10_MODEL:-published}"
if [ "${GB10_MODEL:-}" = smooth ]; then
    export GB10_RESULTS=$B/results
elif [ -d $B/fig_inputs/results ]; then
    export GB10_RESULTS=$B/fig_inputs/results
else
    export GB10_RESULTS=$B/fig_inputs
fi
[ -d $GB10_RESULTS/matpes_r2scan_paper ] || { echo "no results copy in $B/fig_inputs"; exit 5; }
export FIG_OUT=$B/figs
mkdir -p $FIG_OUT
$PY $B/gb10_billion_figures.py
rc=$?
ls -la $FIG_OUT
echo "end rc $rc $(date)"
exit $rc
