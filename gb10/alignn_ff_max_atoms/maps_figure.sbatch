#!/bin/bash
#SBATCH --job-name=ffmapfig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/data/jlee859/ffmax/logs/mapfig_%j.out

# The gate-stack map with one median per element in each layer, how far the
# energy disturbance reaches into each layer (map_profile.py), and the figure
# gb10_energy_maps.png (written next to the script, in /data/jlee859/ffmax).
# Copy energy_maps.py, map_profile.py and gb10_energy_maps_figure.py first.
#   cd /data/jlee859/ffmax && sbatch maps_figure.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export MAPS_DIR=$B/results/$M/maps
R=$B/results/$M
S=$B/structures
cd $B
echo "start $(date) | $(hostname)"
$PY $B/energy_maps.py chunked --structure $S/stack_1e8 \
    --parts "$R/stage10/s10_stack_1e8_forces_rank*_energies.npz" \
    --groups $S/stack_1e8.layer.npy --by_species --axis 1 --pixel 1 \
    --out $MAPS_DIR/stack_1e8
echo "map exit $?"
$PY $B/map_profile.py $MAPS_DIR/stack_1e8.npz
echo "profile exit $?"
$PY $B/gb10_energy_maps_figure.py
rc=$?
echo "figure exit $rc"
ls -la $B/gb10_energy_maps.png
echo "end $(date)"
exit $rc
