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
# With SMOOTH=1 the same for the retrained model (stage 15 stack, maps in
# results/aff_smooth/maps from `SMOOTH=1 sbatch maps.sbatch`), figure
# gb10_energy_maps_smooth.png.
#   cd /data/jlee859/ffmax && sbatch maps_figure.sbatch
#   cd /data/jlee859/ffmax && SMOOTH=1 sbatch maps_figure.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
R=$B/results/$M
S=$B/structures
export MAPS_DIR=$R/maps FIG_OUT=gb10_energy_maps.png
PARTS="$R/stage10/s10_stack_1e8_forces_rank*_energies.npz"
if [ "${SMOOTH:-0}" = 1 ]; then
    export MAPS_DIR=$B/results/aff_smooth/maps FIG_OUT=gb10_energy_maps_smooth.png
    PARTS="$B/results/aff_smooth/stage15/s15_stack_1e8_forces_rank*_energies.npz"
fi
mkdir -p $MAPS_DIR
cd $B
echo "start $(date) | $(hostname) | smooth ${SMOOTH:-0} | maps $MAPS_DIR"
$PY $B/energy_maps.py chunked --structure $S/stack_1e8 \
    --parts "$PARTS" \
    --groups $S/stack_1e8.layer.npy --by_species --axis 1 --pixel 1 \
    --out $MAPS_DIR/stack_1e8
echo "map exit $?"
$PY $B/map_profile.py $MAPS_DIR/stack_1e8.npz
echo "profile exit $?"
$PY $B/gb10_energy_maps_figure.py
rc=$?
echo "figure exit $rc"
ls -la $B/$FIG_OUT
echo "end $(date)"
exit $rc
