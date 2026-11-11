#!/bin/bash
#SBATCH --job-name=ffmaps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/maps_%j.out

# Per-atom energy maps for figures (energy_maps.py), from whichever results
# exist, written to results/matpes_r2scan_paper/maps/ (small files, copy
# these back for analysis/gb10_energy_maps_figure.py):
#   poly_si_150k  relaxed 8-grain polycrystal (stage 7), 5 A slab across z,
#                 atom by atom
#   poly_si_1b    10^9-atom polycrystal (stage 7b), 5 A slab across z,
#                 2 A pixels
#   stack_1e8     Si/SiO2/Al gate stack (stage 10), 5 A slab across y
#                 (an x-z cross-section), 1 A pixels, energies relative to
#                 the median of each element in each layer
# With SMOOTH=1 the same maps of the retrained model are made instead, from
# the stage-13 relaxation and the stage-15 runs, into results/aff_smooth/maps/.
#   cd /data/jlee859/ffmax && sbatch maps.sbatch
#   cd /data/jlee859/ffmax && SMOOTH=1 sbatch maps.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONUNBUFFERED=1
R=$B/results/$M
S=$B/structures
O=$R/maps
RELAX=$R/stage7/relax_poly_si_150k
POLY="$R/stage7/s7_1B_poly_forces $R/stage7/s7_1B_poly_energy"
STACK=$R/stage10/s10_stack_1e8_forces
if [ "${SMOOTH:-0}" = 1 ]; then
    RELAX=$R/stage13/relax_poly_si_150k_smooth
    POLY=$B/results/aff_smooth/stage15/s15_1B_poly_forces
    STACK=$B/results/aff_smooth/stage15/s15_stack_1e8_forces
    O=$B/results/aff_smooth/maps
fi
mkdir -p $O && cd $B
echo "start $(date) | $(hostname) | smooth ${SMOOTH:-0} | maps into $O"
em() { $PY $B/energy_maps.py "$@"; echo "exit $? $(date)"; }

if [ -s ${RELAX}_energies.npy ]; then
    em relaxed --structure ${RELAX}_relaxed \
        --energies ${RELAX}_energies.npy --axis 2 \
        --out $O/poly_si_150k
else
    echo "no relaxation energies yet (${RELAX}_energies.npy)"
fi

for P in $POLY; do
    if ls ${P}_rank*_energies.npz >/dev/null 2>&1; then
        em chunked --structure $S/poly_si_1b \
            --parts "${P}_rank*_energies.npz" --axis 2 --pixel 2 \
            --out $O/poly_si_1b
        break
    fi
    echo "no polycrystal energies at ${P}_rank*_energies.npz"
done

if ls ${STACK}_rank*_energies.npz >/dev/null 2>&1; then
    em chunked --structure $S/stack_1e8 \
        --parts "${STACK}_rank*_energies.npz" \
        --groups $S/stack_1e8.layer.npy --by_species --axis 1 --pixel 1 \
        --out $O/stack_1e8
else
    echo "no device energies yet (${STACK}_rank*_energies.npz)"
fi
ls -la $O
echo "end $(date)"
