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
#   cd /data/jlee859/ffmax && sbatch maps.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONUNBUFFERED=1
R=$B/results/$M
S=$B/structures
O=$R/maps
mkdir -p $O && cd $B
echo "start $(date) | $(hostname)"
em() { $PY $B/energy_maps.py "$@"; echo "exit $? $(date)"; }

if [ -s $R/stage7/relax_poly_si_150k_energies.npy ]; then
    em relaxed --structure $R/stage7/relax_poly_si_150k_relaxed \
        --energies $R/stage7/relax_poly_si_150k_energies.npy --axis 2 \
        --out $O/poly_si_150k
else
    echo "no stage-7 relaxation energies yet"
fi

for TAG in s7_1B_poly_forces s7_1B_poly_energy; do
    if ls $R/stage7/${TAG}_rank*_energies.npz >/dev/null 2>&1; then
        em chunked --structure $S/poly_si_1b \
            --parts "$R/stage7/${TAG}_rank*_energies.npz" --axis 2 --pixel 2 \
            --out $O/poly_si_1b
        break
    fi
done

if ls $R/stage10/s10_stack_1e8_forces_rank*_energies.npz >/dev/null 2>&1; then
    em chunked --structure $S/stack_1e8 \
        --parts "$R/stage10/s10_stack_1e8_forces_rank*_energies.npz" \
        --groups $S/stack_1e8.layer.npy --by_species --axis 1 --pixel 1 \
        --out $O/stack_1e8
else
    echo "no stage-10 device energies yet"
fi
ls -la $O
echo "end $(date)"
