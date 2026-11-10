#!/bin/bash
#SBATCH --job-name=ffmax7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage7_%j.out

# Stage 7: a relaxed Si polycrystal on one GB10, AFF-r2SCAN (folder
# matpes_r2scan_paper). 8 randomly oriented grains in a 146.61 A periodic
# cube (27 Si cells; about 150,000 atoms, grains about 9 nm), relaxed at fixed
# cell with FIRE on one graph with layer checkpointing until every force is
# below 0.05 eV/A (at most 2000 steps), then the energy of every atom of the
# relaxed cell (relax.py). The unrelaxed 40-grain cell of stage 6 had forces
# up to 58 eV/A at its boundaries. Resubmitting resumes from the last saved
# step.
#   cd /data/jlee859/ffmax && sbatch stage7_poly.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M/stage7
S=$B/structures
mkdir -p $R $S && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
$PY $B/cells.py test || { echo "cells.py self-checks failed"; exit 4; }

[ -s $S/poly_si_150k.json ] || $PY $B/cells.py build --kind poly --element Si \
    --box 146.61 --grains 8 --seed 1 --out $S/poly_si_150k
$PY $B/relax.py --model_dir $B/models/$M --structure $S/poly_si_150k --single \
    --fmax 0.05 --max_steps 2000 --out $R/relax_poly_si_150k
echo "relax exit $? end $(date)"
