#!/bin/bash
#SBATCH --job-name=ffrelaxc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/data/jlee859/ffmax/logs/relaxcheck_%j.out

# Stage 7 check: why the FIRE relaxation of the 150k-atom Si polycrystal
# stalls, on one GB10, about 3 minutes (relax_check.py). Run it after
# stage7_poly.sbatch has finished.
#   cd /data/jlee859/ffmax && sbatch relax_check.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M/stage7
cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
[ -s $R/relax_poly_si_150k_relaxed.json ] || { echo "the relaxation has not finished"; exit 5; }
$PY $B/relax_check.py --model_dir $B/models/$M \
    --relaxed $R/relax_poly_si_150k_relaxed --out $R/relax_check_poly_si_150k
rc=$?
echo "exit $rc end $(date)"
exit $rc
