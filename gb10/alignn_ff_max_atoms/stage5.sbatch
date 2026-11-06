#!/bin/bash
#SBATCH --job-name=ffmax5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage5_%j.out

# Stage 5: the stage-1 chunk check on SiGe again, in double precision.
# Same cell as check_forces_* (8,000 atoms of rattled SiGe, k 2, forces,
# checkpointed), each against the full cell:
#   exact halo, all four models            -> results/<model>/f64_sige_exact_<job>.json
#   20 A halo, the three alignn_ff_* models -> results/<model>/f64_sige_h20_<job>.json
# In single precision the alignn_ff_matpes_* models differed by 1e-3 eV/A at
# the exact halo. Near zero here means that was roundoff. The 20 A runs
# measure the halo cut on SiGe (stage 1 measured it on Si only).
#   cd /data/jlee859/ffmax && sbatch stage5.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
for M in matpes_r2scan_paper alignn_ff_mptrj alignn_ff_matpes_pbe alignn_ff_matpes_r2scan; do
    R=$B/results/$M
    $PY $B/ff_max_atoms.py chunk --model_dir $B/models/$M --rep 10 --k 2 \
        --species SiGe --rattle 0.05 --forces --ckpt --validate --float64 \
        --out $R/f64_sige_exact_$J
    echo "$M exact exit $? $(date)"
    [ "$M" = matpes_r2scan_paper ] && continue  # its exact halo is 20.01 A
    $PY $B/ff_max_atoms.py chunk --model_dir $B/models/$M --rep 10 --k 2 \
        --species SiGe --rattle 0.05 --forces --ckpt --validate --float64 \
        --halo 20 --out $R/f64_sige_h20_$J
    echo "$M h20 exit $? $(date)"
done
echo "end $(date)"
