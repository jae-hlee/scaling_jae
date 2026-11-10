#!/bin/bash
#SBATCH --job-name=ffrerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage13rerun_%j.out

# Stage 13 reruns with the retrained model (models/aff_smooth, made by
# stage13_compare.sbatch), on one GB10:
#   1. the stage-7 relaxation of the 155,281-atom Si polycrystal with the same
#      settings (one graph, FIRE, fmax 0.05 eV/A, at most 2,000 steps; 3.8 h
#      if it again runs all 2,000)
#   2. relax_check.py on its result: forces left and the bond-crossing test
#      with trained weights
#   3. the stage-11 evaluation of the 1,066,628-atom STMV virus (about 1 h)
# relax.py resumes from its state file if the job is resubmitted.
#   cd /data/jlee859/ffmax && sbatch stage13_rerun.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
NEW=$B/models/aff_smooth
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage13
S=$B/structures
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
[ -s $NEW/model.md5 ] || { echo "no $NEW (run stage13_compare.sbatch first)"; exit 5; }
(cd $NEW && md5sum -c --quiet model.md5) || { echo "$NEW differs from its model.md5"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
for s in poly_si_150k stmv; do
    [ -s $S/$s.json ] || { echo "no $S/$s (stages 7 and 11 build it)"; exit 6; }
done

rc=0
$PY smooth_eval.py relax.py --model_dir $NEW --structure $S/poly_si_150k \
    --single --fmax 0.05 --max_steps 2000 --out $R/relax_poly_si_150k_smooth || rc=1
echo "relax done, rc $rc $(date)"
$PY smooth_eval.py relax_check.py --model_dir $NEW \
    --relaxed $R/relax_poly_si_150k_smooth_relaxed \
    --out $R/relax_check_smooth_trained_$J || rc=1
echo "relax check done, rc $rc $(date)"
$PY smooth_eval.py ff_any.py chunk --model_dir $NEW --structure $S/stmv \
    --forces --ckpt --cluster_atoms 25000 --save_energies \
    --out $R/stmv_forces_smooth_$J || rc=1
echo "end rc $rc $(date)"
exit $rc
