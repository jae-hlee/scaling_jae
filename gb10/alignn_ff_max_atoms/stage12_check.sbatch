#!/bin/bash
#SBATCH --job-name=ffsmooth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage12check_%j.out

# Stage 12 check: does smooth_messages.patch remove the force jumps, before
# any training? No training here: AFF-r2SCAN's weights with the flag on.
#   1. the patched copy of ALIGNN, $B/alignn_smooth (alignn_6a20fb5 untouched)
#   2. smooth_check.py: flag off reproduces unpatched ALIGNN exactly, the
#      training path agrees with forward_tensors, forces match finite
#      differences
#   3. relax_check.py on the relaxed polycrystal: the bond-crossing test. The
#      force change across 3.5 A should drop from 0.47-0.51 eV/A (job 21603)
#      to the size of the control bond, about 0.005 eV/A. The ratio across/same
#      is about 0.22 for smooth forces but noisy at this size in float32, and
#      the bonds picked can differ from job 21603 because forces differ
#   4. chunked against full cell in float64 on the 772-atom gate stack
#   cd /data/jlee859/ffmax && sbatch stage12_check.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage12
S=$B/structures
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }

if [ ! -d $B/alignn_smooth ]; then
    cp -r $B/alignn_6a20fb5 $B/alignn_smooth
    patch -p1 -d $B/alignn_smooth < $B/smooth_messages.patch \
        || { echo "patch failed"; rm -rf $B/alignn_smooth; exit 5; }
fi
grep -c smooth_messages $B/alignn_smooth/alignn/models/alignn_atomwise_pure.py

rc=0
PYTHONPATH=$B/alignn_smooth $PY prep_smooth.py --src models/$M \
    --out models/aff_smooth_test || rc=1

PYTHONPATH=$B/alignn_6a20fb5 $PY smooth_check.py ref --model_dir models/$M \
    --out $R/smooth_ref_$J.npz || rc=1
PYTHONPATH=$B/alignn_smooth $PY smooth_check.py compare --model_dir models/$M \
    --smooth_dir models/aff_smooth_test --ref $R/smooth_ref_$J.npz || rc=1
echo "checks done, rc $rc $(date)"

PYTHONPATH=$B/alignn_smooth $PY smooth_eval.py relax_check.py \
    --model_dir models/aff_smooth_test \
    --relaxed $B/results/$M/stage7/relax_poly_si_150k_relaxed \
    --out $R/relax_check_smooth_$J || rc=1
echo "crossing test done, rc $rc $(date)"

PYTHONPATH=$B/alignn_smooth $PY smooth_eval.py ff_any.py chunk \
    --model_dir models/aff_smooth_test --structure $S/stack_small \
    --kxyz 2 2 2 --forces --ckpt --validate --float64 \
    --out $R/exact_smooth_stack_small_$J || rc=1
echo "end rc $rc $(date)"
exit $rc
