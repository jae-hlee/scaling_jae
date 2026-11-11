#!/bin/bash
#SBATCH --job-name=ffmax15c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage15check_%j.out

# Stage 15 check, before the billion-atom runs with the retrained model
# (models/aff_smooth): chunked against full cell in double precision, through
# both programs the runs use, with the trained weights.
#   1. ff_stream.py: 8,000 atoms of rattled Si, 2 x 2 x 2 cubes (as stage 9)
#   2. ff_any.py: the 772-atom gate stack, 2 x 2 x 2 cubes (as stage 12)
# Fails (nonzero exit) unless both agree with the full cell to 1e-9 eV/A in
# the forces and 1e-11 eV/atom in the energy; stage15_submit.sh makes the big
# runs wait for this job and starts nothing if it fails. On success it writes
# results/aff_smooth/stage15/check_passed, after which stage15_submit.sh
# skips the check.
#   cd /data/jlee859/ffmax && sbatch stage15_check.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
M=aff_smooth
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M/stage15
J=$SLURM_JOB_ID
mkdir -p $R && cd $B
rm -f $R/check_passed
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
[ -s $B/models/$M/model.md5 ] || { echo "no models/$M (stage13_compare.sbatch makes it)"; exit 5; }
(cd $B/models/$M && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
cat $B/models/$M/model.md5
[ -s $B/structures/stack_small.json ] || { echo "no structures/stack_small (stage 10 check builds it)"; exit 6; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }

rc=0
$PY smooth_eval.py ff_stream.py chunk --model_dir $B/models/$M \
    --kind Si --rep 10 --rattle 0.05 --forces --ckpt --float64 --k 2 --validate \
    --out $R/exact_stream_si_$J || rc=1
echo "ff_stream done, rc $rc $(date)"
$PY smooth_eval.py ff_any.py chunk --model_dir $B/models/$M \
    --structure $B/structures/stack_small --kxyz 2 2 2 --forces --ckpt --float64 --validate \
    --out $R/exact_any_stack_$J || rc=1
echo "ff_any done, rc $rc $(date)"

$PY - $R/exact_stream_si_$J.json $R/exact_any_stack_$J.json <<'EOF' || rc=1
import json, sys
bad = False
for path in sys.argv[1:]:
    try:
        v = json.load(open(path))["validation"]
    except (OSError, KeyError, ValueError) as err:
        print(f"{path}: no validation ({type(err).__name__})")
        bad = True
        continue
    df, de = v.get("max_abs_df"), v.get("max_abs_de_per_atom")
    ok = df is not None and de is not None and df <= 1e-9 and de <= 1e-11
    print(f"{path.rsplit('/', 1)[-1]}: forces differ by {df:.2e} eV/A "
          f"(largest force {v.get('max_abs_f_full', float('nan')):.2f}), "
          f"energy by {de:.2e} eV/atom: {'PASS' if ok else 'FAIL'}")
    bad = bad or not ok
sys.exit(1 if bad else 0)
EOF
[ $rc -eq 0 ] && date > $R/check_passed
echo "end rc $rc $(date)"
exit $rc
