#!/bin/bash
#SBATCH --job-name=ffmax15r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage15ref_%j.out

# Stage 15 reference: the energy of perfect diamond Si (a = 5.43 A, no
# displacements, 8,000 atoms) with the retrained model, needed for the
# grain-boundary excess of the retrained 1B polycrystal (AFF-r2SCAN's
# reference, -8.747559 eV/atom, came from its single-graph sweep in float32).
# AFF-r2SCAN is run the same way as a control and should give -8.74756.
# Each run is ff_stream.py on one task, 2 x 2 x 2 cubes, checked against the
# full cell (--validate), in float32 and float64.
#   cd /data/jlee859/ffmax && sbatch --exclude=atomgptlab03 stage15_ref.sbatch
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/aff_smooth/stage15
J=$SLURM_JOB_ID
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
(cd $B/models/aff_smooth && md5sum -c --quiet model.md5) || { echo "models/aff_smooth differs from its model.md5"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }

rc=0
for P in f32 f64; do
    X=""; [ $P = f64 ] && X="--float64"
    # retrained model: patched ALIGNN and smooth_eval.py, as stream.sbatch with SMOOTH=1
    PYTHONPATH=$B/alignn_smooth $PY smooth_eval.py ff_stream.py chunk \
        --model_dir $B/models/aff_smooth --kind Si --rep 10 --rattle 0 \
        --forces --ckpt --k 2 --validate $X --out $R/ref_si_smooth_${P}_$J || rc=1
    # AFF-r2SCAN control, as stream.sbatch without SMOOTH
    PYTHONPATH=$B/alignn_6a20fb5 $PY ff_stream.py chunk \
        --model_dir $B/models/matpes_r2scan_paper --kind Si --rep 10 --rattle 0 \
        --forces --ckpt --k 2 --validate $X --out $R/ref_si_aff_${P}_$J || rc=1
done

echo "summary:"
for f in $R/ref_si_*_$J.json; do
    $PY - "$f" <<'EOF' || rc=1
import json, sys
d = json.load(open(sys.argv[1]))
v = d.get("validation", {})
print(f"{sys.argv[1].rsplit('/', 1)[-1]}: chunked {d.get('e_per_atom')!r} eV/atom, "
      f"full cell {v.get('e_full_per_atom')!r}, largest force {v.get('max_abs_f_full')!r} eV/A")
EOF
done
echo "end rc $rc $(date)"
exit $rc
