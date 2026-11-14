#!/bin/bash
#SBATCH --job-name=ffmax16s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage16small_%j.out

# Stage 16, part 1: the tests of stages 1, 2 and 5 again with the retrained
# model (models/aff_smooth, smooth_messages), so that the letter reports every
# number with one model. Same cells, options and piece sizes as AFF-r2SCAN:
#   1. largest Si cell per path (e, ckpt, ef, ase), each its own process
#      (normally ends with exit 137 when the host stops the process)
#   2. chunked against full cell on 8,000 atoms of rattled SiGe: float32
#      with and without forces (as check_*_19126), float64 with forces (as
#      f64_sige_exact_19291)
#   3. halo widths 8 to 30 A on 8,000 atoms of rattled Si, float64 (as
#      halo_*_19126)
#   4. 8M atoms of rattled Si on this GB10, with forces (pieces of 117,067
#      atoms) and energy only (330,820), as s2_8M_forces and s2_8M_energy
# Results in results/aff_smooth/stage16/. The 64M runs and the costs by
# structure are separate jobs (stage16_submit.sh submits all of them).
#   cd /data/jlee859/ffmax && sbatch --exclude=atomgptlab03 stage16_small.sbatch
set -uo pipefail
M=aff_smooth
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
MD=$B/models/$M
R=$B/results/$M/stage16
mkdir -p $R && cd $B
echo "start $(date) | $(hostname) | model $M"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $MD && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
cat $MD/model.md5
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2

# the ASE path builds its own model from config.json: make sure it is the
# patched one with smooth_messages on, or the ase sweep would time another model
$PY - $MD <<'EOF' || { echo "the ASE calculator does not build the smooth model"; exit 4; }
import sys
from alignn.ff.calculators import AlignnAtomwiseCalculator
m = AlignnAtomwiseCalculator(path=sys.argv[1], device="cpu").model
on = bool(getattr(m, "smooth_messages", False))
print("ASE calculator model:", type(m).__name__, "| smooth_messages", on)
sys.exit(0 if on else 1)
EOF

ff() { $PY $B/smooth_ffm.py "$@"; }

for p in e ckpt ef ase; do
    ff sweep --model_dir $MD --path $p --out $R/sweep_${p}_$J
    echo "sweep $p exit $? $(date)"
done

ff chunk --model_dir $MD --rep 10 --k 2 --species SiGe --rattle 0.05 \
    --forces --ckpt --validate --out $R/check_forces_$J
echo "check forces exit $? $(date)"
ff chunk --model_dir $MD --rep 10 --k 2 --species SiGe --rattle 0.05 \
    --validate --out $R/check_energy_$J
echo "check energy exit $? $(date)"
ff chunk --model_dir $MD --rep 10 --k 2 --species SiGe --rattle 0.05 \
    --forces --ckpt --validate --float64 --out $R/f64_sige_exact_$J
echo "f64 SiGe exit $? $(date)"

for H in 8 12 16 20 25 30; do
    ff chunk --model_dir $MD --rep 10 --k 2 --rattle 0.05 --forces \
        --ckpt --validate --float64 --halo $H --out $R/halo_${H}_$J
    echo "halo $H exit $? $(date)"
done

ff chunk --model_dir $MD --rep 100 --rattle 0.05 --forces --ckpt \
    --cluster_atoms 117067 --out $R/s16_8M_forces
echo "8M forces exit $? $(date)"
ff chunk --model_dir $MD --rep 100 --rattle 0.05 \
    --cluster_atoms 330820 --out $R/s16_8M_energy
echo "8M energy exit $? $(date)"
echo "end $(date)"
