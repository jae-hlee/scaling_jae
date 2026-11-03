#!/bin/bash
#SBATCH --job-name=ffmax1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage1_%j.out

# Stage 1 for one model on one GB10:
#   1. largest Si cell per path (e, ckpt, ef, ase), each its own process
#   2. chunked vs full cell on 8,000 atoms of rattled SiGe (forces, energy)
#   3. timing sample of the chunked run on an 8M-atom Si cell (3 pieces)
#   4. error of shorter halos against the full cell
#   sbatch stage1.sbatch <model folder under models/>
set -uo pipefail
M=${1:?model folder under /data/jlee859/ffmax/models}
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
R=$B/results/$M
J=$SLURM_JOB_ID
mkdir -p $R && cd $R
ff() { $PY $B/ff_max_atoms.py "$@"; }
echo "start $(date) | $(hostname) | model $M"
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
$PY -c "import matscipy" 2>/dev/null && echo "neighbor list: matscipy" || echo "neighbor list: ase (no matscipy)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2

for p in e ckpt ef ase; do
    ff sweep --model_dir $B/models/$M --path $p --out $R/sweep_${p}_$J
    echo "sweep $p exit $? $(date)"
done

# 60% of the largest cell that fit, as the target size of one piece
cap() { $PY -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; print(int(0.6*r[-1]['natoms']) if r else 20000)" "$1"; }
CF=$(cap $R/sweep_ckpt_$J.json)
CE=$(cap $R/sweep_e_$J.json)
CF=${CF:-20000}
CE=${CE:-20000}
echo "piece size: forces $CF, energy only $CE"

ff chunk --model_dir $B/models/$M --rep 10 --k 2 --species SiGe --rattle 0.05 \
    --forces --ckpt --validate --out $R/check_forces_$J
echo "check forces exit $?"
ff chunk --model_dir $B/models/$M --rep 10 --k 2 --species SiGe --rattle 0.05 \
    --validate --out $R/check_energy_$J
echo "check energy exit $?"

ff chunk --model_dir $B/models/$M --rep 100 --cluster_atoms $CF --forces --ckpt \
    --max_chunks 3 --out $R/sample_forces_$J
echo "sample forces exit $?"
ff chunk --model_dir $B/models/$M --rep 100 --cluster_atoms $CE \
    --max_chunks 3 --out $R/sample_energy_$J
echo "sample energy exit $?"

# shorter halos on 8,000 atoms of rattled Si, double precision, each
# against the full cell (the exact halo is 20 A for 2+2, 40 A for 4+4)
for H in 8 12 16 20 25 30; do
    ff chunk --model_dir $B/models/$M --rep 10 --k 2 --rattle 0.05 --forces \
        --ckpt --validate --float64 --halo $H --out $R/halo_${H}_$J
    echo "halo $H exit $?"
done
echo "end $(date)"
