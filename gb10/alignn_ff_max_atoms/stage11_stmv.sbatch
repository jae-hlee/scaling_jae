#!/bin/bash
#SBATCH --job-name=ffstmv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage11_%j.out

# Stage 11: one evaluation of energy and forces of a complete all-atom virus
# on one GB10, AFF-r2SCAN: the NAMD STMV benchmark, 1,066,628 atoms (capsid,
# RNA, water, ions). About 40 minutes at the rate measured for water (481
# atoms/s), in pieces of 25,000 atoms like the water runs (49 GB). Halo 20.01 A:
# the 52-neighbor cap binds in water, which changed forces by 8e-8 eV/A there.
# This is a test of size and cost: the model was trained on inorganic
# crystals and its accuracy on biomolecules is unknown.
# First, on the login node (the compute nodes may have no internet):
#   mkdir -p /data/jlee859/ffmax/stmv
#   curl -fL -o /data/jlee859/ffmax/stmv/stmv.tar.gz \
#       https://www.ks.uiuc.edu/Research/namd/utilities/stmv.tar.gz
#   cd /data/jlee859/ffmax && sbatch stage11_stmv.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage11
S=$B/structures
D=$B/stmv
mkdir -p $R $S && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
[ -s $D/stmv.tar.gz ] || { echo "download stmv.tar.gz first (see the top of this script)"; exit 5; }
[ -d $D/stmv ] || tar xzf $D/stmv.tar.gz -C $D || { echo "could not unpack"; exit 6; }
ls -la $D/stmv | head -n 12
grep -n -i "cellBasisVector\|cellOrigin" $D/stmv/*.namd

$PY $B/stmv_convert.py --src $D/stmv --out $S/stmv || { echo "conversion failed"; exit 7; }
$PY $B/ff_any.py chunk --model_dir $B/models/$M --structure $S/stmv --forces \
    --ckpt --cluster_atoms 25000 --save_energies --out $R/stmv_forces_$J
rc=$?
echo "exit $rc end $(date)"
exit $rc
