#!/bin/bash
#SBATCH --job-name=ffgsf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage14gsf_%j.out

# Stage 14: can these models describe slip in aluminum? stacking_fault.py
# computes, for each model in double precision on one GB10:
#   1. the lattice constant and bulk modulus of FCC Al
#   2. the stacking-fault energy on (111) along [11-2] and [1-10], as built
#      and relaxed along z (576-atom slab, 24 layers, vacuum above)
#   3. the (111) surface energy
#   4. the as-built gamma surface on a 12 x 21 grid
#   5. the intrinsic fault in a 36-layer slab, as a check on the thickness
# and prints a table next to Table 1 of Choudhary et al., J. Phys.: Condens.
# Matter 27, 015003 (2015). Before any dislocation run: if the intrinsic and
# unstable fault energies are far off, the model cannot describe slip. The
# lines of AFF-r2SCAN may show steps where bonds across the fault pass 3.5 A.
# Run time not measured yet; small cells, expect well under the limit.
#   cd /data/jlee859/ffmax && sbatch stage14_gsf.sbatch
# Other models or settings (MODELS="" runs only the SMOOTH ones):
#   SMOOTH=aff_smooth sbatch stage14_gsf.sbatch      retrained model as well
#   MODELS=matpes_r2scan_paper sbatch stage14_gsf.sbatch
#   ELEMENT=Cu sbatch stage14_gsf.sbatch             (no paper values for Cu)
#   EXTRA="--layers 30" sbatch stage14_gsf.sbatch    more stacking_fault.py options
set -uo pipefail
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
MODELS=${MODELS-"matpes_r2scan_paper alignn_ff_mptrj alignn_ff_matpes_pbe alignn_ff_matpes_r2scan"}
SMOOTH=${SMOOTH-""}
ELEMENT=${ELEMENT:-Al}
EXTRA=${EXTRA-""}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
TAG=gsf_$(echo $ELEMENT | tr A-Z a-z)_$J
cd $B
echo "start $(date) | $(hostname) | $ELEMENT | models: $MODELS | smooth: $SMOOTH"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }

rc=0
ran=0
for M in $MODELS; do
    [ -s models/$M/best_model.pt ] || { echo "no models/$M, skipped"; continue; }
    mkdir -p results/$M/stage14
    echo "== $M $(date)"
    PYTHONPATH=$B/alignn_6a20fb5 $PY stacking_fault.py run \
        --model_dir models/$M --element $ELEMENT \
        --out results/$M/stage14/$TAG $EXTRA || rc=1
    ran=$((ran + 1))
done
for M in $SMOOTH; do
    [ -s models/$M/best_model.pt ] || { echo "no models/$M, skipped"; continue; }
    [ -d $B/alignn_smooth ] || { echo "no $B/alignn_smooth (stage 12 makes it)"; rc=1; continue; }
    mkdir -p results/$M/stage14
    echo "== $M (smooth_messages) $(date)"
    PYTHONPATH=$B/alignn_smooth $PY smooth_eval.py stacking_fault.py run \
        --model_dir models/$M --element $ELEMENT \
        --out results/$M/stage14/$TAG $EXTRA || rc=1
    ran=$((ran + 1))
done
[ $ran -gt 0 ] || { echo "no model ran"; exit 5; }
echo "== summary $(date)"
PYTHONPATH=$B/alignn_6a20fb5 $PY stacking_fault.py summary \
    --glob "results/*/stage14/$TAG.json" || rc=1
echo "end rc $rc $(date)"
exit $rc
