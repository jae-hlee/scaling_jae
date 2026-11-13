#!/bin/bash
#SBATCH --job-name=fftrtscale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/data/jlee859/fftrt/logs/scale_%j.out

# AFF-r2SCAN energies and forces at real piece sizes (trt_scale.py): PyTorch
# eager, one TensorRT 11.4 engine with dynamic sizes, and torch.compile of the
# same traced graph, on rattled Si of 8,000, 32,768, 74,088 and 110,592 atoms
# (the force runs use pieces of 70,000 to 106,000). Speed, memory, and errors
# against PyTorch float32 (and float64 up to 32,768 atoms). If the dynamic
# export or build fails, one static engine per size is built instead.
# Uses the venv of trt_probe.sbatch (venv_trt). About an hour.
#   cd /data/jlee859/fftrt && sbatch --exclude=atomgptlab03 trt_scale.sbatch
set -uo pipefail
B=/data/jlee859/fftrt
FF=/data/jlee859/ffmax
PY=$B/venv_trt/bin/python
M=$FF/models/matpes_r2scan_paper
R=$B/results/scale
J=$SLURM_JOB_ID
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$FF/alignn_6a20fb5:$FF
export PYTHONUNBUFFERED=1
mkdir -p $R && cd $B
echo "start $(date) | $(hostname)"
[ -x $PY ] || { echo "no $PY (run trt_probe.sbatch first)"; exit 5; }
(cd $FF/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }

step() {  # step <label> <args...>
    local label=$1
    shift
    echo
    echo "=== $label  $(date +%T)"
    $PY $B/trt_scale.py "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn|Make sure input .* binding' | cut -c1-4000
    local rc=${PIPESTATUS[0]}
    echo "=== $label: exit $rc"
    return $rc
}

REPS="10 16 21 24"
for REP in $REPS; do
    F64=()
    [ $REP -le 16 ] && F64=(--f64)
    step "prepare rep $REP" prepare --model_dir $M --rep $REP "${F64[@]}" --out $R/si$REP
done

# one dynamic engine for every size
DYN=no
if step "export, dynamic" export --model_dir $M --rep 4 --dynamic --out $R/aff_dyn \
   && step "build, dynamic" build --onnx $R/aff_dyn.onnx --plan $R/aff_dyn.plan \
          --max_data $R/si24 --opt_data $R/si21; then
    DYN=yes
    for REP in $REPS; do
        step "TensorRT dynamic engine, rep $REP" run --plan $R/aff_dyn.plan --data $R/si$REP
    done
fi

# otherwise one static engine per size
if [ $DYN = no ]; then
    echo "dynamic engine not available: static engines per size"
    for REP in $REPS; do
        step "export, static rep $REP" export --model_dir $M --rep $REP --out $R/aff_si$REP \
            && step "build, static rep $REP" build --onnx $R/aff_si$REP.onnx --plan $R/aff_si$REP.plan \
            && step "TensorRT static engine, rep $REP" run --plan $R/aff_si$REP.plan --data $R/si$REP
        rm -f $R/aff_si$REP.plan
    done
fi

for REP in $REPS; do
    step "torch.compile, rep $REP" inductor --model_dir $M --data $R/si$REP
done

echo
echo "summary:"
cp $B/logs/scale_$J.out $B/logs/.scale_$J.copy
grep -E '^(PASS|FAIL) ' $B/logs/.scale_$J.copy | cut -c1-300
rm -f $B/logs/.scale_$J.copy
echo "end $(date)"
