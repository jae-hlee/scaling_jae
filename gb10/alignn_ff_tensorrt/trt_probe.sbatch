#!/bin/bash
#SBATCH --job-name=fftrtprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/jlee859/fftrt/logs/probe_%j.out

# Can ALIGNN-FF run through TensorRT on a GB10 without accuracy loss? Answers
# the open questions before any real work (trt_probe.py has the details):
#   1. do TensorRT 11.4 (tensorrt-cu13) and TensorRT-RTX 1.7 (tensorrt-rtx)
#      install, load and run on this driver (580, CUDA 13.0; both are built
#      with CUDA 13.4), and does the TensorRT-RTX runtime cache work
#   2. does each import ONNX ScatterElements with reduction=add, the sum
#      ALIGNN uses (TensorRT-RTX has no plugins, so probably not)
#   3. can AFF-r2SCAN's energies and forces (with the gradient traced into
#      plain operations) be exported to ONNX, and does an engine built from
#      it reproduce PyTorch, on rattled Si of 512 and 8,000 atoms
# Each step is its own process; a failing step does not stop the others.
# The two libraries go in their own venvs on top of the alignn env, which is
# not changed. About 30 minutes.
#   mkdir -p /data/jlee859/fftrt/logs
#   cd /data/jlee859/fftrt && sbatch --exclude=atomgptlab03 trt_probe.sbatch
set -uo pipefail
B=/data/jlee859/fftrt
FF=/data/jlee859/ffmax
BASE=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
R=$B/results
J=$SLURM_JOB_ID
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$FF/alignn_6a20fb5:$FF
export PYTHONUNBUFFERED=1
mkdir -p $R $B/logs && cd $B
echo "start $(date) | $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
(cd $FF/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }

echo "torch==$($BASE -c 'import torch; print(torch.__version__)')" > $B/constraints.txt
cat $B/constraints.txt
make_venv() {  # make_venv <dir> <packages...>
    local v=$1
    shift
    [ -x $v/bin/python ] || $BASE -m venv --system-site-packages $v || return 1
    $v/bin/python -m pip install -q -c $B/constraints.txt "$@" \
        > $B/logs/pip_$(basename $v)_$J.log 2>&1 \
        || { echo "pip into $v FAILED:"; tail -n 20 $B/logs/pip_$(basename $v)_$J.log; return 1; }
    $v/bin/python - <<'EOF'
import importlib.metadata as md
import torch
print("  torch", torch.__version__, "| cuda available", torch.cuda.is_available())
for p in ("tensorrt-cu13", "tensorrt_cu13_libs", "tensorrt-rtx",
          "tensorrt_rtx_cu13_libs", "onnx", "onnxscript"):
    try:
        print("  ", p, md.version(p))
    except md.PackageNotFoundError:
        pass
assert torch.__version__.startswith("2.13.0"), "pip replaced torch"
EOF
}
VT=$B/venv_trt
VR=$B/venv_rtx
make_venv $VT tensorrt-cu13==11.4.0.106 onnx onnxscript || echo "no TensorRT venv"
make_venv $VR tensorrt-rtx==1.7.1.116 onnx || echo "no TensorRT-RTX venv"

step() {  # step <label> <python> <args...>
    local label=$1 py=$2
    shift 2
    echo
    echo "=== $label  $(date +%T)"
    if [ ! -x "$py" ]; then echo "SKIP (no $py)"; return; fi
    $py $B/trt_probe.py "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn' | cut -c1-4000
    echo "=== $label: exit ${PIPESTATUS[0]}"
}

step "1. TensorRT 11.4, API network" $VT/bin/python basic --lib trt --workdir $R
step "1. TensorRT-RTX 1.7, API network and runtime cache" $VR/bin/python basic --lib rtx --workdir $R
step "2. ONNX scatter-add model" $VT/bin/python make-scatter --out $R/scatter_add.onnx
step "2. TensorRT 11.4, scatter-add" $VT/bin/python scatter --lib trt --onnx $R/scatter_add.onnx
step "2. TensorRT-RTX 1.7, scatter-add" $VR/bin/python scatter --lib rtx --onnx $R/scatter_add.onnx
for REP in 4 10; do
    P=$R/aff_si$REP
    step "3. export AFF-r2SCAN, Si rep $REP" $VT/bin/python export \
        --model_dir $FF/models/matpes_r2scan_paper --rep $REP --out $P
    # as traced, and with accumulating index writes rewritten as scatter_add
    for Q in $P ${P}_sa; do
        [ -s $Q.onnx ] || { echo "no $Q.onnx"; continue; }
        step "3. TensorRT 11.4, $(basename $Q)" $VT/bin/python model --lib trt --prefix $Q
        step "3. TensorRT-RTX 1.7, $(basename $Q)" $VR/bin/python model --lib rtx --prefix $Q
    done
done
echo
echo "summary:"
# grep refuses to read the file it writes to, so read a copy
cp $B/logs/probe_$J.out $B/logs/.probe_$J.copy
grep -E '^(=== .*exit|PASS|FAIL) ' $B/logs/.probe_$J.copy | cut -c1-300
rm -f $B/logs/.probe_$J.copy
echo "end $(date)"
