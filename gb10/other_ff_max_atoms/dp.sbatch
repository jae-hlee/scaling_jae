#!/bin/bash
#SBATCH --job-name=ffodp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/data/jlee859/ffother/logs/dp_%j.out

# DP (Deep Potential, DeePMD-kit 3.2.0, PyTorch backend) on one GB10:
#   install  build DeePMD-kit from source against the env's torch 2.13, with
#            its CUDA kernels (needed for a compressed model on the GPU), into
#            its own venv on top of the alignn env (which is not changed). If
#            the CUDA build fails, a build without the kernels is made and
#            only the uncompressed model can be swept
#   train    an se_e2_a model for Si on the MatPES r2SCAN Si frames
#            (data.sbatch) with the settings of the copper records: cutoff
#            8 A, embedding net 32-64-128, fitting net 240-240-240, double
#            precision; sel from the training data. Then freeze, compress
#            (the 2022 record ran compressed) and errors on the test frames
#   sweep    sweep.py through DeepPot with the vesin neighbor list (linear in
#            the number of atoms), uncompressed and compressed
# This is the Python path, not LAMMPS. Both hand the model a neighbor list
# and evaluate all atoms in one pass.
# A stage whose result exists is skipped, so a resubmitted job continues.
#   STAGES="sweep" sbatch dp.sbatch
#   STEPS=200000 sbatch dp.sbatch           training steps (default 100000)
#   RETRAIN=1 sbatch dp.sbatch              train again from scratch
#   FRESH=1 sbatch dp.sbatch                sweep again from scratch
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 dp.sbatch
set -uo pipefail
B=/data/jlee859/ffother
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
D=$B/data/si
V=$B/venv_dp
T=$B/dp/train
R=$B/results/dp
J=$SLURM_JOB_ID
STAGES=${STAGES:-"install train sweep"}
STEPS=${STEPS:-100000}
export PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p $B/dp $R $B/logs && cd $B
echo "start $(date) | $(hostname) | stages: $STAGES"

# ---------------------------------------------------------------- install
build_dp() {  # build_dp cuda|cpu
    export DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 DP_VARIANT=$1
    export CUDAToolkit_ROOT=/usr/local/cuda
    export PYTORCH_ROOT=$($PY -c "import torch; print(torch.__path__[0])")
    # PyTorch's CMake detects the GB10's compute capability 12.1, turns it
    # into "12.1(2.0)" with a text replacement meant for capability 2.1, and
    # asks nvcc for compute_20, which CUDA 13 does not have (job 21869). An
    # explicit list skips the detection. 9.0+PTX adds portable code that the
    # driver compiles for the GB10 when the library loads, and works whether
    # or not this PyTorch accepts a two-digit capability in this list
    export TORCH_CUDA_ARCH_LIST="9.0+PTX"
    # e3nn: the PyTorch backend imports it, but lists it only in an extra that
    # also pins another torch. --no-cache-dir: a wheel built for one variant
    # must not be reused for the other. An installed deepmd-kit is rebuilt
    # in place (--force-reinstall --no-deps); pip keeps it if the build fails
    local again=()
    $V/bin/python -c "import deepmd" 2>/dev/null && again=(--force-reinstall --no-deps)
    $V/bin/python -m pip install --no-cache-dir -c $B/dp/constraints.txt "${again[@]}" \
        --no-binary deepmd-kit deepmd-kit==3.2.0 > $B/logs/dp_pip_${1}_$J.log 2>&1 || return 1
    $V/bin/python -m pip install --no-cache-dir -c $B/dp/constraints.txt \
        "vesin[torch]" "e3nn>=0.5.9" >> $B/logs/dp_pip_${1}_$J.log 2>&1
}
if [[ " $STAGES " == *" install "* ]]; then
    if [ -x $V/bin/dp ] && [ "$(cat $V/dp_variant 2>/dev/null)" = cuda ]; then
        echo "install: already there, variant cuda"
    elif [ -x $V/bin/dp ] && [ "${NO_CUDA:-0}" = 1 ]; then
        echo "install: keeping the build without CUDA kernels (NO_CUDA=1)"
    else
        [ -x $V/bin/python ] || $PY -m venv --system-site-packages $V \
            || { echo "install: could not make the venv"; exit 5; }
        echo "torch==$($PY -c 'import torch; print(torch.__version__)')" > $B/dp/constraints.txt
        cat $B/dp/constraints.txt
        t0=$SECONDS
        if build_dp cuda; then
            echo cuda > $V/dp_variant
        else
            echo "install: the CUDA build FAILED, errors and last lines of logs/dp_pip_cuda_$J.log:"
            grep -n -m 8 -E 'error|Error|fatal|gencode' $B/logs/dp_pip_cuda_$J.log | cut -c1-300
            tail -n 12 $B/logs/dp_pip_cuda_$J.log | cut -c1-300
            if $V/bin/python -c "import deepmd" 2>/dev/null; then
                echo "install: keeping the earlier build without CUDA kernels"
            else
                echo "install: building without the CUDA kernels"
                build_dp cpu || { echo "install: that FAILED too:"; tail -n 25 $B/logs/dp_pip_cpu_$J.log | cut -c1-300; exit 5; }
            fi
            echo cpu > $V/dp_variant
        fi
        echo "install: built variant $(cat $V/dp_variant) in $((SECONDS - t0)) s"
    fi
fi
[ -x $V/bin/dp ] || { echo "no $V/bin/dp (run the install stage)"; exit 5; }
$V/bin/python -c "import e3nn" 2>/dev/null \
    || $V/bin/python -m pip install -q -c $B/dp/constraints.txt "e3nn>=0.5.9" \
    || { echo "install: could not install e3nn"; exit 5; }
VARIANT=$(cat $V/dp_variant 2>/dev/null || echo unknown)
$V/bin/python - <<'EOF' || { echo "the venv does not import"; exit 5; }
import sys
sys.path.insert(0, "/data/jlee859/ffother")
import py_eval
py_eval.preload_nvrtc()  # as py_eval.py does before DeepPot (see there)
import importlib.metadata as md
import torch
print("torch", torch.__version__, "from", torch.__file__)
print("deepmd-kit", md.version("deepmd-kit"), "| vesin", md.version("vesin"))
assert torch.__version__.startswith("2.13.0"), "pip replaced torch"
assert torch.cuda.is_available()
from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP
print("custom op library loaded:", ENABLE_CUSTOMIZED_OP)
import e3nn
import deepmd.pt.entrypoints.main
import deepmd.pt.infer.deep_eval
print("deepmd PyTorch backend imports, e3nn", md.version("e3nn"))
try:
    from deepmd.env import GLOBAL_CONFIG
    print("built variant:", GLOBAL_CONFIG.get("dp_variant"))
except Exception as err:
    print("built variant: not readable,", type(err).__name__)
import vesin.torch
try:  # the neighbor list of the sweep, on the GPU (it loads libnvrtc)
    p = torch.rand(64, 3, dtype=torch.float64, device="cuda") * 10
    b = torch.eye(3, dtype=torch.float64, device="cuda") * 10
    i, j, S = vesin.torch.NeighborList(cutoff=8.0, full_list=True).compute(
        points=p, box=b, periodic=True, quantities="ijS")
    print("vesin on the GPU:", int(i.shape[0]), "pairs for 64 points")
except Exception as err:
    print("WARNING, vesin on the GPU failed:", type(err).__name__, str(err)[:300])
print("libnvrtc loaded:", py_eval.loaded_nvrtc())
EOF

# ------------------------------------------------------------------ train
if [[ " $STAGES " == *" train "* ]]; then
    [ -d $D/deepmd/train ] || { echo "no $D/deepmd/train (run data.sbatch first)"; exit 6; }
    [ "${RETRAIN:-0}" = 1 ] && rm -rf $T
    mkdir -p $T && cd $T
    # train_done holds the number of steps of the finished training, so a
    # different STEPS continues from the last checkpoint and freezes again
    if [ -s model.pth ] && [ "$(cat train_done 2>/dev/null)" != "$STEPS" ]; then
        rm -f model.pth model-compress.pth train_done
    fi
    if [ -s model.pth ]; then
        echo "train: already there"
    else
        cat > input.json <<EOF
{
 "model": {
  "type_map": ["Si"],
  "descriptor": {"type": "se_e2_a", "sel": "auto", "rcut_smth": 0.5, "rcut": 8.0,
                 "neuron": [32, 64, 128], "resnet_dt": false, "axis_neuron": 16,
                 "type_one_side": true, "seed": 1},
  "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": true, "seed": 1},
  "data_stat_nbatch": 20
 },
 "learning_rate": {"type": "exp", "decay_steps": 5000, "start_lr": 0.001, "stop_lr": 3.51e-8},
 "loss": {"type": "ener", "start_pref_e": 0.02, "limit_pref_e": 1,
          "start_pref_f": 1000, "limit_pref_f": 1, "start_pref_v": 0, "limit_pref_v": 0},
 "training": {
  "training_data": {"systems": "$D/deepmd/train", "batch_size": "auto"},
  "validation_data": {"systems": "$D/deepmd/val", "batch_size": "auto", "numb_btch": 3},
  "numb_steps": $STEPS, "seed": 10, "disp_file": "lcurve.out",
  "disp_freq": 1000, "save_freq": 10000
 }
}
EOF
        RESTART=()
        # the file "checkpoint" names the last complete checkpoint
        CKPT=$(head -n 1 checkpoint 2>/dev/null)
        [ -n "$CKPT" ] && [ -e "$CKPT" ] || CKPT=$(ls -t model.ckpt*.pt 2>/dev/null | head -n 1)
        [ -n "$CKPT" ] && RESTART=(--restart $CKPT) && echo "train: RESUMING from $CKPT"
        echo "train: $(ls $D/deepmd/train | wc -l) training systems, $STEPS steps"
        t0=$SECONDS
        $V/bin/dp --pt train input.json "${RESTART[@]}" > train.log 2>&1
        rc=$?
        echo "train: dp train exit $rc after $((SECONDS - t0)) s"
        if [ $rc -ne 0 ]; then
            echo "train: FAILED, last lines of $T/train.log:"
            tail -n 40 train.log | cut -c1-300
            exit 7
        fi
        grep -i -E 'sel|neighbor|nbor' train.log | head -n 8 | cut -c1-300
        $V/bin/dp --pt freeze -o model.pth > freeze.log 2>&1 \
            || { echo "train: freeze FAILED:"; tail -n 20 freeze.log | cut -c1-300; exit 7; }
        echo "train: frozen to model.pth"
        echo $STEPS > train_done
    fi
    echo "train: lcurve.out, header and last line:"
    head -n 1 lcurve.out | cut -c1-300; tail -n 1 lcurve.out | cut -c1-300
    if [ ! -s model-compress.pth ]; then
        $V/bin/dp --pt compress -i model.pth -o model-compress.pth > compress.log 2>&1 \
            && echo "train: compressed to model-compress.pth" \
            || { echo "train: compress FAILED (only the uncompressed model will be swept):"; tail -n 12 compress.log | cut -c1-300; rm -f model-compress.pth; }
    fi
    # through py_eval.py with DeePMD's own neighbor list, which is fine for
    # these cells of at most 24 atoms (dp test chose vesin, which failed on
    # this GPU in job 21869)
    echo "test frames, DP against MatPES r2SCAN:"
    $V/bin/python $B/py_eval.py errors --kind dp --model model.pth --nlist native \
        --xyz $D/test.extxyz --out $R/test_errors_$J.json \
        || echo "test: FAILED"
    cp input.json lcurve.out $R/ 2>/dev/null
    cd $B
fi

# ------------------------------------------------------------------ sweep
rc=0
if [[ " $STAGES " == *" sweep "* ]]; then
    [ -s $T/model.pth ] || { echo "no $T/model.pth (train first)"; exit 7; }
    # on the GPU vesin reserves room for 512 pairs per atom at an 8 A cutoff
    # (about 14 kB per atom). Diamond Si has 98, so 128 is enough here, and
    # too few is an error, not a wrong result
    export VESIN_CUDA_MAX_PAIRS_PER_POINT=128
    # the neighbor list of DeepPot for the sweep: vesin or nv (both linear in
    # the number of atoms); native builds all pairs and would measure that
    NLIST=${NLIST:-vesin}
    FRESH_FLAG=()
    [ "${FRESH:-0}" = 1 ] && FRESH_FLAG=(--fresh)
    $PY sweep.py python --exe $V/bin/python \
        --exe_args="$B/py_eval.py size --kind dp --model $T/model.pth --nlist $NLIST" \
        --steps 5 --start 10 --work $B/work/dp --out $R/sweep "${FRESH_FLAG[@]}" \
        --note "DeePMD-kit 3.2.0 pt, se_e2_a rcut 8, uncompressed, DeepPot with $NLIST, build $VARIANT, job $J" || rc=1
    if [ "$VARIANT" != cuda ]; then
        echo "sweep: built without the CUDA kernels, so a compressed model cannot run on the GPU: no compressed sweep"
    elif [ -s $T/model-compress.pth ]; then
        $PY sweep.py python --exe $V/bin/python \
            --exe_args="$B/py_eval.py size --kind dp --model $T/model-compress.pth --nlist $NLIST" \
            --steps 5 --start 10 --work $B/work/dp_compressed --out $R/sweep_compressed "${FRESH_FLAG[@]}" \
            --note "DeePMD-kit 3.2.0 pt, se_e2_a rcut 8, compressed, DeepPot with $NLIST, build $VARIANT, job $J" || rc=1
    else
        echo "sweep: no compressed model, so no compressed sweep"
    fi
fi
echo "end rc $rc $(date)"
exit $rc
