#!/bin/bash
#SBATCH --job-name=ffonep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffother/logs/nep_%j.out

# NEP (neuroevolution potential) on one GB10, with GPUMD v5.9.1:
#   build  compile gpumd and nep from source (the makefile's sm_60 and c++14
#          do not work with CUDA 13, so they become sm_121 and c++17)
#   train  a NEP4 for Si on the MatPES r2SCAN Si frames (data.sbatch), with
#          the size settings of the trillion-atom Cu NEP: cutoffs 6 and 4 A,
#          n_max 4 4, basis 8 8, 30 neurons, ZBL below 2 A. Then its errors
#          on the test frames
#   sweep  sweep.py: the largest pristine Si cell gpumd runs with it
# A stage whose result exists is skipped, so a resubmitted job continues.
#   STAGES="train sweep" sbatch nep.sbatch     only some stages
#   GEN=20000 sbatch nep.sbatch                 more training generations
#   RETRAIN=1 sbatch nep.sbatch                 train again from scratch
#   FRESH=1 sbatch nep.sbatch                   sweep again from scratch
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 nep.sbatch
set -uo pipefail
B=/data/jlee859/ffother
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
D=$B/data/si
SRC=$B/src/GPUMD
T=$B/nep/train
P=$B/nep/predict_test
R=$B/results/nep
J=$SLURM_JOB_ID
STAGES=${STAGES:-"build train sweep"}
GEN=${GEN:-10000}
GEN=$(( (GEN + 99) / 100 * 100 ))   # nep writes its model every 100 generations
TAG=v5.9.1
export PATH=/usr/local/cuda/bin:$PATH
# nep compiles kernels for the model at run time and looks for its sources here
export GPUMD_SRC=$SRC/src
export PYTHONUNBUFFERED=1
mkdir -p $B/src $B/nep $R $B/logs && cd $B
echo "start $(date) | $(hostname) | stages: $STAGES"
command -v nvcc >/dev/null || { echo "no nvcc in /usr/local/cuda/bin"; exit 3; }

exe() { find $SRC -maxdepth 3 -type f -name "$1" -perm -u+x 2>/dev/null | head -n 1; }

# ------------------------------------------------------------------ build
if [[ " $STAGES " == *" build "* ]]; then
    if [ -n "$(exe gpumd)" ] && [ -n "$(exe nep)" ]; then
        echo "build: already there"
    else
        [ -d $SRC/.git ] || git clone --depth 1 --branch $TAG \
            https://github.com/brucefan1983/GPUMD.git $SRC \
            || { echo "git clone failed"; exit 4; }
        cd $SRC/src
        echo "makefile as shipped:"; grep -n -E 'std=|sm_[0-9]' makefile
        sed -i 's/-arch=sm_[0-9]*/-arch=sm_121/; s/-std=c++1[14]/-std=c++17/' makefile
        echo "makefile now:"; grep -n -E 'std=|sm_[0-9]' makefile
        t0=$SECONDS
        if env -u OS make -j 16 gpumd nep > $B/logs/nep_build_$J.log 2>&1; then
            echo "build: make ok in $((SECONDS - t0)) s"
        else
            echo "build: make FAILED, last lines of logs/nep_build_$J.log:"
            grep -n -m 5 -i -E 'error' $B/logs/nep_build_$J.log
            tail -n 15 $B/logs/nep_build_$J.log
            echo "build: trying CMake"
            cmake -S $SRC -B $SRC/build -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CUDA_ARCHITECTURES=121 \
                -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
                > $B/logs/nep_cmake_$J.log 2>&1 \
                && cmake --build $SRC/build --parallel 16 >> $B/logs/nep_cmake_$J.log 2>&1 \
                || { echo "build: CMake FAILED too:"; tail -n 25 $B/logs/nep_cmake_$J.log; exit 4; }
            echo "build: CMake ok"
        fi
        cd $B
    fi
fi
GPUMD=$(exe gpumd)
NEP=$(exe nep)
[ -n "$GPUMD" ] && [ -n "$NEP" ] || { echo "no gpumd or nep executable under $SRC"; exit 4; }
echo "gpumd: $GPUMD"
echo "nep:   $NEP"
echo "GPUMD commit: $(git -C $SRC rev-parse HEAD)"

# ------------------------------------------------------------------ train
# loss.out gets one line per 100 generations (and "#" header lines per run).
# Its first column starts again at 100 when nep continues from nep.restart,
# so the generations done are counted from the number of lines
last_gen() {
    if [ -s $T/loss.out ]; then echo $(( 100 * $(grep -vc '^#' $T/loss.out) )); else echo 0; fi
}
errors() {  # errors <folder with energy_train.out and force_train.out>
    awk '{d = $1 - $2; s += (d < 0 ? -d : d); q += d * d}
         END {if (NR) printf "energy: MAE %.2f, RMSE %.2f meV/atom, %d frames\n",
                             1e3 * s / NR, 1e3 * sqrt(q / NR), NR}' $1/energy_train.out
    awk '{for (k = 1; k <= 3; k++) {d = $k - $(k + 3); s += (d < 0 ? -d : d); q += d * d; n++}}
         END {if (n) printf "forces: MAE %.4f, RMSE %.4f eV/A, %d components\n",
                            s / n, sqrt(q / n), n}' $1/force_train.out
}
if [[ " $STAGES " == *" train "* ]]; then
    [ -s $D/train.extxyz ] || { echo "no $D/train.extxyz (run data.sbatch first)"; exit 6; }
    [ "${RETRAIN:-0}" = 1 ] && rm -rf $T $P
    mkdir -p $T && cd $T
    NTRAIN=$(grep -c '^Lattice=' $D/train.extxyz)
    if [ -s nep.txt ] && [ "$(last_gen)" -ge "$GEN" ]; then
        echo "train: already at generation $(last_gen)"
    else
        # nep reads train.xyz and, for monitoring only, test.xyz: the
        # validation frames go there and the test frames stay unseen
        cp $D/train.extxyz train.xyz
        cp $D/val.extxyz test.xyz
        # with nep.restart present nep continues, and "generation" is then
        # the number of further generations, not the total
        [ -f nep.restart ] || rm -f loss.out
        LEFT=$(( GEN - $(last_gen) ))
        if [ $LEFT -le 0 ]; then   # the count says done but there is no model: start over
            rm -f nep.restart loss.out
            LEFT=$GEN
        fi
        cat > nep.in <<EOF
type        1 Si
version     4
cutoff      6 4
n_max       4 4
basis_size  8 8
neuron      30
zbl         2
lambda_e    1
lambda_f    1
lambda_v    0
batch       $NTRAIN
population  50
generation  $LEFT
EOF
        echo "train: $NTRAIN training frames, $(grep -c '^Lattice=' test.xyz) validation frames, $GEN generations in all"
        [ -f nep.restart ] && echo "train: RESUMING after generation $(last_gen), $LEFT to go"
        cat nep.in
        t0=$SECONDS
        $NEP > nep.log 2>&1
        rc=$?
        # nep compiles kernels for this model with nvcc when it starts. If
        # that compile fails it uses its generic kernels, but if the compiled
        # kernels fail to start it stops, so try once more without them
        if [ $rc -ne 0 ] && grep -q -i 'specialized' nep.log; then
            echo "train: the kernels nep compiled at start failed, trying again with nep_compile off:"
            grep -i -m 3 'specialized' nep.log | cut -c1-200
            cp nep.log nep_first_attempt.log
            echo "nep_compile off" >> nep.in
            [ -f nep.restart ] || rm -f loss.out
            $NEP > nep.log 2>&1
            rc=$?
        fi
        grep -i -m 3 -E 'specialized|nep_compile' nep.log | cut -c1-200
        echo "train: nep exit $rc after $((SECONDS - t0)) s, generation $(last_gen)"
        if [ $rc -ne 0 ] || [ ! -s nep.txt ]; then
            echo "train: FAILED, last lines of $T/nep.log:"
            tail -n 30 nep.log
            exit 7
        fi
    fi
    echo "train: loss.out, first and last lines (generation, total, L1, L2, then RMSE of energy, force, virial for train and for validation):"
    grep -v '^#' loss.out | head -n 1; tail -n 1 loss.out
    echo "train: head of nep.txt (the last two numbers of the cutoff line are the neighbor-list sizes):"
    head -n 7 nep.txt | cut -c1-200

    # errors on the unseen test frames, through nep's prediction mode
    rm -rf $P && mkdir -p $P && cd $P
    cp $T/nep.txt . && cp $D/test.extxyz train.xyz
    grep -v -E '^(generation|batch|population)' $T/nep.in > nep.in
    echo "prediction  1" >> nep.in
    $NEP > nep.log 2>&1 || { echo "predict: FAILED"; tail -n 20 nep.log; }
    if [ -s energy_train.out ] && [ -s force_train.out ]; then
        echo "test frames ($(grep -c '^Lattice=' train.xyz)), NEP against MatPES r2SCAN:"
        errors $P | tee $R/test_errors_$J.txt
        echo "first lines of energy_train.out and force_train.out (prediction, then reference):"
        head -n 2 energy_train.out; head -n 2 force_train.out
    else
        echo "predict: no energy_train.out or force_train.out"; tail -n 20 nep.log
    fi
    cp $T/nep.txt $T/nep.in $T/loss.out $R/ 2>/dev/null
    cd $B
fi

# ------------------------------------------------------------------ sweep
rc=0
if [[ " $STAGES " == *" sweep "* ]]; then
    [ -s $T/nep.txt ] || { echo "no $T/nep.txt (train first)"; exit 7; }
    FRESH_FLAG=()
    [ "${FRESH:-0}" = 1 ] && FRESH_FLAG=(--fresh)
    $PY sweep.py gpumd --exe $GPUMD --potential $T/nep.txt \
        --work $B/work/nep --out $R/sweep "${FRESH_FLAG[@]}" \
        --note "GPUMD $TAG $(git -C $SRC rev-parse --short HEAD), NEP4 Si, job $J" || rc=1
fi
echo "end rc $rc $(date)"
exit $rc
