#!/bin/bash
#SBATCH --job-name=ffosnap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/data/jlee859/ffother/logs/snap_%j.out

# SNAP on one GB10, with LAMMPS (stable 30Sep2026) and its KOKKOS package:
#   build  LAMMPS with KOKKOS (CUDA, Blackwell 12.1), ML-SNAP, no MPI
#   train  a quadratic SNAP for Si with FitSNAP on the MatPES r2SCAN Si frames
#          (data.sbatch): 2J = 8 and the quadratic form of the billion-atom
#          carbon SNAP, cutoff 4.8 A as in the published quadratic Si SNAP
#          (the carbon cutoff of 2.7 A would reach only one shell of Si).
#          Needs a second, CPU build of LAMMPS as a shared library
#   sweep  sweep.py: the largest pristine Si cell LAMMPS runs with snap/kk.
#          If training gave no potential, the published Si SNAP that ships
#          with LAMMPS (Si_Zuo_JPCA2020.quadratic, the same 2J, cutoff and
#          form) is used, since memory does not depend on the coefficients
# A stage whose result exists is skipped, so a resubmitted job continues.
#   STAGES="sweep" sbatch snap.sbatch      only some stages
#   FRESH=1 sbatch snap.sbatch             sweep again from scratch
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 snap.sbatch
set -uo pipefail
B=/data/jlee859/ffother
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
D=$B/data/si
L=$B/src/lammps-30Sep2026
V=$B/venv_snap
FS=$B/src/FitSNAP
T=$B/snap/train
R=$B/results/snap
J=$SLURM_JOB_ID
STAGES=${STAGES:-"build train sweep"}
LTAG=stable_30Sep2026_update1
FSHA=1a9b0507fa2ba292a6497ec363707944c1cb89cd
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_ROOT/bin:$PATH
export NVCC_WRAPPER_DEFAULT_COMPILER=$(command -v g++)
export PYTHONUNBUFFERED=1
mkdir -p $B/src $B/snap $R $B/logs && cd $B
echo "start $(date) | $(hostname) | stages: $STAGES"
command -v nvcc >/dev/null || { echo "no nvcc in $CUDA_ROOT/bin"; exit 3; }
LMP=$L/build-kk/lmp

get_lammps() {
    [ -d $L/cmake ] && return 0
    git clone --depth 1 --branch $LTAG https://github.com/lammps/lammps.git $L \
        || git clone --depth 1 --branch stable https://github.com/lammps/lammps.git $L \
        || { echo "git clone of LAMMPS failed"; return 1; }
}

# ------------------------------------------------------------------ build
if [[ " $STAGES " == *" build "* ]]; then
    if [ -x $LMP ]; then
        echo "build: already there"
    else
        get_lammps || exit 4
        echo "LAMMPS: $(git -C $L describe --tags 2>/dev/null) $(git -C $L rev-parse --short HEAD)"
        rm -rf $L/build-kk && mkdir -p $L/build-kk && cd $L/build-kk
        t0=$SECONDS
        cmake -C ../cmake/presets/kokkos-cuda.cmake -D CMAKE_BUILD_TYPE=Release \
            -D BUILD_MPI=no -D PKG_ML-SNAP=yes -D Kokkos_ARCH_BLACKWELL121=yes \
            ../cmake > $B/logs/snap_cmake_$J.log 2>&1 \
            || { echo "build: cmake FAILED:"; grep -n -i -m 8 'error' $B/logs/snap_cmake_$J.log; tail -n 25 $B/logs/snap_cmake_$J.log; exit 4; }
        grep -i -E 'kokkos.*(version|devices|architecture)|CUDA' $B/logs/snap_cmake_$J.log | head -n 12
        cmake --build . -j 18 > $B/logs/snap_build_$J.log 2>&1 \
            || { echo "build: compile FAILED after $((SECONDS - t0)) s, first errors and last lines of logs/snap_build_$J.log:"
                 grep -n -m 8 -E 'error|Error' $B/logs/snap_build_$J.log; tail -n 20 $B/logs/snap_build_$J.log; exit 4; }
        echo "build: ok in $((SECONDS - t0)) s"
        cd $B
    fi
fi

# ------------------------------------------------------------------ train
if [[ " $STAGES " == *" train "* ]]; then
    if [ -s $T/Si_pot.snapcoeff ] && [ -s $T/Si_pot.snapparam ]; then
        echo "train: already there"
    else
        (
        # a subshell: a failure here must not stop the sweep, which can use
        # the published potential
        set -e
        [ -s $D/train.extxyz ] || { echo "no $D/train.extxyz (run data.sbatch first)"; exit 6; }
        get_lammps
        if [ ! -s $L/build-fs/liblammps.so ]; then
            rm -rf $L/build-fs && mkdir -p $L/build-fs && cd $L/build-fs
            cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=no -D BUILD_SHARED_LIBS=yes \
                -D PKG_ML-SNAP=yes ../cmake > $B/logs/snap_fs_cmake_$J.log 2>&1 \
                || { echo "train: cmake of the CPU library FAILED"; tail -n 20 $B/logs/snap_fs_cmake_$J.log; exit 4; }
            cmake --build . -j 18 > $B/logs/snap_fs_build_$J.log 2>&1 \
                || { echo "train: compile of the CPU library FAILED"; grep -n -m 8 -E 'error|Error' $B/logs/snap_fs_build_$J.log; tail -n 20 $B/logs/snap_fs_build_$J.log; exit 4; }
            echo "train: CPU LAMMPS library built"
        fi
        [ -x $V/bin/python ] || $PY -m venv --system-site-packages $V
        $V/bin/python -m pip install -q scikit-learn psutil tabulate > $B/logs/snap_pip_$J.log 2>&1 \
            || { tail -n 15 $B/logs/snap_pip_$J.log; exit 5; }
        if [ ! -d $FS/fitsnap3lib ]; then
            rm -rf $FS && mkdir -p $FS && cd $FS && git init -q \
                && git remote add origin https://github.com/FitSNAP/FitSNAP.git \
                && { git fetch -q --depth 1 origin $FSHA || git fetch -q --depth 1 origin master; } \
                && git checkout -q FETCH_HEAD \
                || { echo "train: could not get FitSNAP"; exit 5; }
        fi
        echo "FitSNAP: $(git -C $FS rev-parse HEAD)"
        export PYTHONPATH=$FS:$L/python${PYTHONPATH:+:$PYTHONPATH}
        export LD_LIBRARY_PATH=$L/build-fs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        $V/bin/python -c "import lammps; l = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none']); print('lammps module:', l.version(), 'ML-SNAP' in l.installed_packages)"

        rm -rf $T && mkdir -p $T && cd $T
        $PY $B/fitsnap_json.py --xyz $D/train.extxyz --out JSON --group Si_train
        $PY $B/fitsnap_json.py --xyz $D/test.extxyz --out JSON --group Si_test
        cat > Si.in <<'EOF'
[BISPECTRUM]
numTypes = 1
twojmax = 8
rcutfac = 4.8
rfac0 = 0.99363
rmin0 = 0.0
wj = 1.0
radelem = 0.5
type = Si
wselfallflag = 0
chemflag = 0
bzeroflag = 0
quadraticflag = 1

[CALCULATOR]
calculator = LAMMPSSNAP
energy = 1
force = 1
stress = 0

[ESHIFT]
Si = 0.0

[SOLVER]
solver = SVD
compute_testerrs = 1
detailed_errors = 1

[SCRAPER]
scraper = JSON

[PATH]
dataPath = JSON

[OUTFILE]
metrics = Si_metrics.md
potential = Si_pot

[REFERENCE]
units = metal
atom_style = atomic
pair_style = zero 10.0
pair_coeff = * *

[GROUPS]
group_sections = name training_size testing_size eweight fweight vweight
group_types = str float float float float float
smartweights = 0
random_sampling = 0
Si_train = 1.0 0.0 100 1 1.0E-8
Si_test = 0.0 1.0 100 1 1.0E-8

[EXTRAS]
dump_descriptors = 0
dump_truth = 0
dump_weights = 0
dump_dataframe = 0

[MEMORY]
override = 0
EOF
        t0=$SECONDS
        $V/bin/python -m fitsnap3 Si.in --overwrite > fitsnap.log 2>&1 \
            || { echo "train: FitSNAP FAILED, last lines of $T/fitsnap.log:"; tail -n 30 fitsnap.log; exit 7; }
        echo "train: FitSNAP ok in $((SECONDS - t0)) s"
        [ -s Si_pot.snapcoeff ] || { echo "train: FitSNAP wrote no Si_pot.snapcoeff"; tail -n 20 fitsnap.log; exit 7; }
        [ -s Si_pot.snapparam ] || { echo "train: FitSNAP wrote no Si_pot.snapparam"; exit 7; }
        echo "train: Si_metrics.md (errors in eV/atom and eV/A; Si_test rows are the unseen frames):"
        head -n 30 Si_metrics.md | cut -c1-220 || true
        head -n 4 Si_pot.snapcoeff; cat Si_pot.snapparam
        cp Si_pot.snapcoeff Si_pot.snapparam Si.in $R/
        cp Si_metrics.md $R/ || true
        )
        trc=$?
        [ $trc -eq 0 ] || echo "train: stage FAILED (rc $trc), the sweep will use the published Si SNAP"
        cd $B
    fi
fi

# ------------------------------------------------------------------ sweep
rc=0
if [[ " $STAGES " == *" sweep "* ]]; then
    [ -x $LMP ] || { echo "no $LMP (run the build stage)"; exit 4; }
    echo "lmp: $LMP"
    $LMP -h 2>&1 | grep -m 3 -i -E 'Massively Parallel|KOKKOS package API|Kokkos library version'
    POT=$B/snap/pot
    rm -rf $POT && mkdir -p $POT
    if [ -s $T/Si_pot.snapcoeff ] && [ -s $T/Si_pot.snapparam ]; then
        cp $T/Si_pot.snapcoeff $POT/Si.snapcoeff
        cp $T/Si_pot.snapparam $POT/Si.snapparam
        WHICH="trained here with FitSNAP"
    elif [ -s $L/potentials/Si_Zuo_JPCA2020.quadratic.snapcoeff ]; then
        cp $L/potentials/Si_Zuo_JPCA2020.quadratic.snapcoeff $POT/Si.snapcoeff
        cp $L/potentials/Si_Zuo_JPCA2020.quadratic.snapparam $POT/Si.snapparam
        WHICH="published Si_Zuo_JPCA2020.quadratic from LAMMPS, NOT trained here"
    else
        echo "no SNAP potential: no $T/Si_pot.snapcoeff and no Si_Zuo_JPCA2020.quadratic in $L/potentials"
        ls $L/potentials | grep -i -E '^Si.*snap' ; exit 7
    fi
    echo "sweep: potential $WHICH"
    cat $POT/Si.snapparam
    printf 'pair_style snap\npair_coeff * * Si.snapcoeff Si.snapparam Si\n' > $POT/pair.in
    FRESH_FLAG=()
    [ "${FRESH:-0}" = 1 ] && FRESH_FLAG=(--fresh)
    $PY sweep.py lammps --exe $LMP \
        --exe_args="-k on g 1 -sf kk -pk kokkos newton on neigh half" \
        --pair_file $POT/pair.in --copy $POT/Si.snapcoeff $POT/Si.snapparam \
        --skin 2.0 --timestep 0.0005 --steps 1 --timeout 10800 \
        --work $B/work/snap --out $R/sweep "${FRESH_FLAG[@]}" \
        --note "LAMMPS $(git -C $L rev-parse --short HEAD) snap/kk, quadratic SNAP 2J 8, potential $WHICH, job $J" || rc=1
fi
echo "end rc $rc $(date)"
exit $rc
