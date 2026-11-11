#!/bin/bash
# Stage 15: the billion-atom runs again with the retrained model of stage 12
# (models/aff_smooth, smooth_messages), the same cells and the same piece
# sizes as with AFF-r2SCAN, results in results/aff_smooth/stage15/. Each run
# is a task array (chunk_any.sbatch or stream.sbatch with SMOOTH=1) and a
# reduce. Every array waits for stage15_check.sbatch (chunked against full
# cell with the trained weights) unless that check has passed before.
# GB10-hours are AFF-r2SCAN's times 1.09, the slowdown seen in stage 13.
#   poly    1,021,299,267-atom Si polycrystal of stage 7b (structure
#           poly_si_1b), forces and per-atom energies, 10 tasks of
#           72,000-atom pieces, about 46 GB10-hours
#   si1b    10^9 atoms of rattled Si (rep 500, 0.05 A), forces, through
#           ff_stream.py with stage 4's 36 x 36 x 36 cubes, 12 tasks (3 x
#           slabs each), about 44 GB10-hours. Stage 4 ran this cell with
#           ff_max_atoms.py, which cannot take the retrained model, so the
#           random displacements differ from stage 4's
#   si2b    2,000,376,000 atoms of rattled Si of stage 9, forces, 20 tasks of
#           106,000-atom pieces, about 73 GB10-hours
#   forces  poly, si1b and si2b (about 163 GB10-hours)
#   energy  energy only: 10^9 Si with stage 4's 20 x 20 x 20 cubes and
#           2,000,376,000 Si as stage 9, 10 tasks each, about 29 GB10-hours
#   stack   the 102,361,540-atom gate stack of stage 10, forces and per-atom
#           energies, 6 tasks, about 11 GB10-hours
#
#   cd /data/jlee859/ffmax && bash stage15_submit.sh forces
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
M=aff_smooth
[ -s models/$M/model.md5 ] || { echo "no models/$M (stage13_compare.sbatch makes it)"; exit 5; }
(cd models/$M && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
grep -q 'SMOOTH' chunk_any.sbatch && grep -q 'SMOOTH' stream.sbatch \
    || { echo "chunk_any.sbatch or stream.sbatch is the old copy without SMOOTH: copy them again"; exit 2; }
export SMOOTH=1
mkdir -p results/$M/stage15
EXC="--exclude=atomgptlab03"

need_gb() {
    local avail
    avail=$(df -BG --output=avail . | tail -n 1 | tr -dc 0-9)
    if [ "$avail" -lt "$1" ]; then
        echo "only $avail GB free here, need $1 GB, nothing submitted"
        exit 1
    fi
}

DEP=""
check() {  # sets DEP so that the runs wait for the check, unless it passed before
    if [ -f results/$M/stage15/check_passed ]; then
        echo "check passed before ($(cat results/$M/stage15/check_passed)), not repeated"
        return 0
    fi
    local jc
    jc=$(sbatch --parsable $EXC stage15_check.sbatch)
    DEP="--dependency=afterok:$jc"
    echo "check $jc (logs/stage15check_$jc.out); the runs below wait for it"
}

stream_run() {  # stream_run <tag> <tasks> <ff_stream options>
    local t=stage15/$1 n=$2 j
    shift 2
    j=$(sbatch --parsable $EXC $DEP --array=0-$((n - 1)) -J s15 \
        stream.sbatch $M $t "$@")
    echo "$t: array $j, reduce $(sbatch --parsable $EXC --dependency=afterok:$j \
        stream_reduce.sbatch $M $t)"
}

poly() {
    need_gb 200  # structure 29 GB if it has to be rebuilt, forces 10 x 12 GB
    local t=stage15/s15_1B_poly_forces jb ja
    jb=$(sbatch --parsable $EXC build.sbatch poly_si_1b --kind poly --element Si \
        --box 2740 --grains 4800 --seed 1)
    local dep="--dependency=afterok:$jb"
    [ -n "$DEP" ] && dep="$dep:${DEP##*:}"
    ja=$(sbatch --parsable $EXC $dep --array=0-9 -J s15_poly \
        chunk_any.sbatch $M poly_si_1b $t --forces --ckpt --cluster_atoms 72000 \
        --save_energies)
    echo "$t: build $jb (does nothing if the structure exists), array $ja, reduce $(sbatch \
        --parsable $EXC --time=04:00:00 --dependency=afterok:$ja reduce_any.sbatch $M $t)"
}

si1b() {
    need_gb 30   # slab force files, about 12 GB
    stream_run s15_1B_si_forces 12 --kind Si --rep 500 --rattle 0.05 \
        --forces --ckpt --k 36
}

si2b() {
    need_gb 50   # slab force files, about 32 GB
    stream_run s15_2B_si_forces 20 --kind Si --rep 630 --rattle 0.05 \
        --forces --ckpt --cluster_atoms 106000
}

energy() {
    stream_run s15_1B_si_energy 10 --kind Si --rep 500 --rattle 0.05 --k 20
    stream_run s15_2B_si_energy 10 --kind Si --rep 630 --rattle 0.05 \
        --cluster_atoms 330820
}

stack() {
    need_gb 20
    local t=stage15/s15_stack_1e8_forces jb ja
    jb=$(sbatch --parsable $EXC build.sbatch stack_1e8 --kind stack \
        --layers Si:400,SiO2:30,Al:200 --lateral 1737.6)
    local dep="--dependency=afterok:$jb"
    [ -n "$DEP" ] && dep="$dep:${DEP##*:}"
    ja=$(sbatch --parsable $EXC $dep --array=0-5 -J s15_stack \
        chunk_any.sbatch $M stack_1e8 $t --forces --ckpt --cluster_atoms 35000 \
        --save_energies)
    echo "$t: build $jb (does nothing if the structure exists), array $ja, reduce $(sbatch \
        --parsable $EXC --dependency=afterok:$ja reduce_any.sbatch $M $t)"
}

case "${1:-}" in
poly|si1b|si2b|energy|stack) check; "$1" ;;
forces) check; poly; si1b; si2b ;;
*) echo "usage: bash stage15_submit.sh forces|poly|si1b|si2b|energy|stack"; exit 2 ;;
esac
