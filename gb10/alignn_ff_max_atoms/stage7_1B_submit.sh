#!/bin/bash
# Stage 7b: a billion-atom Si polycrystal, AFF-r2SCAN (folder
# matpes_r2scan_paper): 4,800 randomly oriented grains (about 20 nm) in a
# 2740 A (0.27 um) periodic cube, as built (not relaxed). The box is 1%
# wider than the 2715 A (10^9 sites) Si cube because removing overlapping
# atoms at the boundaries costs about 0.5-1%, so the cell keeps more than
# 10^9 atoms (about 1.02e9 expected). Evaluated in 10 tasks like stage 4
# (about 72,000-atom pieces, exact 20.01 A halo), keeping the energy of every
# atom (ff_any --save_energies) for maps of the grain boundaries.
#   1. build.sbatch       the structure, about 1 h and 90 GB on one node
#   2. chunk_any.sbatch   10 tasks after the build, about 3.5 h each with
#                         forces (35 GB10-hours at the stage-6 rate) or
#                         about 1 h without
#   3. reduce_any.sbatch  after the tasks
#
#   cd /data/jlee859/ffmax && bash stage7_1B_submit.sh forces     (or: energy)
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
M=matpes_r2scan_paper
MODE=${1:-}

need_gb() {
    local avail
    avail=$(df -BG --output=avail . | tail -n 1 | tr -dc 0-9)
    if [ "$avail" -lt "$1" ]; then
        echo "only $avail GB free here, need $1 GB, nothing submitted"
        exit 1
    fi
}

case "$MODE" in
forces)
    need_gb 200  # structure 29 GB, forces 10 x 12 GB, energies 10 x 1.2 GB
    TAG=stage7/s7_1B_poly_forces
    OPTS="--forces --ckpt --cluster_atoms 72000"
    ;;
energy)
    need_gb 60
    TAG=stage7/s7_1B_poly_energy
    OPTS="--cluster_atoms 330820"
    ;;
*)
    echo "usage: bash stage7_1B_submit.sh forces|energy"
    exit 2
    ;;
esac
jb=$(sbatch --parsable build.sbatch poly_si_1b --kind poly --element Si \
    --box 2740 --grains 4800 --seed 1)
ja=$(sbatch --parsable --dependency=afterok:$jb --array=0-9 -J s7_1B \
    chunk_any.sbatch $M poly_si_1b $TAG $OPTS --save_energies)
jr=$(sbatch --parsable --time=04:00:00 --dependency=afterok:$ja \
    reduce_any.sbatch $M $TAG)
echo "build $jb -> tasks $ja -> reduce $jr ($TAG)"
