#!/bin/bash
# Stage 9: more than a billion atoms with ff_stream.py, AFF-r2SCAN: rattled
# Si (0.05 A), rep 630, 2,000,376,000 atoms in a 3421 A (0.34 um) cube, exact
# 20.01 A halo. No task holds the cell; with forces each task keeps only its
# x slab (about 3 GB). Run stage9_check.sbatch first.
#   energy   10 tasks, 330,820-atom pieces (as the 1B energy run), about 19
#            GB10-hours (about 2 h per task), then reduce
#   forces   20 tasks, 106,000-atom pieces (as the 8M and 64M force runs),
#            about 70 GB10-hours (about 3.5 h per task), then reduce; the
#            slab force files take about 32 GB
#
#   cd /data/jlee859/ffmax && bash stage9_submit.sh energy|forces|both
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
M=matpes_r2scan_paper
CELL="--kind Si --rep 630 --rattle 0.05"

need_gb() {
    local avail
    avail=$(df -BG --output=avail . | tail -n 1 | tr -dc 0-9)
    if [ "$avail" -lt "$1" ]; then
        echo "only $avail GB free here, need $1 GB, nothing submitted"
        exit 1
    fi
}

energy() {
    local t=stage9/s9_2B_energy
    j=$(sbatch --parsable --array=0-9 -J s9_2B_e stream.sbatch $M $t $CELL \
        --cluster_atoms 330820)
    echo "$t: array $j, reduce $(sbatch --parsable --dependency=afterok:$j \
        stream_reduce.sbatch $M $t)"
}

forces() {
    need_gb 50
    local t=stage9/s9_2B_forces
    j=$(sbatch --parsable --array=0-19 -J s9_2B_f stream.sbatch $M $t $CELL \
        --forces --ckpt --cluster_atoms 106000)
    echo "$t: array $j, reduce $(sbatch --parsable --dependency=afterok:$j \
        stream_reduce.sbatch $M $t)"
}

case "${1:-}" in
energy) energy ;;
forces) forces ;;
both) energy; forces ;;
*) echo "usage: bash stage9_submit.sh energy|forces|both"; exit 2 ;;
esac
