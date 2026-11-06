#!/bin/bash
# Stage 4: 1 billion atoms of rattled Si (0.05 A), AFF-r2SCAN (folder
# matpes_r2scan_paper), exact halo, 10 tasks, with and without forces.
#
# Every task holds the whole cell in memory, and the GB10 shares about 117 GB
# between CPU and GPU, so the pieces are set from the 64M memory:
#   no forces  k20, same 261k-atom pieces as 64M, 56 GB reserved
#              + 33 B/atom = about 92 GB
#   forces     k36, about 71k-atom pieces, about 31 GB reserved
#              + 57 B/atom + 12 B/atom while saving = about 103 GB
# A task keeps nothing until it ends, so the pilot runs 3 cubes of the force
# case through the same allocation and save first.
#
#   bash stage4_submit.sh pilot    one task, a few minutes
#   bash stage4_submit.sh full     after the pilot passed, deletes its 12 GB file
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
722d553bf64804413a71a94482b93858  chunk.sbatch
38f3ae7e6e80916fc1a42c5aa65bc001  reduce.sbatch
SUMS
M=matpes_r2scan_paper
R="--rattle 0.05"
PILOT=results/$M/s4_1B_forces_pilot_rank0000

need_gb() {
    local avail
    avail=$(df -BG --output=avail . | tail -n 1 | tr -dc 0-9)
    if [ "$avail" -lt "$1" ]; then
        echo "only $avail GB free here, need $1 GB, nothing submitted"
        exit 1
    fi
}

case "${1:-}" in
pilot)
    need_gb 20
    j=$(sbatch --parsable --array=0 -J s4_pilot chunk.sbatch \
        $M 500 1 72000 s4_1B_forces_pilot $R --k 36 --world 10 --max_chunks 3)
    echo "pilot: job $j, log logs/chunk_${j}_0.out"
    ;;
full)
    # the JSON is written after the force file, so it means the save finished
    size=$(stat -c %s "${PILOT}_forces.npy" 2>/dev/null || echo 0)
    if [ ! -s "$PILOT.json" ] || [ "$size" -lt 12000000000 ]; then
        echo "pilot did not finish, nothing submitted"
        exit 1
    fi
    need_gb 130
    rm -f "${PILOT}_forces.npy"
    j=$(sbatch --parsable --array=0-9 -J s4_1B_energy chunk.sbatch \
        $M 500 0 330820 s4_1B_energy $R --k 20)
    echo "s4_1B_energy: array $j, reduce $(sbatch --parsable \
        --time=04:00:00 --dependency=afterok:$j reduce.sbatch $M s4_1B_energy)"
    j=$(sbatch --parsable --array=0-9 -J s4_1B_forces chunk.sbatch \
        $M 500 1 72000 s4_1B_forces $R --k 36)
    echo "s4_1B_forces: array $j, reduce $(sbatch --parsable \
        --time=04:00:00 --dependency=afterok:$j reduce.sbatch $M s4_1B_forces)"
    ;;
*)
    echo "usage: bash stage4_submit.sh pilot|full"
    exit 2
    ;;
esac
