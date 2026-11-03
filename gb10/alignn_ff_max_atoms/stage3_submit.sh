#!/bin/bash
# Stage 3: 64M atoms of rattled Si (0.05 A) WITHOUT forces, chunked, 6 tasks,
# to match the 64M force runs of stage 2. Same models and piece-size targets as
# the 8M energy runs: AFF-r2SCAN (folder matpes_r2scan_paper) with its exact 20 A
# halo, alignn_ff_mptrj with a 20 A halo.
# Each model's reduce job adds up its tasks. If a task fails, the reduce either
# stops with an error (missing parts) or, after an out-of-memory kill, waits as
# DependencyNeverSatisfied (scancel it).
#   bash stage3_submit.sh
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
722d553bf64804413a71a94482b93858  chunk.sbatch
38f3ae7e6e80916fc1a42c5aa65bc001  reduce.sbatch
SUMS
R="--rattle 0.05"
H="--halo 20"

j=$(sbatch --parsable --array=0-5 -J s3_64M_energy chunk.sbatch \
    matpes_r2scan_paper 200 0 330820 s3_64M_energy $R)
echo "s3_64M_energy AFF-r2SCAN: array $j, reduce $(sbatch --parsable \
    --dependency=afterok:$j reduce.sbatch matpes_r2scan_paper s3_64M_energy)"
j=$(sbatch --parsable --array=0-5 -J s3_64M_energy chunk.sbatch \
    alignn_ff_mptrj 200 0 142996 s3_64M_energy_h20 $R $H)
echo "s3_64M_energy_h20 alignn_ff_mptrj: array $j, reduce $(sbatch --parsable \
    --dependency=afterok:$j reduce.sbatch alignn_ff_mptrj s3_64M_energy_h20)"
