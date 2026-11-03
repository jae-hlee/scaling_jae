#!/bin/bash
# Stage 2: chunked single points of rattled Si (0.05 A) on the GB10s.
#   one GB10: 8M atoms (rep 100), forces and energy only, all 4 models
#   six GB10s: 64M atoms (rep 200), forces, paper model and alignn_ff_mptrj
# Piece sizes are the stage-1 targets (about 39-50 GB peak per piece). The paper
# model uses its exact halo (20 A), the 4+4 models a 20 A halo instead of
# their exact 40 A (stage-1 halo scan on 8,000 atoms of rattled Si: max force
# error 2e-9 to 2e-8 eV/A against the full cell).
# If a task fails, its reduce either runs and stops with an error (missing
# parts) or, after an out-of-memory kill, waits as DependencyNeverSatisfied
# (scancel it).
#   bash stage2_submit.sh
set -euo pipefail
cd /data/jlee859/ffmax
R="--rattle 0.05"
H="--halo 20"
P=matpes_r2scan_paper
one() { echo "$5 $1: $(sbatch --parsable --array=0-0 -J "$5" chunk.sbatch "$@")"; }

one $P 100 1 117067 s2_8M_forces $R
one $P 100 0 330820 s2_8M_energy $R
for m in alignn_ff_mptrj alignn_ff_matpes_pbe alignn_ff_matpes_r2scan; do
    one $m 100 1 58401 s2_8M_forces_h20 $R $H
    one $m 100 0 142996 s2_8M_energy_h20 $R $H
done

j=$(sbatch --parsable --array=0-5 -J s2_64M_forces chunk.sbatch \
    $P 200 1 117067 s2_64M_forces $R)
echo "s2_64M_forces $P: array $j, reduce $(sbatch --parsable \
    --dependency=afterok:$j reduce.sbatch $P s2_64M_forces)"
j=$(sbatch --parsable --array=0-5 -J s2_64M_forces chunk.sbatch \
    alignn_ff_mptrj 200 1 58401 s2_64M_forces_h20 $R $H)
echo "s2_64M_forces_h20 alignn_ff_mptrj: array $j, reduce $(sbatch --parsable \
    --dependency=afterok:$j reduce.sbatch alignn_ff_mptrj s2_64M_forces_h20)"
