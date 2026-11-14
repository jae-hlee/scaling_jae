#!/bin/bash
# Stage 16: every test below 10^8 atoms again with the retrained model
# (models/aff_smooth), so the letter reports all numbers with one model.
# Stage 15 already repeated the runs of 10^8 atoms and more; stage 13 the
# relaxation and STMV. About 5 GB10-hours in all (AFF-r2SCAN's times plus 8%):
#   stage16_small.sbatch  sweeps, SiGe and halo checks, 8M atoms (1 GB10, ~1 h)
#   stage16_cost.sbatch   stage 6 again: costs by structure (1 GB10, ~1 h)
#   64M atoms with forces and energy only, 6 tasks each, then reduce
#                         (AFF-r2SCAN: 2.19 and about 0.5 GB10-hours)
# Results in results/aff_smooth/stage16/.
#   cd /data/jlee859/ffmax && bash stage16_submit.sh
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
M=aff_smooth
[ -s models/$M/model.md5 ] || { echo "no models/$M (stage13_compare.sbatch makes it)"; exit 5; }
(cd models/$M && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
for f in smooth_ffm.py smooth_eval.py ff_any.py summarize_cost.py \
    stage16_small.sbatch stage16_cost.sbatch stage16_chunk.sbatch reduce.sbatch; do
    [ -s "$f" ] || { echo "missing $f here (copy it first)"; exit 2; }
done
mkdir -p results/$M/stage16
EXC="--exclude=atomgptlab03"

echo "small: $(sbatch --parsable $EXC stage16_small.sbatch) (logs/stage16small_<job>.out)"
echo "cost: $(sbatch --parsable $EXC stage16_cost.sbatch) (logs/stage16cost_<job>.out)"
arr() {  # arr <forces 0|1> <piece atoms> <tag>
    local j
    j=$(sbatch --parsable $EXC --array=0-5 -J s16 stage16_chunk.sbatch 200 "$1" "$2" "$3")
    echo "$3: array $j, reduce $(sbatch --parsable $EXC \
        --dependency=afterok:$j reduce.sbatch $M stage16/$3)"
}
arr 1 117067 s16_64M_forces
arr 0 330820 s16_64M_energy
