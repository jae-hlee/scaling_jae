#!/bin/bash
# Stage 10: a device-scale planar gate stack, AFF-r2SCAN: Si 400 A /
# beta-cristobalite SiO2 30 A / Al 200 A in a 1737.6 A (0.17 um) square
# periodic box, 10 A of vacuum above, interfaces as built (about 1.02e8
# atoms; in-plane strain -0.13% for SiO2, +0.01% for Al). Energy and forces
# with per-atom energies, in 35,000-atom pieces (sized for the Al layer), in
# 6 tasks. Run stage10_check.sbatch first; its sample gives the rate.
#   1. build.sbatch       the structure, minutes
#   2. chunk_any.sbatch   6 tasks after the build
#   3. reduce_any.sbatch  after the tasks
#
#   cd /data/jlee859/ffmax && bash stage10_submit.sh
set -euo pipefail
cd "${FFMAX_DIR:-/data/jlee859/ffmax}"
md5sum -c --quiet <<'SUMS'
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
M=matpes_r2scan_paper
TAG=stage10/s10_stack_1e8_forces
avail=$(df -BG --output=avail . | tail -n 1 | tr -dc 0-9)
if [ "$avail" -lt 20 ]; then
    echo "only $avail GB free here, need 20 GB, nothing submitted"
    exit 1
fi
jb=$(sbatch --parsable build.sbatch stack_1e8 --kind stack \
    --layers Si:400,SiO2:30,Al:200 --lateral 1737.6)
ja=$(sbatch --parsable --dependency=afterok:$jb --array=0-5 -J s10_stack \
    chunk_any.sbatch $M stack_1e8 $TAG --forces --ckpt --cluster_atoms 35000 \
    --save_energies)
jr=$(sbatch --parsable --dependency=afterok:$ja reduce_any.sbatch $M $TAG)
echo "build $jb -> tasks $ja -> reduce $jr ($TAG)"
