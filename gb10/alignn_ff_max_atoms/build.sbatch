#!/bin/bash
#SBATCH --job-name=ffbuild
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/build_%j.out

# Builds one cells.py structure into /data/jlee859/ffmax/structures/<name>
# (does nothing if it exists). A billion-atom polycrystal needs about 90 GB
# of memory and 29 GB of disk.
#   sbatch build.sbatch <name> <cells.py build options>
set -uo pipefail
NAME=${1:?structure name}
shift
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONUNBUFFERED=1
cd $B && mkdir -p structures
if [ -s structures/$NAME.json ]; then
    echo "structures/$NAME exists, nothing to do"
    exit 0
fi
echo "start $(date) | $(hostname) | $NAME $*"
free -g | head -2
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v $PY $B/cells.py build "$@" --out structures/$NAME
else
    $PY $B/cells.py build "$@" --out structures/$NAME
fi
rc=$?
echo "exit $rc end $(date)"
exit $rc  # so that --dependency=afterok holds back the next jobs on failure
