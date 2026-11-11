#!/bin/bash
#SBATCH --job-name=ffodata
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/data/jlee859/ffother/logs/data_%j.out

# The pure-Si frames of the MatPES r2SCAN training data (the data AFF-r2SCAN
# was trained on), as extended XYZ and as DeePMD-kit systems, split 80/10/10.
# Uses no GPU; it runs on a GB10 node because the Python env is aarch64.
# About a minute, most of it reading the 513 MB id_prop.json.
#   ELEMENT=Cu sbatch data.sbatch     another element, into data/cu
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 data.sbatch
set -uo pipefail
B=/data/jlee859/ffother
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
ELEMENT=${ELEMENT:-Si}
OUT=$B/data/$(echo $ELEMENT | tr A-Z a-z)
export PYTHONUNBUFFERED=1
cd $B
echo "start $(date) | $(hostname) | $ELEMENT"
ID=$(find /data/jlee859/ffsmooth/data -name 'id_prop.json*' -print -quit)
[ -n "$ID" ] || { echo "no id_prop.json under /data/jlee859/ffsmooth/data"; exit 6; }
$PY si_data.py --data "$(dirname "$ID")" --element $ELEMENT --out $OUT
rc=$?
ls -la $OUT 2>/dev/null
head -n 3 $OUT/train.extxyz 2>/dev/null | cut -c1-400
echo "end rc $rc $(date)"
exit $rc
