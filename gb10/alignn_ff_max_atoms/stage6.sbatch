#!/bin/bash
#SBATCH --job-name=ffmax6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage6_%j.out

# Stage 6: AFF-r2SCAN (folder matpes_r2scan_paper) on structures other than
# diamond Si, on one GB10, with ff_any.py and cells.py. Measures what the v2
# manuscript's Table 4 only estimates from the Si rate.
#   1. cells.py self-checks (decomposition geometry, builders; no GPU)
#   2. chunked against full cell in double precision: FCC Al, a Si
#      polycrystal (uneven cube counts), and liquid-density water. Water has
#      more neighbors within 5 A than the model's cap of 52, which makes
#      the 20 A halo inexact, so water is also run at a 25 A halo and with
#      the cap lifted (exact if the cap is the cause)
#   3. largest single graph for FCC Al, FCC Cu and water, energy only and
#      checkpointed forces, with bonds and bond angles per atom
#   4. throughput of chunked evaluation on ~1M-atom cells of Si, Al, Cu,
#      water and a 40-grain Si polycrystal: 20 pieces each, with forces and
#      energy only, pieces sized at 60% of the largest single graph
#   5. summary table (summarize_cost.py) at the end of this log
#
#   cd /data/jlee859/ffmax && sbatch stage6.sbatch
set -uo pipefail
M=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_6a20fb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage6
S=$B/structures
mkdir -p $R $S && cd $B
echo "start $(date) | $(hostname)"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models && md5sum -c --quiet models.md5) || { echo "model md5 mismatch"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2

# 1. self-checks
$PY $B/cells.py test || { echo "cells.py self-checks failed"; exit 4; }

# structures (built once, reused by later jobs)
build() {
    local name=$1; shift
    [ -s $S/$name.json ] || $PY $B/cells.py build "$@" --out $S/$name
    echo "structure $name: $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['natoms'], 'atoms, box', d['box_A'])" $S/$name.json)"
}
build al_2k --kind Al --rep 8
build water_1k --kind water --rep 8
build poly_si_4k --kind poly --element Si --box 43.44 --grains 4 --seed 2
build si_1m --kind Si --rep 50
build al_1m --kind Al --rep 63
build cu_1m --kind Cu --rep 63
build water_1m --kind water --rep 70
build poly_si_1m --kind poly --element Si --box 271.5 --grains 40

ff() { $PY $B/ff_any.py "$@" --model_dir $B/models/$M; }

# 2. exactness, double precision, forces, against the full cell
ex() {
    local tag=$1; shift
    ff chunk "$@" --forces --ckpt --validate --float64 --out $R/exact_${tag}_$J
    echo "exact $tag exit $? $(date)"
}
ex al_2k --structure $S/al_2k --k 2
ex poly_si_4k --structure $S/poly_si_4k --kxyz 2 3 1
ex water_1k_h20 --structure $S/water_1k --k 2
ex water_1k_h25 --structure $S/water_1k --k 2 --halo 25
ex water_1k_nocap --structure $S/water_1k --k 2 --max_neighbors 10000

# 3. largest single graph
for kind in Al Cu water; do
    for p in e ckpt; do
        ff sweep --kind $kind --path $p --out $R/sweep_${kind}_${p}_$J
        echo "sweep $kind $p exit $? $(date)"
    done
done

# 60% of the largest single graph as the piece size; Si from stage 1
cap() {
    $PY -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; print(int(0.6*r[-1]['natoms']) if r else 0)" "$1" 2>/dev/null || echo 0
}
declare -A CF CE
CF[si]=$(cap $B/results/$M/sweep_ckpt_19126.json)
CE[si]=$(cap $B/results/$M/sweep_e_19126.json)
CF[poly_si]=${CF[si]}
CE[poly_si]=${CE[si]}
for kind in Al Cu water; do
    lk=${kind,,}
    CF[$lk]=$(cap $R/sweep_${kind}_ckpt_$J.json)
    CE[$lk]=$(cap $R/sweep_${kind}_e_$J.json)
done
echo "piece atoms with forces: si ${CF[si]} al ${CF[al]} cu ${CF[cu]} water ${CF[water]}"
echo "piece atoms energy only: si ${CE[si]} al ${CE[al]} cu ${CE[cu]} water ${CE[water]}"

# 4. throughput samples, 20 pieces of each ~1M-atom cell
for s in si al cu water poly_si; do
    if [ "${CF[$s]:-0}" -gt 0 ]; then
        ff chunk --structure $S/${s}_1m --cluster_atoms ${CF[$s]} --forces --ckpt \
            --max_chunks 20 --out $R/sample_forces_${s}_1m_$J
        echo "sample forces $s exit $? $(date)"
    fi
    if [ "${CE[$s]:-0}" -gt 0 ]; then
        ff chunk --structure $S/${s}_1m --cluster_atoms ${CE[$s]} \
            --max_chunks 20 --out $R/sample_energy_${s}_1m_$J
        echo "sample energy $s exit $? $(date)"
    fi
done

# 5. summary
$PY $B/summarize_cost.py $R --job $J
echo "end $(date)"
