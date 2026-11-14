#!/bin/bash
#SBATCH --job-name=ffmax16x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/data/jlee859/ffmax/logs/stage16cost_%j.out

# Stage 16, part 2: stage 6 (cost by structure) again with the retrained model
# (models/aff_smooth), through smooth_eval.py and ff_any.py, on the
# structures stage 6 built:
#   1. chunked against full cell in double precision: FCC Al, the 4-grain Si
#      polycrystal, and water at a 20 A halo, a 25 A halo and with the
#      52-neighbor cap lifted
#   2. largest single graph for FCC Al, FCC Cu and water, energy only and
#      checkpointed forces
#   3. throughput on the ~1M-atom cells of Si, Al, Cu, water and the 40-grain
#      polycrystal, 20 pieces each, with forces and energy only. The pieces
#      have the sizes AFF-r2SCAN used (60% of ITS largest single graphs), so
#      the two models are timed on the same pieces
#   4. summary table (summarize_cost.py) at the end of this log
# Results in results/aff_smooth/stage16/cost/.
#   cd /data/jlee859/ffmax && sbatch --exclude=atomgptlab03 stage16_cost.sbatch
set -uo pipefail
M=aff_smooth
A=matpes_r2scan_paper
B=/data/jlee859/ffmax
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
export PYTHONPATH=$B/alignn_smooth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
R=$B/results/$M/stage16/cost
S=$B/structures
mkdir -p $R && cd $B
echo "start $(date) | $(hostname) | model $M"
md5sum -c --quiet <<'SUMS' || { echo "ff_max_atoms.py is not the stage-4 copy"; exit 2; }
0cc75757ef28971014e6012aca48e079  ff_max_atoms.py
SUMS
(cd $B/models/$M && md5sum -c --quiet model.md5) || { echo "models/$M differs from its model.md5"; exit 2; }
$PY -c "import torch; torch.zeros(1).cuda()" || { echo "no CUDA on $(hostname)"; exit 3; }
for s in al_2k water_1k poly_si_4k si_1m al_1m cu_1m water_1m poly_si_1m; do
    [ -s $S/$s.json ] || { echo "no structures/$s (stage6.sbatch builds it)"; exit 6; }
done

ff() { $PY $B/smooth_eval.py $B/ff_any.py "$@" --model_dir $B/models/$M; }

# 1. exactness, double precision, forces, against the full cell
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

# 2. largest single graph
for kind in Al Cu water; do
    for p in e ckpt; do
        ff sweep --kind $kind --path $p --out $R/sweep_${kind}_${p}_$J
        echo "sweep $kind $p exit $? $(date)"
    done
done

# 3. AFF-r2SCAN's piece sizes: 60% of its largest single graphs
cap() {
    $PY -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; print(int(0.6*r[-1]['natoms']) if r else 0)" "$1" 2>/dev/null || echo 0
}
declare -A CF CE
CF[si]=$(cap $B/results/$A/sweep_ckpt_19126.json)
CE[si]=$(cap $B/results/$A/sweep_e_19126.json)
CF[poly_si]=${CF[si]}
CE[poly_si]=${CE[si]}
for kind in Al Cu water; do
    lk=${kind,,}
    CF[$lk]=$(cap $B/results/$A/stage6/sweep_${kind}_ckpt_19326.json)
    CE[$lk]=$(cap $B/results/$A/stage6/sweep_${kind}_e_19326.json)
done
echo "piece atoms with forces: si ${CF[si]} al ${CF[al]} cu ${CF[cu]} water ${CF[water]}"
echo "piece atoms energy only: si ${CE[si]} al ${CE[al]} cu ${CE[cu]} water ${CE[water]}"

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

# 4. summary
$PY $B/summarize_cost.py $R --job $J
echo "end $(date)"
