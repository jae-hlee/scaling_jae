# ALIGNN-FF scaling on one NVIDIA H200 NVL (139.8 GiB)

Same script, checkpoint (md5 92cfe295...) and protocol as `../../gb10/alignn_ff/`
(energy + forces + stress single points on Si i x i x i cells until OOM).
Result: `bench_NVIDIA_H200_NVL_913733.json` (Skipjack job 913733, node gh204,
2026-09-27): largest cell 74,088 atoms at 144.6 GB, OOM at 85,184, 0.72 s at
21,952 atoms, 1.95 MB/atom, 4.0-4.3x faster than the GB10 at every size.
`job.sh` is the SLURM wrapper (h200 partition; the b200 partition was not
available to this account).
