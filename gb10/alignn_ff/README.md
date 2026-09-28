# ALIGNN-FF scaling on one NVIDIA GB10 (Grace-Blackwell, 121.7 GiB unified)

`bench_ff_scaling.py` (pure-PyTorch ALIGNN 2.0, default `matpes_r2scan` force
field: hidden 128, 2+2 layers, smooth cutoff 5 A, 52 neighbours) evaluates one
**energy + forces + stress** single point per size through the ASE calculator on
the 8-atom conventional Si cell repeated i x i x i, median of three calls, peak
allocated GPU memory, until the GPU runs out of memory. Result:
`bench_GB10_13474.json` (atomgptlab job 13474, 2026-09-27): largest cell
54,872 atoms at 107.1 GB, OOM at 64,000, 3.12 s at 21,952 atoms, 1.95 MB/atom.
`job.sh` is the SLURM wrapper. The H200 run (`../../h200/alignn_ff/`) used the
identical script and checkpoint (md5 92cfe295...).

Differs from the `b200/alignn_ff` sweep, which timed energy only (no forces),
on Cu, with a 12-neighbour cap and the earlier default checkpoint.
