# ALIGNN-FF through TensorRT on GB10 (probe)

Started 2026-11-11 at the user's request: build a TensorRT-RTX engine for ALIGNN-FF, with a runtime
cache, through ONNX or the TensorRT API, without accuracy loss, for all inference cases. Two probe
runs so far (Results, below).

## What was found before running (documentation and source, 2026-11-11)

- TensorRT-RTX 1.7.1 (`pip install tensorrt-rtx`, module `tensorrt_rtx`) has Python 3.10 wheels
  for Linux aarch64 and officially supports DGX Spark, compute capability 12.1, since 1.6. Its
  SBSA packages are built with CUDA 13.4; the GB10 nodes have driver 580 (CUDA 13.0).
- TensorRT-RTX has no plugins, and its Scatter layer has no reduction. Its ONNX parser imports
  `ScatterElements` with `reduction="add"` only through the "ScatterElements" plugin. ALIGNN sums
  its messages into atoms and bonds with exactly this operation (`scatter_add_`), and the
  backward pass that gives the forces adds more, so the model is not expected to import.
- Standard TensorRT 11.4 (`tensorrt-cu13`) supports the platform and ships that plugin (int64
  indices, atomic sums). It rejects `ScatterND` with any reduction, which PyTorch's exporter may
  use for the backward of a gather (`index_put` with accumulate).
- Neither TensorRT computes gradients, so the forces must be exported as plain operations.
  DeePMD-kit does this by tracing `torch.autograd.grad` with `make_fx`. No supported PyTorch route
  documents ONNX export of such a graph.
- Both TensorRT versions accept FP32 networks only when strongly typed, and both use TF32 for
  FP32 matrix products unless `BuilderFlag.TF32` is cleared.

## Files

- `trt_probe.py`, `trt_probe.sbatch`: one job (about 30 minutes, one GB10) that answers the open
  questions with real runs, each step its own process.
  1. Both libraries: install into venvs on top of the `alignn` env, build and run a tiny network
     made with the API, and for TensorRT-RTX save and reload a runtime cache.
  2. Both libraries: import an ONNX model with one `ScatterElements(reduction="add")`.
  3. AFF-r2SCAN on rattled Si of 512 and 8,000 atoms. The bond graph and line graph are built in
     PyTorch exactly as `ff_max_atoms.atom_energies` does. The rest (per-atom energies, and forces
     as the gradient of the total energy) is traced with `make_fx` into plain operations,
     exported to ONNX (three routes tried), built into an engine with TF32 off, and compared
     with PyTorch float32 and float64. Two versions of the graph are exported and tested: as
     traced, and with every accumulating index write (the gradient of a gather) rewritten as
     `scatter_add`, which exports as `ScatterElements` add instead of the `ScatterND` add that
     TensorRT rejects. PyTorch is timed on the same part as the engine (graphs given).

     Shapes are static (one ONNX per cell size). Dynamic shapes come only if this works.

## Results

**Job 21913 (2026-11-12).**

- Both libraries install and run on the GB10's driver (580, CUDA 13.0), and the TensorRT-RTX runtime cache saves and reloads (20 KB).
- TensorRT 11.4 imports `ScatterElements` add, with a maximum error of 5.5e-6 on float32 sums. TensorRT-RTX rejects it: "Unsupported reduction type".
- The export of energies and forces works at 512 and 8,000 atoms. The graph as traced contains 19 `ScatterND` add (gradients of gathers) and 12 `ScatterElements` add; the rewrite turns all 19 into `ScatterElements`.
- Every engine build failed because the exported layer norms keep their mean and inverse standard deviation as extra outputs. Fixed by decomposing layer norm before export.

**Job 21925 (2026-11-12), after that fix.** Errors are the largest force difference from PyTorch float64 (eV/A).

| Cell | Graph | TensorRT 11.4 | TensorRT-RTX 1.7 | Engine error | PyTorch float32 error | Engine (ms) | PyTorch, same part (ms) |
|---|---|---|---|---:|---:|---:|---:|
| 512 Si | as traced | fails: `ScatterND` with reduction | fails: scatter reduction | | | | |
| 512 Si | rewritten | PASS | fails: scatter reduction | 4.5e-5 | 5.8e-5 | 12.9 | 14.0 |
| 8,000 Si | as traced | fails: `ScatterND` with reduction | fails: scatter reduction | | | | |
| 8,000 Si | rewritten | PASS | fails: scatter reduction | 7.2e-5 | 9.4e-5 | 199.4 | 255.7 |

- **Standard TensorRT runs ALIGNN-FF energies and forces as accurately as PyTorch float32**, with TF32 off. Per-atom energies also passed (within 3 times PyTorch float32's own error). The rewrite of the accumulating index writes is required.
- It is 1.08 and 1.28 times faster than PyTorch on the same part, with static shapes (one engine per cell size).
- **TensorRT-RTX cannot import the model**, because it has no scatter reduction and no plugins. Using it would need ALIGNN's message passing and its gradient rewritten without scatter-add.
- Not yet tested: dynamic shapes, the piece sizes of the chunked runs (about 70,000 to 330,000 atoms), and memory use.

## Real piece sizes (job 21928, 2026-11-12)

`trt_scale.py`, `trt_scale.sbatch` (ran in 10 min 49 s on one GB10, venv of the probe). The job runs AFF-r2SCAN energies and forces on rattled Si of 8,000, 32,768, 74,088 and 110,592 atoms; the force runs use pieces of 70,000 to 106,000 atoms. It compares three ways of running them:

- PyTorch eager.
- One TensorRT 11.4 engine with dynamic sizes: traced symbolically, exported with the atom, bond and line-graph counts as dynamic dimensions, and built with one profile up to 1.1 times the largest size. If that fails, one static engine is built per size.
- `torch.compile` of the same traced graph.

For each it records the time, the memory (the drop of the node's available memory), and the errors against PyTorch float32, and against float64 up to 32,768 atoms. Energy-only engines are not part of it.

**Result.** Energies and forces of rattled Si, AFF-r2SCAN, float32, no layer checkpointing; time is the median of repeated calls on the part the engine replaces (line graph given); memory is the drop of available memory, measured the same way for all three.

| Atoms | PyTorch eager | TensorRT 11.4, static engine | `torch.compile` |
|---:|---|---|---|
| 8,000 | 253 ms, 5.0 GB | 200 ms, 7.3 GB | 104 ms, 3.1 GB |
| 32,768 | 1,042 ms, 20.3 GB | 831 ms, 29.4 GB | 439 ms, 12.4 GB |
| 74,088 | 2,358 ms, 46.0 GB | build failed | 995 ms, 28.8 GB |
| 110,592 | 3,465 ms, 68.8 GB | build failed | 1,447 ms, 43.3 GB |

- Every entry that ran passed the accuracy test: within 3 times PyTorch float32's own error against float64 up to 32,768 atoms, and within 1e-3 eV/A (forces) and 1e-4 eV (per-atom energy) of PyTorch float32 above. TensorRT's force difference from float32 was 6.8e-5 and 7.3e-5 eV/A; `torch.compile`'s at 110,592 atoms was 1.0e-4 eV/A and 1.9e-5 eV per atom.
- **The dynamic-size engine did not build**, and neither did the static engines at 74,088 and 110,592 atoms (`build_serialized_network` returned None). All three failures are TensorRT's "Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[...]})": no kernel was found for a fused block of the graph. The engines built at 8,000 and 32,768 atoms, and the one at 32,768 already needed 29.4 GB, so the likely cause is the 32 GB workspace limit of the build (`--workspace_gb`, default 32); a larger limit was not tried. Where TensorRT ran it was 1.25 to 1.27 times faster than PyTorch but used 1.45 times the memory, so it would lower the largest size, not raise it.
- **`torch.compile` of the same traced graph is 2.4 times faster than PyTorch eager at every size and uses 37 to 39% less memory** (0.39 MB/atom at 110,592 atoms, against 0.62). Compiling took 29 s at 110,592 atoms, once per cell size, because the traced graph has static shapes.
- Not yet tested: `torch.compile` inside the chunked runs (pieces differ in size, so either one compile per size, padding to a few fixed sizes, or dynamic shapes), with layer checkpointing, or with the retrained model.

On the cluster everything is in `/data/jlee859/fftrt/` (`venv_trt`, `venv_rtx`, `results/`,
`logs/`).
