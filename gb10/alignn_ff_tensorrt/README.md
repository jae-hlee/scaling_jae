# ALIGNN-FF through TensorRT on GB10 (probe)

Started 2026-11-11 at the user's request: build a TensorRT-RTX engine for ALIGNN-FF, with a runtime
cache, through ONNX or the TensorRT API, without accuracy loss, for all inference cases. **Nothing
here has run yet.**

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

On the cluster everything is in `/data/jlee859/fftrt/` (`venv_trt`, `venv_rtx`, `results/`,
`logs/`).
