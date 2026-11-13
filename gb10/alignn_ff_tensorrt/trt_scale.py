"""Speed, memory and accuracy of AFF-r2SCAN energies and forces at real piece
sizes: PyTorch eager, a TensorRT 11.4 engine, and torch.compile (Inductor) of
the same traced graph. Builds on trt_probe.py (job 21925): energies and forces
traced with make_fx, accumulating index writes rewritten as scatter_add, layer
norm decomposed, exported to ONNX.

  prepare   rattled Si of 8 rep^3 atoms: the inputs (bond graph and line graph
            built in PyTorch as the runs do), references in float32 (and
            float64 with --f64) from ff_max_atoms.atom_energies and autograd,
            and the time and memory of PyTorch eager on the same part the
            engine runs (network and gradient, graphs given)
  export    trace at one size and export ONNX; --dynamic traces symbolically
            and exports with the atom, bond and line-graph counts as dynamic
            dimensions, so one engine serves every size
  build     a TensorRT engine from the ONNX (TF32 off). For a dynamic ONNX,
            one optimization profile from 1 to 1.1 times the largest prepared
            size, tuned at --opt_data
  run       the engine on one prepared size: time, memory, error against the
            references
  inductor  torch.compile of the same traced graph at one prepared size

Memory is the largest drop of the node's MemAvailable during the measured part
(CPU and GPU share it on the GB10), next to torch's own peak where it applies.

    python trt_scale.py prepare --rep 21 --out results/si21
    python trt_scale.py export --rep 4 --dynamic --out results/aff_dyn
    python trt_scale.py build --onnx results/aff_dyn.onnx --max_data results/si24 \\
        --opt_data results/si21 --plan results/aff_dyn.plan
    python trt_scale.py run --plan results/aff_dyn.plan --data results/si21
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

import trt_probe as tp

INPUT_NAMES = tp.INPUT_NAMES


def mem_available_kb():
    """MemAvailable of the node."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


class MemWatch:
    """Largest drop of MemAvailable from the moment it starts."""

    def __init__(self):
        self.base = self.low = mem_available_kb()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            self.low = min(self.low, mem_available_kb())
            time.sleep(0.05)

    def stop(self):
        """GB used at the peak."""
        self._stop = True
        self._thread.join()
        self.low = min(self.low, mem_available_kb())
        return (self.base - self.low) * 1024 / 1e9


def fp32_only():
    """No TF32 anywhere in PyTorch."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def build_inputs(model_dir, rep, dtype):
    """Model and the nine inputs for rattled Si, as trt_probe.export."""
    import torch
    import cells
    import ff_max_atoms as ffm
    dev = torch.device("cuda")
    pos_np, z_np, L = cells.crystal("Si", rep, rattle=0.05, seed=1)
    pos_np = np.asarray(pos_np, dtype=float)
    cell = np.diag(np.asarray(L, dtype=float))
    ffm.DTYPE = dtype
    model, cfg = ffm.load_model(model_dir, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    g = ffm.build_graph(pos_np, cell, z_np, cfg, dev)
    pos = torch.tensor(pos_np, dtype=dtype, device=dev)
    lat = torch.tensor(cell, dtype=dtype, device=dev)
    r = pos[g["dst"]] - pos[g["src"]] + g["shift"] @ lat
    lg_src, lg_dst = tp.line_graph(model, g["src"], g["dst"],
                                   torch.linalg.vector_norm(r, dim=1), len(pos_np))
    ins = [pos, lat, g["src"], g["dst"], g["shift"], g["feats"], g["Z"],
           lg_src, lg_dst]
    return model, ins


def make_joint(model):
    """Energies and forces, autograd inside (traced by make_fx)."""
    import torch

    def joint(pos, lat, src, dst, shift, feats, z_num, lg_src, lg_dst):
        pos = pos.detach().requires_grad_(True)
        e_atom, e_bond = tp.network_energies(model, pos, lat, src, dst, shift,
                                             feats, z_num, lg_src, lg_dst)
        (g,) = torch.autograd.grad(e_atom.sum() + e_bond.sum(), pos)
        return e_atom.detach(), -g
    return joint


def reference(model, ins):
    """atom_energies + autograd, the code path the runs use."""
    import torch
    import ff_max_atoms as ffm
    p = ins[0].clone().requires_grad_(True)
    g = {"src": ins[2], "dst": ins[3], "shift": ins[4], "feats": ins[5],
         "Z": ins[6]}
    e_atom, e_bond = ffm.atom_energies(model, p, ins[1], g)
    (grad,) = torch.autograd.grad(e_atom.sum() + e_bond.sum(), p)
    return e_atom.detach().double().cpu().numpy(), (-grad).double().cpu().numpy()


def timed(fn, repeats=3):
    """Median wall time of fn() after one warm-up call, and its last result."""
    import torch
    out = fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)), out


def prepare(args):
    """Inputs, references and PyTorch eager time and memory at one size."""
    import torch
    fp32_only()
    data = {}
    if args.f64:
        model, ins = build_inputs(args.model_dir, args.rep, torch.float64)
        data["e_atom_f64"], data["forces_f64"] = reference(model, ins)
        del model, ins
        torch.cuda.empty_cache()
    model, ins = build_inputs(args.model_dir, args.rep, torch.float32)
    data["e_atom_f32"], data["forces_f32"] = reference(model, ins)
    joint = make_joint(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mw = MemWatch()
    with torch.enable_grad():
        t_joint, _ = timed(lambda: joint(*ins))
    mem = mw.stop()
    n = int(ins[0].shape[0])
    info = {"atoms": n, "edges": int(ins[2].numel()),
            "line_edges": int(ins[7].numel()), "torch_s": t_joint,
            "torch_mem_gb": mem,
            "torch_peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9}
    np.savez(args.out + ".npz",
             **{name: t.cpu().numpy() for name, t in zip(INPUT_NAMES, ins)},
             **data, **info)
    tp.result(stage="prepare", rep=args.rep, f64=args.f64, **info)
    return tp.verdict(True, f"{n} atoms: PyTorch eager energy and forces "
                      f"{t_joint * 1e3:.1f} ms, {mem:.1f} GB")


def trace(model, ins, dynamic):
    """make_fx of energies and forces with the probe's decompositions, then
    the scatter_add rewrite; checked against eager at each step."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._decomp import core_aten_decompositions, get_decompositions
    aten = torch.ops.aten
    table = core_aten_decompositions()
    table.update(get_decompositions([aten.native_layer_norm,
                                     aten.native_layer_norm_backward]))
    joint = make_joint(model)
    with torch.enable_grad():
        ea, fo = joint(*ins)
        kw = {"tracing_mode": "symbolic", "_allow_non_fake_inputs": True} \
            if dynamic else {}
        gm = make_fx(joint, decomposition_table=table, **kw)(*ins)

    def diff(gm_):
        with torch.no_grad():
            e2, f2 = gm_(*ins)
        return float((e2 - ea).abs().max()), float((f2 - fo).abs().max())

    d0 = diff(gm)
    done, skipped = tp.rewrite_accumulate(gm, ins)
    d1 = diff(gm)
    tp.result(stage="trace", dynamic=dynamic, nodes=len(gm.graph.nodes),
              vs_eager=d0, rewritten=done, skipped=skipped,
              vs_eager_after_rewrite=d1)
    if max(d1[0] / 1e-5, d1[1] / 1e-4) > 1:
        raise RuntimeError(f"traced graph differs from eager: {d1}")
    return gm


def export(args):
    """ONNX of energies and forces, static or with dynamic sizes."""
    import torch
    fp32_only()
    model, ins = build_inputs(args.model_dir, args.rep, torch.float32)
    try:
        gm = trace(model, ins, args.dynamic)
    except Exception as err:
        tp.result(stage="trace", error=f"{type(err).__name__}: {str(err)[:2000]}")
        return tp.verdict(False, f"trace failed ({'dynamic' if args.dynamic else 'static'})")
    names = dict(input_names=INPUT_NAMES, output_names=["e_atom", "forces"])
    ds = None
    if args.dynamic:
        Dim = torch.export.Dim
        N = Dim("N", min=2, max=10**8)
        E = Dim("E", min=2, max=2 * 10**9)
        T = Dim("T", min=2, max=2 * 10**9)
        ds = ({0: N}, None, {0: E}, {0: E}, {0: E}, {0: N}, {0: N},
              {0: T}, {0: T})
    path = args.out + ".onnx"
    errors = {}
    route = None
    for label in ("dynamo", "torch.export then dynamo"):
        try:
            if label == "dynamo":
                torch.onnx.export(gm, tuple(ins), path, dynamo=True,
                                  opset_version=18, external_data=False,
                                  dynamic_shapes=ds, **names)
            else:
                ep = torch.export.export(gm, tuple(ins), dynamic_shapes=ds,
                                         strict=False)
                torch.onnx.export(ep, f=path, opset_version=18,
                                  external_data=False, **names)
            route = label
            break
        except Exception as err:
            errors[label] = f"{type(err).__name__}: {str(err)[:2000]}"
    if route is None:
        tp.result(stage="export", errors=errors)
        return tp.verdict(False, "ONNX export failed")
    import onnx
    m = onnx.load(path)
    shapes = {}
    for i in m.graph.input:
        shapes[i.name] = [d.dim_param or d.dim_value
                          for d in i.type.tensor_type.shape.dim]
    desc = tp.describe_onnx(path)
    tp.result(stage="export", route=route, failed=errors, input_shapes=shapes,
              scatter_nodes=desc["scatter_nodes"],
              layer_norm_nodes=desc["op_types"].get("LayerNormalization", 0),
              onnx_mb=desc["onnx_mb"])
    is_dyn = any(isinstance(d, str) for s in shapes.values() for d in s)
    ok = (not args.dynamic) or is_dyn
    return tp.verdict(ok, f"exported {path} ({route}); dynamic dims: {is_dyn}")


def set_profile(trt, builder, config, network, mx, op):
    """One profile: dynamic dimensions from 1 to 1.1 times the largest
    prepared size, tuned at the opt size."""
    profile = builder.create_optimization_profile()
    shown = {}
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(inp.shape)
        key = tp.input_key(inp.name, mx.files)
        hi_ref, op_ref = mx[key].shape, op[key].shape
        lo = tuple(1 if d == -1 else d for d in shape)
        opt = tuple(op_ref[k] if d == -1 else d for k, d in enumerate(shape))
        hi = tuple(int(hi_ref[k] * 1.1) + 1 if d == -1 else d
                   for k, d in enumerate(shape))
        try:  # the developer guide passes tuples; Dims as a fallback
            profile.set_shape(inp.name, lo, opt, hi)
        except Exception:
            profile.set_shape(inp.name, trt.Dims(lo), trt.Dims(opt), trt.Dims(hi))
        shown[inp.name] = [lo, opt, hi]
    config.add_optimization_profile(profile)
    return shown


def build(args):
    """Engine from ONNX, saved to --plan."""
    trt = tp.trt_module("trt")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    mw = MemWatch()
    try:
        network, _ = tp.parse_onnx(trt, logger, builder, args.onnx)
        config = tp.new_config(trt, "trt", builder, args.workspace_gb)
        dynamic = any(-1 in tuple(network.get_input(i).shape)
                      for i in range(network.num_inputs))
        prof = None
        if dynamic:
            prof = set_profile(trt, builder, config, network,
                               np.load(args.max_data + ".npz"),
                               np.load(args.opt_data + ".npz"))
        t0 = time.perf_counter()
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("build_serialized_network returned None")
        t_build = time.perf_counter() - t0
    except Exception as err:
        tp.result(stage="build", error=f"{type(err).__name__}: {str(err)[:2000]}")
        return tp.verdict(False, f"build failed: {str(err)[:300]}")
    mem = mw.stop()
    with open(args.plan, "wb") as fh:
        fh.write(bytes(plan))
    tp.result(stage="build", dynamic=dynamic, profile=prof, build_s=t_build,
              build_mem_gb=mem, plan_mb=os.path.getsize(args.plan) / 1e6)
    return tp.verdict(True, f"engine {args.plan} ({'dynamic' if dynamic else 'static'}) "
                      f"in {t_build:.0f} s")


def compare(e, f, data):
    """Errors against the references in data."""
    out = {}
    for tag in ("f32", "f64"):
        if f"forces_{tag}" in data.files:
            out[f"d_force_vs_{tag}"] = float(np.abs(f - data[f"forces_{tag}"]).max())
            out[f"d_e_atom_vs_{tag}"] = float(np.abs(e - data[f"e_atom_{tag}"]).max())
    if "forces_f64" in data.files:
        out["torch_f32_d_force_vs_f64"] = float(
            np.abs(data["forces_f32"] - data["forces_f64"]).max())
        out["torch_f32_d_e_atom_vs_f64"] = float(
            np.abs(data["e_atom_f32"] - data["e_atom_f64"]).max())
    return out


def accurate(err):
    """As accurate as PyTorch float32: within 3 times its error against
    float64 where float64 exists, else close to PyTorch float32 itself."""
    if "d_force_vs_f64" in err:
        return (err["d_force_vs_f64"] <= 3 * max(err["torch_f32_d_force_vs_f64"], 1e-5) + 1e-5
                and err["d_e_atom_vs_f64"] <= 3 * max(err["torch_f32_d_e_atom_vs_f64"], 1e-6) + 1e-6)
    return err["d_force_vs_f32"] <= 1e-3 and err["d_e_atom_vs_f32"] <= 1e-4


def run(args):
    """The engine at one prepared size."""
    import torch
    trt = tp.trt_module("trt")
    data = np.load(args.data + ".npz")
    logger = trt.Logger(trt.Logger.WARNING)
    tp.load_plugins(trt, logger)
    with open(args.plan, "rb") as fh:
        plan = fh.read()
    torch.cuda.synchronize()
    mw = MemWatch()
    try:
        engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        ctx = engine.create_execution_context()
        ins = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                arr = data[tp.input_key(name, data.files)]
                ins[name] = torch.from_numpy(arr).to(
                    device="cuda", dtype=tp.torch_dtype(trt, engine.get_tensor_dtype(name)))
        outs, t_first = tp.run_engine(trt, engine, ctx, ins)
        times = [tp.run_engine(trt, engine, ctx, ins)[1] for _ in range(3)]
    except Exception as err:
        mem = mw.stop()
        tp.result(stage="run", atoms=int(data["atoms"]), mem_gb=mem,
                  error=f"{type(err).__name__}: {str(err)[:1500]}")
        return tp.verdict(False, f"engine run failed at {int(data['atoms'])} atoms: "
                          f"{str(err)[:200]}")
    mem = mw.stop()
    keys = list(outs)
    e = outs["e_atom" if "e_atom" in outs else keys[0]].double().cpu().numpy()
    f = outs["forces" if "forces" in outs else keys[1]].double().cpu().numpy()
    err = compare(e, f, data)
    try:
        dev_mem = engine.device_memory_size_v2 / 1e9
    except Exception:
        dev_mem = None
    rep = {"atoms": int(data["atoms"]), "first_run_s": t_first,
           "run_s": float(np.median(times)), "mem_gb": mem,
           "engine_device_memory_gb": dev_mem,
           "torch_s": float(data["torch_s"]),
           "torch_mem_gb": float(data["torch_mem_gb"]), **err}
    tp.result(stage="run", **rep)
    ok = accurate(err)
    return tp.verdict(ok, f"TensorRT, {rep['atoms']} atoms: {rep['run_s'] * 1e3:.1f} ms "
                      f"and {mem:.1f} GB, against PyTorch {rep['torch_s'] * 1e3:.1f} ms "
                      f"and {rep['torch_mem_gb']:.1f} GB; force error vs f32 "
                      f"{err['d_force_vs_f32']:.1e}")


def inductor(args):
    """torch.compile of the traced graph at one prepared size."""
    import torch
    fp32_only()
    data = np.load(args.data + ".npz")
    rep = int(round((int(data["atoms"]) / 8) ** (1 / 3)))
    model, ins = build_inputs(args.model_dir, rep, torch.float32)
    if int(ins[0].shape[0]) != int(data["atoms"]):
        return tp.verdict(False, "could not rebuild the prepared cell")
    try:
        gm = trace(model, ins, False)
        compiled = torch.compile(gm)
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled(*ins)
        torch.cuda.synchronize()
        t_compile = time.perf_counter() - t0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mw = MemWatch()
        with torch.no_grad():
            t_run, out = timed(lambda: compiled(*ins))
        mem = mw.stop()
    except Exception as err:
        tp.result(stage="inductor", error=f"{type(err).__name__}: {str(err)[:2000]}")
        return tp.verdict(False, f"torch.compile failed: {str(err)[:200]}")
    e = out[0].double().cpu().numpy()
    f = out[1].double().cpu().numpy()
    err = compare(e, f, data)
    rep_ = {"atoms": int(data["atoms"]), "compile_s": t_compile, "run_s": t_run,
            "mem_gb": mem, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            "torch_s": float(data["torch_s"]),
            "torch_mem_gb": float(data["torch_mem_gb"]), **err}
    tp.result(stage="inductor", **rep_)
    return tp.verdict(accurate(err), f"torch.compile, {rep_['atoms']} atoms: "
                      f"{t_run * 1e3:.1f} ms and {mem:.1f} GB, against PyTorch "
                      f"{rep_['torch_s'] * 1e3:.1f} ms and {rep_['torch_mem_gb']:.1f} GB")


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--model_dir", required=True)
    s.add_argument("--rep", type=int, required=True)
    s.add_argument("--f64", action="store_true")
    s.add_argument("--out", required=True)
    s = sub.add_parser("export")
    s.add_argument("--model_dir", required=True)
    s.add_argument("--rep", type=int, required=True)
    s.add_argument("--dynamic", action="store_true")
    s.add_argument("--out", required=True)
    s = sub.add_parser("build")
    s.add_argument("--onnx", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--max_data", help="prefix of the largest prepared size")
    s.add_argument("--opt_data", help="prefix of the size to tune for")
    s.add_argument("--workspace_gb", type=float, default=32)
    s = sub.add_parser("run")
    s.add_argument("--plan", required=True)
    s.add_argument("--data", required=True)
    s = sub.add_parser("inductor")
    s.add_argument("--model_dir", required=True)
    s.add_argument("--data", required=True)
    args = p.parse_args()
    return {"prepare": prepare, "export": export, "build": build, "run": run,
            "inductor": inductor}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
