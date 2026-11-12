"""Probe: can ALIGNN-FF (energies and forces) run through standard TensorRT 11.4
or TensorRT-RTX 1.7 on a GB10, in float32 and without accuracy loss?

Each subcommand runs as its own process (trt_probe.sbatch), so that one crash
does not hide the other answers. Every subcommand prints "RESULT {json}" lines
and a final "PASS" or "FAIL" line with the reason.

  basic         a tiny network built with the TensorRT API (matrix multiply
                and bias, dynamic first dimension, TF32 off): build, run,
                compare with numpy. For TensorRT-RTX also the runtime cache:
                create, serialize to a file, load it into a second context.
  make-scatter  writes an ONNX model with one ScatterElements(reduction=add),
                the operation ALIGNN uses to sum messages into atoms and bonds
  scatter       imports that model through the ONNX parser, builds, runs,
                compares with numpy
  export        AFF-r2SCAN on rattled Si: builds the bond graph and the line
                graph in PyTorch exactly as ff_max_atoms.atom_energies does,
                then traces the rest (network, per-atom energies, and forces
                as the gradient of the total energy) with make_fx into one
                graph of plain ATen operations, exports it to ONNX, and saves
                the inputs with PyTorch float32 and float64 references.
                Static shapes: one ONNX per cell size.
  model         builds an engine from that ONNX, runs it on the saved inputs,
                compares energies and forces with the references, and times it
                against PyTorch

    python trt_probe.py basic --lib trt
    python trt_probe.py export --model_dir MODEL --rep 4 --out results/aff_si4
    python trt_probe.py model --lib rtx --prefix results/aff_si4
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def result(**kw):
    """One machine-readable line."""
    print("RESULT " + json.dumps(kw, default=str), flush=True)


def verdict(ok, why):
    """The last line of every subcommand."""
    print(("PASS " if ok else "FAIL ") + why, flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------- TensorRT

def trt_module(lib):
    """tensorrt (standard, 11.x) or tensorrt_rtx."""
    if lib == "rtx":
        import tensorrt_rtx as trt
    else:
        import tensorrt as trt
    return trt


def new_network(trt, builder):
    """A strongly typed network. TensorRT 11 ignores the flag (strong typing
    is always on); the TensorRT-RTX sample passes it to avoid a warning."""
    flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flags is not None and hasattr(flags, "STRONGLY_TYPED"):
        try:
            return builder.create_network(1 << int(flags.STRONGLY_TYPED))
        except Exception as err:
            print("create_network with STRONGLY_TYPED failed:", err)
    return builder.create_network()


def new_config(trt, lib, builder, workspace_gb=16):
    """FP32 only: TF32 off (on by default in both versions). For TensorRT-RTX
    build for this GPU only (without it the engine is built for a portable
    set of GPUs, which the 1.7 notes say can be slower)."""
    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * 2**30))
    if lib == "rtx":
        try:
            config.num_compute_capabilities = 1
            config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)
        except Exception as err:
            print("could not target the current GPU only:", err)
    return config


def dims(trt, shape):
    """A TensorRT Dims from a tuple."""
    return trt.Dims2(*shape) if len(shape) == 2 else trt.Dims(shape)


def load_plugins(trt, logger):
    """Standard TensorRT keeps ScatterElements with a reduction in a plugin;
    TensorRT-RTX has no plugins."""
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(logger, "")


TORCH_DTYPES = {}


def torch_dtype(trt, dt):
    """TensorRT data type to torch."""
    import torch
    if not TORCH_DTYPES:
        TORCH_DTYPES.update({trt.DataType.FLOAT: torch.float32,
                             trt.DataType.INT32: torch.int32,
                             trt.DataType.BOOL: torch.bool})
        if hasattr(trt.DataType, "INT64"):
            TORCH_DTYPES[trt.DataType.INT64] = torch.int64
    return TORCH_DTYPES[dt]


def make_context(trt, lib, engine, cache_file=None):
    """An execution context; for TensorRT-RTX with a runtime cache that is
    loaded from cache_file if it exists. Returns (context, cache or None)."""
    if lib != "rtx":
        return engine.create_execution_context(), None
    rc = engine.create_runtime_config()
    cache = rc.create_runtime_cache()
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "rb") as fh:
            cache.deserialize(fh.read())
    rc.set_runtime_cache(cache)
    try:  # compile shape-specialized kernels up front, not in the background
        rc.dynamic_shapes_kernel_specialization_strategy = \
            trt.DynamicShapesKernelSpecializationStrategy.EAGER
    except Exception as err:
        print("could not set the EAGER kernel strategy:", err)
    return engine.create_execution_context(rc), cache


def run_engine(trt, engine, context, inputs):
    """Run once with PyTorch-owned device buffers; inputs: name -> tensor.
    Returns name -> output tensor and the wall time of the call."""
    import torch
    for name, t in inputs.items():
        context.set_input_shape(name, tuple(t.shape))
        context.set_tensor_address(name, int(t.data_ptr()))
    missing = context.infer_shapes()
    if missing:
        raise RuntimeError(f"input shapes not specified: {missing}")
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        t = torch.empty(shape, dtype=torch_dtype(trt, engine.get_tensor_dtype(name)),
                        device="cuda")
        context.set_tensor_address(name, int(t.data_ptr()))
        outs[name] = t
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if ok is False:
        raise RuntimeError("execute_async_v3 returned False")
    return outs, dt


def build(trt, builder, network, config, profiles=None):
    """Serialized engine, or an exception with the reason."""
    if profiles:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profiles.items():
            profile.set_shape(name, dims(trt, lo), dims(trt, opt), dims(trt, hi))
        config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")
    return plan, time.perf_counter() - t0


def basic(args):
    """Matrix multiply and bias built with the API."""
    import torch
    trt = trt_module(args.lib)
    result(lib=args.lib, version=getattr(trt, "__version__", None),
           torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(trt, builder)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((1, 8)).astype(np.float32)
    x_in = network.add_input("x", trt.float32, trt.Dims2(-1, 4))
    w_c = network.add_constant(trt.Dims2(4, 8), trt.Weights(w)).get_output(0)
    b_c = network.add_constant(trt.Dims2(1, 8), trt.Weights(b)).get_output(0)
    mm = network.add_matrix_multiply(x_in, trt.MatrixOperation.NONE,
                                      w_c, trt.MatrixOperation.NONE)
    out = network.add_elementwise(mm.get_output(0), b_c,
                                  trt.ElementWiseOperation.SUM).get_output(0)
    out.name = "y"
    network.mark_output(out)
    config = new_config(trt, args.lib, builder, 1)
    plan, t_build = build(trt, builder, network, config,
                          {"x": ((1, 4), (1000, 4), (100000, 4))})
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    cache_file = os.path.join(args.workdir, f"basic_{args.lib}.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)
    ctx, cache = make_context(trt, args.lib, engine)
    x = torch.from_numpy(rng.standard_normal((1000, 4)).astype(np.float32)).cuda()
    outs, t1 = run_engine(trt, engine, ctx, {"x": x})
    ref = x.cpu().numpy() @ w + b
    err = float(np.abs(outs["y"].cpu().numpy() - ref).max())
    info = {"build_s": t_build, "first_run_s": t1, "max_abs_err": err}
    if cache is not None:  # TensorRT-RTX: save the cache, load it again
        blob = cache.serialize()
        with open(cache_file, "wb") as fh:
            fh.write(bytes(blob))
        ctx2, _ = make_context(trt, args.lib, engine, cache_file)
        outs2, t2 = run_engine(trt, engine, ctx2, {"x": x})
        info.update(cache_bytes=os.path.getsize(cache_file),
                    run_with_loaded_cache_s=t2,
                    max_abs_err_with_cache=float(
                        np.abs(outs2["y"].cpu().numpy() - ref).max()))
    result(**info)
    return verdict(err < 1e-4, f"basic {args.lib}: max error {err:.2e}")


def make_scatter(args):
    """ONNX: out = ScatterElements(data, indices, updates, axis 0, add)."""
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("ScatterElements", ["data", "indices", "updates"],
                            ["out"], axis=0, reduction="add")
    graph = helper.make_graph(
        [node], "scatter_add",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 8]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, ["E", 8]),
         helper.make_tensor_value_info("updates", TensorProto.FLOAT, ["E", 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                              ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    return verdict(True, f"wrote {args.out}")


def parse_onnx(trt, logger, builder, path):
    """Network from an ONNX file, or an exception listing the parser errors."""
    load_plugins(trt, logger)
    network = new_network(trt, builder)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(path):
        errs = []
        for i in range(parser.num_errors):
            e = parser.get_error(i)
            errs.append(f"{e.node_operator()} {e.node_name()}: {e.desc()}")
        raise RuntimeError("ONNX parse failed: " + " | ".join(errs)[:2000])
    return network, parser


def scatter(args):
    """The ScatterElements(add) model through the ONNX parser."""
    import torch
    trt = trt_module(args.lib)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        network, _ = parse_onnx(trt, logger, builder, args.onnx)
        plan, t_build = build(trt, builder, network,
                              new_config(trt, args.lib, builder, 1),
                              {"data": ((1, 8), (1000, 8), (100000, 8)),
                               "indices": ((1, 8), (30000, 8), (3000000, 8)),
                               "updates": ((1, 8), (30000, 8), (3000000, 8))})
    except Exception as err:
        result(lib=args.lib, stage="parse or build", error=str(err)[:2000])
        return verdict(False, f"scatter-add {args.lib}: {str(err)[:300]}")
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx, _ = make_context(trt, args.lib, engine)
    rng = np.random.default_rng(1)
    n, e = 1000, 30000
    idx = rng.integers(0, n, e)
    upd = rng.standard_normal((e, 8)).astype(np.float32)
    ins = {"data": torch.zeros((n, 8), device="cuda"),
           "indices": torch.from_numpy(np.repeat(idx[:, None], 8, 1)).cuda(),
           "updates": torch.from_numpy(upd).cuda()}
    outs, t1 = run_engine(trt, engine, ctx, ins)
    ref = np.zeros((n, 8), np.float64)
    np.add.at(ref, idx, upd.astype(np.float64))
    err = float(np.abs(outs["out"].cpu().numpy() - ref).max())
    result(lib=args.lib, build_s=t_build, run_s=t1, max_abs_err=err,
           note="float32 sums of about 30 numbers of size 1")
    return verdict(err < 1e-4, f"scatter-add {args.lib}: max error {err:.2e}")


# ------------------------------------------------------------- ALIGNN-FF

def line_graph(model, src, dst, bl, n):
    """The line-graph indices exactly as ff_max_atoms.atom_energies builds
    them (bonds up to the three-body cutoff, pairs sharing an atom)."""
    import torch
    allowed_ids = torch.nonzero(bl <= model.three_body_cutoff).squeeze(-1)
    sub_src = src.index_select(0, allowed_ids)
    order = torch.argsort(sub_src, stable=True)
    sorted_edge_ids = allowed_ids.index_select(0, order)
    sorted_src = sub_src.index_select(0, order)
    node_range = torch.arange(n, device=src.device)
    bucket_start = torch.searchsorted(sorted_src, node_range)
    bucket_end = torch.searchsorted(sorted_src, node_range, right=True)
    A_v = dst.index_select(0, allowed_ids)
    starts = bucket_start.index_select(0, A_v)
    counts = bucket_end.index_select(0, A_v) - starts
    total = int(counts.sum().item())
    lg_src = torch.repeat_interleave(allowed_ids, counts)
    cum = torch.cumsum(counts, dim=0)
    offsets = torch.arange(total, device=src.device) - torch.repeat_interleave(
        cum - counts, counts)
    pos_idx = torch.repeat_interleave(starts, counts) + offsets
    lg_dst = sorted_edge_ids.index_select(0, pos_idx)
    return lg_src, lg_dst


def network_energies(model, pos, lat, src, dst, shift, feats, z_num,
                     lg_src, lg_dst):
    """The rest of ff_max_atoms.atom_energies, given the line graph:
    energy of each atom and of each bond."""
    import torch
    import ff_max_atoms as ffm
    n = pos.shape[0]
    E = src.shape[0]
    r = pos.index_select(0, dst) - pos.index_select(0, src) + shift @ lat
    bl = torch.linalg.vector_norm(r, dim=1)
    r_ij = r.index_select(0, lg_src)
    r_jk = r.index_select(0, lg_dst)
    h_cos = (-(r_ij * r_jk).sum(dim=-1) / (
        torch.linalg.vector_norm(r_ij, dim=-1).clamp_min(1e-12)
        * torch.linalg.vector_norm(r_jk, dim=-1).clamp_min(1e-12))
    ).clamp(-1.0, 1.0)
    x = model.atom_embedding(feats)
    if model.use_element_embedding:
        x = x + model.element_embedding(z_num)
    if model.use_cutoff_function:
        c_off = ffm.cutoff_function_based_edges(
            bl, inner_cutoff=model.inner_cutoff, exponent=model.exponent)
        if model.multiply_cutoff:
            y = model.edge_embedding(bl) * c_off.unsqueeze(-1)
        else:
            y = model.edge_embedding(c_off)
    else:
        y = model.edge_embedding(bl)
    z = model.angle_embedding(h_cos)
    for layer in model.alignn_layers:
        x, y, z = layer.forward_tensors(src, dst, n, lg_src, lg_dst, E, x, y, z)
    for layer in model.gcn_layers:
        x, y = layer.forward_tensors(src, dst, n, x, y)
    e_atom = model.fc(x).squeeze(-1)
    e_bond = torch.zeros_like(bl)
    if model.use_penalty:
        e_bond = e_bond + torch.where(
            bl < model.penalty_threshold,
            model.penalty_factor * (model.penalty_threshold - bl),
            torch.zeros_like(bl))
    if model.use_zbl:
        zf = z_num.to(bl.dtype)
        e_bond = e_bond + ffm.zbl_edge_energies(
            zf.index_select(0, src), zf.index_select(0, dst), bl,
            model.zbl_cutoff)
    return e_atom, e_bond


INPUT_NAMES = ["pos", "lat", "src", "dst", "shift", "feats", "z_num",
               "lg_src", "lg_dst"]


def rewrite_accumulate(gm, ins):
    """Accumulating index writes along dim 0 with one 1-D index
    (index_put with accumulate=True, index_add), which the ONNX exporter
    turns into ScatterND with reduction add (rejected by TensorRT), become
    scatter_add with the index broadcast to the shape of the values, which
    exports as ScatterElements add (imported by standard TensorRT through
    its plugin). The arithmetic is the same: out = self; out[idx[k]] +=
    values[k]. Returns counts of rewritten and skipped nodes."""
    import torch
    from torch.fx.passes.shape_prop import ShapeProp
    aten = torch.ops.aten
    ShapeProp(gm).propagate(*ins)

    def ndim(n):
        tm = n.meta.get("tensor_meta") if hasattr(n, "meta") else None
        return len(tm.shape) if tm is not None else None

    put = {aten.index_put.default}
    if hasattr(aten, "_unsafe_index_put"):
        put.add(aten._unsafe_index_put.default)
    # in-place forms are replaced only when they write into an array nothing
    # else reads (the fresh zeros of a backward formula), where the
    # functional form gives the same result
    put_inplace = {aten.index_put_.default}
    add_ops = {aten.index_add.default}
    add_inplace = {aten.index_add_.default}
    done, skipped = {}, {}
    g = gm.graph
    for node in list(g.nodes):
        if node.op != "call_function":
            continue
        t = node.target
        if t in put or t in put_inplace:
            self_, indices, values = node.args[:3]
            acc = node.args[3] if len(node.args) > 3 else \
                node.kwargs.get("accumulate", False)
            ok = (acc and len(indices) == 1 and indices[0] is not None
                  and (t in put or len(self_.users) == 1))
            idx = indices[0] if ok else None
        elif t in add_ops or t in add_inplace:
            self_, dim, idx, values = node.args[:4]
            alpha = node.args[4] if len(node.args) > 4 else \
                node.kwargs.get("alpha", 1)
            ok = dim == 0 and alpha == 1 and (t in add_ops or len(self_.users) == 1)
        else:
            continue
        nd_v, nd_s, nd_i = ndim(values), ndim(self_), ndim(idx) if ok else None
        if not ok or nd_v is None or nd_v != nd_s or nd_i != 1:
            skipped[str(t)] = skipped.get(str(t), 0) + 1
            continue
        with g.inserting_before(node):
            zeros = g.call_function(aten.zeros_like.default, (values,),
                                    {"dtype": torch.int64})
            idx_b = idx
            for _ in range(nd_v - 1):
                idx_b = g.call_function(aten.unsqueeze.default, (idx_b, -1))
            full = g.call_function(aten.add.Tensor, (zeros, idx_b))
            new = g.call_function(aten.scatter_add.default,
                                  (self_, 0, full, values))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        done[str(t)] = done.get(str(t), 0) + 1
    g.lint()
    gm.recompile()
    return done, skipped


def export_onnx(torch, gm, ins, path, opset):
    """ONNX file by the first route that works; (route, errors)."""
    names = dict(input_names=INPUT_NAMES, output_names=["e_atom", "forces"])

    def via_dynamo():
        torch.onnx.export(gm, tuple(ins), path, dynamo=True,
                          opset_version=opset, external_data=False, **names)

    def via_export_program():
        ep = torch.export.export(gm, tuple(ins), strict=False)
        torch.onnx.export(ep, f=path, opset_version=opset,
                          external_data=False, **names)

    def via_torchscript():
        torch.onnx.export(gm, tuple(ins), path, dynamo=False,
                          opset_version=opset, **names)

    errors = {}
    for label, fn in (("dynamo", via_dynamo),
                      ("torch.export then dynamo", via_export_program),
                      ("torchscript", via_torchscript)):
        try:
            fn()
            return label, errors
        except Exception as err:
            errors[label] = f"{type(err).__name__}: {str(err)[:1500]}"
    return None, errors


def describe_onnx(path):
    """Operator counts and the scatter nodes of an ONNX file."""
    import onnx
    m = onnx.load(path)
    op_types, scatter_modes = {}, {}
    for node in m.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        if node.op_type.startswith("Scatter"):
            red = [a.s.decode() for a in node.attribute if a.name == "reduction"]
            key = f"{node.op_type}:{red[0] if red else 'none'}"
            scatter_modes[key] = scatter_modes.get(key, 0) + 1
    return {"onnx_mb": os.path.getsize(path) / 1e6,
            "opset": [(o.domain, o.version) for o in m.opset_import],
            "inputs": [i.name for i in m.graph.input],
            "outputs": [o.name for o in m.graph.output],
            "op_types": op_types, "scatter_nodes": scatter_modes}


def input_key(name, keys):
    """The saved array for an engine input. Exporters may rename inputs
    ("pos_1", "arg0_1"), so try the exact name, then a name that starts with
    one of ours followed by "_", then the position in "argK" names."""
    if name in keys:
        return name
    for cand in sorted(INPUT_NAMES, key=len, reverse=True):
        if name.startswith(cand + "_"):
            return cand
    digits = "".join(c for c in name.split("_")[0] if c.isdigit())
    if name.startswith("arg") and digits and int(digits) < len(INPUT_NAMES):
        return INPUT_NAMES[int(digits)]
    raise KeyError(f"no saved array for engine input {name}")


def export(args):
    """Trace energy and forces, export to ONNX, save inputs and references."""
    import torch
    import cells
    import ff_max_atoms as ffm
    from torch.fx.experimental.proxy_tensor import make_fx
    dev = torch.device("cuda")
    pos_np, z_np, L = cells.crystal("Si", args.rep, rattle=0.05, seed=1)
    pos_np = np.asarray(pos_np, dtype=float)
    cell = np.diag(np.asarray(L, dtype=float))
    n = len(pos_np)

    def setup(dtype):
        ffm.DTYPE = dtype
        model, cfg = ffm.load_model(args.model_dir, dev)
        for p in model.parameters():
            p.requires_grad_(False)
        g = ffm.build_graph(pos_np, cell, z_np, cfg, dev)
        pos = torch.tensor(pos_np, dtype=dtype, device=dev)
        lat = torch.tensor(cell, dtype=dtype, device=dev)
        r = pos[g["dst"]] - pos[g["src"]] + g["shift"] @ lat
        lg_src, lg_dst = line_graph(model, g["src"], g["dst"],
                                    torch.linalg.vector_norm(r, dim=1), n)
        ins = [pos, lat, g["src"], g["dst"], g["shift"], g["feats"], g["Z"],
               lg_src, lg_dst]
        return model, cfg, ins

    def joint_fn(model):
        # DeePMD's recipe (deepmd/pt_expt): autograd.grad inside the traced
        # function, so make_fx records the backward as plain operations
        def joint(pos, lat, src, dst, shift, feats, z_num, lg_src, lg_dst):
            pos = pos.detach().requires_grad_(True)
            e_atom, e_bond = network_energies(model, pos, lat, src, dst, shift,
                                              feats, z_num, lg_src, lg_dst)
            (g,) = torch.autograd.grad(e_atom.sum() + e_bond.sum(), pos)
            return e_atom.detach(), -g
        return joint

    # references: the code path the runs use (atom_energies + autograd)
    refs = {}
    for tag, dtype in (("f64", torch.float64), ("f32", torch.float32)):
        model, cfg, ins = setup(dtype)
        p = ins[0].clone().requires_grad_(True)
        g = {"src": ins[2], "dst": ins[3], "shift": ins[4], "feats": ins[5],
             "Z": ins[6]}
        e_atom, e_bond = ffm.atom_energies(model, p, ins[1], g)
        (grad,) = torch.autograd.grad(e_atom.sum() + e_bond.sum(), p)
        refs[f"e_atom_{tag}"] = e_atom.detach().double().cpu().numpy()
        refs[f"e_bond_sum_{tag}"] = float(e_bond.sum())
        refs[f"forces_{tag}"] = (-grad).double().cpu().numpy()
    model, cfg, ins = setup(torch.float32)
    joint = joint_fn(model)
    with torch.enable_grad():
        ea, fo = joint(*ins)
    d_e = float(np.abs(ea.double().cpu().numpy() - refs["e_atom_f32"]).max())
    d_f = float(np.abs(fo.double().cpu().numpy() - refs["forces_f32"]).max())
    result(stage="rewritten function against atom_energies, float32",
           atoms=n, edges=int(ins[2].numel()), line_edges=int(ins[7].numel()),
           max_abs_d_e_atom=d_e, max_abs_d_force=d_f)
    if d_e > 1e-5 or d_f > 1e-4:
        return verdict(False, "the function to export does not reproduce "
                       f"atom_energies (energy {d_e:.2e}, forces {d_f:.2e})")

    # eager timing of the code path the runs use
    t = []
    for _ in range(4):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = ins[0].clone().requires_grad_(True)
        g = {"src": ins[2], "dst": ins[3], "shift": ins[4], "feats": ins[5],
             "Z": ins[6]}
        e_atom, e_bond = ffm.atom_energies(model, p, ins[1], g)
        torch.autograd.grad(e_atom.sum() + e_bond.sum(), p)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    torch_s = float(np.median(t[1:]))
    # the part the engine replaces: network and gradient, line graph given
    t = []
    for _ in range(4):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.enable_grad():
            joint(*ins)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    torch_joint_s = float(np.median(t[1:]))
    result(stage="PyTorch timing", atom_energies_and_grad_s=torch_s,
           same_part_as_engine_s=torch_joint_s)

    # trace: one graph of plain ATen ops, forward and backward. Functionalized
    # first (no in-place ops), so that the rewrite below sees every write
    try:
        from torch._decomp import core_aten_decompositions
        table = core_aten_decompositions()
    except Exception as err:
        print("no core ATen decomposition table:", err)
        table = {}
    # Layer norm into basic operations. native_layer_norm is a core op and
    # returns the mean and inverse standard deviation for the backward; the
    # exported LayerNormalization then has three outputs, and TensorRT 11.4
    # and TensorRT-RTX 1.7 accept only the first (probe job 21913)
    try:
        from torch._decomp import get_decompositions
        aten = torch.ops.aten
        table.update(get_decompositions([aten.native_layer_norm,
                                         aten.native_layer_norm_backward]))
    except Exception as err:
        print("could not add the layer-norm decompositions:", err)
    t0 = time.perf_counter()
    traced = None
    for label, fn in (("functionalized", lambda: torch.func.functionalize(
                           joint, remove="mutations_and_views")),
                      ("plain", lambda: joint)):
        try:
            with torch.enable_grad():
                gm = make_fx(fn(), decomposition_table=table)(*ins)
            traced = label
            break
        except Exception as err:
            print(f"make_fx ({label}) failed: {type(err).__name__}: {str(err)[:800]}")
    if traced is None:
        return verdict(False, "make_fx could not trace energy and forces")

    def check(gm_, stage):
        ops = {}
        for node in gm_.graph.nodes:
            if node.op == "call_function":
                ops[str(node.target)] = ops.get(str(node.target), 0) + 1
        with torch.no_grad():  # the traced graph has no autograd left in it
            ea2, fo2 = gm_(*ins)
        d_e2 = float((ea2 - ea).abs().max())
        d_f2 = float((fo2 - fo).abs().max())
        result(stage=stage, nodes=len(gm_.graph.nodes), op_counts=ops,
               max_abs_d_e_atom_vs_eager=d_e2, max_abs_d_force_vs_eager=d_f2)
        return d_e2 <= 1e-5 and d_f2 <= 1e-4

    ok_trace = check(gm, f"make_fx trace ({traced}), {time.perf_counter() - t0:.1f} s")
    data = dict(**{name: t.cpu().numpy() for name, t in zip(INPUT_NAMES, ins)},
                **refs, torch_eager_s=torch_s, torch_joint_s=torch_joint_s,
                atoms=n)
    exported = []

    # as traced
    route, errors = export_onnx(torch, gm, ins, args.out + ".onnx", args.opset)
    if route:
        np.savez(args.out + ".npz", **data)
        result(stage="onnx export, as traced", exporter=route,
               failed_attempts=errors, **describe_onnx(args.out + ".onnx"))
        exported.append("as traced")
    else:
        result(stage="onnx export, as traced", errors=errors)

    # accumulating index writes rewritten as scatter_add
    try:
        done, skipped = rewrite_accumulate(gm, ins)
        ok_rw = check(gm, "after the scatter_add rewrite")
        result(stage="scatter_add rewrite", rewritten=done, skipped=skipped,
               matches_eager=ok_rw)
        if done and ok_rw:
            sa = args.out + "_sa"
            route, errors = export_onnx(torch, gm, ins, sa + ".onnx", args.opset)
            if route:
                np.savez(sa + ".npz", **data)
                result(stage="onnx export, rewritten", exporter=route,
                       failed_attempts=errors, **describe_onnx(sa + ".onnx"))
                exported.append("rewritten")
            else:
                result(stage="onnx export, rewritten", errors=errors)
    except Exception as err:
        result(stage="scatter_add rewrite", error=f"{type(err).__name__}: {str(err)[:1500]}")

    ok = ok_trace and bool(exported)
    return verdict(ok, f"{n} atoms: trace {'matches' if ok_trace else 'DIFFERS FROM'} "
                   f"eager; exported: {', '.join(exported) or 'nothing'}; PyTorch "
                   f"energy and forces {torch_s * 1e3:.1f} ms")


def model_run(args):
    """Engine from the exported ONNX against the PyTorch references."""
    import torch
    trt = trt_module(args.lib)
    data = np.load(args.prefix + ".npz")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        network, _ = parse_onnx(trt, logger, builder, args.prefix + ".onnx")
        names = [network.get_input(i).name for i in range(network.num_inputs)]
        plan, t_build = build(trt, builder, network,
                              new_config(trt, args.lib, builder, args.workspace_gb))
    except Exception as err:
        result(lib=args.lib, stage="parse or build", error=str(err)[:3000])
        return verdict(False, f"model {args.lib}: {str(err)[:300]}")
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    cache_file = args.prefix + f"_{args.lib}.cache"
    ctx, cache = make_context(trt, args.lib, engine)
    ins = {}
    engine_inputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                     if engine.get_tensor_mode(engine.get_tensor_name(i))
                     == trt.TensorIOMode.INPUT]
    result(network_inputs=names, engine_inputs=engine_inputs)
    for name in engine_inputs:
        arr = data[input_key(name, data.files)]
        want = torch_dtype(trt, engine.get_tensor_dtype(name))
        ins[name] = torch.from_numpy(arr).to(device="cuda", dtype=want)
    outs, t_first = run_engine(trt, engine, ctx, ins)
    times = [run_engine(trt, engine, ctx, ins)[1] for _ in range(5)]
    if cache is not None:
        with open(cache_file, "wb") as fh:
            fh.write(bytes(cache.serialize()))
    keys = list(outs)
    e_key = "e_atom" if "e_atom" in outs else keys[0]
    f_key = "forces" if "forces" in outs else keys[1]
    e = outs[e_key].double().cpu().numpy()
    f = outs[f_key].double().cpu().numpy()
    n = int(data["atoms"])
    rep = {"lib": args.lib, "atoms": n, "build_s": t_build,
           "first_run_s": t_first, "run_s_median": float(np.median(times)),
           "torch_eager_s": float(data["torch_eager_s"]),
           "torch_same_part_s": float(data["torch_joint_s"]),
           "outputs": {k: list(v.shape) for k, v in outs.items()}}
    for tag in ("f32", "f64"):
        rep[f"max_abs_d_e_atom_vs_torch_{tag}"] = float(
            np.abs(e - data[f"e_atom_{tag}"]).max())
        rep[f"max_abs_d_force_vs_torch_{tag}"] = float(
            np.abs(f - data[f"forces_{tag}"]).max())
        rep[f"d_energy_per_atom_vs_torch_{tag}"] = float(
            (e.sum() - data[f"e_atom_{tag}"].sum()) / n)
    # PyTorch float32's own error against float64, for scale
    rep["torch_f32_vs_f64_max_abs_d_force"] = float(
        np.abs(data["forces_f32"] - data["forces_f64"]).max())
    rep["torch_f32_vs_f64_max_abs_d_e_atom"] = float(
        np.abs(data["e_atom_f32"] - data["e_atom_f64"]).max())
    result(**rep)
    # as accurate as PyTorch float32: within 3 times its own error against
    # float64, for forces and for every atom's energy
    scale_f = max(rep["torch_f32_vs_f64_max_abs_d_force"], 1e-5)
    scale_e = max(rep["torch_f32_vs_f64_max_abs_d_e_atom"], 1e-6)
    ok = (rep["max_abs_d_force_vs_torch_f64"] <= 3 * scale_f + 1e-5
          and rep["max_abs_d_e_atom_vs_torch_f64"] <= 3 * scale_e + 1e-6)
    return verdict(ok, f"model {args.lib}: forces differ from float64 by "
                   f"{rep['max_abs_d_force_vs_torch_f64']:.2e} eV/A (PyTorch "
                   f"float32: {rep['torch_f32_vs_f64_max_abs_d_force']:.2e}); "
                   f"{rep['run_s_median'] * 1e3:.1f} ms against "
                   f"{rep['torch_same_part_s'] * 1e3:.1f} ms in PyTorch for the "
                   "same part")


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("basic")
    s.add_argument("--lib", choices=["trt", "rtx"], required=True)
    s.add_argument("--workdir", default=".")
    s = sub.add_parser("make-scatter")
    s.add_argument("--out", required=True)
    s = sub.add_parser("scatter")
    s.add_argument("--lib", choices=["trt", "rtx"], required=True)
    s.add_argument("--onnx", required=True)
    s = sub.add_parser("export")
    s.add_argument("--model_dir", required=True)
    s.add_argument("--rep", type=int, default=4, help="Si cell, 8 rep^3 atoms")
    s.add_argument("--opset", type=int, default=18)
    s.add_argument("--out", required=True, help="prefix of .onnx and .npz")
    s = sub.add_parser("model")
    s.add_argument("--lib", choices=["trt", "rtx"], required=True)
    s.add_argument("--prefix", required=True)
    s.add_argument("--workspace_gb", type=float, default=32)
    args = p.parse_args()
    fn = {"basic": basic, "make-scatter": make_scatter, "scatter": scatter,
          "export": export, "model": model_run}[args.mode]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
