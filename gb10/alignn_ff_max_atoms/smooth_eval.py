"""Per-atom energies for models with smooth_messages, for the scripts that go
through ff_max_atoms.atom_energies (ff_any.py, ff_stream.py, relax.py,
relax_check.py). ff_max_atoms.py stays byte-identical (stages 4 and 5 check
its md5), so this file replaces that one function at run time. Models without
smooth_messages evaluate as before. Needs the patched ALIGNN
(smooth_messages.patch); on a smooth_messages model the unpatched function
raises instead of silently evaluating a different model.

    python smooth_eval.py relax_check.py --model_dir M --relaxed R --out O
    python smooth_eval.py ff_any.py chunk --model_dir M --structure S --out O
"""
import runpy
import sys

import torch

import ff_max_atoms as ffm


def atom_energies(model, pos, lat, g):
    """ff_max_atoms.atom_energies with the message weights of the model."""
    src, dst, shift = g["src"], g["dst"], g["shift"]
    n = pos.shape[0]
    r = pos.index_select(0, dst) - pos.index_select(0, src) + shift @ lat
    bl = torch.linalg.vector_norm(r, dim=1)

    E = src.shape[0]
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
    r_ij = r.index_select(0, lg_src)
    r_jk = r.index_select(0, lg_dst)
    h_cos = (-(r_ij * r_jk).sum(dim=-1) / (
        torch.linalg.vector_norm(r_ij, dim=-1).clamp_min(1e-12)
        * torch.linalg.vector_norm(r_jk, dim=-1).clamp_min(1e-12))
    ).clamp(-1.0, 1.0)

    w_g = w_lg = None
    if model.smooth_messages:
        w_g, w_lg = model.message_weights(bl, bl.index_select(0, lg_src),
                                          bl.index_select(0, lg_dst))

    x = model.atom_embedding(g["feats"])
    if model.use_element_embedding:
        x = x + model.element_embedding(g["Z"])
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
        x, y, z = layer.forward_tensors(
            src, dst, n, lg_src, lg_dst, E, x, y, z, w_g, w_lg)
    for layer in model.gcn_layers:
        x, y = layer.forward_tensors(src, dst, n, x, y, w_g)
    e_atom = model.fc(x).squeeze(-1)
    e_bond = torch.zeros_like(bl)
    if model.use_penalty:
        e_bond = e_bond + torch.where(
            bl < model.penalty_threshold,
            model.penalty_factor * (model.penalty_threshold - bl),
            torch.zeros_like(bl))
    if model.use_zbl:
        zf = g["Z"].to(bl.dtype)
        e_bond = e_bond + ffm.zbl_edge_energies(
            zf.index_select(0, src), zf.index_select(0, dst), bl,
            model.zbl_cutoff)
    return e_atom, e_bond


def install():
    """Make every later ffm.atom_energies call use this function."""
    ffm.atom_energies = atom_energies


def main():
    """Run another script of this folder with the function replaced."""
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
