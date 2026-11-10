"""Stage 13: the retrained smooth_messages model (stage 12) against AFF-r2SCAN.

test  energy and force errors against MatPES r2SCAN, through forward(g) as in
      training, on the test frames of the retraining run (its
      ids_train_val_test.json) and on a sample of its training frames. Also
      checks the split: whether it follows from the data order and seed alone,
      and, given AFF-r2SCAN's own split file, whether the two agree. Without
      that file, "AFF-r2SCAN never saw these test frames" rests on its config
      recording the same seed, split sizes and data.
nve   energy conservation: velocity Verlet in float64 through forward_tensors,
      every model from the same start, on rattled Si at 1500 K (1 fs) and on
      water at 300 K (0.5 fs), with and without the 52-neighbor cap (see
      SYSTEMS). A force jump at a cutoff shows as a step change of the total
      energy far above the typical one.

    PYTHONPATH=alignn_smooth python compare_models.py test \
        --models aff=models/matpes_r2scan_paper smooth=models/aff_smooth \
        --data DATA --ids models/aff_smooth/ids_train_val_test.json --out O
    PYTHONPATH=alignn_smooth python compare_models.py nve \
        --models aff=models/matpes_r2scan_paper smooth=models/aff_smooth --out O

Run with the patched ALIGNN: with smooth_messages off it reproduces the
unpatched code (stage 12 check), so both models go through the same code.
"""
import argparse
import hashlib
import json
import os
import random
import time
import zipfile

import numpy as np
import torch

import cells
import ff_max_atoms as ffm
from smooth_check import energy_forces

# name: (builder, temperature K, time step fs, neighbor cap or None for the
# model's 52). The cap never binds in Si (about 28 neighbors within 5 A) but binds
# in water, where a bond appears or vanishes inside the cutoff whenever an
# atom's 52nd and 53rd nearest neighbors swap, a jump that smooth_messages
# does not remove. water_300K_nocap lifts the cap so that this jump can be
# told apart from the one at the 3.5 A three-body cutoff. Without the cap the
# model sees more bonds than in training, so its energies there are not
# meaningful; only its energy conservation is.
SYSTEMS = {
    "si_1500K": (lambda: cells.crystal("Si", 4, rattle=0.05, seed=1), 1500.0, 1.0, None),
    "water_300K": (lambda: cells.water(8, seed=1), 300.0, 0.5, None),
    "water_300K_nocap": (lambda: cells.water(8, seed=1), 300.0, 0.5, 10000),
}


def parse_models(items):
    """name=dir pairs, in the order given."""
    models = {}
    for item in items:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--models takes name=dir, got {item!r}")
        models[name] = path
    return models


def md5(path):
    """md5 of a file, for the record."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_models(dirs, device):
    """Each model in ffm.DTYPE, with its config and checkpoint md5."""
    models, info = {}, {}
    for name, d in dirs.items():
        model, cfg = ffm.load_model(d, device)
        model.config.stresswise_weight = 0.0  # energies and forces only
        models[name] = (model, cfg)
        info[name] = {"dir": os.path.abspath(d),
                      "best_model_md5": md5(os.path.join(d, "best_model.pt")),
                      "smooth_messages": bool(cfg["model"].get("smooth_messages",
                                                               False))}
        print(f"model {name}: {info[name]}", flush=True)
    return models, info


def header():
    """Where and with what this ran."""
    import alignn
    return {"slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename, "torch": torch.__version__,
            "alignn": os.path.dirname(alignn.__file__),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
            else None}


def dump(payload, out):
    """JSON next to the npz, written atomically."""
    tmp = out + ".json.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, out + ".json")
    print("wrote", out + ".json", flush=True)


# ---------------------------------------------------------------- test errors

def load_id_prop(data_dir):
    """Entries of id_prop.json or id_prop.json.zip, as train_alignn reads them."""
    zp = os.path.join(data_dir, "id_prop.json.zip")
    if os.path.exists(zp):
        return json.loads(zipfile.ZipFile(zp).read("id_prop.json"))
    return json.load(open(os.path.join(data_dir, "id_prop.json")))


def check_split(entries, ids, run_cfg, id_tag, aff_ids):
    """Whether the run's split is alignn.data.get_id_train_val_test applied to
    the data in file order, and how it compares with AFF-r2SCAN's split."""
    n_tr, n_va, n_te = (len(ids[k]) for k in ("id_train", "id_val", "id_test"))
    out = {"entries": len(entries), "n_train": n_tr, "n_val": n_va,
           "n_test": n_te, "random_seed": run_cfg.get("random_seed"),
           "keep_data_order": run_cfg.get("keep_data_order")}
    if run_cfg.get("keep_data_order") is False and out["random_seed"] is not None:
        order = list(np.arange(len(entries)))
        random.seed(out["random_seed"])
        random.shuffle(order)
        jid = [e[id_tag] for e in entries]
        out["recomputed_from_file_order"] = (
            [jid[i] for i in order[:n_tr]] == ids["id_train"]
            and [jid[i] for i in order[-n_te:]] == ids["id_test"])
    if aff_ids:
        a = json.load(open(aff_ids))
        test = set(ids["id_test"])
        known = {e[id_tag] for e in entries}
        # a split file of another data set would show no overlap and look
        # clean, so also count how many of its ids are in this data at all
        out.update(aff_ids_file=os.path.abspath(aff_ids),
                   aff_sizes=[len(a[k]) for k in ("id_train", "id_val", "id_test")],
                   aff_ids_in_this_data=sum(
                       len(known.intersection(a[k]))
                       for k in ("id_train", "id_val", "id_test")),
                   aff_same_test_frames=set(a["id_test"]) == test,
                   test_frames_in_aff_train=len(test & set(a["id_train"])),
                   test_frames_in_aff_val=len(test & set(a["id_val"])))
    else:
        out["aff_ids_file"] = None
    return out


def predict(model, cfg, atoms_dict, device):
    """Energy per atom as trained ("out"), the same with the short-bond penalty
    that forward_tensors adds, and forces, from forward((g, lg, lat))."""
    from jarvis.core.atoms import Atoms
    from alignn.torch_graph_builder import build_pure_torch_graph
    atoms = Atoms.from_dict(atoms_dict)
    g, lg = build_pure_torch_graph(
        atoms, two_body_cutoff=float(cfg["cutoff"]),
        three_body_cutoff=float(model.three_body_cutoff),
        max_neighbors=int(cfg["max_neighbors"]),
        atom_features=cfg.get("atom_features", "cgcnn"), device=device)
    lat = torch.tensor(np.array(atoms.lattice_mat), dtype=torch.float32,
                       device=device)
    res = model((g, lg, lat))
    n = atoms.num_atoms
    e = float(res["out"].squeeze())
    pen = 0.0
    if model.config.use_penalty:
        bl = g.edata["r"].norm(dim=1)
        pen = float((model.config.penalty_factor * (
            model.config.penalty_threshold - bl).clamp_min(0.0)).sum())
    return e, e + pen / n, res["grad"].detach().double().cpu().numpy()


def evaluate(models, entries, keys, device, target_key, force_key, label):
    """Per-frame errors of every model; a frame that fails is NaN."""
    res = {}
    n = len(keys)
    for name, (model, cfg) in models.items():
        r = {k: np.full(n, np.nan) for k in
             ("e_pred", "de", "de_pen", "f_abs", "f_sq", "f_max")}
        r["f_comp"] = np.zeros(n, dtype=np.int64)
        failed, t0 = 0, time.perf_counter()
        for k, key in enumerate(keys):
            try:
                ent = entries[key]
                e, e_pen, f = predict(model, cfg, ent["atoms"], device)
                ref_f = np.asarray(ent[force_key], dtype=float).reshape(-1, 3)
                if f.reshape(-1, 3).shape != ref_f.shape:
                    raise ValueError(f"forces {f.shape} against {ref_f.shape}")
            except Exception as err:  # left out of every model's average
                failed += 1
                if failed <= 5:
                    print(f"{label} {name} {key}: {type(err).__name__}: {err}",
                          flush=True)
                continue
            ref_e = float(ent[target_key])
            df = np.abs(f.reshape(-1, 3) - ref_f)
            r["e_pred"][k] = e
            r["de"][k] = e - ref_e
            r["de_pen"][k] = e_pen - ref_e
            r["f_abs"][k] = df.sum()
            r["f_sq"][k] = (df * df).sum()
            r["f_max"][k] = df.max() if df.size else 0.0
            r["f_comp"][k] = df.size
            if (k + 1) % 2000 == 0:
                print(f"{label} {name}: {k + 1:,}/{n:,} frames, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
        r["wall_s"] = time.perf_counter() - t0
        r["failed"] = failed
        res[name] = r
    return res


def summarize(res):
    """Errors over the frames every model evaluated."""
    ok = np.logical_and.reduce([np.isfinite(r["de"]) for r in res.values()])
    out = {}
    for name, r in res.items():
        nc = int(r["f_comp"][ok].sum())
        if not ok.any() or nc == 0:
            out[name] = {"frames": int(ok.sum()), "failed": int(r["failed"]),
                         "wall_s": float(r["wall_s"])}
            continue
        out[name] = {
            "frames": int(ok.sum()), "failed": int(r["failed"]),
            "energy_mae_meV_atom": 1e3 * float(np.abs(r["de"][ok]).mean()),
            "energy_rmse_meV_atom": 1e3 * float(np.sqrt((r["de"][ok] ** 2).mean())),
            "energy_mae_with_penalty_meV_atom":
                1e3 * float(np.abs(r["de_pen"][ok]).mean()),
            "force_mae_eV_A": float(r["f_abs"][ok].sum() / nc),
            "force_rmse_eV_A": float(np.sqrt(r["f_sq"][ok].sum() / nc)),
            "force_max_err_eV_A": float(r["f_max"][ok].max()),
            "wall_s": float(r["wall_s"])}
    return out, ok


def check_pred_csv(path, keys, e_pred):
    """Largest difference from the run's own prediction_results_test_set.csv.
    ALIGNN's train.py (6a20fb5) writes that file only when gradwise_weight
    is 0, so force-field runs have none and this returns None."""
    if not path or not os.path.exists(path):
        return None
    pred = {}
    for line in open(path):
        parts = [x.strip() for x in line.split(",")]
        try:
            pred[parts[0]] = float(parts[-1])
        except (ValueError, IndexError):
            continue  # header
    diffs = [abs(e_pred[k] - pred[key]) for k, key in enumerate(keys)
             if key in pred and np.isfinite(e_pred[k])]
    return {"frames": len(diffs),
            "max_abs_diff_eV_atom": float(max(diffs)) if diffs else None}


def best_epoch(history):
    """1-based epoch of the lowest total validation loss, and the count."""
    if not history or not os.path.exists(history):
        return None
    try:  # train.py appends [total, graph, atom, grad, stress, extra]
        h = json.load(open(history))
        tot = [float(x[0] if isinstance(x, (list, tuple)) else x) for x in h]
        return {"epochs": len(tot), "best_epoch": int(np.argmin(tot)) + 1,
                "best_val_loss": min(tot), "last_val_loss": tot[-1]}
    except Exception as err:  # the record is optional; the errors are not
        return {"error": f"{type(err).__name__}: {err}"}


def test(args, device):
    """Errors of every model on the test frames and a training sample."""
    dirs = parse_models(args.models)
    ids = json.load(open(args.ids))
    run_cfg = json.load(open(args.run_config)) if args.run_config else {}
    t0 = time.perf_counter()
    raw = load_id_prop(args.data)
    print(f"{len(raw):,} entries read in {time.perf_counter() - t0:.0f} s",
          flush=True)
    split = check_split(raw, ids, run_cfg, args.id_tag, args.aff_ids)
    print("split:", json.dumps(split), flush=True)
    entries = {e[args.id_tag]: e for e in raw}
    del raw
    ffm.DTYPE = torch.float32
    models, info = load_models(dirs, device)

    sets = {"test": list(ids["id_test"])}
    if args.n_train_sample:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(ids["id_train"]),
                          size=min(args.n_train_sample, len(ids["id_train"])),
                          replace=False)
        sets["train_sample"] = [ids["id_train"][i] for i in sorted(pick)]

    payload = {**header(), "models": info, "split": split,
               "history": best_epoch(args.history), "sets": {}}
    arrays = {}
    for label, keys in sets.items():
        res = evaluate(models, entries, keys, device, args.target_key,
                       args.force_key, label)
        summary, ok = summarize(res)
        payload["sets"][label] = summary
        arrays[f"{label}_ok"] = ok
        for name, r in res.items():
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{label}_{name}_{k}"] = v
        print(f"\n{label}: {summary[next(iter(summary))]['frames']:,} frames "
              f"evaluated by every model")
        print(f"{'model':<10} {'E MAE':>9} {'E RMSE':>9} {'E+pen MAE':>10} "
              f"{'F MAE':>8} {'F RMSE':>8} {'F max':>8} {'failed':>7}")
        print(f"{'':<10} {'meV/atom':>9} {'meV/atom':>9} {'meV/atom':>10} "
              f"{'eV/A':>8} {'eV/A':>8} {'eV/A':>8}")
        for name, s in summary.items():
            if "energy_mae_meV_atom" not in s:
                print(f"{name:<10} no frame evaluated by every model, "
                      f"{s['failed']} failed", flush=True)
                continue
            print(f"{name:<10} {s['energy_mae_meV_atom']:9.2f} "
                  f"{s['energy_rmse_meV_atom']:9.2f} "
                  f"{s['energy_mae_with_penalty_meV_atom']:10.2f} "
                  f"{s['force_mae_eV_A']:8.4f} {s['force_rmse_eV_A']:8.4f} "
                  f"{s['force_max_err_eV_A']:8.2f} {s['failed']:7d}",
                  flush=True)
        if label == "test" and args.csv_model in res:
            payload["pred_csv_check"] = check_pred_csv(
                args.pred_csv, keys, res[args.csv_model]["e_pred"])
    print("history:", json.dumps(payload["history"]))
    print("prediction csv check:", json.dumps(payload.get("pred_csv_check")))
    np.savez(args.out + ".npz", **arrays)
    dump(payload, args.out)


# ---------------------------------------------------------------- NVE

def nve(args, device):
    """Velocity Verlet in float64 from the same start for every model."""
    from ase import units
    from ase.data import atomic_masses
    ffm.DTYPE = torch.float64
    models, info = load_models(parse_models(args.models), device)
    payload = {**header(), "models": info, "steps": args.steps,
               "seed": args.seed, "systems": {}}
    arrays = {}
    for sname in [s.strip() for s in args.systems.split(",") if s.strip()]:
        build, T, dt_fs, cap = SYSTEMS[sname]
        pos0, z, L = build()
        pos0 = np.asarray(pos0, dtype=float)
        z = np.asarray(z)
        L = np.asarray(L, dtype=float)
        n = len(pos0)
        m = atomic_masses[z.astype(int)][:, None]
        rng = np.random.default_rng(args.seed)
        v0 = rng.normal(size=(n, 3)) * np.sqrt(units.kB * T / m)
        v0 -= (m * v0).sum(0) / m.sum()
        v0 *= np.sqrt(1.5 * (n - 1) * units.kB * T / (0.5 * (m * v0 * v0).sum()))
        dt = dt_fs * units.fs
        payload["systems"][sname] = {"atoms": n, "T0_K": T, "dt_fs": dt_fs,
                                     "box_A": L.tolist(),
                                     "max_neighbors": cap or "model",
                                     "models": {}}
        print(f"\n{sname}: {n} atoms, {T:.0f} K, dt {dt_fs} fs, "
              f"{args.steps} steps, neighbor cap {cap or 'of the model'}",
              flush=True)

        def wrapped(x):
            w = np.mod(x, L)
            w[w >= L] = 0.0
            return w

        for name, (model, cfg) in models.items():
            if cap:
                cfg = dict(cfg, max_neighbors=cap)
            x, v = pos0.copy(), v0.copy()
            t0 = time.perf_counter()
            e, f = energy_forces(model, cfg, wrapped(x), z, L, device)
            epot, ekin = [e], [0.5 * float((m * v * v).sum())]
            for step in range(args.steps):
                v += 0.5 * dt * f / m
                x += dt * v
                e, f = energy_forces(model, cfg, wrapped(x), z, L, device)
                v += 0.5 * dt * f / m
                epot.append(e)
                ekin.append(0.5 * float((m * v * v).sum()))
                if not (np.isfinite(e) and np.isfinite(f).all()):
                    print(f"{name}: not finite at step {step + 1}", flush=True)
                    break
                if (step + 1) % 500 == 0:
                    tot = epot[-1] + ekin[-1] - epot[0] - ekin[0]
                    print(f"{name} step {step + 1}: T "
                          f"{2 * ekin[-1] / (3 * (n - 1) * units.kB):.0f} K, "
                          f"E - E0 {1e3 * tot / n:+.4f} meV/atom, "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
            epot, ekin = np.array(epot), np.array(ekin)
            etot = epot + ekin
            step_change = np.abs(np.diff(etot))
            med = float(np.median(step_change)) if step_change.size else np.nan
            s = {"steps_done": int(etot.size - 1),
                 "T_end_K": float(2 * ekin[-1] / (3 * (n - 1) * units.kB)),
                 "drift_meV_atom": 1e3 * float(etot[-1] - etot[0]) / n,
                 "max_dev_meV_atom": 1e3 * float(np.abs(etot - etot[0]).max()) / n,
                 "step_change_median_meV": 1e3 * med,
                 "step_change_max_meV": 1e3 * float(step_change.max())
                 if step_change.size else None,
                 "steps_over_20x_median": int((step_change > 20 * med).sum())
                 if step_change.size else None,
                 "wall_s": time.perf_counter() - t0}
            payload["systems"][sname]["models"][name] = s
            arrays[f"{sname}_{name}_epot"] = epot
            arrays[f"{sname}_{name}_ekin"] = ekin
            print(f"{name}: {json.dumps(s)}", flush=True)
    np.savez(args.out + ".npz", **arrays)
    dump(payload, args.out)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("test", help="errors on the test frames")
    t.add_argument("--models", nargs="+", required=True, help="name=dir ...")
    t.add_argument("--data", required=True, help="folder with id_prop.json")
    t.add_argument("--ids", required=True, help="the run's ids_train_val_test.json")
    t.add_argument("--run_config", help="the run's config.json (seed, order)")
    t.add_argument("--aff_ids", help="AFF-r2SCAN's ids_train_val_test.json")
    t.add_argument("--pred_csv", help="the run's prediction_results_test_set.csv")
    t.add_argument("--csv_model", default="smooth",
                   help="the model the prediction csv belongs to")
    t.add_argument("--history", help="the run's history_val.json")
    t.add_argument("--n_train_sample", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--id_tag", default="jid")
    t.add_argument("--target_key", default="energy_per_atom")
    t.add_argument("--force_key", default="forces")
    t.add_argument("--out", required=True)
    v = sub.add_parser("nve", help="energy conservation")
    v.add_argument("--models", nargs="+", required=True, help="name=dir ...")
    v.add_argument("--systems", default=",".join(SYSTEMS))
    v.add_argument("--steps", type=int, default=2000)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (test if args.mode == "test" else nve)(args, device)


if __name__ == "__main__":
    main()
