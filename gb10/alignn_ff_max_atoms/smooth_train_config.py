"""Training config for AFF-r2SCAN with smooth_messages on MatPES r2SCAN: the
training settings of the MatPES r2SCAN base run (smooth_train_base.json) with
the model block of AFF-r2SCAN. Prints every top-level setting where the
AFF-r2SCAN config differs from the base, because the settings AFF-r2SCAN was
trained with are not otherwise recorded here.

    python smooth_train_config.py --base smooth_train_base.json \
        --aff models/matpes_r2scan_paper/config.json --out C.json --epochs 2 \
        --n_train 4000 --n_val 400 --n_test 400
"""
import argparse
import json


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True)
    p.add_argument("--aff", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--output_dir", default="smooth_run")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_val", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--keep_data_order", choices=["true", "false"], default=None)
    p.add_argument("--read_existing", action="store_true",
                   help="reuse complete graph caches")
    p.add_argument("--resume", action="store_true",
                   help="continue from current_state.pt when it exists")
    args = p.parse_args()
    base = json.load(open(args.base))
    aff = json.load(open(args.aff))

    print("AFF-r2SCAN config against the training base:")
    for k in sorted(aff):
        if k != "model" and aff[k] != base.get(k):
            print(f"  {k}: AFF-r2SCAN {aff[k]!r}, base {base.get(k)!r}")

    cfg = dict(base)
    for k in ("cutoff", "max_neighbors", "three_body_cutoff", "atom_features"):
        if k in aff:
            cfg[k] = aff[k]
    cfg["model"] = dict(aff["model"])
    cfg["model"].update(smooth_messages=True,
                        two_body_cutoff=float(cfg["cutoff"]))
    cfg.update(epochs=args.epochs, lr_total_epochs=args.epochs,
               batch_size=args.batch_size, output_dir=args.output_dir,
               distributed=False, data_parallel=False,
               read_existing=args.read_existing,
               resume_checkpoint=args.resume)
    for k in ("n_train", "n_val", "n_test", "warmup_steps", "learning_rate"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.keep_data_order:
        cfg["keep_data_order"] = args.keep_data_order == "true"
    json.dump(cfg, open(args.out, "w"), indent=1)
    print("training:", json.dumps({k: cfg.get(k) for k in (
        "epochs", "batch_size", "learning_rate", "keep_data_order", "n_train",
        "n_val", "n_test", "warmup_steps", "read_existing",
        "resume_checkpoint")}))
    print("wrote", args.out)
    print("model:", json.dumps({k: cfg["model"].get(k) for k in (
        "alignn_layers", "gcn_layers", "hidden_features", "graphwise_weight",
        "gradwise_weight", "stresswise_weight", "smooth_messages",
        "two_body_onset", "three_body_onset")}))


if __name__ == "__main__":
    main()
