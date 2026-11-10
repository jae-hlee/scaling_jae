"""A model folder that evaluates an existing checkpoint with smooth_messages
switched on (the same weights, so no retraining), for testing that the patch
removes the force jumps. The energies are not those of a trained model.

    python prep_smooth.py --src models/matpes_r2scan_paper --out models/aff_smooth_test
"""
import argparse
import json
import os


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    cfg["model"].update(smooth_messages=True,
                        two_body_cutoff=float(cfg["cutoff"]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    link = os.path.join(args.out, "best_model.pt")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(args.src, "best_model.pt")), link)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
