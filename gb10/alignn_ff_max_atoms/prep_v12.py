"""Write the v12.2.2024_dft_3d_307k checkpoint (the model of the B200 runs
and their GB10/H200 reruns) as a model folder that ff_max_atoms.load_model,
and so ff_any.py, can read, with the B200 settings: 5 A cutoff, 12 neighbors.

The model config is filtered to ALIGNNAtomWisePureConfig's fields exactly as
gb10/alignn_ff_b200settings/scale_pure_b200settings.py does. energy_mult_natoms
is set to True because load_model insists on it; it changes only forward()'s
readout, not the per-atom energies that ff_any sums. The weights are loaded
into this alignn's model class and saved under its own key names, so the
folder loads without missing or unexpected keys; the script stops if any
checkpoint weight would be missing.

    python prep_v12.py --src <folder with config.json, best_model.pt> --out /data/jlee859/ffmax/models/v12_dft_3d_307k_pure
"""
import argparse
import json
import os

import torch

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    m = dict(cfg["model"])
    original = {k: m.get(k) for k in ("name", "energy_mult_natoms",
                                      "calculate_gradient", "alignn_layers",
                                      "gcn_layers", "hidden_features")}
    m.update(name="alignn_atomwise_pure", calculate_gradient=False,
             stresswise_weight=0, graphwise_weight=1.0, atomwise_weight=0,
             gradwise_weight=0)
    F = ALIGNNAtomWisePureConfig
    keep = set(F.model_fields if hasattr(F, "model_fields") else F.__fields__)
    dropped = sorted(k for k in m if k not in keep)
    m = {k: v for k, v in m.items() if k in keep}
    m["energy_mult_natoms"] = True
    conf = F(**m)
    net = ALIGNNAtomWisePure(conf)
    # every field, defaults included, so header() finds hidden_features etc.
    m = conf.model_dump() if hasattr(conf, "model_dump") else conf.dict()
    sd = torch.load(os.path.join(args.src, "best_model.pt"), map_location="cpu",
                    weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    res = net.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if k != "_species_table"]
    print("source", args.src)
    print("original model settings", original)
    print("config keys dropped (not in the pure config):", dropped)
    print("missing weights:", missing)
    print("unexpected weights:", len(res.unexpected_keys),
          res.unexpected_keys[:10])
    if missing:
        raise SystemExit("checkpoint weights missing, not writing the folder")
    os.makedirs(args.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.out, "best_model.pt"))
    json.dump({"model": m, "cutoff": 5.0, "max_neighbors": 12,
               "atom_features": cfg.get("atom_features", "cgcnn"),
               "source": os.path.abspath(args.src),
               "note": "B200 settings: 5 A cutoff, 12 neighbors; "
                       "energy_mult_natoms forced True (readout only)",
               "original": original},
              open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print("wrote", args.out, "| layers", m.get("alignn_layers"), "+",
          m.get("gcn_layers"), "| hidden", m.get("hidden_features"))


if __name__ == "__main__":
    main()
