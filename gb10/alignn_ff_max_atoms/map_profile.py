"""How far the energy disturbance reaches into each layer of a gate-stack map
from energy_maps.py: for each face of each layer, the depth of the deepest
1 A bin whose mean |energy - layer median| exceeds a threshold. Bins are
searched from the face to the middle of the layer.

    python map_profile.py results/matpes_r2scan_paper/maps/stack_1e8.npz
"""
import json
import sys
import warnings

import numpy as np

THRESHOLDS_EV = (0.001, 0.01, 0.1)


def main():
    """Print the table."""
    d = np.load(sys.argv[1])
    info = json.loads(str(d["info"]))
    dz = info["pixel_A"]
    with warnings.catch_warnings():  # planes between atoms have no data
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prof = np.nanmean(np.abs(d["image"]), axis=0)
    z = (np.arange(len(prof)) + 0.5) * dz
    head = "  ".join(f">{t * 1000:g} meV" for t in THRESHOLDS_EV)
    print(f"depth (A) of the deepest bin above each threshold\n"
          f"layer  face    z (A)   half-thickness  {head}")
    for lay in info["layers"]:
        lo, hi = lay["z_bottom_A"], lay["z_top_A"]
        mid = 0.5 * (lo + hi)
        faces = (("bottom", lo, (z >= lo) & (z < mid)),
                 ("top", hi, (z >= mid) & (z <= hi)))
        for face, zf, inside in faces:
            depths = []
            for t in THRESHOLDS_EV:
                hit = inside & (prof > t)
                depths.append(float(np.abs(z[hit] - zf).max()) if hit.any()
                              else 0.0)
            cols = "  ".join(f"{x:8.1f}" for x in depths)
            print(f"{lay['layer']:5s}  {face:6s}  {zf:7.1f}  "
                  f"{0.5 * (hi - lo):14.1f}  {cols}")


if __name__ == "__main__":
    main()
