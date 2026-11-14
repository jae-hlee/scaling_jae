"""ff_max_atoms.py with the per-atom energies of smooth_messages models, for
the stage-16 sweeps and chunked Si runs with the retrained model.

ff_max_atoms.py stays byte-identical (stages 4 and 5 check its md5). This file
imports it as a module, replaces its atom_energies with
smooth_eval.atom_energies and calls its main(), so every mode and option is
the same. (Running ff_max_atoms.py itself through smooth_eval.py would not
work: it would run as a second copy, __main__, whose atom_energies is not
replaced.) The sweep paths ef and ckpt call the model's own forward_tensors and
the path ase the ALIGNN calculator; both use the patched ALIGNN directly.

    PYTHONPATH=/data/jlee859/ffmax/alignn_smooth \
        python smooth_ffm.py sweep --model_dir M --path e --out O
"""
import ff_max_atoms as ffm
import smooth_eval

smooth_eval.install()

if __name__ == "__main__":
    ffm.main()
