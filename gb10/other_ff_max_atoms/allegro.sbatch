#!/bin/bash
#SBATCH --job-name=ffoalleg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/data/jlee859/ffother/logs/allegro_%j.out

# Allegro on one GB10, with nequip 0.19.1 and nequip-allegro 0.8.3:
#   install  into its own venv on top of the alignn env (which is not changed)
#   train    an Allegro model for Si on the MatPES r2SCAN Si frames
#            (data.sbatch) with the size settings of the SC23 biomolecular
#            model: 2 layers, l_max 2, 64 tensor features, 4.0 A cutoff,
#            latent width 1024, float32. Then its errors on the test frames
#   compile  nequip-compile (AOT Inductor, ASE target), on this GB10
#   sweep    sweep.py through the NequIP ASE calculator (matscipy neighbor
#            list, linear in the number of atoms). If the compile failed, the
#            checkpoint is evaluated uncompiled and the result says so
# This is the Python path, not LAMMPS: the Allegro pair style needs MPI,
# which the nodes do not have, and has open crash reports on GPUs.
# A stage whose result exists is skipped, so a resubmitted job continues.
#   STAGES="compile sweep" sbatch allegro.sbatch
#   EPOCHS=200 sbatch allegro.sbatch        (default 100)
#   RETRAIN=1 sbatch allegro.sbatch         train again from scratch
#   FRESH=1 sbatch allegro.sbatch           sweep again from scratch
#   EAGER=1 sbatch allegro.sbatch           evaluate the checkpoint uncompiled
#   cd /data/jlee859/ffother && sbatch --exclude=atomgptlab03 allegro.sbatch
set -uo pipefail
B=/data/jlee859/ffother
PY=/data/jlee859/miniforge3-aarch64/envs/alignn/bin/python
D=$B/data/si
V=$B/venv_allegro
T=$B/allegro/train
R=$B/results/allegro
J=$SLURM_JOB_ID
STAGES=${STAGES:-"install train compile sweep"}
EPOCHS=${EPOCHS:-100}
export PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p $B/allegro $R $B/logs && cd $B
echo "start $(date) | $(hostname) | stages: $STAGES"

# ---------------------------------------------------------------- install
if [[ " $STAGES " == *" install "* ]]; then
    if [ -x $V/bin/nequip-train ]; then
        echo "install: already there"
    else
        [ -x $V/bin/python ] || $PY -m venv --system-site-packages $V \
            || { echo "install: could not make the venv"; exit 5; }
        echo "torch==$($PY -c 'import torch; print(torch.__version__)')" > $B/allegro/constraints.txt
        cat $B/allegro/constraints.txt
        $V/bin/python -m pip install -c $B/allegro/constraints.txt \
            nequip==0.19.1 nequip-allegro==0.8.3 > $B/logs/allegro_pip_$J.log 2>&1 \
            || { echo "install: pip FAILED:"; tail -n 25 $B/logs/allegro_pip_$J.log; exit 5; }
        grep -E '^Successfully installed' $B/logs/allegro_pip_$J.log | fold -w 200
    fi
fi
[ -x $V/bin/nequip-train ] || { echo "no $V/bin/nequip-train (run the install stage)"; exit 5; }
$V/bin/python - <<'EOF' || { echo "the venv does not import"; exit 5; }
import importlib.metadata as md
import torch
print("torch", torch.__version__, "from", torch.__file__)
print("cuda available", torch.cuda.is_available())
for name in ("nequip", "nequip-allegro", "e3nn", "lightning", "matscipy", "ase", "numpy"):
    print(name, md.version(name))
assert torch.__version__.startswith("2.13.0"), "pip replaced torch"
assert torch.cuda.is_available()
EOF

# ------------------------------------------------------------------ train
ckpt() { find $T/run -name 'best*.ckpt' 2>/dev/null | head -n 1; }
if [[ " $STAGES " == *" train "* ]]; then
    [ -s $D/train.extxyz ] || { echo "no $D/train.extxyz (run data.sbatch first)"; exit 6; }
    [ "${RETRAIN:-0}" = 1 ] && rm -rf $T
    mkdir -p $T && cd $T
    [ -s $D/val.extxyz ] || { echo "no $D/val.extxyz (run data.sbatch first)"; exit 6; }
    # train_done holds the number of epochs of the finished training, so a
    # different EPOCHS trains again
    if [ -n "$(ckpt)" ] && [ "$(cat $T/train_done 2>/dev/null)" = "$EPOCHS" ]; then
        echo "train: already there, $(ckpt)"
    else
        # nequip splits this file itself (90/10); the test frames of
        # data.sbatch are not in it
        cat $D/train.extxyz $D/val.extxyz > trainval.extxyz
        echo "train: $(grep -c '^Lattice=' trainval.extxyz) frames for training and validation, $EPOCHS epochs"
        cat > si.yaml <<EOF
run: [train]

cutoff_radius: 4.0
model_type_names: [Si]
num_scalar_features: 1024
monitored_metric: val0_epoch/weighted_sum

data:
  _target_: nequip.data.datamodule.ASEDataModule
  seed: 123
  split_dataset:
    file_path: $T/trainval.extxyz
    train: 0.9
    val: 0.1
  transforms:
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      model_type_names: \${model_type_names}
    - _target_: nequip.data.transforms.NeighborListTransform
      r_max: \${cutoff_radius}
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 5
    shuffle: true
  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 10
  test_dataloader: \${data.val_dataloader}
  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: \${model_type_names}

trainer:
  _target_: lightning.Trainer
  accelerator: gpu
  max_epochs: $EPOCHS
  log_every_n_steps: 5
  enable_progress_bar: false
  logger:
    _target_: lightning.pytorch.loggers.CSVLogger
    save_dir: \${hydra:runtime.output_dir}
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: \${monitored_metric}
      dirpath: \${hydra:runtime.output_dir}
      filename: best
      save_last: true

training_module:
  _target_: nequip.train.EMALightningModule
  loss:
    _target_: nequip.train.EnergyForceLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 1.0
  val_metrics:
    _target_: nequip.train.EnergyForceMetrics
    coeffs:
      per_atom_energy_mae: 1.0
      forces_mae: 1.0
  test_metrics: \${training_module.val_metrics}
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  model:
    _target_: allegro.model.AllegroModel
    compile_mode: eager
    seed: 456
    model_dtype: float32
    type_names: \${model_type_names}
    r_max: \${cutoff_radius}
    radial_chemical_embed:
      _target_: allegro.nn.TwoBodyBesselScalarEmbed
      num_bessels: 8
      bessel_trainable: true
      polynomial_cutoff_p: 6
    radial_chemical_embed_dim: 128
    scalar_embed_mlp_hidden_layers_depth: 2
    scalar_embed_mlp_hidden_layers_width: 512
    scalar_embed_mlp_nonlinearity: silu
    l_max: 2
    num_layers: 2
    num_scalar_features: \${num_scalar_features}
    num_tensor_features: 64
    allegro_mlp_hidden_layers_depth: 3
    allegro_mlp_hidden_layers_width: \${num_scalar_features}
    allegro_mlp_nonlinearity: silu
    parity: true
    tp_path_channel_coupling: true
    readout_mlp_hidden_layers_depth: 1
    readout_mlp_hidden_layers_width: 128
    readout_mlp_nonlinearity: silu
    avg_num_neighbors: \${training_data_stats:per_type_num_neighbors_mean}
    per_type_energy_shifts: \${training_data_stats:per_atom_energy_mean}
    per_type_energy_scales: \${training_data_stats:forces_rms}
    per_type_energy_scales_trainable: false
    per_type_energy_shifts_trainable: false
EOF
        rm -rf $T/run $T/train_done
        t0=$SECONDS
        $V/bin/nequip-train -cp $T -cn si.yaml hydra.run.dir=$T/run > train.log 2>&1
        rc=$?
        echo "train: nequip-train exit $rc after $((SECONDS - t0)) s"
        if [ $rc -ne 0 ] || [ -z "$(ckpt)" ]; then
            echo "train: FAILED, last lines of $T/train.log:"
            tail -n 40 train.log | cut -c1-300
            find $T -name '*.ckpt' | head
            exit 7
        fi
        echo $EPOCHS > $T/train_done
    fi
    CK=$(ckpt)
    echo "train: checkpoint $CK"
    M=$(find $T/run -name metrics.csv | head -n 1)
    if [ -n "$M" ]; then
        echo "train: $M, header and last two rows:"
        head -n 1 $M | cut -c1-400; tail -n 2 $M | cut -c1-400
        cp $M $R/train_metrics.csv
    fi
    grep -i -E 'number of.*param|trainable' train.log | head -n 5
    cp si.yaml $R/
    cd $B
fi
if [[ " $STAGES " != *" train "* ]] && [[ " $STAGES " != *" compile "* ]] && [[ " $STAGES " != *" sweep "* ]]; then
    echo "end rc 0 $(date)"
    exit 0
fi
CK=$(ckpt)
[ -n "$CK" ] || { echo "no checkpoint under $T/run (train first)"; exit 7; }

# ---------------------------------------------------------------- compile
MODEL=$T/si_ase.nequip.pt2
if [[ " $STAGES " == *" compile "* ]]; then
    if [ -s $MODEL ] && [ $MODEL -nt $CK ]; then
        echo "compile: already there"
    else
        rm -f $MODEL
        t0=$SECONDS
        $V/bin/nequip-compile $CK $MODEL --device cuda --mode aotinductor --target ase \
            > $T/compile.log 2>&1
        rc=$?
        echo "compile: nequip-compile exit $rc after $((SECONDS - t0)) s"
        if [ $rc -ne 0 ] || [ ! -s $MODEL ]; then
            echo "compile: FAILED, last lines of $T/compile.log:"
            tail -n 30 $T/compile.log | cut -c1-300
            rm -f $MODEL
        fi
    fi
fi
if [ -s $MODEL ] && [ "${EAGER:-0}" != 1 ]; then
    HOW=(--model $MODEL)
    NOTE="compiled with AOT Inductor"
else
    HOW=(--model $CK --eager)
    NOTE="checkpoint evaluated uncompiled (no compiled model)"
fi
echo "evaluating: $NOTE"

if [[ " $STAGES " == *" train "* ]] || [[ " $STAGES " == *" compile "* ]]; then
    echo "test frames, Allegro against MatPES r2SCAN:"
    $V/bin/python $B/py_eval.py errors --kind allegro "${HOW[@]}" \
        --xyz $D/test.extxyz --out $R/test_errors_$J.json \
        || echo "errors: FAILED"
fi

# ------------------------------------------------------------------ sweep
rc=0
if [[ " $STAGES " == *" sweep "* ]]; then
    FRESH_FLAG=()
    [ "${FRESH:-0}" = 1 ] && FRESH_FLAG=(--fresh)
    $PY sweep.py python --exe $V/bin/python \
        --exe_args="$B/py_eval.py size --kind allegro ${HOW[*]}" \
        --steps 5 --start 10 --work $B/work/allegro --out $R/sweep "${FRESH_FLAG[@]}" \
        --note "nequip 0.19.1, nequip-allegro 0.8.3, ASE calculator, $NOTE, job $J" || rc=1
fi
echo "end rc $rc $(date)"
exit $rc
