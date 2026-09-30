#!/bin/bash
# Paper settings (Methods): 800x1000 maps, 100 px patches resized to 128, 72 000 pairs per epoch,
# Adam + cosine annealing 1e-4 -> 0 over 100 epochs, batch 50. Needs a GPU with a
# CUDA build of PyTorch; the training scripts pick CUDA automatically (override with --device).
set -e
cd "$(dirname "$0")"
python3 scripts/make_synthetic_dataset.py --preset paper            # or put your own 25 EBSD maps in this format
python3 scripts/train_vae.py --data work/paper/dataset.npz --loss full --img-size 128 --epochs 100 --samples-per-epoch 0
python3 scripts/train_vae.py --data work/paper/dataset.npz --loss pix  --img-size 128 --epochs 100 --samples-per-epoch 0
python3 scripts/train_ddpm.py --data work/paper/dataset.npz --vae work/paper/vae_full/vae.pt --epochs 100 --samples-per-epoch 0
python3 scripts/analyze_fidelity.py --work work/paper --out results/paper
python3 scripts/inverse_design_synthetic.py --work work/paper --out results/paper
