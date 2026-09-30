#!/bin/bash
# End-to-end CPU reproduction (quick preset). Total ~4 h on 4 CPU cores.
# The quick preset takes ~30x fewer optimiser steps than the paper, hence the larger learning rates (1e-3 collapses the physics-aware VAE; 3e-4 is the largest stable value).
set -e
cd "$(dirname "$0")"
# Part A: inverse design with the authors' released latents / properties / surrogates (~6 min)
python3 scripts/inverse_design_official.py
# Part B: synthetic EBSD data -> physics-aware VAE (+ pixel-loss baseline) -> DDPM -> analyses
python3 scripts/make_synthetic_dataset.py --preset quick
python3 scripts/train_vae.py --loss full --epochs 20 --lr 3e-4 --threads 2 &
python3 scripts/train_vae.py --loss pix  --epochs 20 --lr 3e-4 --threads 2 &
wait
# CPU-budget curriculum: the physics-aware loss from scratch resolves boundaries but not orientations within
# ~4k steps, so the model used downstream is warm-started from the pixel-loss VAE and fine-tuned with the full loss
python3 scripts/train_vae.py --loss full --init work/quick/vae_pix/vae.pt --epochs 10 --lr 3e-4 --kl-warmup-epochs 0.01 \
    --out work/quick/vae_curriculum
python3 scripts/train_ddpm.py --vae work/quick/vae_curriculum/vae.pt --epochs 12 --lr 5e-4
python3 scripts/analyze_fidelity.py
python3 scripts/inverse_design_synthetic.py
