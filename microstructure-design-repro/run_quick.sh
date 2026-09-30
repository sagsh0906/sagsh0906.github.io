#!/bin/bash
# End-to-end CPU reproduction (quick preset). Total ~3 h on 4 CPU cores.
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
python3 scripts/train_ddpm.py --epochs 12 --lr 5e-4
python3 scripts/analyze_fidelity.py
python3 scripts/inverse_design_synthetic.py
