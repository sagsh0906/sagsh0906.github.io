"""Train the conditional diffusion refiner (Eqs. 12-15) on top of a trained, frozen VAE.

The condition x_V is the VAE reconstruction (with a sampled latent) of the clean target patch, i.e. the same kind of
input the refiner receives at generation time when latents are sampled from the optimised descriptors.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msdesign.data import PatchDataset  # noqa: E402
from msdesign.models import Diffusion, PhysicsVAE  # noqa: E402
from train_vae import load_patches  # noqa: E402


def vae_condition(vae, x):
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        return vae.decode(vae.reparameterize(mu, logvar))


@torch.no_grad()
def evaluate(model, vae, loader, device='cpu'):
    model.eval()
    acc, n = {}, 0
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)                 # same noise draws for every evaluation
    for _, x in loader:
        x = x.to(device)
        _, parts = model.loss(x, vae_condition(vae, x))
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v.item() * len(x)
        n += len(x)
    torch.random.set_rng_state(g)
    model.train()
    return {k: v / n for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='work/quick/dataset.npz')
    ap.add_argument('--vae', default='work/quick/vae_full/vae.pt')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--samples-per-epoch', type=int, default=6400)
    ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n-test', type=int, default=500)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(os.path.dirname(args.data), 'ddpm')
    os.makedirs(out, exist_ok=True)

    ck = torch.load(args.vae, map_location='cpu')
    vae = PhysicsVAE(img_size=ck['img_size'])
    vae.load_state_dict(ck['state_dict'])
    vae.to(args.device).eval()
    tr, te, patch = load_patches(args.data)
    # clean targets with the six lossless geometric variants (restoration pairs are not needed here)
    ds_tr = PatchDataset(tr, ck['img_size'], degradations=False, seed=args.seed)
    ds_te = PatchDataset(te, ck['img_size'], degradations=False, seed=args.seed + 1)
    rng = np.random.default_rng(args.seed)
    te_loader = DataLoader(Subset(ds_te, rng.choice(len(ds_te), min(args.n_test, len(ds_te)), replace=False)),
                           batch_size=args.batch)

    model = Diffusion().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        n = args.samples_per_epoch or len(ds_tr)
        idx = rng.choice(len(ds_tr), n, replace=n > len(ds_tr))
        loader = DataLoader(Subset(ds_tr, idx), batch_size=args.batch, shuffle=True, drop_last=True)
        run = {}
        for _, x in loader:
            x = x.to(args.device)
            loss, parts = model.loss(x, vae_condition(vae, x))
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v.item() / len(loader)
        sched.step()
        rec = dict(epoch=epoch, train=run, test=evaluate(model, vae, te_loader, args.device), seconds=time.time() - t0)
        history.append(rec)
        print(f"[ddpm] epoch {epoch:3d} train " + ' '.join(f'{k} {v:.4f}' for k, v in run.items()) +
              ' | test ' + ' '.join(f'{k} {v:.4f}' for k, v in rec['test'].items()) + f" ({rec['seconds']:.0f}s)",
              flush=True)
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, img_size=ck['img_size'], args=vars(args)),
                   os.path.join(out, 'ddpm.pt'))
        with open(os.path.join(out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)


if __name__ == '__main__':
    main()
