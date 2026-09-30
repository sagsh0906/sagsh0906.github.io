"""Train the physics-aware VAE (Eq. 11) or the pixel-loss-only baseline on the augmented patch data set.

Paper settings: Adam, cosine annealing from 1e-4 to 0 over 100 epochs, batch 50, 72 000 pairs, 128 px inputs.
  python scripts/train_vae.py --data work/paper/dataset.npz --img-size 128 --epochs 100 --samples-per-epoch 0
CPU quick preset (default): 64 px patches, 16 epochs of 9 600 randomly drawn pairs each.
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
from msdesign.data import PatchDataset, crop_patches  # noqa: E402
from msdesign.losses import physics_loss  # noqa: E402
from msdesign.models import PhysicsVAE  # noqa: E402


def load_patches(npz_path):
    d = np.load(npz_path)
    patch = int(d['patch'])
    tr = np.concatenate([crop_patches(d['images'][i], patch) for i in d['train_idx']])
    te = np.concatenate([crop_patches(d['images'][i], patch) for i in d['test_idx']])
    return tr, te, patch


@torch.no_grad()
def evaluate(model, loader, kl_weight, device='cpu'):
    model.eval()
    acc, n = {}, 0
    for x_in, x in loader:
        x_in, x = x_in.to(device), x.to(device)
        _, parts, _ = model.loss(x_in, x, terms=('pix', 'edge', 'ssim'), kl_weight=kl_weight)
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v.item() * len(x)
        n += len(x)
    model.train()
    return {k: v / n for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='work/quick/dataset.npz')
    ap.add_argument('--loss', choices=['full', 'pix'], default='full')
    ap.add_argument('--img-size', type=int, default=None, help='network input size (default: patch size)')
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--samples-per-epoch', type=int, default=9600, help='0 = all 72 000-style pairs')
    ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--kl-weight', type=float, default=1.0)
    ap.add_argument('--kl-warmup-epochs', type=float, default=4.0,
                    help='linearly ramp the KL weight from 0 over this many epochs (avoids posterior collapse)')
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--init', default=None, help='warm-start from a VAE checkpoint (e.g. the pixel-loss model)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(os.path.dirname(args.data), f'vae_{args.loss}')
    os.makedirs(out, exist_ok=True)

    tr, te, patch = load_patches(args.data)
    img_size = args.img_size or patch
    ds_tr = PatchDataset(tr, img_size, seed=args.seed)
    ds_te = PatchDataset(te, img_size, seed=args.seed + 1)
    rng = np.random.default_rng(args.seed)
    te_idx = rng.choice(len(ds_te), min(args.n_test, len(ds_te)), replace=False)
    te_loader = DataLoader(Subset(ds_te, te_idx), batch_size=args.batch)
    tr_eval_idx = rng.choice(len(ds_tr), min(args.n_test, len(ds_tr)), replace=False)
    tr_eval_loader = DataLoader(Subset(ds_tr, tr_eval_idx), batch_size=args.batch)
    print(f'train pairs {len(ds_tr)}, test pairs {len(ds_te)}, patch {patch}px -> input {img_size}px')

    model = PhysicsVAE(img_size=img_size)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu')['state_dict'])
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    terms = ('pix', 'edge', 'ssim') if args.loss == 'full' else ('pix',)
    history = []
    step, steps_per_epoch = 0, (args.samples_per_epoch or len(ds_tr)) // args.batch
    warmup_steps = max(1, int(args.kl_warmup_epochs * steps_per_epoch))
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        n = args.samples_per_epoch or len(ds_tr)
        idx = rng.choice(len(ds_tr), n, replace=n > len(ds_tr))
        loader = DataLoader(Subset(ds_tr, idx), batch_size=args.batch, shuffle=True, drop_last=True)
        run = 0.0
        for x_in, x in loader:
            x_in, x = x_in.to(args.device), x.to(args.device)
            beta = args.kl_weight * min(1.0, step / warmup_steps)
            loss, parts, _ = model.loss(x_in, x, terms=terms, kl_weight=beta)
            step += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        rec = dict(epoch=epoch, lr=opt.param_groups[0]['lr'], kl_beta=beta, train_objective=run / len(loader),
                   train=evaluate(model, tr_eval_loader, args.kl_weight, args.device),
                   test=evaluate(model, te_loader, args.kl_weight, args.device), seconds=time.time() - t0)
        history.append(rec)
        print(f"[{args.loss}] epoch {epoch:3d}  train pix {rec['train']['pix']:.4f} edge {rec['train']['edge']:.4f} "
              f"ssim {rec['train']['ssim']:.4f} kl {rec['train']['kl']:.4f} | test pix {rec['test']['pix']:.4f} "
              f"edge {rec['test']['edge']:.4f} ssim {rec['test']['ssim']:.4f}  ({rec['seconds']:.0f}s)", flush=True)
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, img_size=img_size, patch=patch, args=vars(args)),
                   os.path.join(out, 'vae.pt'))
        with open(os.path.join(out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)


if __name__ == '__main__':
    main()
