"""Reconstruction fidelity and latent-space analysis (paper Fig. 2, Fig. 3, Supplementary Figs. 4, 6, 8).

Compares the physics-aware model (L_pix + L_edge + L_SSIM, VAE + DDPM refiner) with the pixel-loss-only VAE:
  fig2_losses.png           training / test loss curves of both models              (Fig. 2a, b)
  fig2c_reconstructions.png originals vs reconstructions                            (Fig. 2c)
  fig3_single_patch.png     six-feature distributions for one random test patch     (Fig. 3a-h)
  fig3_aggregate_<m>.png    mean features, original vs reconstructed, all patches   (Fig. 3i-n, Supp. Fig. 6)
  misorientation.png        neighbour-grain misorientation distributions            (Supp. Fig. 4)
  latent_space.png          PCA of the latent space, interpolation / extrapolation  (Supp. Fig. 8)
  fidelity_metrics.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msdesign import plotting as P  # noqa: E402
from msdesign.data import crop_patches  # noqa: E402
from msdesign.features import (FEATURE_NAMES, FEATURE_UNITS, grain_table, mean_features,  # noqa: E402
                               neighbour_misorientations)
from msdesign.losses import edge_loss, pixel_loss, ssim_loss  # noqa: E402
from msdesign.pipeline import decode, encode, load_ddpm, load_vae, quantize, reconstruct  # noqa: E402

LABELS = {'area': 'Grain area', 'major_axis': 'Major axis', 'minor_axis': 'Minor axis',
          'euler1': 'Euler1', 'euler2': 'Euler2', 'euler3': 'Euler3'}


def image_losses(a, b):
    import torch
    ta = torch.from_numpy(a).permute(0, 3, 1, 2)
    tb = torch.from_numpy(b).permute(0, 3, 1, 2)
    return dict(pix=pixel_loss(ta, tb).item(), edge=edge_loss(ta, tb).item(), ssim=ssim_loss(ta, tb).item())


def feature_matrix(images, pixel_um):
    return np.array([mean_features(im, pixel_um=pixel_um, method=SEG) for im in images])


def aggregate(F, n_maps=25, n_patches=80):
    Fm = F.reshape(n_maps, n_patches, -1)
    return np.nanmean(Fm, 1), np.nanstd(Fm, 1)


def agg_metrics(Fo, Fr, idx):
    mo, _ = aggregate(Fo)
    mr, _ = aggregate(Fr)
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        a, b = mo[idx, j], mr[idx, j]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        out[name] = dict(mean_abs_rel_err=float(np.mean(np.abs(b - a) / np.abs(a))),
                         pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float('nan'),
                         n_valid=int(ok.sum()))
    return out


SEG = 'watershed'   # common grain segmentation for originals and reconstructions (see features.segment_grains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work/quick')
    ap.add_argument('--out', default='results/synthetic')
    ap.add_argument('--no-ddpm', action='store_true')
    ap.add_argument('--seg', choices=['watershed', 'dark'], default='watershed')
    args = ap.parse_args()
    global SEG
    SEG = args.seg
    os.makedirs(args.out, exist_ok=True)
    P.setup()
    plt = P.plt
    d = np.load(os.path.join(args.work, 'dataset.npz'))
    pixel_um, patch = float(d['pixel_um']), int(d['patch'])
    train_idx, test_idx = d['train_idx'], d['test_idx']
    patches = np.concatenate([crop_patches(im, patch) for im in d['images']])          # (2000, p, p, 3), map order
    map_of = np.repeat(np.arange(25), 80)
    is_test = np.isin(map_of, test_idx)

    vae_f, _ = load_vae(os.path.join(args.work, 'vae_full', 'vae.pt'))
    vae_p, _ = load_vae(os.path.join(args.work, 'vae_pix', 'vae.pt'))
    curr_path = os.path.join(args.work, 'vae_curriculum', 'vae.pt')
    vae_c = load_vae(curr_path)[0] if os.path.exists(curr_path) else None
    vae_main = vae_c if vae_c is not None else vae_f       # the VAE the DDPM refiner was trained on
    ddpm_path = os.path.join(args.work, 'ddpm', 'ddpm.pt')
    ddpm = None if args.no_ddpm or not os.path.exists(ddpm_path) else load_ddpm(ddpm_path)

    # reconstructions are cached; the VAE part is deterministic given the seed, so the DDPM refinement (computed
    # later, once the refiner is trained) is conditioned on exactly the cached VAE outputs
    cache_vae = os.path.join(args.work, 'reconstructions_vae.npz')
    cache_ddpm = os.path.join(args.work, 'reconstructions_ddpm.npz')
    cached = dict(np.load(cache_vae)) if os.path.exists(cache_vae) else {}
    for key, vae in (('full', vae_f), ('pix', vae_p), ('curr', vae_c)):
        if vae is not None and key not in cached:
            cached[key] = quantize(reconstruct(vae, patches, None, seed=0)[0])
            np.savez_compressed(cache_vae, **cached)
    rec_fd = None
    if ddpm is not None:
        if os.path.exists(cache_ddpm):
            rec_fd = np.load(cache_ddpm)['full_ddpm']
        else:
            _, rec_fd = reconstruct(vae_main, patches, ddpm, seed=0)
            rec_fd = quantize(rec_fd)
            np.savez_compressed(cache_ddpm, full_ddpm=rec_fd)
    models = {'pixel-loss VAE': cached['pix'], 'physics-aware VAE (from scratch)': cached['full']}
    if 'curr' in cached:
        models['physics-aware VAE (pixel warm start)'] = cached['curr']
    if rec_fd is not None:
        models['physics-aware VAE + DDPM'] = rec_fd
    orig = patches.astype(np.float32) / 255.0
    metrics = {'image_losses_test': {m: image_losses(r[is_test], orig[is_test]) for m, r in models.items()}}
    print(json.dumps(metrics, indent=1))

    # ---- Fig. 2a, b: loss curves (all three terms are logged for every model, whatever it was trained on)
    keys = [k for k in ('pix', 'full', 'curriculum') if os.path.exists(os.path.join(args.work, f'vae_{k}',
                                                                                        'history.json'))]
    hist = {k: json.load(open(os.path.join(args.work, f'vae_{k}', 'history.json'))) for k in keys}
    titles = {'pix': r'trained with $L_{pix}$ only', 'full': r'$L_{pix}+L_{edge}+L_{SSIM}$ from scratch',
              'curriculum': r'$L_{pix}$ (ep. 1-20) $\rightarrow$ full loss (ep. 21-30)'}
    fig, axs = plt.subplots(1, len(keys), figsize=(4.1 * len(keys), 3.2), sharey=True)
    for ax, key in zip(np.atleast_1d(axs), keys):
        h = hist['pix'] + hist[key] if key == 'curriculum' else hist[key]
        ep = np.arange(1, len(h) + 1)
        for comp, col in zip(('pix', 'edge', 'ssim'), (P.BLUE, P.ORANGE, P.AQUA)):
            ax.plot(ep, [r['train'][comp] for r in h], c=col, label=f'$L_{{{comp}}}$ train')
            ax.plot(ep, [r['test'][comp] for r in h], c=col, ls='--', lw=1.4, label=f'$L_{{{comp}}}$ test')
        if key == 'curriculum':
            ax.axvline(20.5, c=P.GRAY, lw=1, ls=':')
        ax.set_xlabel('Epoch')
        ax.set_title(titles[key], fontsize=9)
    np.atleast_1d(axs)[0].set_ylabel('Loss')
    np.atleast_1d(axs)[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig2_losses.png'))
    plt.close(fig)
    metrics['final_losses'] = {k: dict(train=hist[k][-1]['train'], test=hist[k][-1]['test']) for k in hist}

    # ---- Fig. 2c: qualitative comparison on random test patches
    rng = np.random.default_rng(3)
    sel = rng.choice(np.nonzero(is_test)[0], 6, replace=False)
    rows = [('original', orig)] + list(models.items())
    fig, axs = plt.subplots(len(rows), 6, figsize=(9.5, 1.6 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        P.show_images(axs[r], imgs[sel])
        axs[r, 0].set_ylabel(name, rotation=0, ha='right', va='center', fontsize=8, color=P.INK2)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.28)
    fig.savefig(os.path.join(args.out, 'fig2c_reconstructions.png'))
    plt.close(fig)

    # ---- Fig. 3a-h: one random test patch, six-feature distributions
    best = list(models)[-1]
    k = sel[0]
    to, tr = grain_table(orig[k], pixel_um=pixel_um, method=SEG), grain_table(models[best][k], pixel_um=pixel_um, method=SEG)
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(2, 4)
    for j, (im, title) in enumerate(((orig[k], 'original'), (models[best][k], 'reconstructed'))):
        ax = fig.add_subplot(gs[j, 0])
        P.show_images([ax], [im])
        ax.set_title(title, fontsize=8)
    for j, name in enumerate(FEATURE_NAMES):
        ax = fig.add_subplot(gs[j // 3, 1 + j % 3])
        a, b = to[name], tr[name]
        both = np.r_[a, b]
        if len(both) == 0:
            continue
        bins = np.linspace(both.min(), both.max() + 1e-9, 12)
        for vals, col, lab in ((a, P.BLUE, 'original'), (b, P.ORANGE, 'reconstructed')):
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, density=True, color=col, alpha=0.3)
            if len(vals) > 2 and np.std(vals) > 0:
                xs = np.linspace(bins[0], bins[-1], 200)
                ax.plot(xs, gaussian_kde(vals)(xs), c=col, lw=1.6, label=lab)
        ax.set_xlabel(f'{LABELS[name]} ({FEATURE_UNITS[FEATURE_NAMES.index(name)]})', fontsize=8)
        ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig3_single_patch.png'))
    plt.close(fig)

    # ---- Fig. 3i-n / Supp. Fig. 6: mean features, all 2000 patches + per-map aggregates
    Fo = feature_matrix(orig, pixel_um)
    metrics['aggregate_feature_fidelity'] = {}
    for mname, rec in models.items():
        Fr = feature_matrix(rec, pixel_um)
        mo, so = aggregate(Fo)
        mr, sr = aggregate(Fr)
        fig, axs = plt.subplots(1, 6, figsize=(15, 2.9))
        for j, (ax, name) in enumerate(zip(axs, FEATURE_NAMES)):
            ax.scatter(Fo[:, j], Fr[:, j], s=4, c=P.GRAY, alpha=0.25, lw=0, label='patches (2000)')
            for idx, col, mk, lab in ((train_idx, P.BLUE, 'o', 'train maps (20)'), (test_idx, P.RED, 's',
                                                                                    'test maps (5)')):
                ax.errorbar(mo[idx, j], mr[idx, j], xerr=so[idx, j], yerr=sr[idx, j], fmt=mk, ms=5, c=col,
                            mec='white', mew=0.6, elinewidth=0.8, capsize=0, label=lab)
            lim = np.nanpercentile(np.r_[Fo[:, j], Fr[:, j]], [0.5, 99.5]) + np.array([0.0, 1e-9])
            ax.plot(lim, lim, ls='--', lw=1, c=P.INK2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            unit = FEATURE_UNITS[j]
            ax.set_xlabel(f'Original {LABELS[name].lower()} ({unit})', fontsize=8)
            ax.set_ylabel(f'Reconstructed ({unit})', fontsize=8)
        axs[0].legend(fontsize=6, loc='upper left')
        fig.suptitle(mname, fontsize=10)
        fig.tight_layout()
        tag = mname.replace(' ', '_').replace('+', 'plus').replace('-', '_')
        fig.savefig(os.path.join(args.out, f'fig3_aggregate_{tag}.png'))
        plt.close(fig)
        metrics['aggregate_feature_fidelity'][mname] = dict(train=agg_metrics(Fo, Fr, train_idx),
                                                            test=agg_metrics(Fo, Fr, test_idx))

    # ---- Supp. Fig. 4: neighbour misorientation distributions on the test maps
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bins = np.linspace(0, 63, 22)
    mis_o = np.concatenate([neighbour_misorientations(im, method=SEG) for im in orig[is_test]] + [np.zeros(0)])
    ax.hist(mis_o, bins=bins, density=True, color=P.GRAY, alpha=0.45, label='original')
    metrics['misorientation_twin_fraction'] = {'original': float(np.mean(np.abs(mis_o - 60) < 3))}
    for (mname, rec), col in zip(models.items(), (P.BLUE, P.ORANGE, P.AQUA, P.VIOLET)):
        mis = np.concatenate([neighbour_misorientations(im, method=SEG) for im in rec[is_test]] + [np.zeros(0)])
        h, _ = np.histogram(mis, bins=bins, density=True)
        ax.step(bins[:-1], h, where='post', c=col, lw=1.5, label=mname)
        metrics['misorientation_twin_fraction'][mname] = float(np.mean(np.abs(mis - 60) < 3))
    ax.set_xlabel('Neighbour misorientation (deg)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'misorientation.png'))
    plt.close(fig)

    # ---- Supp. Fig. 8: latent space PCA, interpolation M1 -> M2 and extrapolation M2 -> M3
    from sklearn.decomposition import PCA
    Z = encode(vae_main, patches)
    pca = PCA(2).fit(Z)
    Z2 = pca.transform(Z)
    m1, m2 = sel[1], sel[2]
    alphas = np.linspace(0, 2, 21)
    path = np.array([(1 - a) * Z[m1] + a * Z[m2] for a in alphas])
    path_imgs = quantize(decode(vae_main, path, ddpm, out_size=patch))
    feats = np.array([mean_features(im, pixel_um=pixel_um, method=SEG) for im in path_imgs])
    ngr = [len(grain_table(im, pixel_um=pixel_um, method=SEG)['area']) for im in path_imgs]
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(3, 11)
    ax = fig.add_subplot(gs[:2, :4])
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=Fo[:, 3], s=6, cmap='Blues', lw=0)
    p_path = pca.transform(path)
    ax.plot(p_path[:11, 0], p_path[:11, 1], c=P.ORANGE, lw=1.5)
    ax.plot(p_path[10:, 0], p_path[10:, 1], c=P.RED, lw=1.5, ls='--')
    for lab, pt in (('M1', p_path[0]), ('M2', p_path[10]), ('M3', p_path[-1])):
        ax.annotate(lab, pt, xytext=(4, 4), textcoords='offset points', color=P.INK)
    fig.colorbar(sc, ax=ax, label='mean Euler1 (deg)')
    ax.set_xlabel('Latent PC1')
    ax.set_ylabel('Latent PC2')
    for i in range(11):
        a2 = fig.add_subplot(gs[2, i])
        P.show_images([a2], [path_imgs[2 * i]])
        a2.set_title(f'{alphas[2 * i]:.1f}', fontsize=7)
    for r, (vals, lab) in enumerate(((feats[:, 0], 'mean grain area (um$^2$)'), (feats[:, 3], 'mean Euler1 (deg)'),
                                     (np.array(ngr), 'number of grains'))):
        a3 = fig.add_subplot(gs[:2, 5 + 2 * r:7 + 2 * r])
        a3.plot(alphas, vals, c=P.BLUE, marker='o', ms=3)
        a3.axvline(1.0, c=P.GRAY, lw=1, ls=':')
        a3.set_xlabel(r'$\alpha$ (M1 $\to$ M2 $\to$ M3)', fontsize=8)
        a3.set_title(lab, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'latent_space.png'))
    plt.close(fig)
    metrics['latent_path_features'] = dict(alpha=alphas.tolist(), mean_area=feats[:, 0].tolist(),
                                           mean_euler1=feats[:, 3].tolist(), n_grains=ngr)
    with open(os.path.join(args.out, 'fidelity_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps(metrics['aggregate_feature_fidelity'], indent=1))


if __name__ == '__main__':
    main()
