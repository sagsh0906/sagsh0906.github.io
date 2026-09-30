"""Part B - closed-loop inverse design with our own trained models on the synthetic data (Fig. 4, Fig. 5, Supp. 10-14).

1. encode the 25 x 80 patches with the physics-aware VAE -> mu / sigma descriptors -> PCA (3 + 3);
   Supp. Fig. 10: fidelity of re-generated maps as a function of the number of retained PCs;
2. GBR surrogates on the crystal-plasticity-proxy labels (same randomised search as the official notebook);
3. NSGA-II (pop 10, 500 generations, [-2, 2]^6) -> Delta-HV curve, PC1 distribution, Pareto candidates;
4. decode 80 latents per candidate (VAE + DDPM), stitch 8 x 10 maps, grain-area distributions (Fig. 5a-d);
5. "virtual validation": the CP proxy evaluates the generated maps (stand-in for CPFE, Fig. 5g-h / Supp. Fig. 14).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msdesign import plotting as P  # noqa: E402
from msdesign.cp_proxy import properties  # noqa: E402
from msdesign.data import crop_patches, stitch_patches  # noqa: E402
from msdesign.features import grain_table, mean_features, size_distribution_stats  # noqa: E402
from msdesign.inverse import (DescriptorPCA, fit_gbr, latent_descriptors, run_nsga2, sample_latents,  # noqa: E402
                              wasserstein_selection)
from msdesign.nsga2 import non_dominated_fronts  # noqa: E402
from msdesign.pipeline import decode, encode, load_ddpm, load_vae, quantize  # noqa: E402


def safe_properties(img, pixel_um, return_curve=False):
    try:
        return properties(img, pixel_um=pixel_um, return_curve=return_curve)
    except ValueError:                      # no grains could be segmented in the image
        nan_curve = dict(eng_strain=np.zeros(1), eng_stress=np.zeros(1))
        return (np.nan, np.nan, nan_curve) if return_curve else (np.nan, np.nan)


SEG = 'watershed'   # common grain segmentation for originals and reconstructions (see features.segment_grains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='work/quick')
    ap.add_argument('--out', default='results/synthetic')
    ap.add_argument('--n-gen', type=int, default=500)
    ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='+', default=[212] + list(range(5)))
    ap.add_argument('--n-candidates', type=int, default=5)
    ap.add_argument('--no-ddpm', action='store_true')
    ap.add_argument('--vae', default=None, help='default: vae_curriculum if present, else vae_full')
    ap.add_argument('--pc-study', type=int, nargs='*', default=[1, 2, 3, 6, 12, 24])
    ap.add_argument('--seg', choices=['watershed', 'dark'], default='watershed')
    args = ap.parse_args()
    global SEG
    SEG = args.seg
    os.makedirs(args.out, exist_ok=True)
    P.setup()
    plt = P.plt
    d = np.load(os.path.join(args.work, 'dataset.npz'))
    pixel_um, patch = float(d['pixel_um']), int(d['patch'])
    Y = d['props']
    patches = np.concatenate([crop_patches(im, patch) for im in d['images']])
    vae_path = args.vae or next(p for p in (os.path.join(args.work, 'vae_curriculum', 'vae.pt'),
                                            os.path.join(args.work, 'vae_full', 'vae.pt')) if os.path.exists(p))
    vae, _ = load_vae(vae_path)
    print('VAE:', vae_path)
    ddpm_path = os.path.join(args.work, 'ddpm', 'ddpm.pt')
    ddpm = None if args.no_ddpm or not os.path.exists(ddpm_path) else load_ddpm(ddpm_path)
    metrics = dict(ddpm=ddpm is not None, vae=vae_path)

    # 1) descriptors + PCA
    Z = encode(vae, patches)
    z_mean, z_std = latent_descriptors(Z)
    pca = DescriptorPCA(3, 3).fit(z_mean, z_std)
    X = pca.transform(z_mean, z_std)
    metrics['pca_explained_3'] = dict(mean=float(pca.pca_mean.explained_variance_ratio_.sum()),
                                      std=float(pca.pca_std.explained_variance_ratio_.sum()))

    # Supp. Fig. 10: how many PCs are needed to regenerate a faithful microstructure?
    if args.pc_study:
        rng = np.random.default_rng(0)
        Fo = np.array([mean_features(p, pixel_um=pixel_um, method=SEG) for p in patches.astype(np.float32) / 255.0])
        Fo_map = np.nanmean(Fo.reshape(25, 80, -1), 1)
        errs = {}
        for k in args.pc_study:
            pk = DescriptorPCA(k, k).fit(z_mean, z_std)
            mu_k, sd_k, _ = pk.inverse(pk.transform(z_mean, z_std))
            e = []
            for m in range(25):
                imgs = quantize(decode(vae, sample_latents(mu_k[m], sd_k[m], 80, rng), None, out_size=patch))
                Fg = np.nanmean([mean_features(im, pixel_um=pixel_um, method=SEG) for im in imgs], 0)
                e.append(np.abs(Fg[:3] - Fo_map[m, :3]) / Fo_map[m, :3])
            errs[k] = float(np.mean(e))
            print(f'PCs {k:2d}: mean relative error of morphology features {errs[k]:.3f}', flush=True)
        metrics['pc_study_rel_err'] = errs
        fig, ax = plt.subplots(figsize=(3.8, 2.9))
        ax.plot(list(errs), list(errs.values()), marker='o', c=P.BLUE)
        ax.set_xlabel(r'retained PCs (per $\mu$ / $\sigma$ descriptor)')
        ax.set_ylabel('rel. error of grain morphology')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'suppfig10_pc_study.png'))
        plt.close(fig)

    # 2) surrogates on the proxy labels
    m_ys, info_ys = fit_gbr(X, Y[:, 0], split_seed=75)
    m_el, info_el = fit_gbr(X, Y[:, 1], split_seed=98)
    for name, info in (('yield_strength', info_ys), ('elongation', info_el)):
        metrics[name] = {k: v for k, v in info.items() if k != 'split'}
        print(f"{name}: R2 train {info['r2_train']:.3f} test {info['r2_test']:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, info, model, lab, unit in ((axs[0], info_ys, m_ys, 'Yield strength', 'MPa'),
                                       (axs[1], info_el, m_el, 'Elongation', '%')):
        X_tr, X_te, y_tr, y_te = info['split']
        ax.scatter(y_tr, model.predict(X_tr), s=26, c=P.GRAY, edgecolors='white', label='train')
        ax.scatter(y_te, model.predict(X_te), s=34, c=P.RED, edgecolors='white', label='test')
        lo, hi = Y[:, 0 if unit == 'MPa' else 1].min(), Y[:, 0 if unit == 'MPa' else 1].max()
        ax.plot([lo, hi], [lo, hi], ls='--', lw=1, c=P.INK2)
        ax.set_xlabel(f'Proxy {lab.lower()} ({unit})')
        ax.set_ylabel(f'GBR prediction ({unit})')
        ax.set_title(f"{lab}: test $R^2$ = {info['r2_test']:.2f}")
        ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4bc_parity.png'))
    plt.close(fig)

    # 3) NSGA-II
    runs = {s: run_nsga2([m_ys, m_el], X, Y, pop_size=args.pop, n_gen=args.n_gen, seed=s) for s in args.seeds}
    dhv = np.stack([r['delta_hv'] for r in runs.values()])
    main_run = runs[args.seeds[0]]
    ref = main_run['ref']
    metrics['nsga2'] = dict(hv0=main_run['hv0'], final_delta_hv_mean=float(dhv[:, -1].mean()),
                            final_delta_hv_std=float(dhv[:, -1].std()))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    g = np.arange(dhv.shape[1])
    ax.fill_between(g, dhv.mean(0) - dhv.std(0), dhv.mean(0) + dhv.std(0), color=P.BLUE, alpha=0.18, lw=0)
    ax.plot(g, dhv.mean(0), c=P.BLUE, label=f'mean of {len(runs)} runs')
    ax.set_xlabel('Generation')
    ax.set_ylabel(r'$\Delta$HV (MPa $\cdot$ %)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4d_delta_hv.png'))
    plt.close(fig)

    Xf, Ff = main_run['X'], -main_run['F']
    contrib = np.prod(Ff - ref, axis=1)
    order = np.argsort(-contrib)
    cand = []
    for i in order:                                     # distinct candidates with the highest Delta-HV
        if not any(np.allclose(Ff[i], Ff[j]) for j in cand):
            cand.append(i)
        if len(cand) == args.n_candidates:
            break
    cand = np.array(cand)

    # 4) decode the candidates: 80 latents each -> VAE (+ DDPM) -> 8 x 10 mosaic
    mu, sd, n_clip = pca.inverse(Xf[cand])
    rng = np.random.default_rng(1)
    gen_maps, gen_patches, zs = [], [], []
    for k in range(len(cand)):
        z = sample_latents(mu[k], sd[k], 80, rng)
        imgs = quantize(decode(vae, z, ddpm, seed=k, out_size=patch))
        zs.append(z)
        gen_patches.append(imgs)
        gen_maps.append(stitch_patches(imgs, 8, 10))
    gen_patches = np.stack(gen_patches)
    metrics['latent_std_clipped_entries'] = n_clip

    # PC1 distributions of initial vs generated maps (Fig. 4e), measured by re-encoding the generated patches
    Zg = np.concatenate([encode(vae, (p * 255).round().astype(np.uint8)) for p in gen_patches])
    zg_mean, zg_std = latent_descriptors(Zg, n_maps=len(cand))
    Xg = pca.transform(zg_mean, zg_std)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, j, lab in ((axs[0], 0, r'PC1 of $\mu$'), (axs[1], 3, r'PC1 of $\sigma$')):
        ax.hist(X[:, j], bins=10, color=P.BLUE, alpha=0.45, label='initial 25 maps')
        ax.scatter(Xg[:, j], np.full(len(Xg), 0.3), marker='^', s=60, c=P.RED, edgecolors='white',
                   label='generated (re-encoded)', zorder=3)
        ax.scatter(Xf[cand, j], np.full(len(cand), 0.8), marker='v', s=50, c=P.ORANGE, edgecolors='white',
                   label='NSGA-II target', zorder=3)
        ax.set_xlabel(lab)
        ax.set_ylabel('count')
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4e_pc1.png'))
    plt.close(fig)
    metrics['generated_outside_initial_range'] = float(np.mean(np.any((Xg < X.min(0)) | (Xg > X.max(0)), 1)))

    # Fig. 5a-d: generated vs original patches and grain-area distributions
    orig = patches.astype(np.float32) / 255.0
    a_orig = np.concatenate([grain_table(p, pixel_um=pixel_um, method=SEG)['area'] for p in orig])
    a_gen = np.concatenate([grain_table(p, pixel_um=pixel_um, method=SEG)['area'] for p in gen_patches.reshape(-1, patch, patch, 3)])
    fine = np.median(a_orig) / 4            # same 'fine grain' threshold for both populations
    st_o, st_g = size_distribution_stats(a_orig, fine_um2=fine), size_distribution_stats(a_gen, fine_um2=fine)
    metrics['grain_area'] = dict(original=st_o, generated=st_g)
    per_map = [size_distribution_stats(np.concatenate([grain_table(p, pixel_um=pixel_um, method=SEG)['area']
                                                       for p in crop_patches(im, patch)]))['bimodality']
               for im in d['images'].astype(np.float32) / 255.0]
    metrics['bimodality_per_training_map'] = per_map
    metrics['bimodality_per_generated_map'] = [
        size_distribution_stats(np.concatenate([grain_table(p, pixel_um=pixel_um, method=SEG)['area'] for p in gp]))['bimodality']
        for gp in gen_patches]
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 8)
    rs = np.random.default_rng(5)
    o_sel = rs.choice(len(orig), 8, replace=False)
    g_sel = rs.choice(gen_patches.reshape(-1, patch, patch, 3).shape[0], 8, replace=False)
    for i in range(4):
        P.show_images([fig.add_subplot(gs[0, i])], [orig[o_sel[i]]])
        P.show_images([fig.add_subplot(gs[1, i])], [gen_patches.reshape(-1, patch, patch, 3)[g_sel[i]]])
    fig.axes[0].set_title('original patches', loc='left', fontsize=9)
    fig.axes[1].set_title('optimised (generated) patches', loc='left', fontsize=9)
    bins = np.linspace(np.log10(max(a_orig.min(), 0.1)), np.log10(max(a_orig.max(), a_gen.max(initial=0))), 30)
    for r, (a, col, lab, st) in enumerate(((a_orig, P.BLUE, 'original (2000 patches)', st_o),
                                           (a_gen, P.RED, f'generated ({gen_patches.shape[0] * 80} patches)', st_g))):
        ax = fig.add_subplot(gs[r, 4:])
        if len(a):
            ax.hist(np.log10(a), bins=bins, density=True, color=col, alpha=0.55)
        ax.set_xlabel(r'log$_{10}$ grain area ($\mu$m$^2$)')
        ax.set_ylabel('density')
        ax.set_title(f"{lab}: bimodality coeff. {st['bimodality']:.2f}, fine fraction {st['fine_fraction']:.2f}",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig5abcd_grain_size.png'))
    plt.close(fig)

    # 5) virtual validation of the generated maps with the CP proxy (stand-in for CPFE)
    val = []
    curves = []
    for k in range(len(cand)):
        sel, _ = wasserstein_selection(zs[k], n_select=2)
        ys_map, el_map, cur = safe_properties(gen_maps[k], pixel_um, return_curve=True)
        ys_sel = [safe_properties(gen_patches[k][i], pixel_um) for i in sel]
        val.append(dict(pred_ys=float(Ff[cand[k], 0]), pred_el=float(Ff[cand[k], 1]), proxy_ys_map=ys_map,
                        proxy_el_map=el_map, proxy_selected_patches=[list(map(float, v)) for v in ys_sel]))
        curves.append(cur)
        print(f"candidate {k + 1}: GBR {Ff[cand[k], 0]:.0f} MPa / {Ff[cand[k], 1]:.1f} %   "
              f"proxy on generated map {ys_map:.0f} MPa / {el_map:.1f} %", flush=True)
    metrics['validation'] = val
    # same candidates decoded by the VAE alone (no diffusion refinement): separates refiner artefacts (speckle read
    # as dislocation density by the KAM -> GND step) from what the optimised latent descriptors encode
    val_vae = []
    for k in range(len(cand)):
        m_vae = stitch_patches(quantize(decode(vae, zs[k], None, out_size=patch)), 8, 10)
        ys_v, el_v = safe_properties(m_vae, pixel_um)
        val_vae.append(dict(pred_ys=float(Ff[cand[k], 0]), pred_el=float(Ff[cand[k], 1]), proxy_ys_map=ys_v,
                            proxy_el_map=el_v))
        print(f"candidate {k + 1}: proxy on VAE-only map {ys_v:.0f} MPa / {el_v:.1f} %", flush=True)
    metrics['validation_vae_only'] = val_vae
    if len(cand) > 2:
        pred = np.array([[v['pred_ys'], v['pred_el']] for v in val])
        for tag, vv in (('ddpm', val), ('vae_only', val_vae)):
            got = np.array([[v['proxy_ys_map'], v['proxy_el_map']] for v in vv])
            metrics[f'validation_corr_{tag}'] = dict(
                ys=float(np.corrcoef(pred[:, 0], got[:, 0])[0, 1]), el=float(np.corrcoef(pred[:, 1], got[:, 1])[0, 1]))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    P.show_images([axs[0]], [gen_maps[0]])
    axs[0].set_title('candidate 1: generated 8x10 mosaic', fontsize=9)
    for k, cur in enumerate(curves[:3]):
        axs[1].plot(cur['eng_strain'] * 100, cur['eng_stress'], c=(P.RED, P.ORANGE, P.VIOLET)[k],
                    label=f'candidate {k + 1}')
    best_orig = int(np.argmax(np.prod(Y - ref, 1)))
    _, _, cur0 = properties(d['images'][best_orig].astype(np.float32) / 255.0, pixel_um=pixel_um, return_curve=True)
    axs[1].plot(cur0['eng_strain'] * 100, cur0['eng_stress'], c=P.BLUE, ls='--', label=f'best initial (#{best_orig + 1})')
    axs[1].set_xlabel('Engineering strain (%)')
    axs[1].set_ylabel('Engineering stress (MPa)')
    axs[1].legend(fontsize=7)
    axs[2].scatter(Y[:, 0], Y[:, 1], s=26, c=P.BLUE, edgecolors='white', label='initial maps (proxy)')
    nd0 = non_dominated_fronts(-Y)[0]
    o = nd0[np.argsort(Y[nd0, 0])]
    axs[2].plot(Y[o, 0], Y[o, 1], c=P.BLUE, lw=1, ls='--')
    for k, v in enumerate(val):
        axs[2].annotate('', (v['proxy_ys_map'], v['proxy_el_map']), (v['pred_ys'], v['pred_el']),
                        arrowprops=dict(arrowstyle='->', color=P.GRAY, lw=0.8))
    axs[2].scatter([v['pred_ys'] for v in val], [v['pred_el'] for v in val], marker='s', s=40, c=P.RED,
                   edgecolors='white', label='GBR prediction')
    axs[2].scatter([v['proxy_ys_map'] for v in val], [v['proxy_el_map'] for v in val], marker='D', s=36,
                   c=P.ORANGE, edgecolors='white', label='CP proxy on generated map')
    axs[2].set_xlabel('Yield strength (MPa)')
    axs[2].set_ylabel('Elongation (%)')
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig5gh_validation.png'))
    plt.close(fig)
    P.plt.imsave(os.path.join(args.out, 'generated_map_candidate1.png'), gen_maps[0])
    np.savez_compressed(os.path.join(args.work, 'generated.npz'), maps=np.stack(gen_maps), latents=np.stack(zs),
                        pcs=Xf[cand], props_pred=Ff[cand])
    with open(os.path.join(args.out, 'inverse_design_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(json.dumps({k: metrics[k] for k in ('yield_strength', 'elongation', 'nsga2', 'grain_area')}, indent=1,
                     default=str))


if __name__ == '__main__':
    main()
