"""Part A - reproduce Fig. 4b-f with the data released in the official repository.

Inputs (MIT-licensed, copied from github.com/nwpuai4msegroup/microstructures_design):
  data/official/z_i.csv      latent vectors (2000 x 128) of the 25 x 80 EBSD patches from the authors' trained VAE
  data/official/dataset.csv  processing conditions, yield strength and elongation of the 25 alloys

Outputs go to results/official/ : parity plots, Delta-HV curves, PC1 distributions, Pareto front, metrics.json and
optimized_latents.npz (the 80 latent vectors of every optimised candidate, ready to be decoded by a trained VAE).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msdesign import plotting as P  # noqa: E402
from msdesign.inverse import (DescriptorPCA, TreeEnsemble, fit_gbr, fit_gpr, latent_descriptors,  # noqa: E402
                              run_nsga2, sample_latents, wasserstein_selection)
from msdesign.nsga2 import non_dominated_fronts  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'official'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'results', 'official'))
    ap.add_argument('--n-gen', type=int, default=500)
    ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='+', default=[212] + list(range(10)))
    ap.add_argument('--surrogate', choices=['official', 'gbr', 'gpr'], default='official',
                    help="'official': the authors' saved GBR models (data/official/gbr_official_trees.npz); "
                         "'gbr'/'gpr': refit with the notebook's randomised search / a GP")
    ap.add_argument('--robustness', type=int, default=20,
                    help='number of random train/test splits for the surrogate-variance analysis (0 = skip)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    P.setup()
    plt = P.plt

    z = np.genfromtxt(os.path.join(args.data, 'z_i.csv'), delimiter=',', skip_header=1)[:, 1:]
    tab = np.genfromtxt(os.path.join(args.data, 'dataset.csv'), delimiter=',', skip_header=1)
    Y = tab[:, 4:6]                       # yield strength (MPa), elongation (%)
    print('latents', z.shape, 'properties', Y.shape)

    # 1) descriptors + invertible PCA (3 + 3)
    z_mean, z_std = latent_descriptors(z)
    pca = DescriptorPCA(3, 3).fit(z_mean, z_std)
    X = pca.transform(z_mean, z_std)
    full_mean = DescriptorPCA(24, 24).fit(z_mean, z_std)
    evr = dict(mean=full_mean.pca_mean.explained_variance_ratio_.tolist(),
               std=full_mean.pca_std.explained_variance_ratio_.tolist())
    print('explained variance (first 3 PCs): mean %.3f  std %.3f' % (sum(evr['mean'][:3]), sum(evr['std'][:3])))
    print('PC ranges:', X.min(0).round(2), X.max(0).round(2))

    # 2) surrogates, same train/test splits as the official notebook (seed 75 for YS, 98 for EL)
    metrics = dict(explained_variance=evr, surrogate=args.surrogate)
    if args.surrogate == 'official':
        from sklearn.model_selection import train_test_split
        npz = os.path.join(args.data, 'gbr_official_trees.npz')
        m_ys, m_el = TreeEnsemble(npz, 'GBR_ys'), TreeEnsemble(npz, 'GBR_el')
        infos = []
        for m, col, seed in ((m_ys, 0, 75), (m_el, 1, 98)):
            X_tr, X_te, y_tr, y_te = train_test_split(X, Y[:, col], test_size=0.2, random_state=seed)
            # the EL model was fitted on the seed-75 split (its init constant = mean of that training set);
            # also report the score on its true held-out samples
            tr75, te75 = train_test_split(np.arange(25), test_size=0.2, random_state=75)
            infos.append(dict(r2_train=r2_score(y_tr, m.predict(X_tr)), r2_test=r2_score(y_te, m.predict(X_te)),
                              r2_true_heldout=r2_score(Y[te75, col], m.predict(X[te75])),
                              best_params=m.best_params, internal_cv_r2=m.cv_best_score, n_train=m.n_train,
                              split=(X_tr, X_te, y_tr, y_te)))
        info_ys, info_el = infos
    else:
        fit = fit_gbr if args.surrogate == 'gbr' else fit_gpr
        m_ys, info_ys = fit(X, Y[:, 0], split_seed=75)
        m_el, info_el = fit(X, Y[:, 1], split_seed=98)
    for name, info in (('yield_strength', info_ys), ('elongation', info_el)):
        metrics[name] = {k: v for k, v in info.items() if k != 'split'}
        print(f"{name}: R2 train {info['r2_train']:.3f}  test {info['r2_test']:.3f}  {info.get('best_params', '')}")
    if args.surrogate != 'gpr':  # extra, more robust estimate with only 25 samples: leave-one-out CV
        for name, col, info in (('yield_strength', 0, info_ys), ('elongation', 1, info_el)):
            bp = info['best_params']
            bp = eval(bp) if isinstance(bp, str) else bp  # noqa: S307  (dict literal from our own export)
            pred = cross_val_predict(GradientBoostingRegressor(random_state=0, **bp), X, Y[:, col], cv=LeaveOneOut())
            metrics[name]['r2_loocv_same_hyperparams'] = r2_score(Y[:, col], pred)
            print(f'{name}: LOO-CV R2 with the same hyper-parameters {r2_score(Y[:, col], pred):.3f}')
    if args.robustness:  # how much does the 5-sample test R2 depend on the split / search seed?
        rob = {}
        for name, col in (('yield_strength', 0), ('elongation', 1)):
            vals = [fit_gbr(X, Y[:, col], split_seed=s, search_seed=s)[1]['r2_test'] for s in range(args.robustness)]
            rob[name] = dict(r2_test=vals, median=float(np.median(vals)),
                             frac_above_0p92=float(np.mean(np.array(vals) > 0.92)))
            print(f'{name}: test R2 over {args.robustness} random splits: median {np.median(vals):.2f}, '
                  f'IQR {np.percentile(vals, 25):.2f}..{np.percentile(vals, 75):.2f}, >0.92 in '
                  f'{100 * np.mean(np.array(vals) > 0.92):.0f} %')
        metrics['split_robustness'] = rob
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for k, (name, c) in enumerate((('yield_strength', P.BLUE), ('elongation', P.ORANGE))):
            v = np.array(rob[name]['r2_test'])
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)),
                       np.clip(v, -1.5, 1), s=18, c=c, alpha=0.8, lw=0)
            ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, c=P.INK, lw=1.5)
        ax.axhline(0.92, ls='--', lw=1, c=P.INK2)
        ax.text(1.45, 0.93, 'paper: 0.92', ha='right', va='bottom', color=P.INK2, fontsize=8)
        ax.set_xticks([0, 1], ['Yield strength', 'Elongation'])
        ax.set_ylabel('Test $R^2$ (5 held-out alloys)')
        ax.set_xlim(-0.6, 1.6)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'surrogate_split_robustness.png'))
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, info, lab, unit in ((axs[0], info_ys, 'Yield strength', 'MPa'), (axs[1], info_el, 'Elongation', '%')):
        X_tr, X_te, y_tr, y_te = info['split']
        model = m_ys if ax is axs[0] else m_el
        ax.scatter(y_tr, model.predict(X_tr), s=26, c=P.GRAY, label='train', edgecolors='white', linewidths=0.6)
        ax.scatter(y_te, model.predict(X_te), s=34, c=P.RED, label='test', edgecolors='white', linewidths=0.6)
        lo, hi = min(y_tr.min(), y_te.min()), max(y_tr.max(), y_te.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls='--', lw=1, c=P.INK2)
        ax.set_xlabel(f'Measured {lab.lower()} ({unit})')
        ax.set_ylabel(f'Predicted ({unit})')
        ax.set_title(f"{lab}: test $R^2$ = {info['r2_test']:.2f}")
        ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4bc_parity.png'))
    plt.close(fig)

    # 3) NSGA-II runs (pop 10, 500 generations, box [-2, 2]^6)
    runs = {}
    for seed in args.seeds:
        res = run_nsga2([m_ys, m_el], X, Y, pop_size=args.pop, n_gen=args.n_gen, seed=seed)
        runs[seed] = res
        print(f'seed {seed}: final Delta-HV {res["delta_hv"][-1]:.0f}  (HV0 = {res["hv0"]:.0f})')
    dhv = np.stack([r['delta_hv'] for r in runs.values()])
    main_run = runs[args.seeds[0]]
    metrics['nsga2'] = dict(hv0=main_run['hv0'], ref=main_run['ref'].tolist(),
                            final_delta_hv_mean=float(dhv[:, -1].mean()), final_delta_hv_std=float(dhv[:, -1].std()),
                            gen_to_95pct=int(np.argmax(dhv.mean(0) >= 0.95 * dhv.mean(0)[-1])))

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    g = np.arange(dhv.shape[1])
    ax.fill_between(g, dhv.mean(0) - dhv.std(0), dhv.mean(0) + dhv.std(0), color=P.BLUE, alpha=0.18, lw=0)
    ax.plot(g, dhv.mean(0), c=P.BLUE, label=f'mean of {len(runs)} runs')
    ax.plot(g, main_run['delta_hv'], c=P.ORANGE, lw=1.2, label=f'seed {args.seeds[0]} (official)')
    ax.set_xlabel('Generation')
    ax.set_ylabel(r'$\Delta$HV (MPa $\cdot$ %)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4d_delta_hv.png'))
    plt.close(fig)

    # 4) where do the generated solutions sit relative to the 25 initial maps? (Fig. 4e)
    allX = np.concatenate([h[0] for r in runs.values() for h in r['history']])
    allF = -np.concatenate([h[1] for r in runs.values() for h in r['history']])
    ref = main_run['ref']
    contrib = np.prod(allF - ref, axis=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    hi = contrib >= np.quantile(contrib, 0.9)
    ax.scatter(allX[~hi, 0], allX[~hi, 3], s=8, c=P.GRAY, alpha=0.35, lw=0, label='generated')
    ax.scatter(allX[hi, 0], allX[hi, 3], s=22, marker='^', c=P.RED, alpha=0.7, lw=0,
               label=r'generated, top-10 % $\Delta$HV')
    ax.scatter(X[:, 0], X[:, 3], s=40, c=P.BLUE, edgecolors='white', linewidths=0.8, label='initial 25 maps')
    ax.set_xlabel(r'PC1 of $\mu$ descriptor')
    ax.set_ylabel(r'PC1 of $\sigma$ descriptor')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4e_pc1.png'))
    plt.close(fig)
    lo_b, hi_b = X.min(0), X.max(0)
    outside = np.any((allX < lo_b) | (allX > hi_b), axis=1)
    metrics['nsga2']['fraction_outside_initial_box'] = float(outside.mean())
    metrics['nsga2']['fraction_top10_outside_initial_box'] = float(outside[hi].mean())

    # 5) final Pareto front and the five best candidates (Fig. 4f)
    Xf, Ff = main_run['X'], -main_run['F']
    c_final = np.prod(Ff - ref, axis=1)
    top = np.argsort(-c_final)[:5]
    nd0 = non_dominated_fronts(-Y)[0]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(Y[:, 0], Y[:, 1], s=34, c=P.BLUE, edgecolors='white', linewidths=0.8, label='initial data (25)')
    o = nd0[np.argsort(Y[nd0, 0])]
    ax.plot(Y[o, 0], Y[o, 1], c=P.BLUE, lw=1, ls='--', label='initial Pareto front')
    ax.scatter(Ff[top, 0], Ff[top, 1], s=46, marker='s', c=P.RED, edgecolors='white', linewidths=0.8,
               label='optimised candidates (top 5)')
    uniq = [i for n, i in enumerate(top) if not any(np.allclose(Ff[i], Ff[j]) for j in top[:n])]
    for k, i in enumerate(uniq[:2]):   # label the two best distinct candidates, as points 1 and 2 of Fig. 4f
        ax.annotate(str(k + 1), (Ff[i, 0], Ff[i, 1]), xytext=(6, 4), textcoords='offset points', color=P.INK)
    ax.set_xlabel('Yield strength (MPa)')
    ax.set_ylabel('Elongation (%)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4f_pareto.png'))
    plt.close(fig)
    metrics['candidates'] = [dict(pcs=Xf[i].round(4).tolist(), ys_pred=float(Ff[i, 0]), el_pred=float(Ff[i, 1]))
                             for i in top]

    # 6) back to latent space: 80 latent vectors per candidate + Wasserstein-based representative patches
    mu, sd, n_clip = pca.inverse(Xf[top])
    rng = np.random.default_rng(0)
    Z = np.stack([sample_latents(mu[k], sd[k], 80, rng) for k in range(len(top))])
    sel = [wasserstein_selection(Z[k])[0] for k in range(len(top))]
    metrics['latent_std_clipped_entries'] = n_clip
    np.savez(os.path.join(args.out, 'optimized_latents.npz'), z=Z, mu=mu, sd=sd, pcs=Xf[top], props=Ff[top],
             wasserstein_selected=np.array(sel))

    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(json.dumps({k: metrics[k] for k in ('yield_strength', 'elongation', 'nsga2')}, indent=1, default=str))
    print('candidates:', [(round(c['ys_pred']), round(c['el_pred'], 1)) for c in metrics['candidates']])


if __name__ == '__main__':
    main()
