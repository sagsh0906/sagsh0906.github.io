"""Property-driven sampling of the VAE latent space (paper Fig. 4, Methods "GBR" / "NSGA-II").

Pipeline (identical to the official ``inverse_design.ipynb``):
1. latents of the 80 patches of each of the 25 maps -> per-dimension mean and std over patches (2 x 128 per map);
2. PCA (fitted on the 25 maps) on the mean- and std-descriptors separately, first 3 PCs each -> 6-D vector;
3. gradient-boosting surrogates for yield strength and elongation (randomised hyper-parameter search, 5-fold CV);
4. NSGA-II (pop 10, 500 generations, box [-2, 2]^6) maximising both predicted properties;
5. inverse PCA -> new (mu, sigma) descriptors -> 80 latent vectors z ~ N(mu, sigma^2) -> decoder (+ DDPM);
6. hypervolume improvement Delta-HV_t = HV_t - HV_0 (Eq. 16) to monitor the optimisation;
7. Wasserstein-distance based choice of representative patches for CPFE (Supplementary Note 6).
"""
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from .nsga2 import nsga2, non_dominated_fronts


def latent_descriptors(z, n_maps=25, n_patches=80):
    """z: (n_maps * n_patches, d) latents ordered map by map -> (mean, std) descriptors, each (n_maps, d)."""
    z = np.asarray(z).reshape(n_maps, n_patches, -1)
    return z.mean(axis=1), z.std(axis=1)


class DescriptorPCA:
    """Invertible PCA on the mean and std descriptors (3 + 3 components by default)."""

    def __init__(self, n_mean=3, n_std=3):
        self.pca_mean = PCA(n_components=n_mean, svd_solver='full')
        self.pca_std = PCA(n_components=n_std, svd_solver='full')
        self.n_mean = n_mean

    def fit(self, z_mean, z_std):
        for pca, data in ((self.pca_mean, z_mean), (self.pca_std, z_std)):
            pca.fit(data)
            # scikit-learn >= 1.5 changed the svd_flip sign convention; restore the one used by the authors
            # (sklearn 0.23: the largest-magnitude score of every component is positive) so that their saved
            # surrogates see the same axes
            scores = pca.transform(data)
            flip = np.sign(scores[np.argmax(np.abs(scores), axis=0), np.arange(scores.shape[1])])
            pca.components_ *= flip[:, None]
        return self

    def transform(self, z_mean, z_std):
        return np.concatenate([self.pca_mean.transform(z_mean), self.pca_std.transform(z_std)], axis=1)

    def inverse(self, X, min_std=1e-3):
        X = np.atleast_2d(X)
        mu = self.pca_mean.inverse_transform(X[:, :self.n_mean])
        sd = self.pca_std.inverse_transform(X[:, self.n_mean:])
        n_clipped = int((sd < min_std).sum())
        return mu, np.maximum(sd, min_std), n_clipped


def sample_latents(mu, sd, n_patches=80, rng=None):
    """Latent vectors of one new microstructure: z_ij ~ N(mu_j, sd_j^2), shape (n_patches, d)."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(mu[None, :], sd[None, :], size=(n_patches, len(mu)))


GBR_SEARCH_SPACE = {
    # paper: n_estimators in [10, 500], max_depth in [1, 20]; the official notebook also searches these two
    'n_estimators': stats.randint(low=10, high=500),
    'max_depth': stats.randint(low=1, high=20),
    'min_samples_split': stats.randint(low=2, high=10),
    'learning_rate': [0.1, 0.75, 0.05, 0.04, 0.03, 0.02, 0.01, 0.05],
}


def fit_gbr(X, y, test_size=0.2, split_seed=75, search_seed=0, n_iter=50, n_jobs=-1):
    """Randomised-search GBR exactly as in the official notebook; returns model and train/test R^2."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=split_seed)
    search = RandomizedSearchCV(GradientBoostingRegressor(random_state=search_seed), GBR_SEARCH_SPACE, cv=5,
                                n_iter=n_iter, n_jobs=n_jobs, random_state=search_seed)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    return model, dict(r2_train=r2_score(y_tr, model.predict(X_tr)), r2_test=r2_score(y_te, model.predict(X_te)),
                       best_params=search.best_params_, split=(X_tr, X_te, y_tr, y_te))


class TreeEnsemble:
    """Predictor for the authors' GBR models exported by scripts/export_official_gbr.py (portable tree dump)."""

    def __init__(self, npz_path, name):
        d = np.load(npz_path)
        self.init = float(d[f'{name}_init'])
        self.lr = float(d[f'{name}_lr'])
        self.n_train = int(d[f'{name}_train_n'])
        self.best_params = str(d[f'{name}_best_params'])
        self.cv_best_score = float(d[f'{name}_cv_best_score'])
        sizes = d[f'{name}_n_nodes']
        offs = np.concatenate([[0], np.cumsum(sizes)])
        cols = ('children_left', 'children_right', 'feature', 'threshold', 'value')
        self.trees = [tuple(d[f'{name}_{c}'][offs[i]:offs[i + 1]] for c in cols) for i in range(len(sizes))]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.full(len(X), self.init)
        rows = np.arange(len(X))
        for left, right, feat, thr, val in self.trees:
            node = np.zeros(len(X), int)
            active = left[node] >= 0
            while active.any():
                n = node[active]
                go_left = X[rows[active], feat[n]] <= thr[n]
                node[active] = np.where(go_left, left[n], right[n])
                active = left[node] >= 0
            out += self.lr * val[node]
        return out


def fit_gpr(X, y, test_size=0.2, split_seed=75):
    """Alternative surrogate (the Results section calls the model 'Gaussian process regression')."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=split_seed)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-2)
    ym, ys = y_tr.mean(), y_tr.std()
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=5, random_state=0)
    model.fit(X_tr, (y_tr - ym) / ys)

    class _Wrapped:
        def predict(self, Xq):
            return model.predict(Xq) * ys + ym

    w = _Wrapped()
    return w, dict(r2_train=r2_score(y_tr, w.predict(X_tr)), r2_test=r2_score(y_te, w.predict(X_te)),
                   split=(X_tr, X_te, y_tr, y_te))


def average_hv(Y, ref):
    """Eq. (16): mean over the non-dominated solutions of prod_k (y_k - ref_k); Y are maximised objectives."""
    Y = np.asarray(Y, float)
    nd = non_dominated_fronts(-Y)[0]
    return float(np.mean(np.prod(Y[nd] - ref[None, :], axis=1)))


def run_nsga2(models, X_ref, Y_ref, bounds=(-2.0, 2.0), pop_size=10, n_gen=500, seed=212):
    """Maximise all surrogate predictions with NSGA-II; returns result dict incl. Delta-HV per generation."""
    n_var = X_ref.shape[1]
    xl, xu = np.full(n_var, bounds[0]), np.full(n_var, bounds[1])

    def evaluate(X):
        return -np.stack([m.predict(X) for m in models], axis=1)

    res = nsga2(evaluate, xl, xu, pop_size=pop_size, n_gen=n_gen, seed=seed)
    ref = Y_ref.min(axis=0)
    hv0 = average_hv(Y_ref, ref)
    dhv = np.array([average_hv(-F, ref) - hv0 for _, F in res['history']])
    res.update(delta_hv=dhv, hv0=hv0, ref=ref)
    return res


def wasserstein_selection(z_patches, n_select=10):
    """Rank the 80 generated patches by the 1-D Wasserstein distance between each patch's latent values and the
    pooled latent distribution of the candidate (official notebook, 'Candidate microstructure selection')."""
    pooled = z_patches.ravel()
    wd = np.array([wasserstein_distance(pooled, z_patches[i]) for i in range(len(z_patches))])
    return np.argsort(wd)[:n_select], wd
