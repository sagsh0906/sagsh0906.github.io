"""A compact NSGA-II (Deb et al., 2002) mirroring the pymoo defaults used by the official notebook.

pymoo's ``NSGA2(pop_size=10)`` uses random initial sampling, binary tournament selection on (rank, crowding),
simulated binary crossover (SBX, eta=15, p_c=0.9, per-variable exchange probability 0.5), polynomial mutation
(eta=20, per-variable probability 1/n_var) and rank-and-crowding survival.  The same operators are implemented here
so that the optimisation can be run without pymoo and every generation can be recorded (for the Delta-HV curves).
All objectives are minimised.
"""
import numpy as np


def non_dominated_fronts(F):
    """Fast non-dominated sorting. Returns a list of index arrays (front 0 first)."""
    n = len(F)
    dominates = (np.all(F[:, None, :] <= F[None, :, :], axis=2) & np.any(F[:, None, :] < F[None, :, :], axis=2))
    n_dominated_by = dominates.sum(axis=0)
    fronts = []
    current = np.nonzero(n_dominated_by == 0)[0]
    assigned = np.zeros(n, bool)
    while len(current):
        fronts.append(current)
        assigned[current] = True
        n_dominated_by = n_dominated_by - dominates[current].sum(axis=0)
        current = np.nonzero((n_dominated_by == 0) & ~assigned)[0]
    return fronts


def crowding_distance(F):
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind='stable')
        f = F[order, j]
        span = f[-1] - f[0]
        cd[order[0]] = cd[order[-1]] = np.inf
        if span > 0:
            cd[order[1:-1]] += (f[2:] - f[:-2]) / span
    return cd


def rank_and_crowding(F):
    rank = np.empty(len(F), int)
    crowd = np.empty(len(F))
    for r, fr in enumerate(non_dominated_fronts(F)):
        rank[fr] = r
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd


def _sbx(p1, p2, xl, xu, rng, eta=15.0, prob=0.9, prob_var=0.5):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > prob_var or abs(p1[i] - p2[i]) < 1e-14:
            continue
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        lo, hi = xl[i], xu[i]
        rand = rng.random()

        def beta_q(beta):
            alpha = 2.0 - beta ** -(eta + 1.0)
            if rand <= 1.0 / alpha:
                return (rand * alpha) ** (1.0 / (eta + 1.0))
            return (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

        b1 = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
        b2 = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
        v1 = 0.5 * ((y1 + y2) - beta_q(b1) * (y2 - y1))
        v2 = 0.5 * ((y1 + y2) + beta_q(b2) * (y2 - y1))
        v1, v2 = np.clip(v1, lo, hi), np.clip(v2, lo, hi)
        if rng.random() < 0.5:
            v1, v2 = v2, v1
        c1[i], c2[i] = v1, v2
    return c1, c2


def _polynomial_mutation(x, xl, xu, rng, eta=20.0, prob_var=None):
    x = x.copy()
    n = len(x)
    prob_var = 1.0 / n if prob_var is None else prob_var
    for i in range(n):
        if rng.random() > prob_var:
            continue
        lo, hi = xl[i], xu[i]
        d1, d2 = (x[i] - lo) / (hi - lo), (hi - x[i]) / (hi - lo)
        r = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if r < 0.5:
            xy = 1.0 - d1
            val = 2.0 * r + (1.0 - 2.0 * r) * xy ** (eta + 1.0)
            dq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - d2
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy ** (eta + 1.0)
            dq = 1.0 - val ** mut_pow
        x[i] = np.clip(x[i] + dq * (hi - lo), lo, hi)
    return x


def _tournament(rank, crowd, rng, n):
    idx = rng.integers(0, len(rank), size=(n, 2))
    a, b = idx[:, 0], idx[:, 1]
    better_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    pick = np.where(better_a, a, b)
    coin = rng.random(n) < 0.5
    return np.where(tie & coin, a, pick)


def nsga2(evaluate, xl, xu, pop_size=10, n_gen=500, seed=0, callback=None):
    """Run NSGA-II. ``evaluate(X) -> F`` (batch, minimised). Returns dict with final X, F and full history."""
    rng = np.random.default_rng(seed)
    xl, xu = np.asarray(xl, float), np.asarray(xu, float)
    X = xl + rng.random((pop_size, len(xl))) * (xu - xl)
    F = evaluate(X)
    history = [(X.copy(), F.copy())]
    for gen in range(1, n_gen):
        rank, crowd = rank_and_crowding(F)
        parents = _tournament(rank, crowd, rng, pop_size)
        kids = []
        for k in range(0, pop_size, 2):
            p1, p2 = X[parents[k]], X[parents[(k + 1) % pop_size]]
            c1, c2 = _sbx(p1, p2, xl, xu, rng)
            kids += [_polynomial_mutation(c1, xl, xu, rng), _polynomial_mutation(c2, xl, xu, rng)]
        Xo = np.array(kids[:pop_size])
        Fo = evaluate(Xo)
        Xa, Fa = np.vstack([X, Xo]), np.vstack([F, Fo])
        keep = []
        for fr in non_dominated_fronts(Fa):
            if len(keep) + len(fr) <= pop_size:
                keep += fr.tolist()
            else:
                cd = crowding_distance(Fa[fr])
                keep += fr[np.argsort(-cd, kind='stable')[:pop_size - len(keep)]].tolist()
                break
        X, F = Xa[keep], Fa[keep]
        history.append((X.copy(), F.copy()))
        if callback is not None:
            callback(gen, X, F)
    return dict(X=X, F=F, history=history)
