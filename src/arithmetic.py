"""Pure arithmetic; importing this module performs no estimation."""
import numpy as np

Q = np.array([.1, .5, .9], dtype=np.float64)


def pinball(y, p):
    r = np.asarray(y, dtype=np.float64)[:, None] - np.asarray(p, dtype=np.float64)
    assert r.shape == (len(y), 3) and np.isfinite(r).all()
    return r * (Q - (r < 0))


def metric(y, p, w):
    return float(np.sum(np.asarray(w) * pinball(y, p).mean(axis=1)))


def point_width(p):
    p = np.asarray(p, dtype=np.float64)
    return p[:, 1], np.log1p(np.abs(p[:, 2] - p[:, 0]))


def capacity(score, weights, ties):
    """Largest whole-person weighted prefix at <=80%; fixed hash tie order."""
    order = np.lexsort((ties, score))
    c = np.cumsum(weights[order], dtype=np.float64)
    n = int(np.searchsorted(c, .8, side='right'))
    selected = order[:n]
    cov = float(c[n - 1]) if n else 0.
    next_w = float(weights[order[n]]) if n < len(order) else 0.
    assert cov <= .8 and (n == len(order) or cov + next_w > .8 - 1e-14)
    return selected, dict(n_selected=n, coverage=cov, next_person_weight=next_w,
                          shortfall=.8 - cov, whole_person_bound_pass=True)


def standardized(f, u, moments=None, weights=None):
    a = np.column_stack([f, u]).astype(np.float64)
    if moments is None:
        mu = np.sum(weights[:, None] * a, axis=0)
        variance = np.sum(weights[:, None] * (a - mu)**2, axis=0)
        variance[np.ptp(a, axis=0) == 0] = 0.  # exact constant column, despite summation roundoff
        moments = dict(mean=mu.tolist(), scale=np.sqrt(variance).tolist())
    mu, scale = np.array(moments['mean']), np.array(moments['scale'])
    z = np.zeros_like(a)
    good = scale > 0
    z[:, good] = (a[:, good] - mu[good]) / scale[good]
    return np.column_stack([np.ones(len(a)), z]), moments


def calibration_system(z, y, f, weights, prior):
    e = np.log1p(np.abs(y - f))
    penalty = np.diag([0., 1., 1.])
    a = z.T @ (weights[:, None] * z) + penalty
    b = z.T @ (weights * e) + penalty @ np.array(prior)
    return a, b


def calibration_solve(a, b):
    """One estimation unit; called only from terminal pilot diagnostics."""
    np.linalg.cholesky(a)
    beta = np.linalg.solve(a, b)
    residual = float(np.linalg.norm(a @ beta - b) / max(np.linalg.norm(b), np.finfo(float).tiny))
    assert np.isfinite(beta).all() and residual <= 1e-10
    return beta, residual
