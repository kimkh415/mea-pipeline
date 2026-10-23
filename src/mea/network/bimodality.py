"""Bimodality test on inter-peak distances; drops echo peaks (port of test_bimodality.m)."""
from __future__ import annotations
import numpy as np
from scipy import stats

THRESHOLD_BC = 0.555
N_BURST_MIN = 8
MIN_DISTANCE = 4.0


def _norm_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _fit_1(x):
    """Single-Gaussian MLE fit; returns (log-likelihood, [mu])."""
    mu = float(x.mean())
    var = float(x.var()) or 1e-12
    ll = float(np.log(_norm_pdf(x, mu, var) + 1e-300).sum())
    return ll, np.array([mu])


def _fit_2(x, rng, n_replicates, max_iter, reg):
    """Two-component 1-D Gaussian mixture by EM; best log-likelihood over random restarts.

    Mirrors MATLAB fitgmdist with Replicates and RegularizationValue; seeded from rng.
    """
    n = x.size
    rng = rng if rng is not None else np.random.default_rng(0)
    best = (-np.inf, np.array([np.nan, np.nan]))
    for _ in range(max(1, n_replicates)):
        mu = rng.choice(x, size=2, replace=False).astype(float)
        var = np.full(2, float(x.var()) + reg)
        w = np.array([0.5, 0.5])
        ll_old = -np.inf
        for _ in range(max_iter):
            p = np.stack([w[k] * _norm_pdf(x, mu[k], var[k]) for k in range(2)])
            tot = p.sum(0) + 1e-300
            r = p / tot
            ll = float(np.log(tot).sum())
            nk = r.sum(1) + 1e-300
            w = nk / n
            mu = (r * x).sum(1) / nk
            var = (r * (x - mu[:, None]) ** 2).sum(1) / nk + reg
            if abs(ll - ll_old) < 1e-8 * max(1.0, abs(ll)):
                break
            ll_old = ll
        if ll > best[0]:
            best = (ll, mu.copy())
    return best


def bimodality(data, rng=None, n_replicates=5, max_iter=500, reg=1e-5,
               threshold_bc=THRESHOLD_BC):
    """Bimodality coefficient plus a 1-vs-2 component GMM test (AIC and BIC).

    Uses unbiased skew/kurtosis (MATLAB defaults are biased); scipy kurtosis is already excess.
    """
    x = np.asarray(data, float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    out = {"n": n}
    if n < 4:
        out.update(bimodality_coefficient=np.nan, BC_suggests_bimodal=False,
                   GMM_suggests_bimodal=False, final=False, mu=(np.nan, np.nan))
        return out

    s = float(stats.skew(x, bias=False))
    k_excess = float(stats.kurtosis(x, fisher=True, bias=False))
    bc = (s ** 2 + 1) / (k_excess + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    out["bimodality_coefficient"] = bc
    out["BC_suggests_bimodal"] = bool(bc > threshold_bc)

    ll1, mu1 = _fit_1(x)
    ll2, mu2 = _fit_2(x, rng, n_replicates, max_iter, reg)
    # free parameters: 2 for one component, 5 for two
    aic = lambda ll, k: 2 * k - 2 * ll
    bic = lambda ll, k: k * np.log(n) - 2 * ll
    out["AIC_1comp"], out["AIC_2comp"] = aic(ll1, 2), aic(ll2, 5)
    out["BIC_1comp"], out["BIC_2comp"] = bic(ll1, 2), bic(ll2, 5)
    out["GMM_suggests_bimodal"] = bool(out["AIC_2comp"] < out["AIC_1comp"]
                                       and out["BIC_2comp"] < out["BIC_1comp"])
    out["mu"] = tuple(np.sort(mu2))
    out["final"] = bool(out["BC_suggests_bimodal"] and out["GMM_suggests_bimodal"])
    return out


def drop_echo_peaks(bursts, rng=None, n_burst_min=N_BURST_MIN,
                    min_distance=MIN_DISTANCE, threshold_bc=THRESHOLD_BC):
    """Drop peaks whose distance to the previous peak falls in the short mode.

    Only when n > n_burst_min, distances are bimodal, and modes differ by > min_distance x.
    """
    n = len(bursts)
    if n <= n_burst_min:
        return bursts, None
    dist = np.array([bursts[i]["peak"] - bursts[i - 1]["peak"] for i in range(1, n)])
    bi = bimodality(dist, rng, threshold_bc=threshold_bc)
    d1, d2 = bi["mu"]
    if not (bi["final"] and d2 > d1 * min_distance):
        return bursts, bi
    keep = [True] * n
    for i in range(n - 1, 0, -1):
        if abs(dist[i - 1] - d1) < abs(dist[i - 1] - d2):
            keep[i] = False
    return [b for b, k in zip(bursts, keep) if k], bi
