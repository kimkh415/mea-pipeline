"""Cluster network bursts into families (port of cluster_neuronal_events.m).

D = w_scalar * Euclid(z-scored metrics) + w_population * Jaccard(unit sets); classical MDS, Ward, silhouette k.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

N_SCALAR = 5
W_SCALAR = 0.35
W_POPULATION = 0.65
MAX_K = 4
LINKAGE_METHOD = "ward"
ALPHA = 0.05
MIN_SIZE_CLUSTER = 0
SIL_THRESHOLD = 0.40
MIN_DIFF = 0.10


def zscore(x, nan_aware=True):
    """Column-wise z-score; NaN/inf set to 0.

    nan_aware=False reproduces MATLAB, where one NaN zeroes the whole column.
    """
    x = np.asarray(x, float)
    if nan_aware:
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(x, axis=0)
            # <2 observations: treat column as constant
            n_ok = np.sum(np.isfinite(x), axis=0)
            sd = np.where(n_ok > 1, np.nanstd(x, axis=0, ddof=1), 0.0)
    else:
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def scale_distances(d, mode="robust", pct=95.0):
    """Scale distances to [0,1]: 'robust' divides by the pct percentile and clips; 'legacy_max' by the max."""
    d = np.asarray(d, float)
    if d.size == 0:
        return d
    if mode == "legacy_max":
        return d / (d.max() + np.finfo(float).eps)
    s = np.percentile(d, pct)
    if s <= 0:
        s = d.max() + np.finfo(float).eps
    return np.clip(d / s, 0.0, 1.0)


def jaccard_distances(pop_binary, empty_pair="zero"):
    """Pairwise Jaccard distance; two empty sets get 0 ('zero') or 1 ('legacy_one', MATLAB)."""
    b = np.asarray(pop_binary, bool)
    n = b.shape[0]
    if n < 2:
        return np.zeros(0)
    d = pdist(b.astype(float), "jaccard")
    # SciPy returns 0 (not NaN) for empty unions, so find empty pairs explicitly
    empty = ~b.any(axis=1)
    if not empty.any():
        return np.nan_to_num(d, nan=0.0)
    iu = np.triu_indices(n, 1)
    both_empty = empty[iu[0]] & empty[iu[1]]
    fill = 1.0 if empty_pair == "legacy_one" else 0.0
    d = np.nan_to_num(d, nan=fill)
    d[both_empty] = fill
    return d


def composite_distance(scalar_metrics, pop_binary, w_scalar=W_SCALAR,
                       w_population=W_POPULATION, nan_aware=True,
                       scale_mode="robust", empty_pair="zero"):
    """Weighted sum of scalar and Jaccard distances (condensed); also returns both blocks."""
    z = zscore(scalar_metrics, nan_aware)
    d_sc = scale_distances(pdist(z, "euclidean"), scale_mode)
    d_pop = jaccard_distances(pop_binary, empty_pair)
    return w_scalar * d_sc + w_population * d_pop, d_sc, d_pop


def cmdscale(d_square):
    """Classical MDS; returns coordinates (positive-eigenvalue dims only) and all eigenvalues, descending."""
    d = np.asarray(d_square, float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j.dot(d ** 2).dot(j)
    b = (b + b.T) / 2.0
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-10
    y = vecs[:, keep] * np.sqrt(vals[keep])
    return y, vals


def silhouette(x, labels):
    """Mean silhouette over all points; NaN when there is only one cluster."""
    x = np.asarray(x, float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2 or x.shape[0] < 3:
        return np.nan
    d = squareform(pdist(x, "euclidean"))
    s = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            s[i] = 0.0
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def _embed_and_cluster(scalar_metrics, pop_binary, max_k=MAX_K, linkage_method=LINKAGE_METHOD, **kw):
    d_comp, _, _ = composite_distance(scalar_metrics, pop_binary, **kw)
    n = scalar_metrics.shape[0]
    if n < 3:
        return np.ones(n, int), np.full(max_k + 1, np.nan), np.zeros((n, 1)), 1
    y, _ = cmdscale(squareform(d_comp))
    if y.shape[1] == 0:
        y = np.zeros((n, 1))
    z = linkage(y, linkage_method)
    sil = np.full(max_k + 1, np.nan)
    for k in range(2, min(max_k, n - 1) + 1):
        sil[k] = silhouette(y, fcluster(z, k, criterion="maxclust"))
    best = int(np.nanargmax(sil)) if np.isfinite(sil).any() else 1
    return fcluster(z, best, criterion="maxclust"), sil, y, best


def permutation_test(scalar_metrics, pop_binary, rng, n_perm=1000, max_k=MAX_K,
                     plus_one=True, linkage_method=LINKAGE_METHOD, **kw):
    """Permutation test of the silhouette per k, shuffling scalar and unit-set rows independently.

    p = (1 + #{perm >= real}) / (1 + n_valid) with plus_one, else hits / n_valid.
    """
    n = scalar_metrics.shape[0]
    _, real_sil, _, _ = _embed_and_cluster(scalar_metrics, pop_binary, max_k,
                                           linkage_method, **kw)
    perm = np.full((n_perm, max_k + 1), np.nan)
    for i in range(n_perm):
        sm = scalar_metrics[rng.permutation(n)]
        pb = pop_binary[rng.permutation(n)]
        _, s, _, _ = _embed_and_cluster(sm, pb, max_k, linkage_method, **kw)
        perm[i] = s
    p = np.full(max_k + 1, np.nan)
    n_valid = np.zeros(max_k + 1, int)
    for k in range(2, max_k + 1):
        col = perm[:, k]
        ok = np.isfinite(col)
        n_valid[k] = int(ok.sum())
        if not np.isfinite(real_sil[k]) or n_valid[k] == 0:
            continue
        hits = int((col[ok] >= real_sil[k]).sum())
        p[k] = (1 + hits) / (1 + n_valid[k]) if plus_one else hits / n_valid[k]
    return {"p_values": p, "real_silhouette": real_sil,
            "perm_silhouette": perm, "n_valid": n_valid}


def bootstrap_optimal_k(scalar_metrics, pop_binary, rng, n_boot=200,
                        max_k=MAX_K, linkage_method=LINKAGE_METHOD, **kw):
    """How often each k wins the silhouette across resampled event sets."""
    n = scalar_metrics.shape[0]
    votes = np.zeros(max_k + 1, int)
    sil_sum = np.zeros(max_k + 1)
    sil_n = np.zeros(max_k + 1, int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(idx).size < 3:
            continue
        _, s, _, best = _embed_and_cluster(scalar_metrics[idx], pop_binary[idx],
                                           max_k, linkage_method, **kw)
        votes[best] += 1
        for k in range(2, max_k + 1):
            if np.isfinite(s[k]):
                sil_sum[k] += s[k]; sil_n[k] += 1
    total = max(votes.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil = np.where(sil_n > 0, sil_sum / np.maximum(sil_n, 1), np.nan)
    return {"vote_ratio": votes / total, "votes": votes, "silhouette": sil,
            "optimal": int(np.argmax(votes)) if votes.any() else 1}


def cluster_events(scalar_metrics, pop_binary, rng, n_perm=1000, n_boot=200,
                   max_k=MAX_K, alpha=ALPHA, sil_threshold=SIL_THRESHOLD,
                   has_secondary=False, linkage_method=LINKAGE_METHOD,
                   plus_one=True, **kw):
    """Assign families: labels, k, and clusteringQuality 0/1/2.

    2: silhouette-optimal k has p < alpha; 1: smallest k with p < alpha_sig and sil >= sil_sig
    (alpha/4, 1.2x threshold when there are no secondaries); 0: one family.
    """
    n = scalar_metrics.shape[0]
    labels, sil, y, best_k = _embed_and_cluster(scalar_metrics, pop_binary, max_k,
                                                linkage_method, **kw)
    out = {"label": labels, "n_cluster": int(best_k), "silhouette": sil,
           "coords": y, "clusteringQuality": 0}
    if best_k <= 1 or n < 3:
        out.update(label=np.ones(n, int), n_cluster=1)
        return out

    boot = bootstrap_optimal_k(scalar_metrics, pop_binary, rng, n_boot, max_k,
                               linkage_method=linkage_method, **kw)
    perm = permutation_test(scalar_metrics, pop_binary, rng, n_perm, max_k,
                            plus_one=plus_one, linkage_method=linkage_method, **kw)
    out["bootstrap"] = boot
    out["permutation"] = perm

    alpha_sig = alpha if has_secondary else alpha / 4
    sil_sig = sil_threshold if has_secondary else sil_threshold * 1.2

    optimal_k, quality = 1, 0
    if np.isfinite(perm["p_values"][best_k]) and perm["p_values"][best_k] < alpha:
        optimal_k, quality = best_k, 2
    else:
        cand = [k for k in range(2, max_k + 1)
                if np.isfinite(perm["p_values"][k]) and perm["p_values"][k] < alpha_sig]
        if cand:
            k = cand[0]
            if np.isfinite(sil[k]) and sil[k] >= sil_sig:
                optimal_k, quality = k, 1
    if optimal_k == 1:
        out.update(label=np.ones(n, int), n_cluster=1, clusteringQuality=0)
    else:
        z = linkage(y, linkage_method)
        out.update(label=fcluster(z, optimal_k, criterion="maxclust"),
                   n_cluster=int(optimal_k), clusteringQuality=int(quality))
    return out
