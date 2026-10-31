"""Ranking metrics, cluster bootstrap confidence intervals and paired tests."""
import numpy as np
from scipy import stats


def rank_of(true, ranked):
    """1-based rank of `true` in the ranked list, or np.inf when absent."""
    try:
        return ranked.index(true) + 1
    except ValueError:
        return np.inf


def summarise_ranks(ranks, ks=(1, 5, 10)):
    r = np.asarray(ranks, dtype=float)
    out = {f"R@{k}": float(np.mean(r <= k)) for k in ks}
    out["MRR"] = float(np.mean(np.where(np.isfinite(r), 1.0 / r, 0.0)))
    out["n"] = int(len(r))
    return out


def cluster_bootstrap(values, clusters, stat=np.mean, B=2000, seed=2026, alpha=0.05):
    """Percentile CI that resamples whole clusters (projects), since requirements of one project are correlated."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        bs[b] = stat(values[idx])
    return float(stat(values)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_cluster_bootstrap_diff(a, b, clusters, B=2000, seed=2026):
    """Difference in means (a - b) with cluster bootstrap CI and two-sided bootstrap p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    est, lo, hi = cluster_bootstrap(d, clusters, B=B, seed=seed)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed + 1)
    centred = d - d.mean()
    cnt = 0
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        if abs(centred[idx].mean()) >= abs(d.mean()):
            cnt += 1
    return est, lo, hi, (cnt + 1) / (B + 1)


def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return n10, n01, p


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def reciprocal_rank_fusion(rankings, k=60, top=None):
    """rankings: list of ranked lists of ids. Returns fused ranked list."""
    score = {}
    for rl in rankings:
        for r, x in enumerate(rl):
            score[x] = score.get(x, 0.0) + 1.0 / (k + r + 1)
    fused = sorted(score, key=lambda x: -score[x])
    return fused[:top] if top else fused
