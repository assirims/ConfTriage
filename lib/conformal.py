"""Distribution-free uncertainty quantification used to turn model scores into risk-controlled assistance.

* Split conformal prediction with the LAC score (Sadinle et al., 2019; Angelopoulos & Bates, 2023):
  prediction sets that contain the true label with probability >= 1 - alpha (marginal coverage).
* Mondrian (class-conditional) conformal prediction (Vovk et al., 2005): coverage >= 1 - alpha within every class.
* Risk-controlled automation with Learn-then-Test (Angelopoulos et al., 2021) and exact binomial bounds as in
  selection with guaranteed risk (Geifman & El-Yaniv, 2017): a confidence threshold such that, with probability
  >= 1 - delta over the calibration sample, the error rate among automatically accepted items is <= eps.
* Conformal risk control (Angelopoulos et al., 2024) for a bounded monotone loss, used (i) to guarantee that the
  expected share of items that are automatically labelled AND wrong is <= eps (risk-controlled automation), and
  (ii) to guarantee the expected per-requirement recall of retained trace links.
"""
import math

import numpy as np
from scipy import stats


def conformal_quantile(scores, alpha):
    """Finite-sample corrected (1 - alpha) quantile of calibration non-conformity scores; +inf if n is too small."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return np.inf
    return s[k - 1]


def lac_threshold(p_cal, y_cal, alpha):
    """LAC non-conformity score: 1 - probability of the true label."""
    return conformal_quantile(1.0 - p_cal[np.arange(len(y_cal)), y_cal], alpha)


def lac_sets(p, q):
    return (1.0 - p) <= q + 1e-12


def mondrian_thresholds(p_cal, y_cal, alpha, n_classes):
    q = np.full(n_classes, np.inf)
    for c in range(n_classes):
        m = y_cal == c
        if m.any():
            q[c] = conformal_quantile(1.0 - p_cal[m, c], alpha)
    return q


def mondrian_sets(p, q):
    return (1.0 - p) <= q[None, :] + 1e-12


def binom_upper(errors, n, level):
    """One-sided exact (Clopper-Pearson) upper confidence bound for an error rate, holding with prob. >= 1 - level."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(stats.beta.ppf(1 - level, errors + 1, n - errors))


LTT_GRID = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])   # fixed in advance


def ltt_threshold(conf_cal, correct_cal, eps, delta, grid=LTT_GRID):
    """Learn-then-Test (Angelopoulos et al., 2021) with Bonferroni correction over a fixed grid of confidence
    thresholds. For each threshold the null hypothesis 'error rate among accepted items > eps' is tested with the exact
    binomial (Clopper-Pearson) bound at level delta / |grid|. Returns the lowest threshold whose null is rejected
    (largest automation), or +inf if none is. With probability >= 1 - delta over the calibration sample, every
    returned threshold has selective error <= eps."""
    conf_cal = np.asarray(conf_cal, float)
    correct_cal = np.asarray(correct_cal, bool)
    level = delta / len(grid)
    best = np.inf
    for lam in grid:
        acc = conf_cal >= lam
        if binom_upper(int((acc & ~correct_cal).sum()), int(acc.sum()), level) < eps:
            best = min(best, lam)
    return best


def crc_threshold(loss_matrix, lambdas, alpha, B=1.0):
    """Conformal risk control. `lambdas` are ordered from the least to the most conservative setting (here: from high
    to low score thresholds, i.e. from few to many retained links); loss_matrix[i, j] is the loss of calibration
    unit i at lambdas[j] and must be non-increasing along that order. Returns the index of the least conservative
    setting with (n / (n + 1)) * mean_loss + B / (n + 1) <= alpha, which guarantees E[loss] <= alpha on a new
    exchangeable unit."""
    L = np.asarray(loss_matrix, float)
    n = L.shape[0]
    adj = (n / (n + 1.0)) * L.mean(0) + B / (n + 1.0)
    ok = np.where(adj <= alpha)[0]
    return int(ok[0]) if len(ok) else len(lambdas) - 1


AUTO_GRID = np.concatenate([np.linspace(0.0, 1.0, 1001), [np.inf]])     # ascending: from 'accept all' to 'accept none'


def crc_automation_threshold(conf_cal, wrong_cal, eps, grid=AUTO_GRID):
    """Smallest confidence threshold such that E[1{confidence >= threshold and prediction wrong}] <= eps on a new
    exchangeable item (conformal risk control; the loss is non-increasing in the threshold)."""
    conf_cal = np.asarray(conf_cal, float)
    wrong_cal = np.asarray(wrong_cal, bool)
    L = (conf_cal[:, None] >= grid[None, :]) & wrong_cal[:, None]
    return grid[crc_threshold(L.astype(float), grid, eps)]
