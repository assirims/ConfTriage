"""Simulation checks of the conformal procedures (run: python tests/test_conformal.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.conformal import crc_threshold, lac_sets, lac_threshold, ltt_threshold, mondrian_sets, mondrian_thresholds  # noqa: E402


def sim(n, K, rng):
    logits = rng.normal(size=(n, K)) * 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=pi) for pi in p])          # calibrated model
    return p, y


def main():
    rng = np.random.default_rng(0)
    K, alpha, cov, mcov, sel_ok, autom = 6, 0.1, [], [], [], []
    for _ in range(300):
        pc, yc = sim(300, K, rng)
        pt, yt = sim(300, K, rng)
        q = lac_threshold(pc, yc, alpha)
        cov.append(lac_sets(pt, q)[np.arange(len(yt)), yt].mean())
        qm = mondrian_thresholds(pc, yc, alpha, K)
        mcov.append(mondrian_sets(pt, qm)[np.arange(len(yt)), yt].mean())
        lam = ltt_threshold(pc.max(1), pc.argmax(1) == yc, eps=0.1, delta=0.1)
        autom.append((pt.max(1) >= lam).mean())
        acc = pt.max(1) >= lam
        sel_ok.append(acc.sum() == 0 or (pt.argmax(1)[acc] != yt[acc]).mean() <= 0.1 + 0.05)
    print(f"LAC coverage {np.mean(cov):.3f} (target >= {1 - alpha})")
    print(f"Mondrian coverage {np.mean(mcov):.3f} (target >= {1 - alpha})")
    print(f"LTT: share of runs with test selective error <= eps + 0.05: {np.mean(sel_ok):.3f}; mean automation {np.mean(autom):.3f}")
    assert np.mean(cov) >= 1 - alpha - 0.01 and np.mean(mcov) >= 1 - alpha - 0.01 and np.mean(sel_ok) >= 0.9
    # conformal risk control on a recall-type loss
    lam = np.linspace(1, 0, 101)
    risk = []
    for _ in range(300):
        def units(n):
            s = rng.uniform(size=(n, 5))
            return np.array([[1 - (s[i] >= l).mean() for l in lam] for i in range(n)])
        j = crc_threshold(units(100), lam, 0.2)
        risk.append(units(200)[:, j].mean())
    print(f"CRC mean loss {np.mean(risk):.3f} (target <= 0.2)")
    assert np.mean(risk) <= 0.2 + 0.01
    print("all conformal tests passed")


if __name__ == "__main__":
    main()


def test_crc_automation():
    from lib.conformal import crc_automation_threshold
    rng = np.random.default_rng(5)
    risks = []
    for _ in range(300):
        pc, yc = sim(250, 6, rng)
        pt, yt = sim(250, 6, rng)
        lam = crc_automation_threshold(pc.max(1), pc.argmax(1) != yc, 0.05)
        risks.append(((pt.max(1) >= lam) & (pt.argmax(1) != yt)).mean())
    print(f"CRC automation: mean share auto-labelled and wrong {np.mean(risks):.4f} (target <= 0.05)")
    assert np.mean(risks) <= 0.05 + 0.005


if __name__ == "__main__":
    test_crc_automation()
