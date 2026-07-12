"""Detector evaluation: real (label 0) vs fabricated (label 1).

Reports per-test ROC-AUC and a composite (grouped logistic regression). Also
breaks evasion down by condition (D3/D6) and scale by tier (H3). Checks KC1.

All tables are capped to the same n (D1 matching) before feature use is already
handled upstream; here we just consume the feature vectors.

Usage: python detect.py --fab fabricated_pilot.json --out detector_pilot.json
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import digit_tests as dt
from common import SUMMARIES


def _load(fn):
    return json.loads((SUMMARIES / fn).read_text(encoding="utf-8"))


def _matrix(records):
    X, keys = [], dt.FEATURE_KEYS
    for r in records:
        f = r["features"]
        X.append([f.get(k, np.nan) for k in keys])
    return np.array(X, float), keys


def main(fab_fn, real_fn, out):
    real = _load(real_fn)["tables"]
    fab = _load(fab_fn)["records"]

    Xr, keys = _matrix(real)
    Xf, _ = _matrix(fab)
    X = np.vstack([Xr, Xf])
    y = np.array([0] * len(Xr) + [1] * len(Xf))
    # impute NaNs by column median (rare; from tiny-n tables)
    col_med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, col_med)

    # per-test AUC (direction-agnostic: report max(auc, 1-auc) with sign)
    per_test = {}
    for j, k in enumerate(keys):
        try:
            a = roc_auc_score(y, X[:, j])
        except ValueError:
            a = float("nan")
        per_test[k] = round(float(max(a, 1 - a)), 3)

    # composite: grouped CV so a table never spans train/test
    groups = np.array([f"real:{i}" for i in range(len(Xr))] +
                      [r["table"] for r in fab])
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    composite_auc = round(float(roc_auc_score(y, oof)), 3)

    # evasion breakdown: composite score of fabricated tables by condition
    fab_scores = oof[len(Xr):]
    by_cond, by_tier = {}, {}
    for r, s in zip(fab, fab_scores):
        by_cond.setdefault(r["condition"], []).append(s)
        by_tier.setdefault(r["tier"], []).append(s)
    # first-digit-only AUC per condition (to show C2 evades it)
    benford_j = keys.index("benford_mad")
    fd_by_cond = {}
    for cond in sorted(by_cond):
        idx = [i for i, r in enumerate(fab) if r["condition"] == cond]
        yy = np.array([0] * len(Xr) + [1] * len(idx))
        XX = np.concatenate([Xr[:, benford_j], Xf[idx, benford_j]])
        XX = np.where(np.isfinite(XX), np.nanmedian(XX), XX)
        try:
            a = roc_auc_score(yy, XX); fd_by_cond[cond] = round(float(max(a, 1 - a)), 3)
        except ValueError:
            fd_by_cond[cond] = float("nan")

    summary = {
        "n_real": len(Xr), "n_fab": len(Xf),
        "per_test_auc": dict(sorted(per_test.items(), key=lambda kv: -kv[1])),
        "composite_auc": composite_auc,
        "composite_mean_score_by_condition": {c: round(float(np.mean(v)), 3) for c, v in sorted(by_cond.items())},
        "composite_mean_score_by_tier": {t: round(float(np.mean(v)), 3) for t, v in sorted(by_tier.items())},
        "firstdigit_benford_auc_by_condition": fd_by_cond,
        "KC1_pass": bool(composite_auc >= 0.65 or max(per_test.values()) >= 0.60),
    }
    (SUMMARIES / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"real={len(Xr)} fab={len(Xf)}  composite AUC={composite_auc}")
    print("top per-test AUC:", list(summary["per_test_auc"].items())[:4])
    print("first-digit Benford AUC by condition:", fd_by_cond)
    print("composite score by tier (higher=more detectable):", summary["composite_mean_score_by_tier"])
    print("KC1_pass:", summary["KC1_pass"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fab", default="fabricated_pilot.json")
    ap.add_argument("--real", default="real_corpus.json")
    ap.add_argument("--out", default="detector_pilot.json")
    a = ap.parse_args()
    main(a.fab, a.real, a.out)
