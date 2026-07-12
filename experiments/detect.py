"""Detector evaluation: real (label 0) vs fabricated (label 1).

Reports, and checks against the pre-registered pilot gates in brain/REDTEAM.md:
  per-test ROC-AUC (all, and open-models-only)      -> G-signal
  terminal-digit effect size per frontier model      -> G-frontier
  first-digit AUC and composite AUC by condition      -> G-evasion
  composite grouped-CV AUC                             -> overall

At pilot n the AUC point estimates are noisy, so the gates lean on large
effect sizes (last-digit entropy gap, round-number-rate gap), not thin AUCs.

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

OPEN = ("small_open", "large_open")


def _load(fn):
    return json.loads((SUMMARIES / fn).read_text(encoding="utf-8"))


def _matrix(records):
    keys = dt.FEATURE_KEYS
    X = [[r["features"].get(k, np.nan) for k in keys] for r in records]
    return np.array(X, float), keys


def _auc(y, s):
    try:
        a = roc_auc_score(y, s)
        return float(max(a, 1 - a))
    except ValueError:
        return float("nan")


def main(fab_fn, real_fn, out):
    real = _load(real_fn)["tables"]
    fabdata = _load(fab_fn)
    fab = fabdata["records"]

    Xr, keys = _matrix(real)
    Xf, _ = _matrix(fab)
    col_med = np.nanmedian(np.vstack([Xr, Xf]), axis=0)
    Xr = np.where(np.isfinite(Xr), Xr, col_med)
    Xf = np.where(np.isfinite(Xf), Xf, col_med)

    # ---- per-test AUC: all fabricated, and open-models-only (G-signal) ----
    y_all = np.array([0] * len(Xr) + [1] * len(Xf))
    X_all = np.vstack([Xr, Xf])
    per_test_all = {k: round(_auc(y_all, X_all[:, j]), 3) for j, k in enumerate(keys)}

    open_idx = [i for i, r in enumerate(fab) if r["category"] in OPEN]
    Xo = Xf[open_idx]
    y_open = np.array([0] * len(Xr) + [1] * len(Xo))
    Xopen = np.vstack([Xr, Xo])
    per_test_open = {k: round(_auc(y_open, Xopen[:, j]), 3) for j, k in enumerate(keys)}
    terminal_keys = ["lastdigit_entropy_bits", "round_end0_rate", "lastdigit_mad", "round_5or0_rate"]
    best_terminal_open = max(per_test_open[k] for k in terminal_keys)

    # ---- frontier survival (G-frontier): last-digit entropy + round rate vs real ----
    ent_j = keys.index("lastdigit_entropy_bits")
    end0_j = keys.index("round_end0_rate")
    real_ent, real_end0 = float(np.mean(Xr[:, ent_j])), float(np.mean(Xr[:, end0_j]))
    frontier_survival = {}
    for tier in sorted(set(r["tier"] for r in fab if r["category"] == "frontier")):
        # use non-terminal-targeting conditions so C3 evasion does not mask the native signal
        idx = [i for i, r in enumerate(fab) if r["tier"] == tier and r["condition"] in ("C0", "C1")]
        if not idx:
            continue
        ent = float(np.mean(Xf[idx, ent_j]))
        end0 = float(np.mean(Xf[idx, end0_j]))
        frontier_survival[tier] = {
            "n": len(idx),
            "lastdigit_entropy": round(ent, 3), "real_entropy": round(real_ent, 3),
            "entropy_gap": round(real_ent - ent, 3),
            "round_end0_rate": round(end0, 3), "real_end0": round(real_end0, 3),
            "end0_gap": round(end0 - real_end0, 3),
            "signal_present": bool((real_ent - ent) > 0.3 or (end0 - real_end0) > 0.1),
        }

    # ---- composite grouped-CV AUC (all) ----
    groups = np.array([f"real:{i}" for i in range(len(Xr))] + [r["table"] for r in fab])
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y_all))
    for tr, te in GroupKFold(n_splits=n_splits).split(X_all, y_all, groups):
        sc = StandardScaler().fit(X_all[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X_all[tr]), y_all[tr])
        oof[te] = clf.predict_proba(sc.transform(X_all[te]))[:, 1]
    composite_auc = round(_auc(y_all, oof), 3)

    # ---- evasion by condition (G-evasion): first-digit AUC + composite score ----
    benford_j = keys.index("benford_mad")
    fab_oof = oof[len(Xr):]
    by_cond = {}
    for cond in ("C0", "C1", "C2", "C3"):
        idx = [i for i, r in enumerate(fab) if r["condition"] == cond]
        if not idx:
            continue
        yy = np.array([0] * len(Xr) + [1] * len(idx))
        fd_auc = _auc(yy, np.concatenate([Xr[:, benford_j], Xf[idx, benford_j]]))
        by_cond[cond] = {
            "n": len(idx),
            "firstdigit_benford_auc": round(fd_auc, 3),
            "composite_mean_score": round(float(np.mean(fab_oof[idx])), 3),
        }

    # ---- gate evaluation ----
    g_signal = best_terminal_open >= 0.80
    g_frontier = any(v["signal_present"] for v in frontier_survival.values())
    c0 = by_cond.get("C0", {}).get("firstdigit_benford_auc", float("nan"))
    c2 = by_cond.get("C2", {}).get("firstdigit_benford_auc", float("nan"))
    g_evasion = bool(np.isfinite(c0) and np.isfinite(c2) and (c0 - c2) > 0.05
                     and by_cond.get("C2", {}).get("composite_mean_score", 0) > 0.5)

    summary = {
        "n_real": len(Xr), "n_fab": len(Xf),
        "refusal_rate_by_cell": fabdata.get("refusal_rate_by_cell", {}),
        "n_refusals": fabdata.get("n_refusals", 0),
        "per_test_auc_all": dict(sorted(per_test_all.items(), key=lambda kv: -kv[1])),
        "per_test_auc_open_only": dict(sorted(per_test_open.items(), key=lambda kv: -kv[1])),
        "best_terminal_auc_open": round(best_terminal_open, 3),
        "composite_auc_all": composite_auc,
        "frontier_survival": frontier_survival,
        "evasion_by_condition": by_cond,
        "gates": {
            "G_signal_open>=0.80": bool(g_signal),
            "G_frontier_signal_survives": bool(g_frontier),
            "G_evasion_real": bool(g_evasion),
        },
    }
    (SUMMARIES / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"real={len(Xr)} fab={len(Xf)} refusals={summary['n_refusals']}")
    print("best terminal-digit AUC (open only):", summary["best_terminal_auc_open"])
    print("composite AUC (all):", composite_auc)
    print("frontier survival:", {k: v["signal_present"] for k, v in frontier_survival.items()})
    print("first-digit AUC by cond:", {c: v["firstdigit_benford_auc"] for c, v in by_cond.items()})
    print("GATES:", summary["gates"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fab", default="fabricated_pilot.json")
    ap.add_argument("--real", default="real_corpus.json")
    ap.add_argument("--out", default="detector_pilot.json")
    a = ap.parse_args()
    main(a.fab, a.real, a.out)
