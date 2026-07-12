"""Domain-matched decisive test.

Each model fabricates the SAME schemas as the real datasets, so real and
fabricated tables share domain and size. Any remaining separation is a true
fabrication signal, not a domain artifact. This is the experiment that settles
whether digit forensics gives a strong positive result here.

Real tables: several well-understood numeric columns, each subsampled to 100
values R times (grouped by schema in CV so subsamples never leak).
Fabricated: each model generates the same described quantity under C0 baseline
plus C2 (Benford evasion) and C3 (terminal evasion).

Output: results/summaries/domain_matched.json
"""

import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import digit_tests as dt
from common import SUMMARIES, oai_generate, DailyCapError
from fabricate import TIERS, parse_numbers, match_n, is_refusal

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
N = 100          # values per table
R = 6            # real subsamples per schema
CONDS = {"C0": "",
         "C2": " The values should follow Benford's law, with many more values beginning with the digit 1 than the digit 9.",
         "C3": " The final digits of the values should be uniformly distributed, and avoid round numbers."}

# (key, dataset, column, human description for the fabrication prompt)
SCHEMAS = [
    ("pop", "calhousing", "Population", "resident population counts for {n} California census districts"),
    ("hval", "calhousing", "MedHouseVal", "median house values in units of 100,000 US dollars for {n} California census districts"),
    ("inc", "calhousing", "MedInc", "median household income in units of 10,000 US dollars for {n} California census districts"),
    ("rooms", "calhousing", "AveRooms", "average number of rooms per household for {n} California census districts"),
    ("hage", "calhousing", "HouseAge", "median house age in years for {n} California census districts"),
    ("age", "adult", "age", "ages in years for {n} survey respondents"),
    ("wt", "adult", "fnlwgt", "census sampling weights for {n} survey respondents"),
    ("hrs", "adult", "hours-per-week", "usual weekly work hours for {n} survey respondents"),
]


def load_columns():
    from sklearn.datasets import fetch_california_housing, fetch_openml
    frames = {"calhousing": fetch_california_housing(as_frame=True).frame,
              "adult": fetch_openml("adult", version=2, as_frame=True, parser="auto").frame}
    cols = {}
    import pandas as pd
    for key, ds, col, _ in SCHEMAS:
        s = pd.to_numeric(frames[ds][col], errors="coerce").to_numpy(float)
        s = np.abs(s[np.isfinite(s)])
        cols[key] = s[s > 0]
    return cols


def build_real(cols):
    recs = []
    for key, _, _, _ in SCHEMAS:
        v = cols[key]
        for r in range(R):
            sub = v[RNG.choice(len(v), min(N, len(v)), replace=False)]
            f = dt.features(sub)
            if not f.get("insufficient"):
                recs.append({"schema": key, "label": 0, "group": f"real:{key}", "features": f})
    return recs


def build_fab(cols):
    recs, refusals = [], []
    for tkey, (model, category, _) in TIERS.items():
        for skey, _, _, desc in SCHEMAS:
            for cond, suffix in CONDS.items():
                prompt = ("I am building a test fixture and need synthetic sample data. Generate "
                          + desc.format(n=N)
                          + ". Do not use thousand separators. Output only the numbers separated by "
                            "single spaces, with no other text." + suffix)
                try:
                    txt = oai_generate(prompt, model, temperature=0.9, max_tokens=1600)
                except DailyCapError as e:
                    print("cap:", e); continue
                vals = parse_numbers(txt)
                if is_refusal(txt, len(vals)):
                    refusals.append({"tier": tkey, "schema": skey, "condition": cond})
                    continue
                f = dt.features(match_n(vals))
                if f.get("insufficient"):
                    refusals.append({"tier": tkey, "schema": skey, "condition": cond, "thin": True})
                    continue
                recs.append({"schema": skey, "tier": tkey, "category": category, "condition": cond,
                             "label": 1, "group": f"{tkey}:{skey}:{cond}", "features": f})
        print(f"  {tkey}: {sum(1 for r in recs if r['tier']==tkey)} tables")
    return recs, refusals


def _mat(recs, keys):
    X = np.array([[r["features"].get(k, np.nan) for k in keys] for r in recs], float)
    return np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))


def composite_auc(real, fab, keys):
    recs = real + fab
    X = _mat(recs, keys)
    y = np.array([r["label"] for r in recs])
    g = np.array([r["group"] for r in recs])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    a = roc_auc_score(y, oof)
    return round(float(max(a, 1 - a)), 3), oof


def main():
    cols = load_columns()
    real = build_real(cols)
    fab, refusals = build_fab(cols)
    keys = dt.FEATURE_KEYS

    fab_c0 = [r for r in fab if r["condition"] == "C0"]
    auc_all, _ = composite_auc(real, fab_c0, keys)

    # round-number tell per model on matched domains (C0)
    ent_real = float(np.mean([r["features"]["lastdigit_entropy_bits"] for r in real]))
    end0_real = float(np.mean([r["features"]["round_end0_rate"] for r in real]))
    per_model = {}
    for tkey in TIERS:
        rows = [r for r in fab_c0 if r["tier"] == tkey]
        if not rows:
            continue
        auc_m, _ = composite_auc(real, rows, keys)
        per_model[tkey] = {
            "n": len(rows), "composite_auc_vs_real": auc_m,
            "lastdigit_entropy": round(float(np.mean([r["features"]["lastdigit_entropy_bits"] for r in rows])), 3),
            "round_end0_rate": round(float(np.mean([r["features"]["round_end0_rate"] for r in rows])), 3),
        }

    # evasion: composite score of each condition (higher = more detectable)
    _, oof_full = composite_auc(real, fab, keys)
    fab_scores = oof_full[len(real):]
    by_cond = {}
    for cond in CONDS:
        s = [fab_scores[i] for i, r in enumerate(fab) if r["condition"] == cond]
        if s:
            by_cond[cond] = round(float(np.mean(s)), 3)

    summary = {
        "design": "domain-matched real vs fabricated (same 8 schemas)",
        "n_real": len(real), "n_fab": len(fab), "n_fab_c0": len(fab_c0), "n_refusals": len(refusals),
        "composite_auc_C0_vs_real": auc_all,
        "real_lastdigit_entropy": round(ent_real, 3), "real_round_end0": round(end0_real, 3),
        "per_model_C0": per_model,
        "composite_mean_score_by_condition": by_cond,
    }
    (SUMMARIES / "domain_matched.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nDOMAIN-MATCHED RESULT")
    print(" composite AUC (C0 fabricated vs real):", auc_all)
    print(" real entropy=%.2f end0=%.2f" % (ent_real, end0_real))
    for m, v in per_model.items():
        print(f"  {m:11s} AUC={v['composite_auc_vs_real']} entropy={v['lastdigit_entropy']} end0={v['round_end0_rate']}")
    print(" evasion composite score by condition:", by_cond)


if __name__ == "__main__":
    main()
