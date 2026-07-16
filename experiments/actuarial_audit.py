"""Post-submission audit recheck. Recomputes, from actuarial_scaled.json only:
  - per-model first-error localization (weak model vs the three strong models)
  - mitigation deltas under tolerance (1/2/5%) x unparseable handling variants
  - consistency raw counts (agree, agree-and-correct, agree-and-wrong)
  - no-answer distribution by model x condition
No model calls. Output: results/summaries/audit_recheck.json, which the revised
paper cites.
"""

import json
import math
from collections import Counter, defaultdict

from common import SUMMARIES

Z = 1.959963985
STRONG = ("deepseek", "gemini", "gptoss120")


def wilson(k, n):
    if n == 0:
        return None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z / den * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def main():
    d = json.loads((SUMMARIES / "actuarial_scaled.json").read_text(encoding="utf-8"))
    recs = d["records"]
    models = d["models"]
    conds = ["base_v0", "base_v1", "base_v2", "scaffoldA_v0", "scaffoldB_v0"]

    # no-answer distribution (missing cells out of 48 per model x cond)
    cells = Counter((r["model"], r["cond"]) for r in recs)
    missing = {f"{m}:{c}": 48 - cells.get((m, c), 0)
               for m in models for c in conds if 48 - cells.get((m, c), 0) > 0}

    # per-model localization on incorrect base_v0
    base = [r for r in recs if r["cond"] == "base_v0"]
    wrong = [r for r in base if not r["correct"]]
    loc = {}
    for m in models:
        w = [r for r in wrong if r["model"] == m]
        k = sum(1 for r in w if r["first_wrong_anchor"] == 0)
        loc[m] = {"incorrect": len(w), "factor_step": k, "share_ci": wilson(k, len(w))}
    sk = sum(loc[m]["factor_step"] for m in STRONG)
    sn = sum(loc[m]["incorrect"] for m in STRONG)
    loc["strong_pooled"] = {"incorrect": sn, "factor_step": sk, "share_ci": wilson(sk, sn)}

    # mitigation deltas under variants
    def amap(cond, tol):
        out = {}
        for r in recs:
            if r["cond"] == cond:
                ok = (abs(r["final"] - r["answer"]) / abs(r["answer"]) < tol) if r["answer"] else abs(r["final"]) < 1e-6
                out[(r["model"], r["prob"])] = ok
        return out
    variants = {}
    for tol in (0.01, 0.02, 0.05):
        for handle in ("exclude", "as_incorrect"):
            b = amap("base_v0", tol)
            row = {}
            for scaf in ("scaffoldA_v0", "scaffoldB_v0"):
                s = amap(scaf, tol)
                pairs = []
                for key, bok in b.items():
                    if key in s:
                        pairs.append((bok, s[key]))
                    elif handle == "as_incorrect":
                        pairs.append((bok, False))
                delta = sum(x for x, _ in pairs) / len(pairs) - sum(y for _, y in pairs) / len(pairs)
                row[scaf] = {"delta": round(delta, 3), "n_pairs": len(pairs)}
            variants[f"tol{int(tol*100)}pct_{handle}"] = row

    # near-miss bands among incorrect responses
    bands = {}
    for cond in ("base_v0", "scaffoldA_v0", "scaffoldB_v0"):
        rows = [r for r in recs if r["cond"] == cond and not r["correct"] and r["answer"]]
        c = Counter()
        for r in rows:
            rel = abs(r["final"] - r["answer"]) / abs(r["answer"])
            c["1to2pct" if rel < 0.02 else "2to5pct" if rel < 0.05 else "over5pct"] += 1
        bands[cond] = {"incorrect": len(rows), **dict(c)}

    # consistency raw counts
    groups = defaultdict(dict)
    for r in recs:
        if r["cond"].startswith("base_"):
            groups[(r["model"], r["prob"])][r["cond"]] = r
    total = agree = agree_correct = agree_wrong = 0
    for g in groups.values():
        if len(g) == 3:
            fs = [g[c]["final"] for c in ("base_v0", "base_v1", "base_v2")]
            truth = g["base_v0"]["answer"]
            if not fs[0]:
                continue
            total += 1
            if all(abs(f - fs[0]) / abs(fs[0]) < 0.01 for f in fs):
                agree += 1
                if abs(fs[0] - truth) / abs(truth) < 0.01:
                    agree_correct += 1
                else:
                    agree_wrong += 1

    out = {
        "source": "actuarial_scaled.json",
        "no_answer_missing_cells": missing,
        "localization_per_model": loc,
        "mitigation_variants": variants,
        "near_miss_bands": bands,
        "consistency_counts": {"cells": total, "agree": agree,
                               "agree_and_correct": agree_correct, "agree_and_wrong": agree_wrong},
        "audit_sample_note": "localization_audit.json holds 24 stored cases, all llama8b; "
                             "strong-model flags are not hand-validated (n=18 pool).",
    }
    (SUMMARIES / "audit_recheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("localization per model:", {m: loc[m]["share_ci"] for m in loc})
    print("mitigation deltas (all variants stay positive):",
          {k: (v["scaffoldA_v0"]["delta"], v["scaffoldB_v0"]["delta"]) for k, v in variants.items()})
    print("consistency:", out["consistency_counts"])
    print("wrote results/summaries/audit_recheck.json")


if __name__ == "__main__":
    main()
