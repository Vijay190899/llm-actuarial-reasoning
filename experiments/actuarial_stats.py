"""Statistics for the manuscript: Wilson 95% CIs on accuracies and bootstrap CIs
on the mitigation effect. Reads the scaled results, writes a traceable summary
so every number in the paper comes from a file, not by hand.

Usage: python actuarial_stats.py --summary actuarial_scaled.json
Output: results/summaries/actuarial_stats.json
"""

import argparse
import json
import math

import numpy as np

from common import SUMMARIES

Z = 1.959963985  # 95%


def wilson(k, n):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / d
    half = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return round(p, 3), round(center - half, 3), round(center + half, 3)


def boot_delta(pairs, iters=5000, seed=0):
    """pairs: list of (base_correct, scaf_correct) 0/1; return mean delta + 95% CI."""
    rng = np.random.default_rng(seed)
    a = np.array([b for b, _ in pairs], float)
    b = np.array([s for _, s in pairs], float)
    base = a.mean() - b.mean()
    idx = rng.integers(0, len(pairs), (iters, len(pairs)))
    deltas = a[idx].mean(1) - b[idx].mean(1)
    return round(float(base), 3), round(float(np.percentile(deltas, 2.5)), 3), round(float(np.percentile(deltas, 97.5)), 3)


def main(summary):
    d = json.loads((SUMMARIES / summary).read_text(encoding="utf-8"))
    recs = d["records"]
    models = d["models"]

    # per-model accuracy (base_v0) with Wilson CI
    acc = {}
    for m in models:
        rows = [r for r in recs if r["model"] == m and r["cond"] == "base_v0"]
        k = sum(r["correct"] for r in rows)
        acc[m] = {"n": len(rows), "correct": k, "acc_ci": wilson(k, len(rows))}

    # overall accuracy across all models (base_v0)
    base_all = [r for r in recs if r["cond"] == "base_v0"]
    k_all = sum(r["correct"] for r in base_all)
    overall_acc = {"n": len(base_all), "acc_ci": wilson(k_all, len(base_all))}

    # mitigation deltas (base_v0 - scaffold), paired by (model, prob), pooled
    def pairs_for(scaf):
        out = []
        for r in base_all:
            s = next((x for x in recs if x["model"] == r["model"] and x["prob"] == r["prob"]
                      and x["cond"] == scaf), None)
            if s is not None:
                out.append((int(r["correct"]), int(s["correct"])))
        return out
    mitigation = {scaf: {"n_pairs": len(pairs_for(scaf)),
                         "delta_base_minus_scaffold_ci": boot_delta(pairs_for(scaf))}
                  for scaf in ("scaffoldA_v0", "scaffoldB_v0")}

    # localization: proportion of incorrect base_v0 whose first wrong step is the factor (anchor 0)
    wrong = [r for r in base_all if not r["correct"]]
    kfac = sum(1 for r in wrong if r["first_wrong_anchor"] == 0)
    localization = {"n_incorrect": len(wrong), "factor_step": kfac,
                    "factor_share_ci": wilson(kfac, len(wrong))}

    # consistency (recompute agreement across rewordings, with Wilson CI)
    probs = sorted({r["prob"] for r in recs})
    agree = 0; total = 0
    for m in models:
        for pi in probs:
            fs = [r["final"] for r in recs if r["model"] == m and r["prob"] == pi
                  and r["cond"].startswith("base_")]
            if len(fs) == 3 and fs[0]:
                total += 1
                if all(abs(f - fs[0]) / abs(fs[0]) < 0.01 for f in fs):
                    agree += 1
    consistency = {"n": total, "consistent": agree, "rate_ci": wilson(agree, total)}

    out = {"summary_source": summary, "per_model_accuracy": acc, "overall_accuracy": overall_acc,
           "mitigation": mitigation, "localization_factor_step": localization,
           "consistency": consistency}
    (SUMMARIES / "actuarial_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("per-model acc (p, lo, hi):", {m: acc[m]["acc_ci"] for m in models})
    print("overall acc:", overall_acc["acc_ci"])
    print("mitigation delta base-scaffold (delta, lo, hi):",
          {k: v["delta_base_minus_scaffold_ci"] for k, v in mitigation.items()})
    print("localization factor-step share:", localization["factor_share_ci"])
    print("consistency rate:", consistency["rate_ci"])


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--summary", default="actuarial_scaled.json")
    main(a.parse_args().summary)
