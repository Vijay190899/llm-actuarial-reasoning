"""Run models on actuarial problems; grade final answers, localize first-error
step via anchors, and measure consistency across rewordings.

Localization is done only on models that expose their working (instruct models
asked to show steps), per the council risk note on reasoning-model CoT opacity.

Output: results/summaries/actuarial_pilot.json
"""

import argparse
import json
import re

import numpy as np

import actuarial_problems as ap
from common import SUMMARIES, oai_generate, DailyCapError

MODELS = {
    "llama8b":  "meta-llama/llama-3.1-8b-instruct",
    "deepseek": "deepseek/deepseek-chat-v3.1",
    "gemini":   "google/gemini-2.5-flash",
    "gptoss120": "openai/gpt-oss-120b",
}
REL = 0.01  # relative tolerance for "matches"


def nums(txt):
    txt = txt.replace(",", "")
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", txt)]


def final_answer(txt):
    m = re.findall(r"ANSWER:\s*\$?(-?\d[\d,]*\.?\d*)", txt, re.IGNORECASE)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    ns = nums(txt)
    return ns[-1] if ns else None


def matches(val, target):
    if target == 0:
        return abs(val) < 1e-6
    return abs(val - target) / abs(target) < REL


def localize(response, anchors):
    """Return (first_wrong_index, all_found) using anchor presence in the text."""
    found = nums(response)
    first_wrong = None
    for idx, (_, tgt) in enumerate(anchors):
        if not any(matches(v, tgt) for v in found):
            first_wrong = idx
            break
    return first_wrong, sum(any(matches(v, t) for v in found) for _, t in anchors)


def run(model_keys, n_per_family, out):
    probs = ap.generate(n_per_family=n_per_family, seed=7)
    recs, refusals, extractable = [], 0, 0
    for mk in model_keys:
        model = MODELS[mk]
        for pi, p in enumerate(probs):
            variants = {"v0": p["text"], "v1": p["reword"][0], "v2": p["reword"][1]}
            # base condition: all three phrasings (for consistency) + scaffold on v0
            calls = {f"base_{vk}": vt + ap.BASE_INSTRUCTION for vk, vt in variants.items()}
            calls["scaffold_v0"] = p["text"] + " " + ap.SCAFFOLD
            for cond, prompt in calls.items():
                try:
                    txt = oai_generate(prompt, model, temperature=0.3, max_tokens=1200)
                except DailyCapError as e:
                    print("cap:", e); continue
                if not isinstance(txt, str) or not txt.strip():
                    refusals += 1; continue
                fa = final_answer(txt)
                if fa is None:
                    refusals += 1; continue
                correct = matches(fa, p["answer"])
                fw, nfound = localize(txt, p["anchors"])
                # KC1 = did the model SHOW parseable working (numbers beyond the final),
                # so a first-divergence step can be identified? This is parse-ability,
                # not correctness (a wrong-but-shown answer is still localizable).
                shows_work = len(nums(txt)) >= 3
                if shows_work:
                    extractable += 1
                recs.append({"model": mk, "family": p["family"], "prob": pi, "cond": cond,
                             "final": fa, "answer": p["answer"], "correct": bool(correct),
                             "first_wrong_anchor": fw, "n_anchors": len(p["anchors"]),
                             "anchors_found": int(nfound)})
        print(f"  {mk}: {sum(1 for r in recs if r['model']==mk)} responses")

    # ---- aggregate ----
    def acc(rows):
        return round(np.mean([r["correct"] for r in rows]), 3) if rows else None
    base = [r for r in recs if r["cond"].startswith("base_")]
    base_v0 = [r for r in recs if r["cond"] == "base_v0"]
    scaf = [r for r in recs if r["cond"] == "scaffold_v0"]

    # RQ1 accuracy by model (base_v0)
    acc_by_model = {mk: acc([r for r in base_v0 if r["model"] == mk]) for mk in model_keys}
    acc_by_family = {f: acc([r for r in base_v0 if r["family"] == f]) for f in ap.RNG_ORDER}

    # RQ2 localization: distribution of first-wrong-anchor among INCORRECT base_v0 responses
    wrong = [r for r in base_v0 if not r["correct"]]
    loc_dist = {}
    for r in wrong:
        k = "final_only" if r["first_wrong_anchor"] is None else f"anchor_{r['first_wrong_anchor']}"
        loc_dist[k] = loc_dist.get(k, 0) + 1

    # RQ3 consistency: per (model, prob) do v0,v1,v2 finals agree with each other?
    consist = []
    for mk in model_keys:
        for pi in range(len(probs)):
            fs = [r["final"] for r in base if r["model"] == mk and r["prob"] == pi]
            if len(fs) == 3:
                agree = all(matches(fs[a], fs[0]) for a in range(3))
                consist.append(agree)
    consistency_rate = round(np.mean(consist), 3) if consist else None

    # RQ4 mitigation: accuracy base_v0 vs scaffold_v0 (paired by model,prob)
    def paired_acc(rows):
        return round(np.mean([r["correct"] for r in rows]), 3) if rows else None

    summary = {
        "models": model_keys, "n_problems": len(probs), "n_responses": len(recs),
        "n_refusals_no_answer": refusals,
        "KC1_extractable_rate": round(extractable / max(len(recs), 1), 3),
        "RQ1_accuracy_by_model_base": acc_by_model,
        "RQ1_accuracy_by_family_base": acc_by_family,
        "RQ2_first_wrong_anchor_dist_among_incorrect": loc_dist,
        "RQ3_consistency_rate_across_rewordings": consistency_rate,
        "RQ4_accuracy_base_vs_scaffold": {"base_v0": paired_acc(base_v0), "scaffold_v0": paired_acc(scaf)},
        "records": recs,
    }
    (SUMMARIES / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nACTUARIAL PILOT")
    print(" KC1 extractable rate:", summary["KC1_extractable_rate"])
    print(" RQ1 accuracy by model:", acc_by_model)
    print(" RQ1 accuracy by family:", acc_by_family)
    print(" RQ2 first-wrong-anchor among incorrect:", loc_dist)
    print(" RQ3 consistency across rewordings:", consistency_rate)
    print(" RQ4 base vs scaffold accuracy:", summary["RQ4_accuracy_base_vs_scaffold"])
    return summary


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--models", default="llama8b,deepseek,gemini,gptoss120")
    ap_.add_argument("--n-per-family", type=int, default=2)
    ap_.add_argument("--out", default="actuarial_pilot.json")
    a = ap_.parse_args()
    run(a.models.split(","), a.n_per_family, a.out)
