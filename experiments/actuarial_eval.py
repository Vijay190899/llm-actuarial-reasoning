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


def get_response(prompt, model):
    """Fetch, and retry once with more tokens + a nudge if the content is empty
    (empty content is the main parse-failure cause, usually a token cutoff)."""
    txt = oai_generate(prompt, model, temperature=0.3, max_tokens=1200)
    if isinstance(txt, str) and txt.strip():
        return txt, False
    txt2 = oai_generate(prompt + " Be concise; end with 'ANSWER: <number>'.", model,
                        temperature=0.3, max_tokens=2600)
    return (txt2, False) if isinstance(txt2, str) and txt2.strip() else (txt2, True)


def run(model_keys, n_per_family, out):
    probs = ap.generate(n_per_family=n_per_family, seed=7)
    recs, no_answer, extractable = [], [], 0
    audit = []  # localization spot-check sample
    for mk in model_keys:
        model = MODELS[mk]
        for pi, p in enumerate(probs):
            variants = {"v0": p["text"], "v1": p["reword"][0], "v2": p["reword"][1]}
            calls = {f"base_{vk}": vt + ap.BASE_INSTRUCTION for vk, vt in variants.items()}
            calls["scaffoldA_v0"] = p["text"] + " " + ap.SCAFFOLD_A
            calls["scaffoldB_v0"] = p["text"] + " " + ap.SCAFFOLD_B
            for cond, prompt in calls.items():
                try:
                    txt, empty = get_response(prompt, model)
                except DailyCapError as e:
                    print("cap:", e); continue
                fa = None if empty else final_answer(txt)
                if fa is None:
                    no_answer.append({"model": mk, "family": p["family"], "cond": cond,
                                      "empty": bool(empty)})
                    continue
                correct = matches(fa, p["answer"])
                fw, nfound = localize(txt, p["anchors"])
                if len(nums(txt)) >= 3:
                    extractable += 1
                recs.append({"model": mk, "family": p["family"], "prob": pi, "cond": cond,
                             "final": fa, "answer": p["answer"], "correct": bool(correct),
                             "first_wrong_anchor": fw, "n_anchors": len(p["anchors"])})
                # collect a localization audit sample (incorrect base_v0 cases)
                if cond == "base_v0" and not correct and len(audit) < 24:
                    anchor_str = "; ".join(f"{n}={v:.4f}" for n, v in p["anchors"])
                    audit.append({"model": mk, "family": p["family"], "answer": round(p["answer"], 4),
                                  "flagged_anchor": fw, "anchors": anchor_str,
                                  "model_tail": txt[-260:]})
        print(f"  {mk}: {sum(1 for r in recs if r['model']==mk)} responses")
    (SUMMARIES / "localization_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

    # ---- aggregate ----
    def acc(rows):
        return round(np.mean([r["correct"] for r in rows]), 3) if rows else None
    base_v0 = [r for r in recs if r["cond"] == "base_v0"]

    acc_by_model = {mk: acc([r for r in base_v0 if r["model"] == mk]) for mk in model_keys}
    acc_by_family = {f: acc([r for r in base_v0 if r["family"] == f]) for f in ap.RNG_ORDER}

    # RQ2 localization among INCORRECT base_v0
    loc_dist = {}
    for r in [x for x in base_v0 if not x["correct"]]:
        k = "final_only" if r["first_wrong_anchor"] is None else f"anchor_{r['first_wrong_anchor']}"
        loc_dist[k] = loc_dist.get(k, 0) + 1

    # RQ3 consistency (FIXED): separate consistency from accuracy.
    #  consistent          = all 3 rewordings agree with each other (right or wrong)
    #  consistent_correct  = all 3 agree AND match the true answer
    #  mean_cov            = mean coefficient of variation of the 3 finals (dispersion)
    con_agree, con_correct, covs = [], [], []
    for mk in model_keys:
        for pi in range(len(probs)):
            fs = [r["final"] for r in recs
                  if r["model"] == mk and r["prob"] == pi and r["cond"].startswith("base_")]
            truth = next((r["answer"] for r in recs
                          if r["model"] == mk and r["prob"] == pi), None)
            if len(fs) == 3 and fs[0]:
                agree = all(matches(f, fs[0]) for f in fs)
                con_agree.append(agree)
                con_correct.append(agree and truth is not None and matches(fs[0], truth))
                m = np.mean(fs)
                if m:
                    covs.append(np.std(fs) / abs(m))
    consistency = {
        "consistent_across_rewordings": round(np.mean(con_agree), 3) if con_agree else None,
        "consistent_and_correct": round(np.mean(con_correct), 3) if con_correct else None,
        "mean_coeff_of_variation": round(float(np.mean(covs)), 3) if covs else None,
    }

    # RQ4 mitigation: base vs TWO scaffolds
    mitigation = {c: acc([r for r in recs if r["cond"] == c])
                  for c in ("base_v0", "scaffoldA_v0", "scaffoldB_v0")}
    mit_by_model = {mk: {c: acc([r for r in recs if r["model"] == mk and r["cond"] == c])
                         for c in ("base_v0", "scaffoldA_v0", "scaffoldB_v0")} for mk in model_keys}

    n_empty = sum(1 for x in no_answer if x["empty"])
    summary = {
        "models": model_keys, "n_problems": len(probs), "n_responses": len(recs),
        "n_no_answer_after_retry": len(no_answer), "n_empty_content": n_empty,
        "KC1_extractable_rate": round(extractable / max(len(recs), 1), 3),
        "RQ1_accuracy_by_model_base": acc_by_model,
        "RQ1_accuracy_by_family_base": acc_by_family,
        "RQ2_first_wrong_anchor_dist_among_incorrect": loc_dist,
        "RQ3_consistency": consistency,
        "RQ4_mitigation_overall": mitigation,
        "RQ4_mitigation_by_model": mit_by_model,
        "records": recs,
    }
    (SUMMARIES / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nACTUARIAL PILOT (corrected)")
    print(" no-answer after retry:", len(no_answer), "(empty:", n_empty, ")")
    print(" RQ1 accuracy by model:", acc_by_model)
    print(" RQ2 localization:", loc_dist)
    print(" RQ3 consistency:", consistency)
    print(" RQ4 mitigation overall:", mitigation)
    return summary


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--models", default="llama8b,deepseek,gemini,gptoss120")
    ap_.add_argument("--n-per-family", type=int, default=2)
    ap_.add_argument("--out", default="actuarial_pilot.json")
    a = ap_.parse_args()
    run(a.models.split(","), a.n_per_family, a.out)
