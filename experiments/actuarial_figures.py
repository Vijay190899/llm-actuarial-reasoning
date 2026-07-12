"""Publication figures for the actuarial diagnostic study.

Reads a results summary (records) and renders the paper's figure set with a
colorblind-safe categorical palette (Okabe-Ito), thin marks, recessive axes,
direct labels, and a single value axis per panel. Generalizes to any number of
models present in the data.

Figures:
  F1 accuracy heatmap (model x problem family)      -> RQ1
  F2 first-error localization by model (stacked)     -> RQ2
  F3 rewording consistency by model                  -> RQ3
  F4 mitigation: base vs scaffold accuracy by model  -> RQ4

Usage: python actuarial_figures.py --summary actuarial_pilot.json
Output: results/figures/*.pdf and *.png
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SUMMARIES
import actuarial_problems as ap

FIGDIR = SUMMARIES.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe categorical palette (fixed order)
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
FAM_LABEL = {"A_ann_imm": "annuity-imm", "B_ann_due": "annuity-due", "C_accum": "accumulation",
             "D_pure_endow": "pure endow", "E_term_ins": "term ins", "F_temp_annuity": "temp annuity"}
MODEL_LABEL = {"llama8b": "Llama-3.1-8B", "deepseek": "DeepSeek-V3.1",
               "gemini": "Gemini-2.5-Flash", "gptoss120": "gpt-oss-120B"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.dpi": 120})


def _recs(summary):
    d = json.loads((SUMMARIES / summary).read_text(encoding="utf-8"))
    return d["records"], d.get("models") or sorted({r["model"] for r in d["records"]})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_accuracy(recs, models):
    fams = ap.RNG_ORDER
    M = np.full((len(models), len(fams)), np.nan)
    for mi, m in enumerate(models):
        for fj, f in enumerate(fams):
            rows = [r for r in recs if r["model"] == m and r["family"] == f and r["cond"] == "base_v0"]
            if rows:
                M[mi, fj] = np.mean([r["correct"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 0.7 + 0.5 * len(models)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([FAM_LABEL[f] for f in fams], rotation=25, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models])
    for mi in range(len(models)):
        for fj in range(len(fams)):
            if np.isfinite(M[mi, fj]):
                ax.text(fj, mi, f"{M[mi,fj]:.2f}", ha="center", va="center",
                        color="white" if M[mi, fj] < 0.6 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("accuracy", color=INK)
    ax.set_title("Final-answer accuracy by model and problem family", color=INK, loc="left")
    fig.tight_layout(); _save(fig, "F1_accuracy_heatmap")


def fig_localization(recs, models):
    cats = ["factor step (anchor 0)", "later step", "final arithmetic only"]
    data = np.zeros((len(models), 3))
    for mi, m in enumerate(models):
        wrong = [r for r in recs if r["model"] == m and r["cond"] == "base_v0" and not r["correct"]]
        for r in wrong:
            fw = r["first_wrong_anchor"]
            k = 2 if fw is None else (0 if fw == 0 else 1)
            data[mi, k] += 1
        tot = data[mi].sum()
        if tot:
            data[mi] /= tot
    fig, ax = plt.subplots(figsize=(7.2, 0.8 + 0.5 * len(models)))
    left = np.zeros(len(models)); y = range(len(models))
    for k in range(3):
        ax.barh(y, data[:, k], left=left, color=OI[k], height=0.6, label=cats[k],
                edgecolor="white", linewidth=1.2)
        for mi in range(len(models)):
            if data[mi, k] > 0.08:
                ax.text(left[mi] + data[mi, k] / 2, mi, f"{data[mi,k]:.0%}", ha="center", va="center",
                        color="white", fontsize=8)
        left += data[:, k]
    ax.set_yticks(list(y)); ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models])
    ax.set_xlim(0, 1); ax.set_xlabel("share of incorrect answers", labelpad=4); _clean(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=8)
    ax.set_title("Where the first error occurs", color=INK, loc="left")
    fig.subplots_adjust(bottom=0.42)
    _save(fig, "F2_error_localization")


def fig_consistency(recs, models):
    rates = []
    for m in models:
        base = [r for r in recs if r["model"] == m and r["cond"].startswith("base_")]
        probs = sorted({r["prob"] for r in base})
        agree = []
        for pi in probs:
            fs = [r["final"] for r in base if r["prob"] == pi]
            if len(fs) == 3 and fs[0]:
                agree.append(all(abs(f - fs[0]) / abs(fs[0]) < 0.01 for f in fs))
        rates.append(np.mean(agree) if agree else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = range(len(models))
    ax.bar(x, rates, color=OI[0], width=0.6, edgecolor="white", linewidth=1.2)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.02, f"{r:.0%}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("problems with identical answer\nacross 3 rewordings"); _clean(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Self-consistency under semantically-equivalent rewording", color=INK, loc="left")
    fig.tight_layout(); _save(fig, "F3_consistency")


def fig_mitigation(recs, models):
    base, scaf = [], []
    for m in models:
        b = [r["correct"] for r in recs if r["model"] == m and r["cond"] == "base_v0"]
        s = [r["correct"] for r in recs if r["model"] == m and r["cond"] == "scaffold_v0"]
        base.append(np.mean(b) if b else 0.0); scaf.append(np.mean(s) if s else 0.0)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(models)); w = 0.38
    ax.bar(x - w / 2, base, w, color=OI[0], label="baseline prompt", edgecolor="white", linewidth=1.2)
    ax.bar(x + w / 2, scaf, w, color=OI[1], label="structured scaffold", edgecolor="white", linewidth=1.2)
    for xi, (b, s) in enumerate(zip(base, scaf)):
        ax.text(xi - w / 2, b + 0.02, f"{b:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(xi + w / 2, s + 0.02, f"{s:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); _clean(ax)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Mitigation: baseline vs structured scaffold", color=INK, loc="left")
    fig.tight_layout(); _save(fig, "F4_mitigation")


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def main(summary):
    recs, models = _recs(summary)
    print(f"figures from {summary}: {len(models)} models, {len(recs)} responses")
    fig_accuracy(recs, models)
    fig_localization(recs, models)
    fig_consistency(recs, models)
    fig_mitigation(recs, models)
    print("done ->", FIGDIR)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--summary", default="actuarial_pilot.json")
    main(a.parse_args().summary)
