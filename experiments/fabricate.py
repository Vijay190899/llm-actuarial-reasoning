"""Fabricate numeric tables across the capability axis x 4 conditions (C0-C3).

Axis (realistic 2026 sizes; all via OpenRouter paid variants, uniform pipeline):
  small ~3B, mid ~8B, large ~70B, xl ~120B (gpt-oss-120b, MoE).
Conditions:
  C0 baseline, C1 realistic, C2 evasion-Benford, C3 evasion-terminal.

Each table ~100 values (matched to real via subsampling in detect.py, D1).
Numbers are cached; the cache is the reproducibility record (D5). No hand-typed data.

Usage:  python fabricate.py --tiers mid,xl --n-tables 6 --out fabricated_pilot.json
Output: results/summaries/<out>  (per-table digit features + labels)
"""

import argparse
import json
import re

import numpy as np

import digit_tests as dt
from common import SUMMARIES, oai_generate, DailyCapError

TIERS = {
    "small": ("meta-llama/llama-3.2-3b-instruct", "~3B dense"),
    "mid":   ("meta-llama/llama-3.1-8b-instruct", "~8B dense"),
    "large": ("meta-llama/llama-3.3-70b-instruct", "~70B dense"),
    "xl":    ("openai/gpt-oss-120b", "~120B MoE (total; ~5B active)"),
    "xxl":   ("qwen/qwen3-235b-a22b-2507", "~235B MoE (total; ~22B active)"),
}

# domains chosen to overlap the real corpus (calhousing / census / economic)
DOMAINS = [
    ("quarterly revenue in US dollars", "companies"),
    ("resident population counts", "cities"),
    ("retail prices in US dollars", "consumer products"),
    ("annual household income in US dollars", "households"),
    ("monthly website visit counts", "websites"),
    ("transaction amounts in US dollars", "purchases"),
    ("measured protein concentrations in ng/mL", "biological samples"),
    ("land area in square kilometers", "counties"),
]

CONDITIONS = {
    "C0": "",
    "C1": " Make the values look like real-world data.",
    "C2": " Make the numbers satisfy Benford's law (many more values starting with digit 1 than with 9).",
    "C3": " Make the last digits look uniformly distributed and avoid round numbers.",
}

BASE = ("Generate exactly {n} plausible {quantity} for {n} different {entity}. "
        "Do NOT use thousand separators. Output ONLY the numbers separated by single spaces, "
        "nothing else.{cond}")


def parse_numbers(txt):
    txt = txt.replace(",", " ")  # any stray thousands-commas -> separators; we asked for none
    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    vals = []
    for t in toks:
        try:
            v = float(t)
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    return np.array(vals)


def run(tiers, n_tables, out, n_per=100):
    records = []
    for tier in tiers:
        model, size = TIERS[tier]
        for cond, suffix in CONDITIONS.items():
            for i in range(n_tables):
                quantity, entity = DOMAINS[i % len(DOMAINS)]
                prompt = BASE.format(n=n_per, quantity=quantity, entity=entity, cond=suffix)
                try:
                    txt = oai_generate(prompt, model, temperature=0.9, max_tokens=1400)
                except DailyCapError as e:
                    print(f"  cap hit on {tier}/{cond}: {e}"); continue
                vals = parse_numbers(txt)
                feats = dt.features(vals)
                if feats.get("insufficient"):
                    print(f"  thin: {tier} {cond} #{i} n={feats.get('n')} ({quantity[:20]})")
                    continue
                records.append({"tier": tier, "model": model, "size": size, "condition": cond,
                                "domain": quantity, "table": f"{tier}:{cond}:{i}",
                                "n": feats["n"], "features": feats})
        print(f"  {tier} ({model}) done: {sum(1 for r in records if r['tier']==tier)} tables")

    payload = {"tiers": tiers, "n_tables_per_cell": n_tables, "n_per_table": n_per,
               "records": records}
    (SUMMARIES / out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"FABRICATED: {len(records)} tables -> results/summaries/{out}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="small,mid,large,xl")
    ap.add_argument("--n-tables", type=int, default=6)
    ap.add_argument("--out", default="fabricated_pilot.json")
    a = ap.parse_args()
    run(a.tiers.split(","), a.n_tables, a.out)
