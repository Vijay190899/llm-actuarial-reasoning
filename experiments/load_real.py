"""Build the REAL-data corpus: numeric columns from public datasets, spanning
BOTH regimes the red team flagged (D2):
  - clean / Benford-eligible magnitudes (span >= 1 order of magnitude)
  - naturally-heaped real data (prices, rounded admin/survey counts)

Each numeric column = one "table" (the detector's unit). We tag domain + regime,
compute digit features, and write a summary. No API, no generation. CPU only.

Output: results/summaries/real_corpus.json  (per-table features + labels)
        + a short human-readable interpretation printed at the end.
"""

import json
import warnings

import numpy as np

import digit_tests as dt
from common import SUMMARIES, atomic_write

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
N_MATCH = 100  # subsample every table to a common size so features are not confounded by n (D1)


def match_n(v):
    v = np.asarray(v, float)
    if len(v) > N_MATCH:
        return v[RNG.choice(len(v), N_MATCH, replace=False)]
    return v


def _tables_from_frame(df, source, max_cols=12):
    """Yield (name, values, domain) for numeric columns with enough positive values."""
    import pandas as pd
    out = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        v = s[np.isfinite(s)].to_numpy(dtype=float)
        v = np.abs(v[v > 0])
        if len(v) >= 80 and len(np.unique(v)) >= 15:
            out.append((f"{source}:{col}", v, source))
        if len(out) >= max_cols:
            break
    return out


def collect():
    """Pull a spread of real numeric tables. Each source is best-effort; failures skip."""
    tables = []

    # --- sklearn built-ins (small download, then cached) ---
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        tables += _tables_from_frame(d.frame, "calhousing")  # MedHouseVal=heaped prices, Population spans OOM
    except Exception as e:
        print("skip calhousing:", e)

    # --- OpenML: economic / admin / census style (network) ---
    for name, ver in [("adult", 2), ("diabetes", 1), ("credit-g", 1), ("bank-marketing", 1)]:
        try:
            from sklearn.datasets import fetch_openml
            d = fetch_openml(name, version=ver, as_frame=True, parser="auto")
            tables += _tables_from_frame(d.frame, f"openml_{name}")
        except Exception as e:
            print(f"skip openml/{name}:", e)

    return tables


def main():
    tables = collect()
    records = []
    for name, v, domain in tables:
        elig = bool(dt.is_benford_eligible(v))
        feats = dt.features(match_n(v))  # features at matched n (D1)
        if feats.get("insufficient"):
            continue
        records.append({"table": name, "domain": domain, "benford_eligible": elig,
                        "n": feats["n"], "features": feats})

    # Regime split by eligibility (proxy for clean-vs-heaped)
    elig = [r for r in records if r["benford_eligible"]]
    heap = [r for r in records if not r["benford_eligible"]]

    def _avg(rs, k):
        xs = [r["features"][k] for r in rs if isinstance(r["features"].get(k), (int, float))
              and np.isfinite(r["features"][k])]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n_tables": len(records),
        "n_benford_eligible": len(elig),
        "n_heaped_or_ineligible": len(heap),
        "by_regime": {
            "eligible": {k: _avg(elig, k) for k in
                         ["benford_mad", "lastdigit_entropy_bits", "round_end0_rate", "mantissa_ks_p"]},
            "ineligible": {k: _avg(heap, k) for k in
                           ["benford_mad", "lastdigit_entropy_bits", "round_end0_rate", "mantissa_ks_p"]},
        },
        "tables": records,
    }
    atomic_write(SUMMARIES / "real_corpus.json", json.dumps(summary, indent=1))

    print(f"REAL CORPUS: {len(records)} tables "
          f"({len(elig)} Benford-eligible, {len(heap)} heaped/ineligible)")
    e, h = summary["by_regime"]["eligible"], summary["by_regime"]["ineligible"]
    print("  eligible  : last_entropy=%.2f end0_rate=%.3f benford_mad=%.4f"
          % (e["lastdigit_entropy_bits"], e["round_end0_rate"], e["benford_mad"]))
    print("  ineligible: last_entropy=%.2f end0_rate=%.3f benford_mad=%.4f"
          % (h["lastdigit_entropy_bits"], h["round_end0_rate"], h["benford_mad"]))
    print("  -> D2 check: if 'ineligible' real data already heaps (low entropy / high end0),"
          " it is the honest false-positive source the paper must report.")


if __name__ == "__main__":
    main()
