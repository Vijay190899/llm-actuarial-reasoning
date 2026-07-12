"""Classical digit-distribution forensics -> per-table feature vector.

All tests are standard (Benford/Nigrini). Given a 1-D array of positive numeric
values from one table/column, `features()` returns the detector feature dict.
No hand-typed constants leak into the paper: the paper reads the JSON these
scripts emit, never a number typed here.
"""

import numpy as np
from scipy import stats

BENFORD_P1 = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])  # first digit 1..9


def _clean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.abs(x)
    return x[x > 0]


def first_digit(x):
    x = _clean(x)
    lead = (x / 10 ** np.floor(np.log10(x))).astype(int)
    lead = np.clip(lead, 1, 9)
    return lead


def _digit_at(x, place):
    """place=1 -> first significant digit; place=2 -> second, etc."""
    x = _clean(x)
    norm = x / 10 ** np.floor(np.log10(x))  # in [1,10)
    shifted = norm * 10 ** (place - 1)
    return (np.floor(shifted) % 10).astype(int)


def last_digit(x):
    # terminal digit of the integer part; for decimals use rounded value
    x = _clean(x)
    return (np.round(x).astype(np.int64) % 10)


def _chi2_uniform(counts):
    counts = np.asarray(counts, float)
    n = counts.sum()
    if n < 1:
        return np.nan, np.nan
    exp = np.full(len(counts), n / len(counts))
    chi2, p = stats.chisquare(counts, exp)
    return chi2, p


def features(x, min_n=30):
    """Return detector features for one numeric column. NaNs where n too small."""
    x = _clean(x)
    n = len(x)
    f = {"n": int(n)}
    if n < min_n:
        return {**f, "insufficient": True}

    # First-digit Benford: MAD (Nigrini) + chi2 p
    fd = first_digit(x)
    obs1 = np.array([(fd == d).sum() for d in range(1, 10)], float)
    p1 = obs1 / obs1.sum()
    f["benford_mad"] = float(np.mean(np.abs(p1 - BENFORD_P1)))          # Nigrini MAD
    _, f["benford_chi2_p"] = stats.chisquare(obs1, BENFORD_P1 * obs1.sum())

    # Second-digit uniformity-ish (Benford 2nd-digit is near-uniform-ish; use chi2 vs Benford2)
    sd = _digit_at(x, 2)
    obs2 = np.array([(sd == d).sum() for d in range(0, 10)], float)
    _, f["second_digit_chi2_p"] = _chi2_uniform(obs2)

    # Last-digit uniformity (real ratio-scale data -> ~uniform; fabrication heaps)
    ld = last_digit(x)
    obsl = np.array([(ld == d).sum() for d in range(0, 10)], float)
    f["lastdigit_chi2, lastdigit_chi2_p".split(", ")[0]], f["lastdigit_chi2_p"] = _chi2_uniform(obsl)
    pl = obsl / obsl.sum()
    f["lastdigit_mad"] = float(np.mean(np.abs(pl - 0.1)))

    # Round-number heaping: fraction ending in 0 or 00, and "typical" multiples
    xi = np.round(x).astype(np.int64)
    f["round_end0_rate"] = float(np.mean(xi % 10 == 0))
    f["round_end00_rate"] = float(np.mean(xi % 100 == 0))
    f["round_5or0_rate"] = float(np.mean(np.isin(xi % 10, [0, 5])))

    # Terminal-digit entropy (bits); low entropy = heaping
    with np.errstate(divide="ignore"):
        ent = -np.sum(pl[pl > 0] * np.log2(pl[pl > 0]))
    f["lastdigit_entropy_bits"] = float(ent)

    # Mantissa uniformity: Benford <=> log10 mantissa ~ Uniform(0,1).
    # Keep the KS p-value for reference, but the DETECTOR uses mantissa_mad,
    # a sample-size-robust effect size (p-values scale with n and would let the
    # detector cheat on table size instead of on fabrication).
    mant = np.log10(x) % 1.0
    f["mantissa_ks_p"] = float(stats.kstest(mant, "uniform").pvalue)
    mhist, _ = np.histogram(mant, bins=10, range=(0.0, 1.0))
    f["mantissa_mad"] = float(np.mean(np.abs(mhist / mhist.sum() - 0.1)))

    # Unique-value ratio (fabrication often repeats "nice" values).
    # Only comparable across tables when n is held fixed (see N_MATCH in callers).
    f["unique_ratio"] = float(len(np.unique(xi)) / n)
    return f


# Detector features: all sample-size robust (proportions, entropies, MADs).
# p-value features are computed above for reference but excluded here so the
# detector cannot separate tables by their size (D1).
FEATURE_KEYS = [
    "benford_mad", "lastdigit_mad", "lastdigit_entropy_bits",
    "round_end0_rate", "round_end00_rate", "round_5or0_rate",
    "mantissa_mad", "unique_ratio",
]


def is_benford_eligible(x, min_n=50, min_orders=1.0):
    """Nigrini eligibility: positive, ratio-scale, spanning >= min_orders magnitudes."""
    x = _clean(x)
    if len(x) < min_n:
        return False
    return (np.log10(x.max()) - np.log10(x.min())) >= min_orders


if __name__ == "__main__":
    # self-test: real-like (Benford) vs fabricated-like (heaped) columns
    rng = np.random.default_rng(0)
    benford_like = 10 ** (rng.uniform(0, 4, 2000))              # spans 4 orders, mantissa-uniform
    heaped = rng.choice([100, 150, 200, 250, 500, 1000, 1500, 2000, 5000], 2000)
    heaped = heaped * (1 + rng.integers(0, 3, 2000))           # still very round
    fr = features(benford_like)
    fh = features(heaped)
    print("REAL-like  benford_mad=%.4f last_entropy=%.3f end0=%.3f mant_ks_p=%.3g"
          % (fr["benford_mad"], fr["lastdigit_entropy_bits"], fr["round_end0_rate"], fr["mantissa_ks_p"]))
    print("FAB-like   benford_mad=%.4f last_entropy=%.3f end0=%.3f mant_ks_p=%.3g"
          % (fh["benford_mad"], fh["lastdigit_entropy_bits"], fh["round_end0_rate"], fh["mantissa_ks_p"]))
    assert fh["lastdigit_entropy_bits"] < fr["lastdigit_entropy_bits"], "heaping should lower last-digit entropy"
    assert fh["round_end0_rate"] > fr["round_end0_rate"], "fabricated should heap on round numbers"
    print("SELF-TEST PASS: terminal-digit signal separates heaped from Benford-like.")
