"""Actuarial problem instances with code-computed ground truth and labelled
anchor steps, built on the validated engine in actuarial_gen.py.

Each problem is self-contained: any life-table values the model needs are given
in the prompt (as integers), and the reference answer is computed from those
SAME integers, so a correct method reproduces it exactly. Every problem carries:
  - text (v0 standard, plus reworded variants v1/v2 for the consistency axis)
  - answer (float)
  - anchors: ordered list of (name, value) canonical intermediate quantities that
    any correct solution must produce (used for first-error localization)
  - family, params
"""

import math

from actuarial_gen import tpx, a_immediate_n, a_due_n, s_immediate_n, X0

RNG_ORDER = ["A_ann_imm", "B_ann_due", "C_accum", "D_pure_endow", "E_term_ins", "F_temp_annuity"]


def _lx_integers(x_min, x_max):
    """Integer life-table values from the validated Makeham SULT (l_20=100000)."""
    return {x: round(100000 * tpx(X0, x - X0)) for x in range(x_min, x_max + 1)}


def _fmt(i):
    return f"{i:.3f}".rstrip("0").rstrip(".")


def make_family(fam, rng):
    """Return one problem dict for the given family, parameters drawn from rng."""
    if fam == "A_ann_imm":
        pmt = rng.choice([1000, 2500, 5000, 800]); n = int(rng.choice([6, 8, 10, 12, 15, 20])); i = rng.choice([0.03, 0.04, 0.05, 0.06, 0.07])
        factor = a_immediate_n(n, i); ans = pmt * factor
        text = (f"An annuity pays {pmt} at the END of each year for {n} years. "
                f"The annual effective interest rate is {_fmt(i)}. Find the present value.")
        v1 = (f"At an annual effective rate of {_fmt(i)}, a series of {n} year-end payments of {pmt} is made. "
              f"What is its present value today?")
        v2 = (f"A pension will pay {pmt} at the end of every year for the next {n} years. "
              f"Using an interest rate of {_fmt(i)} per year, compute the present value.")
        anchors = [("annuity-immediate factor a_n", factor), ("present value", ans)]
        params = {"pmt": pmt, "n": n, "i": i}

    elif fam == "B_ann_due":
        pmt = rng.choice([1000, 2000, 3000, 1500]); n = int(rng.choice([6, 8, 10, 12, 15, 20])); i = rng.choice([0.03, 0.04, 0.05, 0.06, 0.07])
        factor = a_due_n(n, i); ans = pmt * factor
        text = (f"An annuity pays {pmt} at the BEGINNING of each year for {n} years. "
                f"The annual effective interest rate is {_fmt(i)}. Find the present value.")
        v1 = (f"Payments of {pmt} are made at the START of each year for {n} years. "
              f"At {_fmt(i)} annual effective interest, find the present value.")
        v2 = (f"A lease requires {pmt} paid in advance at the beginning of each of {n} years. "
              f"With interest at {_fmt(i)} per year, what is the present value?")
        anchors = [("annuity-due factor a-due_n", factor), ("present value", ans)]
        params = {"pmt": pmt, "n": n, "i": i}

    elif fam == "C_accum":
        pmt = rng.choice([1000, 1200, 2000, 500]); n = int(rng.choice([8, 10, 15, 20, 25])); i = rng.choice([0.03, 0.04, 0.05, 0.06, 0.07])
        factor = s_immediate_n(n, i); ans = pmt * factor
        text = (f"Deposits of {pmt} are made at the END of each year for {n} years into a fund earning "
                f"{_fmt(i)} annual effective interest. Find the accumulated value just after the last deposit.")
        v1 = (f"Each year for {n} years, {pmt} is deposited at year-end into an account crediting {_fmt(i)}. "
              f"What is the balance immediately after the final deposit?")
        v2 = (f"A saver contributes {pmt} at the end of every year for {n} years; the fund grows at {_fmt(i)} per year. "
              f"Find the accumulated value at the time of the last contribution.")
        anchors = [("accumulation factor s_n", factor), ("accumulated value", ans)]
        params = {"pmt": pmt, "n": n, "i": i}

    elif fam == "D_pure_endow":
        x = rng.choice([35, 40, 45, 50, 55, 60]); n = int(rng.choice([5, 8, 10, 12])); i = rng.choice([0.05, 0.06]); ben = rng.choice([10000, 25000, 50000])
        lx = _lx_integers(x, x + n); v = 1 / (1 + i)
        npx = lx[x + n] / lx[x]; ans = ben * (v ** n) * npx
        tbl = ", ".join(f"l_{a}={lx[a]}" for a in (x, x + n))
        text = (f"A pure endowment pays {ben} at the end of {n} years to a life now aged {x}, if the life survives. "
                f"Interest is {_fmt(i)} annual effective. From the life table: {tbl}. Find the expected present value.")
        v1 = (f"Given {tbl} and interest {_fmt(i)}, find the EPV of {ben} payable in {n} years to (aged {x}) "
              f"contingent on survival.")
        v2 = (f"A life aged {x} will receive {ben} in {n} years only if still alive. With {tbl} and rate {_fmt(i)}, "
              f"what is the expected present value of this pure endowment?")
        anchors = [("survival prob n_p_x", npx), ("discount v^n", v ** n), ("expected present value", ans)]
        params = {"x": x, "n": n, "i": i, "benefit": ben, "lx": lx}

    elif fam == "E_term_ins":
        x = rng.choice([45, 50, 55, 60, 65]); n = int(rng.choice([2, 3, 4])); i = rng.choice([0.05, 0.06]); ben = rng.choice([10000, 100000, 50000])
        lx = _lx_integers(x, x + n); v = 1 / (1 + i)
        factor = sum((v ** (k + 1)) * (lx[x + k] - lx[x + k + 1]) / lx[x] for k in range(n))
        ans = ben * factor
        tbl = ", ".join(f"l_{a}={lx[a]}" for a in range(x, x + n + 1))
        text = (f"A {n}-year term life insurance pays {ben} at the end of the year of death to a life aged {x}. "
                f"Interest is {_fmt(i)} annual effective. From the life table: {tbl}. "
                f"Find the expected present value (death benefit payable end of year of death).")
        v1 = (f"With {tbl} and interest {_fmt(i)}, compute the EPV of a {n}-year term assurance of {ben} on (aged {x}), "
              f"benefit at end of year of death.")
        v2 = (f"A life aged {x} buys {n}-year term cover of {ben}, paid at the end of the year of death. "
              f"Using {tbl} and rate {_fmt(i)}, find the expected present value.")
        anchors = [("term-insurance factor A", factor), ("expected present value", ans)]
        params = {"x": x, "n": n, "i": i, "benefit": ben, "lx": lx}

    else:  # F_temp_annuity
        x = rng.choice([45, 50, 55, 60, 65]); n = int(rng.choice([2, 3, 4])); i = rng.choice([0.05, 0.06]); pmt = rng.choice([1000, 5000, 2000])
        lx = _lx_integers(x, x + n - 1); v = 1 / (1 + i)
        factor = sum((v ** k) * (lx[x + k] / lx[x]) for k in range(n))
        ans = pmt * factor
        tbl = ", ".join(f"l_{a}={lx[a]}" for a in range(x, x + n))
        text = (f"A {n}-year temporary life annuity-due pays {pmt} at the START of each year to a life aged {x}, "
                f"while alive. Interest is {_fmt(i)} annual effective. From the life table: {tbl}. "
                f"Find the expected present value.")
        v1 = (f"Given {tbl} and interest {_fmt(i)}, find the EPV of a {n}-year annuity-due of {pmt} on (aged {x}), "
              f"payments in advance while alive.")
        v2 = (f"A life aged {x} receives {pmt} at the beginning of each year for up to {n} years, while living. "
              f"With {tbl} and rate {_fmt(i)}, compute the expected present value.")
        anchors = [("temporary annuity-due factor a-due_x:n", factor), ("expected present value", ans)]
        params = {"x": x, "n": n, "i": i, "pmt": pmt, "lx": lx}

    return {"family": fam, "text": text, "reword": [v1, v2], "answer": float(ans),
            "anchors": [(nm, float(val)) for nm, val in anchors], "params": params}


# Two scaffold designs, so a null mitigation result is not attributed to one weak prompt.
SCAFFOLD_A = ("Solve step by step. First state the exact formula you will use and the timing convention "
              "(payments in advance vs arrears; benefit timing). Then compute each intermediate quantity "
              "explicitly and label it. Do arithmetic carefully. End with a line 'ANSWER: <number>'.")
SCAFFOLD_B = ("Work in two passes. Pass 1: identify the exact formula and the timing convention, and list "
              "every input value. Pass 2: compute each intermediate value, then recompute the final answer "
              "a second, independent way to check it; if the two disagree, find and fix the error. "
              "End with a line 'ANSWER: <number>'.")
SCAFFOLD = SCAFFOLD_A  # backward compat

BASE_INSTRUCTION = " Show your work briefly and end with a line 'ANSWER: <number>'."


def generate(n_per_family=2, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    probs = []
    for fam in RNG_ORDER:
        for _ in range(n_per_family):
            probs.append(make_family(fam, rng))
    return probs


if __name__ == "__main__":
    ps = generate(1, seed=1)
    for p in ps:
        print(f"[{p['family']}] answer={p['answer']:.4f} anchors={[f'{n}={v:.4f}' for n,v in p['anchors']]}")
        print("   ", p["text"][:110])
