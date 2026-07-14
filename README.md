# Diagnostic Evaluation of Frontier LLM Actuarial Calculation

A study of not just whether frontier language models get actuarial calculations right, but where and why they break, how consistent they are when the same case is reworded, and whether a simple intervention helps.

## Question

Two 2026 benchmarks measure LLM actuarial accuracy. This project adds the diagnostic layer they lack:

1. Accuracy on core actuarial calculations (annuities, life contingencies, premiums), by model.
2. Error-mechanism localization: on a multi-step problem, which step is the first to go wrong, and is it a conceptual error (for example confusing annuity-in-advance with annuity-in-arrears) or an arithmetic slip.
3. Self-consistency: does the same problem, validly reworded, produce the same answer.
4. Mitigation: does a structured solution scaffold reduce the dominant error class.

## Why the ground truth is trustworthy

Problems are procedurally generated with randomized parameters, so the exact items are not in any training set. Every answer and every intermediate quantity is computed in code. The engine reproduces the published Standard Ultimate Life Table and the exact actuarial identities before any model is called, so the reference values are correct by construction rather than by assertion.

## Layout

- `experiments/actuarial_gen.py` the validated actuarial engine (Makeham Standard Ultimate Life Table, interest and life-contingent functions). Run it to see the ground-truth validation checks.
- `experiments/actuarial_problems.py` problem instances with code-computed answers, labelled anchor steps, reworded variants, and a mitigation scaffold.
- `experiments/actuarial_eval.py` runs models, grades final answers, localizes the first wrong step, and measures consistency.
- `experiments/common.py` cached calls to an OpenAI-compatible endpoint (OpenRouter).
- `experiments/actuarial_stats.py` Wilson confidence intervals and bootstrap intervals for the reported numbers.
- `experiments/actuarial_figures.py` renders the paper figures from the results.
- `results/summaries/` machine-readable results. Every number reported traces back to one of these.
- `paper/` the manuscript and figures.

## Reproducibility

Model responses are cached by a hash of the request, so reruns are free and reproduce the same outputs. Model identifiers and run dates are recorded with the results. API keys live in a local `.env` file that is never committed.

## Setup

Create a `.env` file in this folder with your key:

```
OPENROUTER_API_KEY=your_key_here
```

Then run the scripts from `experiments/` using the project virtual environment. Start with `python actuarial_gen.py` to confirm the ground-truth validation passes.
