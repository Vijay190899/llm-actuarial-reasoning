# Digit Forensics for LLM-Fabricated Numeric Tables

A study of whether simple, training-free digit-distribution tests can flag numeric data tables that were fabricated by language models, and how easily a model can be prompted to evade those tests.

## Question

Classical forensic accounting uses digit tests to spot invented numbers. First-digit tests follow Benford's law, and terminal-digit tests expect the last digits of genuine measurement data to be close to uniform. This project asks three things:

1. Do tables fabricated by language models deviate from these expectations more than real-world data does, and by how much.
2. Which test carries the signal. The hypothesis is that terminal-digit structure separates fabricated from real tables better than the well-known first-digit test.
3. Whether a model told to satisfy Benford's law, or to flatten its terminal digits, can defeat the tests, and whether a combined detector still holds.

## Status

Pilot stage. The real-data corpus and the digit-test module are built and checked. The fabrication run across the model set is gated: it expands to the full sample only if the pilot meets pre-registered criteria recorded in `brain/REDTEAM.md`.

## Model set

Two open models with published parameter counts (Llama 3.2 3B and Llama 3.3 70B) and three consumer frontier models (DeepSeek V3.1, Gemini 2.5 Flash, a frontier model). All are reached through a single OpenRouter key. Each model fabricates tables under four prompt conditions: plain, realistic, first-digit evasion, and terminal-digit evasion.

## Layout

- `experiments/` runnable scripts. `digit_tests.py` computes the forensic features, `load_real.py` builds the real corpus, `fabricate.py` generates fabricated tables, `detect.py` scores real against fabricated.
- `results/summaries/` machine-readable result files. Every number reported in the paper traces back to one of these.
- `brain/` project notes: state, decisions, findings, questions, and the red-team log.
- `experiment_design.md`, `council_protocol.md`, `pivot_ladder.md` the plan, the review protocol, and the roadblock policy.
- `paper/` the manuscript.

## Reproducibility

Every model response is cached on disk by a hash of the request, so reruns cost nothing and reproduce the same tables. Exact model identifiers and run dates are recorded with the results. API keys live in a local `.env` file that is never committed.

## Setup

Create a `.env` file in this folder with your OpenRouter key:

```
OPENROUTER_API_KEY=your_key_here
```

Then run the scripts from `experiments/` using the project virtual environment.
