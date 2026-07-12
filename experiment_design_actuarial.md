# Diagnostic Evaluation of Frontier LLM Actuarial Calculation

**Field:** AI. **Niche:** AI x actuarial science (quantitative professional reasoning).
**Date:** 2026-07-12. **Status:** design, pre-G1.

## Positioning (the increment over prior work)

Two 2026 benchmarks exist and both measure ACCURACY only:
- ActuBench (arXiv 2604.20273): multi-agent generated actuarial items, MCQ + LLM-judge scoring. Confirmed it does NOT test self-consistency, does NOT localize where a multi-step calculation breaks, tests NO mitigation.
- Insurance LLM benchmark (arXiv 2511.07794): actuarial sub-items, found most models score below 40; reasoning models beat instruct.

Our contribution is diagnostic, not another accuracy score:
1. **Error-mechanism localization.** For multi-step problems, find the FIRST step where the model diverges from the code-computed reference, and classify the error (wrong formula, wrong table value, arithmetic slip, timing/convention error such as annuity-due vs annuity-immediate). This is the "where and why," which accuracy benchmarks cannot give.
2. **Self-consistency.** Same problem, semantically-equivalent rewordings (reorder givens, reword, rename entities); measure answer agreement. A high-stakes calculation should be invariant to phrasing.
3. **Mitigation.** One lightweight intervention (structured actuarial scaffold prompt and/or a calculator tool) tested for whether it removes the dominant error class.

## Why this is robust and high-prior (the properties digit forensics lacked)

- **Objective ground truth by construction.** Problems are procedurally generated; the reference answer AND every intermediate quantity are computed in code. No labeling ambiguity.
- **Anti-contamination by construction.** Randomized numeric parameters mean the exact items are not in any training set, so we measure reasoning, not memorization. This is a genuine methodological advantage over static benchmarks.
- **Failure is near-certain (high prior).** The insurance benchmark already shows sub-40 scores, so there will be errors to localize; the question is the structure of the errors, which is guaranteed to yield a result.

## Research questions

- RQ1 (accuracy): final-answer accuracy per problem family and model (reasoning vs non-reasoning).
- RQ2 (localization, primary increment): where does the first error occur, and what is the error-type distribution per family and model.
- RQ3 (consistency): answer agreement across semantically-equivalent rewordings of the same problem.
- RQ4 (mitigation): does the scaffold/tool intervention reduce the dominant error class and raise accuracy.

## Problem families (each with code-computed answer + labelled intermediate steps)

Compound interest / present value; level annuity-immediate and annuity-due (timing convention is a known trap); life annuity (uses a standard life table); net single premium and net level premium for term/whole life; simple reserves. Each family has a canonical solution path with named anchor quantities (discount factor v, annuity value a-double-dot, mortality-weighted EPV) that any correct method must produce, so localization does not depend on the model using our exact path.

## Metrics

Final-answer accuracy (within relative tolerance); first-divergence step index; error-type rate; cross-rewording answer-agreement rate; mitigation delta (accuracy and dominant-error-class rate).

## Kill / pre-registration gates (checked before scaling)

- **KC0 (linchpin): the generator must be validated against authoritative values BEFORE any model eval.** Cross-check every family against textbook example values and, where possible, an independent actuarial library. If our reference answers are wrong, the study is invalid. This is the first thing G1 must enforce.
- **KC1 (localization feasibility):** we must be able to reliably extract the model's intermediate quantities and match them to reference anchor quantities on a pilot of >= 80% of responses. If step extraction is unreliable, fall back to coarse localization (final-answer + single mid-anchor) or to the consistency+accuracy story.
- **KC2 (signal):** if pilot accuracy is uniformly ~100% (no errors to localize) OR ~0% (models refuse/garble), the localization story is empty; pivot emphasis to consistency (RQ3), which is robust regardless.

## Fallbacks within the topic (no single point of failure)

- If localization extraction is hard, the consistency audit (RQ3) alone is a robust, publishable result.
- If models are surprisingly accurate, "accurate but inconsistent across phrasings" is a clean finding.
- If models are uniformly poor, the error taxonomy + mitigation is the contribution.
- The annuity-due vs annuity-immediate timing error is a likely clean, systematic, conceptual failure (analogous to the documented tax AGI/taxable-income placement error); even alone it is a crisp finding.

## Feasibility

Generator in Python (first-principles actuarial math + a standard life table), validated. Prompts are text, cheap on OpenRouter; reasoning models cost more output tokens. Pilot ~5 models x ~10 problems x {base, reworded, mitigation} approx 150 calls, well under $1. 1-2 day analysis. Reuses the existing harness, repo, council, and pivot machinery.
