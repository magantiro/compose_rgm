# K41 budgeted preference sweep — result, read once against the frozen gate

**Gate: `R1 ∧ R2 ∧ B`, preregistered in
`BUDGETED_PREFERENCE_SWEEP_PREREGISTRATION.md` before implementation.**

## Verdict: **FAIL**

| test | criterion | observed | |
|---|---|---|---|
| **B — budget** | ≤10,000 unique oracle evals on **every** source | max **6,888**; `within_budget` true 12/12; **cap never binding** | ✅ **PASS** |
| **R1 — quality** | bootstrap 95 % CI lower on `mean(r_HV)` > 0.90 | mean 5.018, CI lower **1.425** | ⚠️ passes, but see below |
| **R2 — breadth** | bootstrap 95 % CI lower on `mean(r_ND)` > 0.90 | mean 1.018, CI lower **0.737** | ❌ **FAIL** |

**Per the frozen stop rule: the compression branch closes. No second algorithm,
no re-specification, no rescue.**

## The budget mechanism itself worked exactly as designed

This part is unambiguous and worth keeping regardless of the verdict.

- **12/12 sources within budget.** Max spend 6,888 against a 10,000 cap.
- **`cap_was_binding` false on every source** — the hard cap never fired, so
  nothing here is an artifact of a mid-traversal stop or DFS order. That was the
  entire reason `K_sweep` was derived from the guard (`240 × 41 = 9,840`) rather
  than the observed median expansion count.
- `R_θ` shortlisting cut oracle spend from a median **75,910 → 4,113**, about
  **18×**, with no budget violation anywhere.

## R1 "passes" for the wrong reason — the ceiling is not a ceiling

**`mean(r_HV) = 5.018`. A retention ratio cannot exceed 1 if the denominator is
an upper bound.** It does here, on 10 of 12 sources, reaching **40.0× on source
006**. That is not a retention measurement; it is evidence the estimand was
mis-specified.

The mechanical reason is already documented and was found *before* this run,
when offline replay failed: **the budgeted sweep is not a sub-traversal of the
full sweep.** It scores a different candidate subset, so its partition has
different winners, so it descends a **different tree**. It is therefore free to
land on endpoints the full sweep never visited — and to expand *more* states:

| source | expansions, full sweep | expansions, budgeted |
|---|---|---|
| **006** | **14** | **76** |
| 000 | 60 | 71 |
| 008 | 109 | 131 |

Source 006's full sweep terminated after 14 expansions; the budgeted sweep ran
76. Calling the former a "full-information **ceiling**" is simply wrong: it is
the full-information *preference partition*, and the greedy trajectory it
induces is **not an upper bound on attainable hypervolume**.

**This does not convert the FAIL into a PASS.** R2 is measured on the same
ratio scale and fails on its own terms — the budgeted sweep genuinely produces
fewer nondominated points on 6 of 12 sources (as low as 0.515), and the CI
lower bound of 0.737 sits well under the 0.90 floor. The intersection–union
test fails whatever R1 is doing.

## What is NOT being done here

**No re-specified estimand is proposed in this document, and none should be
inferred from it.** Noticing that a denominator was mislabelled *immediately
after* the gate failed is precisely the situation the preregistration exists to
constrain. The stop rule was frozen in advance for this exact moment:

> collapse → **close the compression branch. No second algorithm.**

If the comparison is ever revisited, it requires a **fresh preregistration,
written before any outcome is re-read**, and it must confront the fact that the
two sweeps traverse different trees rather than papering over it.

## The honest scientific position, unchanged from the preregistration

Stated *before* this result was seen, so it is not a consolation:

> the **full-information preference sweep** demonstrates the latent richness of
> the controller, while the **five-weight / budgeted** controller demonstrates
> what is achievable at contemporary sample budgets.

That pair stands. P0c's finding is untouched: five weights badly undersample a
genuinely preference-rich controller (median 69.5 vs 3.5 endpoints, +64.2 % HV,
12/12). What K41 does **not** establish is that this richness can be exposed at
a 10,000-query budget with front breadth preserved.

## Provenance

12/12 shards, `editing_v2/r_theta_run/pareto_budgeted_sweep/`. `K_sweep = 41`
from `floor(10000/240)`, both inputs frozen before P0c ran. Q1–Q3 qualification
passed 17/17 before any outcome was opened. Per-source hypervolume uses a
reference from the union of both arms' endpoints so the ratio is within-source
and symmetric; these are development numbers and **not** the frozen panel
estimand.
