# Post-smoke plan — preregistered, **NOT AUTHORISED**

**Artifact status: `DESIGN_ONLY`.** Nothing here is implemented. Nothing here
touches the running smoke. It is committed **before the smoke's results exist**
so the decision tree in §5 is fixed in advance rather than chosen once the
numbers are visible — which is the entire reason it is written down now.

Recorded 2026-08-13, when 1 of 12 sources had completed and no aggregate
existed.

---

## 1. Batch the scorer — an exact vectorisation, gated by a regression test

`objective_vector()` calls `oracle.margin_many([key])` with a **one-element
list**. That is an implementation defect, not an algorithmic property.
Replacing ~586 single-molecule forwards with one batched forward over 586
molecules is an **exact vectorisation of the same computation**.

**Gate, under fixed seeds. All six must hold; the first five are identity
conditions.**

| # | condition | if it fails |
|---|---|---|
| 1 | identical canonical candidate set | not a vectorisation — must not land |
| 2 | identical potency scores to tolerance | not a vectorisation — must not land |
| 3 | identical selected action | not a vectorisation — must not land |
| 4 | identical complete trajectory | not a vectorisation — must not land |
| 5 | identical `N_drd2` | not a vectorisation — must not land |
| 6 | **lower** scorer invocation / batch count | no benefit — pointless |

If any of 1–5 differ it is a **change of computation wearing a performance
label**, and it does not land. Once it lands, **all future Pareto work uses the
batched implementation.**

### What batching does NOT fix — and this stands as a finding

Even perfectly batched, the algorithm still requests **~586 × (13–23) distinct
potency evaluations per source**. If potency were a docking workflow or a
wet-lab assay, batching would save **nothing conceptually**. **Exhaustive
full-fiber preference control is genuinely objective-query-intensive.** That is
an algorithmic property, and it survives the implementation fix.

## 2. Split the counters — even where they are numerically identical today

Replace the fused counter with:

```
N_drd2_requests        N_drd2_unique        N_drd2_evaluator_calls
N_descriptor_requests  N_scorer_batches
```

This is **not an experiment change**. It makes future accounting *incapable* of
silently conflating quantities that happen to coincide.

The hazard is the one the audit already surfaced: a version that skips DRD2 when
developability binds would make `N_drd2` and `N_descriptor` diverge **while the
fused counter kept reporting a single number**, and nothing would flag it.

> **The completed smoke's unified count remains reconstructible.** Nothing
> short-circuited in the code that produced it: `objective_vector()` computed
> DRD2, QED and cLogP together on every evaluation, unconditionally. So for
> these shards `N_drd2 = N_descriptor = N_all_objective =
> native_oracle_calls`, exactly, and the split counters can be back-filled from
> the existing artifact without rerunning anything.

## 3. Do NOT measure the pre-alias mark count

The oracle sees **~586 distinct canonical successors**, and that is the quantity
relevant to expensive scoring. The mark-to-successor collapse ratio is
interesting for **kernel** efficiency and **does not change the oracle-demand
conclusion**.

**No kernel instrumentation will be added for it.** Closed, not deferred.

## 4. Do NOT headline the raw ratio

At matched kernel computation, exhaustive control uses **vastly more objective
information** than generate-and-rank. That is correct, and it is **not** the
same statement as *"COMPOSE needs 3,000× more oracle evaluations for the same
Pareto quality"* — **we have not seen Pareto quality yet.**

> **The scientific object is HV versus unique potency evaluations.** Maybe 10×
> the queries buys 5× the coverage; maybe 1,000× buys nothing. **The curve
> decides.** Until the curve exists, the counts appear raw, with the withdrawal
> caveat attached, and never as a headline ratio.

## 5. The decision tree — fixed BEFORE results

### 5a. If Pareto control is WEAK

*Poor HV, weak preference ordering, poor coverage **and** high query demand.*

→ **Pareto is not a strong pillar in its current form.** We do **not** build an
efficiency method to rescue a capability that is not compelling. No efficiency
experiment is justified by a weak capability.

### 5b. If Pareto control is STRONG and oracle demand is the only ugly axis

→ **One** efficiency experiment is justified, and it starts with the simplest
structure-preserving option:

```
legal fiber  ->  reference-prioritised shortlist under R_theta  ->  expensive
                 scoring and control
```

The question: **how much frontier is retained when expensive evaluation is
restricted to a small `R_theta`-prioritised subset?**

That directly tests **why learning `R_theta` is useful**, and connects to the
already-preregistered reference-law ablation this lane hosts. It is **not a new
`h_phi` project.**

### 5c. NO K-SEARCH — binding

**Do not** run `K in {4, 8, 16, 32, 64, 128}` and select the attractive point.

**Pre-register ONE operating point**, derived from a *desired reduction in
oracle demand* — not from which value performs best. Then report **fraction of
HV retained** against **fraction of potency evaluations retained**.

A sweep followed by a choice is a K-search with the selection step hidden in the
write-up, and it is the same defect family this lane has now caught seven times.

---

## Authorisation status

| item | status |
|---|---|
| batch the scorer | **NOT AUTHORISED** — post-smoke |
| split the counters | **NOT AUTHORISED** — post-smoke |
| pre-alias mark count | **CLOSED** — will not be measured |
| headline the ratio | **FORBIDDEN** |
| efficiency experiment | **CONDITIONAL** on 5b, and not otherwise |

Nothing in this document is started until the frozen smoke has landed and been
reported against the six questions, the five watch-items and the accounting
audit.
