# Benchmark rulings after the editing-competence and SA audits

**Project-wide. Binds Lanes 2, 3, 5, 6 and the main lane.** Extends
`docs/COMPARATOR_ROLES_CANONICAL.md` and
`docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`. Nothing here reopens a frozen result.

---

## 1. DDSBM is the tier-1 conventional framework comparison

**Task: DDSBM's ZINC logP 2→4.** Its paired CSV ships in the repo, so the
23,936 / 5,984 split is exactly reproducible, and its metrics are model-agnostic.

> **COMPOSE runs alone. DDSBM's Table 1 is cited as reported.**

This fills the `FRAMEWORK_NEIGHBOR` slot that has been empty since the role
amendment, and it does so on a **distribution-shift task, not on DRD2** — the
part most likely to be misread, and therefore recorded in the status and the
manifest as well as here.

**NLL is excluded from any quoted comparison.** It is defined against DDSBM's
*own* reference process, so quoting it would compare each method to a different
yardstick — and it happens to be the column carrying their headline win.
Excluding it is not a courtesy; including it would be an error.

## 2. The similarity-constrained DRD2 task is DEMOTED from primary

**No longer a headline external comparison.** Secondary competence/stress task
only, unless and until an explicit **mutually fair resource protocol** is defined.

The reason is structural, not incidental. That benchmark budgets **candidates,
not oracle calls**, because its native methods never query the property oracle
inside the sampling loop — GrIDDD's DRD2 calls are dataset labelling, post-hoc
metrics and success evaluation. Their algorithmic oracle requests are **zero**;
COMPOSE's would be thousands.

> **The absence of an oracle cap in a published protocol is an unstated
> assumption, not permission.**

Same shape as the HN-GFN surrogate asymmetry, now pointed at us. Spending
thousands of queries under a protocol that assumed none and calling it matched
would be exactly the error we refused to accept from others.

## 3. T3 — do not rescue the 1,000-evaluation budget

**Do not silently change it. Do not force-run it either.**

If a single budget-6 trajectory already exposes ~3,500 candidates, then 1,000
evaluations does not test *"COMPOSE under a hard budget"* — it prevents the
native controller from completing even a meaningful short run. The comparison
would be structurally uninterpretable, not merely unfavourable.

**Classify and preserve as:**

```
T3: FEASIBILITY-FAILED UNDER FROZEN 1K BUDGET
```

**Barred:** retroactively moving 1k → 5k → 10k. Any future query-efficiency
experiment must be defined **prospectively**, with an externally motivated
budget, and ideally only after the final Pareto / query-efficient controller is
frozen. Rewriting a frozen budget until the algorithm fits is how a freeze stops
meaning anything.

## 4. InVirtuoGen — permanent provenance restriction

> **No InVirtuoGen-generated output may ever become `R_θ` training data.**

Its `LICENSE` contradicts itself (CC BY-NC-SA 4.0 vs Attribution-NonCommercial
4.0), GitHub reports `NOASSERTION`, and its weights terms forbid using its output
to train molecular-generation models. Recorded in the provenance ledger and
permanent.

Whether it can be cited, or used for an evaluation-only comparison, is a separate
question. **There is no scientific reason to let its outputs near the training
corpus regardless** — which is what makes this cheap to honour now and expensive
to discover later.

## 5. SA constraint — run the census, do not yet run the experiment

**Promising enough for the fiber census. Not yet promising enough for a
claim-bearing experiment.**

The geometry is what a good feasibility gate should look like: with **τ = 3.0
fixed before measurement** from the external CDD definition, median starting
slack is **0.1097**. The constraint is therefore likely to *bite* rather than be
vacuous, while support-retention failure remains **genuinely possible**. A gate
that could not fail would not be a gate.

Once the Gate-0 mount-path defect is resolved, run the census under the
already-frozen A2 gates (retention ≥ 0.10, mask-empty ≤ 0.05, spread > 1/3, plus
demonstrable headroom). **If it fails, kill SA. Do not shop τ.**

Two of the lane's own choices are endorsed and become binding:

- **Do not proxy ΔSA per edit.** Local proxy reasoning has already badly
  mischaracterised what the exact molecular fiber can do — a proxy once called a
  DRD2 threshold unreachable in four edits and the real fiber climbed +4.42 in
  three.
- **The objective is potency subject to SA ≤ τ**, never SA itself. "Optimize SA
  subject to SA ≤ τ" is degenerate.

## 6. Unchanged

**P0c stays blinded until all 12 sources finish.** The expensive sources
finishing last is expected and triggers no interpretation or intervention.

**cLogP n=48 still waits** on the forced real-kill and torn-write checkpoint
regression. A synthetic harness interruption and an actual `SIGKILL` mid-commit
are **not equivalent failure modes**, and only the second is what Modal
preemption actually does.
