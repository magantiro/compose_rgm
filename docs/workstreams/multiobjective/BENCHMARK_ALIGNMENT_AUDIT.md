# Published-protocol alignment audit — GSK3β / JNK3

**Artifact status: `DESIGN_ONLY`.** No compute. No baseline executed. Nothing
installed. Every cell resolves to a file:line in an official repository, a paper
section, or a committed artifact in this repo.

**Question.** Can GSK3β/JNK3 be frozen as Panel B — COMPOSE run alone under it,
external numbers cited **as reported** from the original papers?

**Verdict: NO. Freeze declined.** Two dimensions fail independently, and each is
sufficient on its own. The published GSK3β/JNK3 values are therefore
**contextual, non-head-to-head**.

The freeze was evaluated on external protocol overlap alone. No COMPOSE outcome
entered the decision, and the P3/P4 repair was not consulted or waited for.

---

## Verdict table

| # | dimension | verdict | one-line reason |
|---|---|---|---|
| 1 | oracle files / score definitions | **FAIL** | HN-GFN scores GSK3β/JNK3 with a **1024-bit** ECFP4 and its own RandomForest pickles; TDC — which InversionGNN calls — uses a **2048-bit** ECFP4 and different pickles. Different input dimension, therefore necessarily different models. |
| 2 | allowed query / oracle budget | **FAIL** | The published budget is **1,000** true-oracle evaluations. COMPOSE's *cheapest guided* arm spends **2,187** per single preference trajectory; its primary arm spends **53,977**. COMPOSE cannot complete one trajectory inside the whole published budget. |
| 3 | hypervolume reference point | `PENDING` | extraction in progress |
| 4 | top-K and preference weighting | `PENDING` | extraction in progress |
| 5 | validity and canonicalization | **FAIL (contributory)** | HN-GFN assigns invalid molecules a score of `0.`, inside the scored set. COMPOSE's support is legal by construction and has no invalid endpoints, so the two denominators are not the same set. |
| 6 | seeds and reporting convention | `PENDING` | extraction in progress |

Dimensions 1 and 2 already settle the verdict. 3, 4 and 6 are being completed
for the record, because a future benchmark decision will need them and because a
dimension left blank tends to be read later as a dimension that passed.

---

## Dimension 1 — the two papers do not share an oracle

### What TDC's oracle is

`tdc/chem_utils/oracle/oracle.py`, read from
`raw.githubusercontent.com/mims-harvard/TDC/main` on 2026-08-13:

```
682:    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
686:    gsk3_score = gsk3_model.predict_proba(fp)[0, 1]
```
```
710:        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
714:        jnk3_score = self.jnk3_model.predict_proba(fp)[0, 1]
```

Model files, lines 660-664 and 701-706:

```
660: def load_gsk3b_model():
661:     gsk3_model_path = "oracle/gsk3b.pkl"
662:     if SKLEARN_VERSION >= version.parse("0.24.0"):
663:         gsk3_model_path = "oracle/gsk3b_current.pkl"
```

### What HN-GFN's oracle is

`oracle/scorer/kinase_scorer.py:18` loads `kinase_rf/%s.pkl`, and the
fingerprint comes from `utils/chem.py:62-66`:

```
62: def fingerprints_from_mol(mol):
63:     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
```

Its shipped models, hashed 2026-08-13:

| file | sha256 | bytes |
|---|---|---|
| `oracle/scorer/kinase_rf/gsk3b.pkl` | `60b2e6ebafd151d0c4e50ef6363eeba9d971c6467383e8dae51a6bf0b53be28c` | 97,423,867 |
| `oracle/scorer/kinase_rf/jnk3.pkl` | `a03dd44a89e922250bd0827c44f480be8e8685c93d43b17557d1dcd42e9b775c` | 37,529,090 |

(The repo also ships `oracle/gsk3b.pkl` and `oracle/jnk3.pkl` at top level.
Those are **not** the scorer's models — no code path references them; the only
reference to a kinase pickle anywhere in the repo is `kinase_scorer.py:18`.
They appear to be leftovers from the MARS lineage the scorer header cites.)

### What InversionGNN's oracle is

`molecular/denovo.py:9,25-26`:

```
 9: from tdc import Oracle
25: jnk = Oracle('JNK3')
26: gsk = Oracle('GSK3B')
```

Confirmed identically in `molecular/labelling.py:18-23`, `molecular/rebuttal.py:5-6`
and `molecular/inference_utils.py:11`. InversionGNN uses TDC's oracle
throughout.

### Why this is fatal rather than cosmetic

A RandomForest trained on 1,024 features cannot consume a 2,048-feature vector;
sklearn raises. So these are not two implementations of one function with a
tolerance between them — they are **two different functions**, trained on
different feature spaces, returning different numbers for the same molecule.

> **HN-GFN's published GSK3β/JNK3 values and InversionGNN's published GSK3β/JNK3
> values were not produced by the same oracle.** They are already not
> head-to-head with each other, before COMPOSE is mentioned at all.

Placing both in one table under a shared column header would assert an
equivalence neither paper established.

### A second, independent hazard inside TDC

TDC selects `gsk3b.pkl` or `gsk3b_current.pkl` **based on the installed sklearn
version** (lines 661-664, 703-706). So "the TDC GSK3β oracle" is not one object;
it is two, chosen by an environment variable nobody reports. Any paper citing
`Oracle('GSK3B')` numbers has an unstated sklearn dependency, and neither paper
states which pickle it used.

This is the same class of defect the project already fixed for DRD2 by
extracting parameters into a version-independent numpy artifact
(`artifacts/oracles/drd2_svm_v1`, whose manifest notes the extracted arrays
"cannot drift with the environment"). No equivalent extraction exists for
GSK3β/JNK3.

### Could alignment be manufactured?

Not by us. Rerunning HN-GFN under TDC oracles is precisely the 10-13 V100-hour
rerun the redirect forbids, and it would produce numbers that are ours rather
than the authors'. **Declined.**

### But one repair may already exist, and it changes what we cite

There is a distinction worth keeping sharp:

- **Combining HN-GFN's own reported numbers with InversionGNN's own reported
  numbers is invalid**, for the reason above. That is settled.
- **If InversionGNN reran HN-GFN inside its own harness, under TDC oracles**,
  then InversionGNN's table is *internally* oracle-consistent, and citing **that
  single table** — rather than two papers — would be a legitimate "as reported"
  source with one oracle behind every row.

Which of these holds is `PENDING` extraction. It does **not** change the verdict,
because dimension 2 fails independently and by one to two orders of magnitude.
It changes only **what the contextual citation should point at**:

> If InversionGNN reran HN-GFN: cite InversionGNN's table as the single source,
> naming TDC's oracle and the budget.
> If it did not: cite each paper separately, and state that they used different
> GSK3β/JNK3 implementations.

Recording the distinction now, before the answer arrives, so the citation is not
chosen after the fact.

---

## Dimension 2 — the budget is off by one to two orders of magnitude

### The published budget

**HN-GFN**, verified in code — `main_mobo.py:50-52`: `num_init_examples=200`,
`num_outer_loop_iters=8`, `num_samples=100`. The true oracle is called once per
outer round at `main_mobo.py:393`. Total: **1,000 true-oracle evaluations.**

The paper frames the setting as "an evaluation budget of N rounds with fixed
batches of size b", starting from "a random initial dataset".

**InversionGNN's** stated budget for the two-objective setting is `PENDING`
extraction; the paper describes surrogate pretraining calls plus
`N_weight × 1,000` optimization calls, so its total is larger than HN-GFN's.

**This does not soften the verdict, and the arithmetic says why.** Even against
PMO's far more generous 10,000-call convention, COMPOSE's primary arm spends
**5.4× the entire budget per trajectory**. The conclusion is stable across every
budget any of these benchmarks uses, which is precisely why it is reported as a
property of the benchmark *class*.

### What COMPOSE spends

Computed from the committed artifact
`diagnostics/pareto_semantic_oracle_accounting.json` (12 sources, 5 preferences,
`BUDGET = 6`), on `native_oracle_calls` — the distinct-valid-canonical counter,
which is the one comparable to a benchmark's oracle budget:

| arm | per source (mean) | **per single preference trajectory** | vs the 1,000-call budget |
|---|---:|---:|---:|
| `unguided` (preference-blind floor) | 5.0 | 1.0 | 0.001× |
| `gen_rank@greedy` | 2.6 | 0.5 | 0.0005× |
| `gen_rank@verified` | 57.5 | 11.5 | 0.01× |
| **`greedy_pref`** | 10,936 | **2,187** | **2.2×** |
| **`verified_pref`** (primary) | 269,884 | **53,977** | **54×** |

A full five-preference COMPOSE front costs **270× the entire published budget**
on the primary arm, and **11×** on the cheapest guided arm.

### Why this is structural, not an implementation inefficiency

COMPOSE's control interrogates the **exact legal successor fiber at every
decision**. The census fiber width is ~586, and the smoke records 612.2 unique
evaluations per kernel call against it — so the counter is sound and the cost is
what exhaustive-fiber control *is*. A budget-6 trajectory necessarily touches
roughly `6 × 586 ≈ 3,500` candidates before any rollout.

This is the project's own recorded expectation, met exactly:

> COMPOSE covers the Pareto front with far fewer molecular trajectories.
> COMPOSE covers the Pareto front with *more* oracle evaluations.

Both remain true. It is a legitimate and informative property, not a defect —
and it is the reason a 1,000-call benchmark cannot host COMPOSE's primary arm.

### The generalization that matters more than GSK3β/JNK3

Standard molecular-optimization benchmarks budget oracle calls in the
**10³–10⁴** range; PMO's convention is 10,000. COMPOSE's primary arm needs
~5.4 × 10⁴ **per trajectory**.

> **Changing benchmark does not fix this.** The misalignment is with
> oracle-budget benchmarking as a class, not with GSK3β/JNK3 in particular.

The one recorded idea that would change the arithmetic — reference-prioritized
shortlisting, `legal support → R_θ plausibility shortlist → expensive goal
evaluation → control` — is already in `docs/QUEUED_EXPERIMENTS.md` as
`QUEUED_CONDITIONAL — NO DESIGN / NO RUN`. This audit identifies it as the
blocking dependency for any head-to-head oracle-budget comparison. **It is named
here as a dependency and is not designed, scoped or run.**

---

## Dimension 5 — the scored set is not the same set

HN-GFN, `oracle/scorer/scorer.py:36`:

```
36: scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
```

An unparseable molecule receives **0.0 and stays in the batch**. It counts
against the budget and enters the reported set.

COMPOSE's support is legal by construction — every canonical successor in the
fiber is a valid molecule, which is why Lane 4's `native_oracle_calls` and
Workstream D's `unique_valid_canonical_evaluations` agree numerically for
COMPOSE and would not agree for a method that can emit an invalid proposal.

So the two "top-K sets" are drawn from populations with different validity
semantics. On its own this would be a caveat; alongside dimensions 1 and 2 it is
a third reason the sets are not interchangeable.

---

## What we do instead

### Published values are labelled contextual, non-head-to-head

Every external GSK3β/JNK3 number that appears in the paper carries its oracle
provenance and its budget, and is never placed in a shared-header column with a
COMPOSE number. The caption states which dimension failed. Concretely:

> HN-GFN and InversionGNN report GSK3β/JNK3 results under **different oracle
> implementations** (1024-bit versus 2048-bit ECFP4, different fitted models) and
> at a **1,000-call budget**, against which COMPOSE's exhaustive-fiber control
> spends 2,187–53,977 evaluations per preference trajectory. These values are
> reported for context and are not head-to-head.

### Panel B is not frozen on GSK3β/JNK3

And on this evidence it should not be frozen on any oracle-budget benchmark
while COMPOSE's primary arm costs 5.4 × 10⁴ evaluations per trajectory.

### The fallback, in preference order

1. **No Panel B numeric row; contextual citation only.** Cheapest, most
   defensible, and consistent with the standing rule that `N/A` is preferable to
   a distorted adaptation. The Pareto claim rests on Panel A, which is where the
   scientific content is.
2. **A resource frontier instead of a point.** Report COMPOSE's HV against
   oracle evaluations as a curve on TDC's GSK3β/JNK3 oracles, with the published
   external results drawn as clearly-labelled reference markers at their own
   budgets and their own oracles. This follows the existing project rule to
   report "fraction of quality retained against fraction of resource spent,
   which is a curve with a meaning, rather than a single ratio at a chosen
   point". It is honest about the budget gap instead of hiding it, and it needs
   no rerun of anyone's method. **It still requires a GSK3β/JNK3 oracle
   extraction that does not exist yet** — see cost below.
3. **Do nothing until reference-prioritized shortlisting exists.** If a
   shortlist ever brings COMPOSE's per-trajectory demand near 10³, the
   head-to-head becomes available on its own merits. Not a reason to build it
   now, and explicitly not a design.

**Recommendation: option 1**, with option 2 available if the lead wants a figure
rather than a sentence.

### Not recommended

- Rerunning HN-GFN or InversionGNN under a common oracle. Forbidden by the
  redirect, expensive, and it would replace the authors' numbers with ours.
- Reducing COMPOSE's `BUDGET`, shortlist, or fiber interrogation to fit a
  benchmark. That is changing the method to fit the table, and the frozen
  objects are not this lane's to touch.
- Picking a different two-objective task hoping the budget aligns. It will not;
  the gap is with the benchmark class.

---

## Cost of running COMPOSE alone under a GSK3β/JNK3 benchmark

Reported for the record, since it was asked, and **not** a recommendation to run.

**Compute.** From the smoke's own measurement, one source costs ~58 minutes on 8
CPUs for all five arms at `BUDGET = 6` with five preferences, of which ~46 min is
kernel enumeration and ~12 min scoring. A 12-source held-in panel is therefore
roughly **12 core-hours**, CPU-only — well within reach and consistent with the
lane's CPU-only constraint. Compute is **not** the obstacle.

**Oracle work.** ~270,000 unique evaluations per source on the primary arm,
~3.2 million across 12 sources. GSK3β/JNK3 are RandomForest models, materially
slower per call than the DRD2 SVM.

**The actual blocker is an artifact that does not exist.** COMPOSE cannot call
`tdc.Oracle` — installing PyTDC into the project environment is barred, and TDC's
sklearn-version-dependent model selection would make the oracle drift with the
environment, which is the exact failure `artifacts/oracles/drd2_svm_v1` was
built to prevent. Running COMPOSE on TDC oracles therefore requires a
**version-independent extraction of the GSK3β and JNK3 RandomForests**, done once
in a throwaway venv, hash-bound in a committed manifest, with a parity panel —
the same route already taken for DRD2.

That extraction is **strictly harder than DRD2's**: an RBF-SVM is a few dense
arrays, while these are large RandomForests (97 MB and 38 MB) whose exported
tree structure must reproduce `predict_proba` exactly. Estimate: **1-2 days of
engineering, zero GPU, plus a parity test.** It is not authorized and not
started.

Given that dimensions 1 and 2 fail regardless, **that extraction should not be
built to serve this comparison.** It would only be worth building if COMPOSE
needs TDC oracles for some independent reason.
