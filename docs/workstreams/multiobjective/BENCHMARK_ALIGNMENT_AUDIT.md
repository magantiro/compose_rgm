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
| 2 | allowed query / oracle budget | **FAIL** | Published budget **1,000** (HN-GFN's own). COMPOSE spends **2,187** per preference trajectory on its cheapest guided arm and **53,977** on its primary arm. And InversionGNN attributes **70,000** to HN-GFN — a **70×** disagreement between the two papers about the same method. |
| 3 | hypervolume reference point | **FAIL** | HN-GFN uses `ref_point = zeros`, over the *entire accumulated oracle-scored dataset*. InversionGNN states **no molecular reference point at all**. COMPOSE uses a frozen held-in p5 nadir with a p99 normalizer. Three conventions, one of them unstated. |
| 4 | top-K and preference weighting | **FAIL** | HN-GFN's MOBO weights are **5 i.i.d. Dirichlet draws resampled every round**; its own synthetic experiment uses 5 *evenly spaced* ones; InversionGNN uses 5 *uniformly random angles*; COMPOSE uses 5 *fixed frozen* ones. HN-GFN does not even agree with itself. |
| 5 | validity and canonicalization | **FAIL** | HN-GFN scores invalid molecules `0.` and **keeps them in the set**; InversionGNN **drops** them via a vocabulary-subset check that is not RDKit sanitization; COMPOSE's support is legal by construction. |
| 6 | seeds and reporting convention | **FAIL** | HN-GFN's `±` is over **3 random seeds**. InversionGNN's `±` is over **5 weight vectors** and the word "seed" never appears in the paper. Neither defines `±` as std or CI. The two dispersions are different quantities. |

**All six dimensions fail.** The audit was written expecting one or two to fail
and the rest to pass; the result is that no dimension aligns.

Sources for this section: HN-GFN repo @ `90078b8` and the paper text extraction
of arXiv:2302.04040v2; InversionGNN repo @ `cfdf1d9` and the HTML extraction of
arXiv:2503.01488v1. Line citations `hngfn.txt:N` / `inv.txt:N` refer to those
extractions; every quotation below was re-verified against them directly.

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

### The one possible repair — RESOLVED, and it closes

Before the extraction arrived, this audit recorded a distinction and committed to
it in advance, so the citation target could not be chosen after the fact:

> If InversionGNN reran HN-GFN inside its own harness under TDC oracles, then
> InversionGNN's table is internally oracle-consistent, and citing **that single
> table** would be a legitimate "as reported" source with one oracle behind every
> row. If it did not, cite each paper separately and state the oracle difference.

**It reran HN-GFN — but not at HN-GFN's published protocol, and the gap is 70×.**

InversionGNN's Table 3 carries an explicit `Oracle Calls` column, defined at
`inv.txt:428` as *"'A+B' which means allocate A Oracle call budget for
pretraining and B for optimization"*. Reading the flattened table rows directly
(`inv.txt:395-403` for HN-GFN, `inv.txt:413-417` for InversionGNN), the
GSK3β+JNK3 entries are:

| method as listed by InversionGNN | Nov | Div | APS | **Oracle Calls** |
|---|---|---|---|---|
| DST | 100% | 0.750 | 0.827 | 10K+5K |
| MOGFN-AL | 100% | 0.673 | 0.742 | 50K+20K |
| **HN-GFN** | 100% | 0.784 | 0.725 | **50K+20K = 70,000** |
| I-LS | 100% | 0.693 | 0.823 | 10K+5K |
| **InversionGNN** | 100% | 0.768 | **0.841** | **10K+5K = 15,000** |

HN-GFN's own paper states its budget as **1,000**: *"we consider starting with
|D0| = 200 random molecules and further querying the oracle N = 8 rounds with
batch size b = 100"* (`hngfn.txt:489-491`), matching `main_mobo.py:50-52` exactly.

So the two papers disagree by **70×** about how many oracle calls HN-GFN spends.
The two reported hypervolumes disagree accordingly: HN-GFN reports its own as
**0.669 ± 0.061** (Table 5), InversionGNN reports HN-GFN's as **0.592 ± 0.042**
(Table 4).

**This closes the last route to a valid head-to-head table**, in both directions:

- Cite each paper's own numbers → different oracles (dimension 1).
- Cite InversionGNN's single table → internally consistent, but it does not
  report HN-GFN under HN-GFN's published protocol, so quoting it as "HN-GFN's
  result" would misrepresent HN-GFN by a factor of seventy in budget.

The 0.592-versus-0.669 gap must be recorded as a **protocol mismatch, not a
reproduction failure**, and neither paper's number may be presented as *the*
HN-GFN GSK3β/JNK3 result.

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

**The per-trajectory column is a conservative lower bound, and the direction is
worth stating.** `native_oracle_calls` counts *distinct* molecules scored, and
distinctness is measured per arm across all five preference trajectories. Where
two preferences visit the same molecule, that molecule is counted once. So
dividing by five *under*-states what a single standalone trajectory would cost,
and the true figure is somewhat higher. The finding is therefore reported
against the number that flatters COMPOSE.

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

## Dimension 3 — three hypervolume conventions, one of them unstated

**HN-GFN: reference point is the zero vector**, at all three construction sites
in the repo — `dataset.py:37`, `main_mobo.py:397`, `utils/metrics.py:87`, each
`Hypervolume(ref_point=torch.zeros(len(args.objectives)))`. The paper agrees:
hypervolume is *"bounded from below by the preference point (0, 0)"*
(`hngfn.txt:436-438`; "preference point" is evidently a typo for "reference
point").

Crucially, the headline metric is computed over the **entire accumulated
oracle-scored dataset** — `dataset.py:131-135 compute_hypervolume()` runs over
`self.scores`, which grows to 200 + i×100 — not over a fixed-size front. So
HN-GFN's HV is a function of everything it has ever evaluated.

**InversionGNN: no molecular reference point is stated anywhere.** The string
"reference point" occurs exactly once in the paper, at `inv.txt:208`: *"We set
the reference point as (1,1) in this task."* That sentence sits in the **synthetic**
task's metrics paragraph, and that task **minimizes** `1 - exp(-||x ∓ 1/√n||²)`,
so (1,1) is an upper bound — structurally incompatible with the molecular
maximization setting. The paper's only provenance for its molecular HV is
*"we follow the recent multi-objective method HN-GFN … and report the
Hypervolume (HV)"* (`inv.txt:453`), which is not a specification. There is **no
hypervolume code anywhere in the repo's `molecular/` directory**.

**COMPOSE:** a frozen held-in **p5 nadir** with a **p99 normalizer**, from
`diagnostics/pareto_tradeoff_census.json :: frozen_scales`, deliberately *not*
bounded by 1.

Three different conventions, one entirely unstated, and none of them the frozen
COMPOSE one. An HV column carrying all three would be three different quantities
under one header.

## Dimension 4 — nobody uses the same weights, and HN-GFN differs from itself

| source | preference vectors | spacing |
|---|---|---|
| **HN-GFN MOBO** (`main_mobo.py:483`) | `np.random.dirichlet(alpha_vector, 5*(2**(n_obj-2)))` → 5 for two objectives | **i.i.d. random, resampled every outer round** |
| **HN-GFN synthetic** (`main.py:493`) | `circle_points(K=5, min_angle=0.1, max_angle=π/2-0.1)` | **evenly spaced in angle**, then L1-normalized |
| **InversionGNN** (Algorithm 3) | 5, *"to serve as independent trials"* (`inv.txt:430`) | *"Generate a uniformly distributed variable, u … θ = (π/2)u"* — **uniformly random angle** |
| **COMPOSE** (`pareto_control_app.py:103`) | `(0.1, 0.3, 0.5, 0.7, 0.9)` | **fixed and frozen** |

The GSK3β/JNK3 MOBO result — the one a Panel B row would cite — uses **random,
per-round-resampled** weights. Only HN-GFN's *other* experiment uses evenly
spaced ones. A reader who assumes "5 evenly spaced preferences" from the paper's
§5.1 language and applies it to the §5.2 numbers is reading a different protocol
than the one that produced them.

**Top-K also differs three ways.** HN-GFN keeps the top 20 by acquisition value
per weight, 5 × 20 = 100 (`main_mobo.py:382,390`), ranked by the **scalarized UCB
acquisition value, not the oracle score**. InversionGNN uses top-100 for APS,
top-20 for Non-Uniformity (`inv.txt:433`), and *"all the solutions of all
weights"* for Table 3 (`inv.txt:453`) — three conventions inside one paper.

## Dimension 6 — the two `±` symbols denote different quantities

**HN-GFN:** *"Each experiment is repeated with 3 random seeds"* (`hngfn.txt:491`).
Its `±` is dispersion **over seeds**.

**InversionGNN:** *"the 5 different weight vectors to serve as independent
trials"* (`inv.txt:430`) and *"For each weight, we compute the results separately
and report the average results across all 5 trials"* (`inv.txt:433`). Its `±` is
dispersion **over preference weights**. The word "seed" does not appear anywhere
in the paper.

These are not the same statistic. One measures run-to-run variability at fixed
preference; the other measures spread *across the front* — which for a
preference-conditioned method is part of the signal, not noise. Placing them in
one column, or comparing their widths, would be unsound.

Neither paper states whether `±` is a standard deviation or a confidence
interval. That remains `UNVERIFIED` for both, and no interval derived from either
may be given an interpretation the source does not license.

## Dimension 5 — the scored set is not the same set

HN-GFN canonicalizes by a SMILES round-trip, `utils/chem.py:37-43`:

```
37: def standardize_smiles(mol):
38:     try:
39:         smiles = Chem.MolToSmiles(mol)
40:         mol = Chem.MolFromSmiles(smiles)
41:         return mol
42:     except Exception:
43:         return None
```

applied at `oracle/scorer/scorer.py:23`, and then, at `scorer.py:35`:

```
35: scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
```

An unparseable molecule receives **0.0 and stays in the batch**. It counts
against the budget and enters the reported set.

**InversionGNN does the opposite, and not with RDKit.** `chemutils.py:177-180`:

```
177: def is_valid(smiles):
178:     word_lst = smiles2word(smiles)
179:     word_set = set(word_lst)
180:     return word_set.issubset(vocabulary)
```

That is a **vocabulary-membership test, not sanitization**. A molecule failing
it is **dropped from the population entirely** (`inference_utils.py:98-99,
111-112, 133-134, 154-155` all `return set()`), never scored, never counted.

**COMPOSE's support is legal by construction** — every canonical successor in the
fiber is a valid molecule, which is why Lane 4's `native_oracle_calls` and
Workstream D's `unique_valid_canonical_evaluations` agree numerically for COMPOSE
and would not agree for a method that can emit an invalid proposal.

So there are three different treatments of an invalid molecule: **scored as zero
and retained** (HN-GFN), **silently dropped** (InversionGNN), **cannot occur**
(COMPOSE). The three "top-K sets" are drawn from populations constructed by
different rules, and their denominators are not the same.

One consequence worth noting because it cuts *against* over-reading this
dimension: since HN-GFN's HV reference point is `(0,0)`, a zero-scored invalid
molecule contributes exactly nothing to its hypervolume. So the validity
difference distorts HN-GFN's HV less than it might appear — it distorts the
diversity and top-K statistics, and it makes the budget denominators
incomparable.

---

## What we do instead

### Published values are labelled contextual, non-head-to-head

Every external GSK3β/JNK3 number that appears in the paper carries its oracle
provenance and its budget, and is never placed in a shared-header column with a
COMPOSE number. The caption states which dimension failed. Concretely:

> HN-GFN and InversionGNN report GSK3β/JNK3 results under **different oracle
> implementations** (1024-bit versus 2048-bit ECFP4, different fitted models),
> **different hypervolume reference conventions** (zero vector over an
> accumulating dataset; unstated), **different preference-weight schemes**
> (per-round random Dirichlet; uniformly random angle), and **budgets differing
> by 70×** — HN-GFN reports 1,000 oracle calls for itself, while InversionGNN
> attributes 50K+20K to it. Their `±` intervals denote different quantities
> (3 seeds; 5 weight vectors). COMPOSE's exhaustive-fiber control spends
> 2,187–53,977 evaluations per preference trajectory. These values are reported
> for context and are **not** head-to-head, with each other or with COMPOSE.

**Citation rule, now settled by the extraction.** Cite **each paper's own number
for its own method only**, each carrying its own oracle, budget, reference point
and `±` semantics. Do **not** quote InversionGNN's Table 4 value of 0.592 as
"HN-GFN's hypervolume" — that run used a 70× different budget than HN-GFN
published. Do **not** quote HN-GFN's 0.669 beside InversionGNN's 0.763 as a
comparison. The 0.592-versus-0.669 gap is a **protocol mismatch, not a
reproduction failure**, and saying which it is protects both papers.

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
