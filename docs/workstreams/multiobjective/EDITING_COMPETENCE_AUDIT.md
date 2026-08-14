# Source-conditioned editing competence — protocol-alignment audit

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute.

**The question, and only this:** ignoring Pareto entirely, is COMPOSE a credible
source-conditioned molecular editor? **Reviewer insurance and conventional
competence — not another scientific pillar**, and deliberately not built like one.

**The set is two external methods.** DDSBM as `FRAMEWORK_NEIGHBOR` if the
protocol genuinely aligns, plus **one** practical editor. No third.

---

## Headline

> **COMPOSE could physically enter the classic similarity-constrained DRD2
> benchmark tomorrow — it already holds every frozen component — and it should
> not be entered as tier 1.** The protocol carries no oracle-call budget because
> its native methods do not query the oracle at inference. COMPOSE queries it at
> every decision. The absence of a budget is an unstated assumption, not
> permission.

That is the mirror image of the HN-GFN finding, pointed at us. There, the
baseline had a hidden amortization advantage and I refused to let it look cheap.
Here COMPOSE would have the hidden advantage, and the same refusal applies.

---

## Part 1 — can COMPOSE enter a published protocol directly?

### The candidate: the classic Jin et al. similarity-constrained DRD2 task

Protocol as stated by GrIDDD (paper §5.2), which runs it natively:

> "selecting 800 molecules from the test set with DRD2 activity score ≤ 0.05.
> For each molecule, we sample **20 candidates** from different latents. We
> consider the optimization successful if at least one among the candidates has a
> DRD2 score ≥ 0.5 and a fingerprint Tanimoto similarity with the starting
> molecule ≥ 0.4. We report the success rate across the 800 molecules."

### What COMPOSE already holds — verified, hash-bound, committed

This is not a plan to build alignment. It is an inventory of alignment that
already exists in `artifacts/oracles/drd2_svm_v1/`.

| protocol component | COMPOSE's frozen equivalent | status |
|---|---|---|
| **DRD2 oracle** | the same `clf_py36.pkl`, sha256 `dbc473fc…111100f`, extracted to `drd2_svm_parameters.npz` | **exact**; parity verdict *"numpy reimplementation reproduces the original estimator"*, max probability gap **2.19e-14** |
| **success threshold** `DRD2 ≥ 0.5` | `margin_many` docstring, `drd2_oracle.py:274-282`: *"Zero is a perfectly good margin — it is exactly P = 0.5, the success threshold"* | **exact**; `margin ≥ 0 ⟺ P ≥ 0.5`, no conversion ambiguity |
| **similarity** ECFP4 Tanimoto ≥ 0.4 | manifest `similarity.definition`: `AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)` with Tanimoto; `tanimoto_to()` at `drd2_oracle.py:66` | **exact** |
| **source molecules** (800, DRD2 ≤ 0.05) | 200 of them, with reference scores, embedded in the manifest — probability range **0.000037 – 0.048158**, none ≥ 0.5, from `iclr19-graph2graph data/drd2/test.txt` | **200 / 800 held**; the remaining 600 are a trivial public fetch |
| **candidate allowance** (20 per source) | expressible — COMPOSE returns endpoints | **expressible** |

Three independent groups converged on this oracle: Jin et al. published it,
GrIDDD repackaged the pickle into `clf_py36_weights.npz` with `gamma=0.015625`,
and COMPOSE extracted the same pickle into `drd2_svm_parameters.npz`. The
fingerprint definitions match character for character.

**So the usual blockers are all absent.** No oracle mismatch. No threshold
conversion. No similarity ambiguity. No missing dataset. This is the closest to
tier 1 anything in this workstream has come.

### Why it is nevertheless tier 2

The classic benchmark budgets **candidates**, not oracle calls. My first reading
was that this dissolves the budget-class problem that blocked Panel B — COMPOSE's
oracle appetite would simply be unbudgeted. Checking properly reversed that.

**GrIDDD does not query the DRD2 oracle during generation.** Its call sites are:

| site | what it is |
|---|---|
| `griddd/datasets/zinc250k_dataset.py:62,128` | **training-set labelling**, offline |
| `griddd/freegress.py:803-805` | **post-generation** metric collection |
| `griddd/freegress.py:1018-1020` | **success evaluation**, `if drd2(smiles) >= 0.5` |

**No call inside the sampling loop.** Guidance comes from classifier-free
guidance on a property the model was trained to condition on. Per input molecule,
GrIDDD's oracle usage is ~20 post-hoc scorings, and in this lane's vocabulary its
`algorithmic_oracle_requests` — objective information consumed *to make
decisions* — is **zero**. The same holds for JT-VAE, CG-VAE and GCPN, which are
trained translators and RL policies, not oracle-guided searchers.

COMPOSE consults the frozen evaluator at **every decision** over a ~586-wide
fiber. Its `algorithmic_oracle_requests` on this task would be in the thousands
per source.

> **The protocol has no oracle-call budget because it never occurred to anyone
> that a method would need one. Entering COMPOSE into it would exploit an
> unbudgeted resource that every native method declines to use.**

A 5.0% success rate obtained without test-time oracle access and a success rate
obtained with thousands of test-time oracle queries are not the same measurement,
and putting them in one column would assert that they are.

**Verdict: tier 2.** COMPOSE may be run under this protocol and reported, but the
external numbers stay **contextual, non-head-to-head**, and any COMPOSE row must
carry its `algorithmic_oracle_requests` beside it. That is not a hedge — it is
the same rule that produced `INCOMPARABLE_SURROGATE_ASYMMETRY`, applied in the
direction that costs us rather than the direction that flatters us.

### The one thing that would make it tier 1

A **declared oracle-call budget**, reported for every arm. That is a *modified*
protocol, so it cannot be cited as the published one — but it is honest, cheap,
and it is what the resource-frontier discipline already requires. Recorded as an
option; not designed here.

---

## Part 2 — per-method tier verdicts

### DDSBM — `FRAMEWORK_NEIGHBOR`

`PENDING` primary-source audit. The bar, stated before the evidence lands:

- is it **per-molecule source-conditioned** at inference, or only
  distribution-to-distribution between two datasets?
- is its published molecular protocol the classic similarity-constrained task, or
  its own?
- are checkpoints released, or is training required?

If its published protocol is one COMPOSE can reproduce exactly, it is tier 1 for
citation. If it is source-conditioned but published on a task we cannot
reproduce, it is tier 3 — and it is the **one** rerun this lane would ever
endorse, because it is the framework counterfactual and that slot is still empty.

### One practical editor — GraphXForm or InVirtuoGen

`PENDING`. Selection criteria, fixed in advance:

1. natively **source-conditioned**;
2. an action space that can **delete**, not only append — the predecessor lane
   already recorded GraphXForm as `inappropriate` for the deletion arm of its
   frozen T4, *"its action space is strictly constructive and cannot remove an
   atom or bond"*;
3. a published protocol we can either reproduce or cite cleanly.

Criterion 2 is not a tiebreak. A comparator that cannot delete cannot express
half of what a source-conditioned editor does, and reporting it as *the*
practical editor without saying so would misdescribe the field.

---

## Part 3 — costed plan for the COMPOSE-only run

**A projection, not a measurement, and not a recommendation to run.** Assumptions
are stated so the arithmetic can be checked rather than trusted.

**Basis.** The Pareto smoke measured ~58 min per source on 8 CPUs for all five
arms, of which ~46 min was kernel enumeration and ~12 min scoring. Summing the
committed per-source kernel calls across arms gives ~612, so **one kernel call
costs roughly 4.5 s**. Per-trajectory costs from
`diagnostics/pareto_semantic_oracle_accounting.json` at `BUDGET = 6`:

| arm | kernel calls / trajectory | oracle evals / trajectory |
|---|---:|---:|
| `greedy_pref` | **3.2** | 2,187 |
| `verified_pref` | **79.8** | 53,977 |

The 25× kernel gap between the arms dominates everything below, so a single
average would be misleading in both directions.

**Projection at 20 candidates per source** (the protocol's allowance), treating
one candidate as one trajectory:

| arm | per source | **200 sources** | **800 sources** |
|---|---:|---:|---:|
| greedy control | ~7 min (8 CPUs) | **~190 core-h** | **~750 core-h** |
| verified control | ~2 h (8 CPUs) | **~3,200 core-h** | **~12,800 core-h** |

**All CPU-only. No GPU anywhere.** Compute is not the blocker; the spread is the
decision.

**What is not costed, because it is not this lane's to choose.** How COMPOSE
produces 20 candidates — 20 independent trajectories, one trajectory yielding 20
endpoints, or a shortlist — is a design choice that changes the cost by an order
of magnitude and changes what the number means. The projection above assumes the
most expensive reading (20 independent trajectories) so it cannot flatter.

**The cheap first move**, if the lead ever authorizes anything here: the 200
sources already embedded in the frozen manifest, greedy arm, **~190 core-hours**.
That answers "is COMPOSE in the right range at all?" before anyone spends the
800-source figure, and it needs no new dataset acquisition.

---

## Part 4 — what the project has already frozen, and a conflict

The predecessor lane's `CONVENTIONAL_SUITE.md` (branch
`codex/compose-baseline-qualification`) already froze **T3 — similarity-constrained
lead optimization**:

| field | frozen value |
|---|---|
| oracle | DRD2 log-odds subject to ECFP4 Tanimoto to source ≥ 0.4 |
| starting state | one designated source per instance |
| **budget** | **1,000 `unique_valid_canonical_evaluations`** |
| success | fraction of sources satisfying both the similarity floor and a preregistered potency threshold |
| seeds | 3 per source |
| note | *"this is the task that separates the methods"* |

It also records that candidates failing the similarity floor **still count** —
*"they are failed proposals, not free actions"* — which is the right call.

**The conflict, and it must be surfaced rather than resolved here.** T3 imposes a
1,000-evaluation budget; the published classic benchmark imposes none. These are
two different tasks wearing similar descriptions:

- **T3** is a budgeted, COMPOSE-native, source-conditioned task. Its numbers are
  ours and are not literature-comparable.
- **The classic benchmark** is unbudgeted and literature-comparable, and COMPOSE
  enters it only with the disclosure above.

Running one and captioning it as the other is the failure this document exists to
prevent. And COMPOSE's own arithmetic says T3's budget is tight: a single
budget-6 trajectory over a ~586-wide fiber touches ~3,500 candidates, so **1,000
unique evaluations does not cover two edit steps** of exhaustive-fiber control.
Whether T3 is runnable as frozen is a real question, and it belongs to whoever
owns that suite.
