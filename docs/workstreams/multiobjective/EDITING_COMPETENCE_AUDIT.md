# Source-conditioned editing competence — protocol-alignment audit

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute.

**The question, and only this:** ignoring Pareto entirely, is COMPOSE a credible
source-conditioned molecular editor? **Reviewer insurance and conventional
competence — not another scientific pillar**, and deliberately not built like one.

**The set is two external methods.** DDSBM as `FRAMEWORK_NEIGHBOR` if the
protocol genuinely aligns, plus **one** practical editor. No third.

---

## Headline

Two findings, pointing opposite ways.

> **1. COMPOSE could physically enter the classic similarity-constrained DRD2
> benchmark tomorrow — it already holds every frozen component — and it should
> not be entered as tier 1.** The protocol carries no oracle-call budget because
> its native methods do not query the oracle at inference. COMPOSE queries it at
> every decision. The absence of a budget is an unstated assumption, not
> permission.

That is the mirror image of the HN-GFN finding, pointed at us. There, the
baseline had a hidden amortization advantage and I refused to let it look cheap.
Here COMPOSE would have the hidden advantage, and the same refusal applies.

> **2. There is nevertheless a tier-1 path, and it is the first this workstream
> has found.** DDSBM is genuinely source-conditioned, its ZINC logP-shift split is
> shipped in its own repo and therefore exactly reproducible, and its metrics are
> model-agnostic apart from NLL. **COMPOSE can be run alone on DDSBM's own task
> and DDSBM's Table 1 cited as reported.**

So the framework-counterfactual slot — empty since the role amendment — **can be
filled at tier 1**, on a distribution-shift task rather than on DRD2.

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

### DDSBM — `FRAMEWORK_NEIGHBOR`, and the tier **splits by task**

*Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation*,
Kim et al., **ICLR 2025** (OpenReview `tQyh0gnfqW`, arXiv:2410.01500). Code
`github.com/junhkim1226/DDSBM`, last push 2025-04-15.

**It is genuinely source-conditioned. YES, verified in code, not inferred.**
`diffusion_model_discrete.py:720 predict_step` → `:727-731` uses `target="0"`
with the comment *"using original data to generate new data"* → `:740` `X_0, E_0
= dense_data_0.X, dense_data_0.E` → `:750 sample_forward_bridge_batch(X_0=X_0,
E_0=E_0, …)`.

> **A trap worth recording**, because reading one function would have given the
> wrong answer: `on_test_epoch_end` (`:629-633`) *discards* the source and
> substitutes prior noise. That path is reachable **only** for the unconditional
> datasets (`train_helper.py:310-311`). A shallower audit would have called
> DDSBM de novo and been wrong.

So the conceptual placement holds: DDSBM is the nearest alternative
source-conditioned graph-transformation abstraction, exactly as the role
amendment assigned.

#### But its published protocol is not an optimization benchmark

| dimension | DDSBM's published value |
|---|---|
| task | ZINC250k **logP distribution shift**, Gaussians centred at 2 and 4 (§5.2) |
| oracle | **pure RDKit** — `Crippen.MolLogP`, `QED.qed`, `sascorer` (`analysis/prop_diff.py:24,81,89`). No learned model, no pickle, **no DRD2** |
| budget | **neither candidates nor oracle calls** — there is no budget because there is **no oracle in the loop**. `predict_step` emits exactly **one** output per input, keyed by index (`:767-787`, `:828`) |
| success criterion | **none defined or reported anywhere** |
| similarity constraint | **none enforced**, during generation or post-hoc. Structure retention is *measured* (NLL, QED/SA MAD), never gated. Tanimoto appears only to build a training coupling for an ablation |
| seeds | *"average of three independent training runs with different random seeds"* (§5.1) — three **training** runs, so error bars carry full retraining cost |

**`DRD2` occurs zero times in the paper.** DDSBM shares no evaluation surface
with the similarity-constrained benchmark: different oracle, no budget concept, no
success criterion, no similarity gate.

#### Verdict — tier 1 on its own task, tier 4 on ours

**Tier 1, for the ZINC logP-shift task, on the model-agnostic metrics.** The
paired CSV `data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv` is **shipped in
the repo**, so the 23,936 / 5,984 split is exactly reproducible. COMPOSE could be
run alone and DDSBM's Table 1 cited **as reported** on: validity, uniqueness,
novelty, NSPDK, **logP $W_1$**, QED MAD, SAscore MAD, FCD.

**NLL is excluded, and this matters.** It is defined against DDSBM's *own*
reference process (§4.3, §D.6) — model-relative, not protocol-neutral. It is also
the column carrying DDSBM's headline win (160.461 vs DBM's 288.572). Quoting it
beside a COMPOSE number would be comparing each method to a different yardstick.

**Tier 4 for anything DRD2 or similarity-constrained.** Getting a DRD2 number
requires constructing a new paired dataset (inactive ↔ active, Hungarian-matched)
and **full retraining** — DDSBM has no conditioning knob to retarget a trained
model at a new property. Verified blockers:

- **No checkpoints.** README TODO line 14, `- [ ] Checkpoints update using
  Zenodo` — **unchecked**; Zenodo API `q=DDSBM` returns **0 records** (I ran both
  checks directly). A search summary claiming checkpoints exist on Zenodo is
  **false** — it misread the unchecked box.
- **No licence.** GitHub API `license: null`; `/license` endpoint returns
  **404**. Default copyright: not vendorable, same stop as GrIDDD and OP-GFN.
- **4× RTX A4000**, 300 epochs × 6 IMF iterations, ×3 seeds. Wall-clock
  GPU-hours **`UNVERIFIED`** — the paper reports no timing figure, so no number
  goes in the table.
- GPU training is barred for this lane regardless.

**So the framework-counterfactual slot can be filled at tier 1 — but on a
distribution-shift task, not on DRD2.** That is a real and useful finding: it is
the first tier-1 path this workstream has produced.

#### The same asymmetry, much weaker

COMPOSE would consult `Crippen.MolLogP` while editing; DDSBM consults it zero
times at inference. Structurally identical to the DRD2 problem above — but far
weaker in force, because logP is a **free deterministic descriptor**, not a costly
oracle, and the benchmark measures distribution match rather than oracle
efficiency. The disclosure still travels with the row; it simply does not
disqualify it.

### One practical editor — **InVirtuoGen selected**, on the criteria fixed in advance

The criteria were recorded before the evidence arrived:

1. natively **source-conditioned**;
2. an action space that can **delete**, not only append;
3. a published protocol we can either reproduce or cite cleanly.

**GraphXForm fails criterion 2 outright, and the failure is a capability, not a
fit.** Verified directly in the clone at `867bdcf9`:

- `grep -rniE "RemoveAtom|RemoveBond|delete_atom|delete_bond" --include=*.py`
  returns **zero hits** across the repository.
- The action space (`molecule_design.py:18-26`) is Level 0 terminate / create
  atom / pick atom, Level 1 pick second atom, Level 2 bond order. Purely
  constructive.
- The authors say so twice. §3.3.3: *"Extending the action space to include atom
  and bond removal is straightforward within our framework; however, we leave the
  exploration of this possibility for future work."* And in the Conclusion:
  *"we aim to extend the action space by allowing the agent to remove bonds and
  atoms."*

This independently confirms the predecessor lane's finding, which recorded
GraphXForm as `inappropriate` for the deletion arm of frozen T4.

**GraphXForm also fails criterion 3 on budget currency.** Its headline numbers are
GuacaMol goal-directed best-of-run under an **8-hour wall-clock cap on one H100**,
and §3.1 explicitly *rejects* the oracle-call convention: *"While this is useful
for comparing sample efficiency, it offers limited insight into overall
efficiency when objective evaluations are inexpensive."* A wall-clock budget on
named hardware is not citable across labs, and this lane is CPU-only regardless.

It is otherwise the best-licensed and best-published thing in this audit — **MIT**
(`LICENSE:1`), a live 347 MB pretrained checkpoint, peer-reviewed in *Digital
Discovery* 4:1052–1065 (2025), DOI `10.1039/d4dd00339j`. None of that repairs an
append-only action space.

> **GraphXForm: tier 4, capability-disqualified for source-conditioned editing.**
> Cite in related work as a strong de novo constructive designer. Do not report it
> as a source-conditioned editing comparator.

#### InVirtuoGen — selected, tier 2, with real caveats

*Refine Drugs, Don't Complete Them: Uniform-Source Discrete Flows for
Fragment-Based Drug Discovery*, Kaech, Wyss, Borgwardt, Grasso. **arXiv:2509.26405**,
NeurIPS 2025 **workshop** (AI4D3) — main-conference or journal venue `UNVERIFIED`.
Code `github.com/invirtuolabs/InVirtuoGen_results` @ `b50bb3ae`.

It wins on the pre-registered criteria:

| criterion | InVirtuoGen |
|---|---|
| source-conditioned | **YES** for lead optimization — seed molecule, QED ≥ 0.6, SA ≤ 4, **Tanimoto ≥ δ ∈ {0.4, 0.6}** — structurally the same shape as ours |
| can delete | **YES** — `delete_atom` and `delete_cyclic_bond` in `in_virtuo_reinforce/ga/mutate.py` |
| budget currency | **oracle calls, 10,000** — the same currency we use |

And it is unusually disciplined about budget honesty, publicly noting that GenMol
and f-RAG prescreen all of ZINC250k — *"while they nominally report results with
10k oracle calls, the effective budget is closer to 260,000"* — and reporting
both regimes.

**But every caveat below must travel with it.**

1. **Its citable numbers are on the wrong task.** The PMO `drd2` value (0.985
   no-prescreen, 0.995 with) is **de novo** — no source molecule, no edit budget,
   no similarity constraint. Citing it says nothing about source-conditioned
   editing.
2. **Its one matching experiment is unreproducible in the release.** `README.md:302`
   documents `python -m in_virtuo_reinforce.ppo_docking --max_oracle_calls 1000
   …`, but the shipped `ppo_docking.py` is a **results-aggregation script**: its
   argparse accepts only `--results_root`, `--reference_table`,
   `--exclude_prescreen`, `--include_std`, `--ablation_mode`, `--results_paths`,
   `--model_names`, and it contains **zero** occurrences of `torch`, `vina` or
   `ckpt`. Only **2 of 5 receptors** ship (`jak2`, `parp1`; `fa7`, `5ht1b`,
   `braf` missing). The lead-optimization objective is **QuickVina2 docking**, not
   DRD2, in any case.
3. **Its editing power is Graph GA's, not its own.** `ga/mutate.py:1-10` states
   *"This file has been taken from jensengroup/GB_GA"*. The neural component is a
   discrete flow over fragmented-SMILES tokens; the delete operators are
   SMARTS-based GA mutations. Attributing editing competence to the flow model
   would be wrong, and the honest citation for those operators is GB-GA.
4. **Licence, and this one is actionable.** `LICENSE` is **internally
   inconsistent** — line 3 says CC BY-NC-**SA** 4.0, line 5 says
   "Attribution-NonCommercial 4.0"; the GitHub API reports `NOASSERTION`. A
   separate `WEIGHTS_TERMS_OF_USE.md` restricts the weights to non-commercial use
   and states *"You **must not** use nor allow others to use"* the output to train
   models for molecular generation, and *"You **must not** publish or share
   InVirtuoGEN model parameters"*.

   > **Concrete consequence: no InVirtuoGen output may ever become `R_θ`
   > training data.** Running it as a baseline appears fine for non-commercial
   > academic work; training on anything it emits is barred. Recorded as a
   > decision for main, not an assumption.
5. Fragmented SMILES **discards stereochemistry** (their own Limitations).

> **InVirtuoGen: tier 2.** Cite its PMO `drd2` number as *reported*, labelled de
> novo, never as a source-conditioned editing comparison. A tier-3 rerun on our
> task would need a new oracle function in its intact PMO harness plus a
> from-scratch rewrite of the lead-optimization path — not recommended.

#### The honest summary of Part 2

**Neither practical editor's native benchmark aligns with source-conditioned
DRD2-style editing.** GraphXForm cannot delete and budgets wall-clock;
InVirtuoGen budgets oracle calls and can delete, but its citable numbers are de
novo and its matching experiment does not run. InVirtuoGen is selected because it
wins on criteria fixed before the evidence, not because it aligns.

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
