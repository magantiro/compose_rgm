# Comparator matrix — organized by ROLE

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute
spent. Every cost figure is a **published figure or a projection**, never a
measurement by this lane. Cells that cannot be resolved to a primary source say
`UNVERIFIED` rather than carrying an estimate.

**Governed by `docs/COMPARATOR_ROLES_CANONICAL.md`**, which supersedes the
comparator-selection guidance in
`docs/BASELINE_PHILOSOPHY_AND_MAIN_LANE_DECISIONS.md` (whose four main-lane
decisions and stop rules remain in force).

> **Comparator selection is framework-first.** A method does not become a primary
> baseline merely because it optimizes the same property. Framework-neighbor
> comparisons establish methodological novelty; matched internal controls
> establish causality; task-specialist baselines establish practical competence.

**This document was reorganized from a task matrix into a role matrix.** The
previous organization sorted methods by which experiment they could enter, which
silently encoded the assumption that entering more experiments made a method more
important. It does not. A method may hold different roles in different
experiments, and the role is recorded per method with its reason.

---

## The execution tiers

From `docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`, which extends the role taxonomy:

> **Use published baseline results whenever we can reproduce the exact published
> evaluation protocol on COMPOSE. Rerun an external method only when a direct
> comparison genuinely requires it.** Baseline execution is the exception
> justified by the scientific question, never the default reflex.

| tier | condition | action |
|---|---|---|
| **1** | exact alignment on every protocol dimension | run **COMPOSE only**; cite baselines as **"reported"**, never "our rerun" |
| **2** | partial alignment | **contextual literature numbers**, never head-to-head |
| **3** | question is on our custom task **and** a thin native adapter exists | authorize a rerun of the **smallest necessary set** |
| **4** | code broken, checkpoints absent, licence restrictive, or adaptation requires method invention | **cite and discuss; do not reconstruct** |

The **role** says what a method would establish. The **tier** says whether we
could ever get it from publication or would have to execute it. They are
orthogonal, and both are recorded.

## Role and tier assignments at a glance

| method | role | **tier** | one-line reason |
|---|---|---|---|
| **DDSBM** | `FRAMEWORK_NEIGHBOR` | **3** | nearest alternative graph-CTMC/bridge abstraction; a native rerun on our exact sources is meaningful because no published result covers them |
| **GrIDDD** | `FRAMEWORK_NEIGHBOR` *(qualification pending)* | **4 for task, framework tier pending** | discrete graph diffusion with node insertion **and deletion**. Its task row is tier 4: the official lead set is `exact_unresolved` in our own frozen contract |
| **generate-and-rank P3/P4** | `MATCHED_CAUSAL_CONTROL` | **n/a — internal** | same executor, same budget, purpose applied only after generation |
| **empirical-family reference ablation** | `MATCHED_CAUSAL_CONTROL` | **n/a — internal** | same control, `R_θ` replaced — isolates what the learned reference law buys |
| **preference-blind floor (`unguided`)** | `MATCHED_CAUSAL_CONTROL` | **n/a — internal** | same process, no objective consulted — the floor Q1 must clear |
| **greedy vs verified control** | `MATCHED_CAUSAL_CONTROL` | **n/a — internal** | same everything, lookahead removed |
| **HN-GFN** | `TASK_COMPETENCE` | **2** | all six protocol dimensions fail → contextual only. Never substitutes for source-conditioned P3/P5/P6 |
| **InversionGNN** | `TASK_COMPETENCE` | **2 to cite, 4 to run** | numbers usable as context; no checkpoint, an arity defect and **no licence** put execution in tier 4 |
| **GraphXForm** | `TASK_COMPETENCE` | **3** | named in the amendment: a native rerun on our exact sources is meaningful if no published result covers them |
| **InVirtuoGen / MolEditRL** | `TASK_COMPETENCE` | `UNVERIFIED` | no primary source read by this lane |
| **OP-GFN** | **excluded entirely** | **4** | not preference-conditioned; CC BY-NC-**ND** forbids distributing the adapter |
| **Edit Flows** | `CONCEPTUAL_LINEAGE_ONLY` | **4** | edit-based CTMC over variable-length **sequences**; a graph port is method invention |
| **Expanding Flow Maps** | `CONCEPTUAL_LINEAGE_ONLY` | **4** | variable-size **de novo** generation, not source-conditioned legal rewriting |
| **MOG-DFM** | `CONCEPTUAL_LINEAGE_ONLY` | **4** | fixed-length token sequences; uniform init; local one-step guidance |
| **PepTune** | `CONCEPTUAL_LINEAGE_ONLY` | **4** | peptide SMILES tokens; de novo; **Pareto dominance filtering, so not preference-conditioned** |
| **pCoMole** | `UNVERIFIED` | **4** | primary document not read — see escalation |
| **AReUReDi** | `CONCEPTUAL_LINEAGE_ONLY`, *conditionally reassessable* | **4**, conditionally **3** | the only same-lab candidate for direct numerical comparison, under four conditions below |
| **CDD** | hard-constraint lineage — **Lane 6 owns** | **2**, with its **task definition** reusable | reuse CDD's published `SA(y) ≤ τ` predicate as an externally defined task; **do not rebuild CDD** |
| **PRODIGY / ConStruct** | hard-constraint lineage — **Lane 6 owns** | **Lane 6's call** | not audited here; no claim about them appears in this workstream |

**Reading the tiers.** Only DDSBM and GraphXForm sit at tier 3 — the only two
methods this lane would ever expect to see executed, and only on our exact
sources where no published result exists. Everything else is cited or discussed.
**No method in this workstream is tier 1**, because tier 1 requires exact
protocol alignment and the alignment audit found none.

---

## `FRAMEWORK_NEIGHBOR` — primary external evidence for novelty

These answer the question closest to what COMPOSE actually contributes: *why an
executable graph CTMC over canonical legal-rewrite fibers, rather than another
graph-CTMC or bridge abstraction?*

### DDSBM

Promoted to the principal numerical external process comparator. **Not audited by
this lane** — `baselines/ddsbm/README.md` and `environment.lock` exist on branch
`codex/compose-baseline-qualification`, and the comparator map in
`PARALLEL_WORKSTREAMS_AND_HANDOFF.md` already records its qualification
questions. Recorded here for role completeness only.

The standing constraint from the baseline policy still binds: *"the official
graph-transformation implementation. Molecular input/objective adaptation may be
unavoidable; the learned bridge/process stays their algorithm."* And from the
philosophy doc: if it cannot natively say "this labeled core must remain present
at every state", **we do not invent that mechanism for it.**

### GrIDDD

**Qualification in progress.** Verdict and evidence recorded in
`FRAMEWORK_NEIGHBOR_GRIDDD.md` when the primary-source sweep completes. Main row
only if its native task and conditioning semantics align **without substantial
adaptation**.

The bar it must clear is the same one that excluded HN-GFN from Panel A, applied
to a different question. For a `FRAMEWORK_NEIGHBOR` the decisive axes are:

- does it natively accept a **supplied source molecule** as the starting state?
- are its **insert/delete** operations over chemical graph states, and is the node
  count genuinely variable along a trajectory?
- can objective conditioning attach **without retraining** the base process?
- would our adapter only evaluate and count, or would it have to supply
  mechanism?

If the last answer is "supply mechanism", it drops to `CONCEPTUAL_LINEAGE_ONLY`
by the same rule that bars a homemade Edit Flows port.

---

## `MATCHED_CAUSAL_CONTROL` — primary evidence for why each component matters

Often more informative than adding a named model, because these hold executor,
state, objective and budget fixed and change exactly one thing. They are
Category 2 comparators under `docs/BASELINE_IMPLEMENTATION_POLICY.md` — *"we
implement these ourselves, because they are defined relative to COMPOSE"* — and
they are Panel A members. **Lane 4 owns the runs.**

| control | the one thing it changes | what it isolates |
|---|---|---|
| generate-and-rank P3/P4 | *when* purpose is applied | closed-loop control vs endpoint ranking |
| empirical-family ablation | the reference law `R_θ` | what learning the reference process buys |
| `unguided` floor | whether the objective is consulted at all | that preference response is not chance |
| greedy vs verified | lookahead depth | future-awareness, **already claimed elsewhere** |
| hard mask vs post-hoc vs soft guidance | how constraints are imposed | exact support restriction vs ordinary guidance (**Lane 2**) |

**A standing caution that the role rename does not repeal.** The greedy-vs-verified
control carries a *structural sign guarantee* on per-preference scalarized value
and **no guarantee** on set-level hypervolume. The two must stay separate, and
Pareto does not carry a second "planning beats greedy" headline.

---

## `TASK_COMPETENCE` — credibility, not novelty

> These establish that COMPOSE performs credibly on ordinary molecular design.
> **They do not define the novelty claim, and they never substitute for a
> source-conditioned control.**

The Panel A/B separation this lane built already enforced exactly this. The
amendment makes it project-wide; the separation is unchanged and
`assert_not_cross_panel` still holds it.

### HN-GFN — `TASK_COMPETENCE`, and Panel B carries no numeric row

| field | value | source |
|---|---|---|
| paper | Zhu et al., NeurIPS 2023 | arXiv:2302.04040 |
| code | `github.com/violet-sto/HN-GFN` @ `90078b8ceeee3e907deeced9b096a6e395b71177` | clone, verified |
| licence | MIT | `LICENSE` |
| checkpoints | none for the GFlowNet; a pretrained **proxy** at `data/pretrained_proxy/` | tree listing |
| native scalarization | linear weighted sum, `--scalar` default `WeightedSum` | `main.py:54`, `main_mobo.py:62`, `main.py:226-227` |
| starting state | **empty block molecule, every rollout** | `main.py:145` |
| true-oracle budget | 200 + 8×100 = **1,000** | `main_mobo.py:50-52` |
| surrogate in loop | yes — GFlowNet reward is the proxy | `main_mobo.py:211`, `main_mobo.py:393` |
| published runtime | *"our proposed HN-GFN costs 10 hours"* on *"1 Tesla V100 GPU"*, +33% with hindsight training | paper Appendix B.4 |
| environment blocker | BoTorch symbols removed upstream; no dependency pins ship at all | upstream issue #1 |

**The GPU cost is now moot** — under the redirect we cite rather than rerun. And
the alignment audit found **all six protocol dimensions fail**, so Panel B
carries no numeric row at all. See `BENCHMARK_ALIGNMENT_AUDIT.md`.

### InversionGNN — `TASK_COMPETENCE`, conditional

| field | value | source |
|---|---|---|
| paper | Niu et al., ICLR 2025 | arXiv:2503.01488 |
| code | `github.com/ivanniu/InversionGNN` @ `cfdf1d9a981ca4ce5dd7293dc38373ddf9377718` | clone, verified |
| licence | **none in tree** — default all-rights-reserved; **not vendorable** | `find -iname '*licen*'` → 0 |
| checkpoints | **none**; `model_ckpt = ""` then `torch.load(model_ckpt)` | `molecular/denovo.py:75-76` |
| blocking defect | six arguments passed to a four-parameter function | `denovo.py:107` vs `inference_utils.py:153` |
| objectives | GSK3β, JNK3, QED, SA. **No DRD2.** | paper §5.2 |

### OP-GFN — excluded entirely

Not preference-conditioned in its own mode (`seh_frag_moo.py:398,414`);
objectives asserted closed to `{seh, qed, sa, mw}` (`:62`); and **CC BY-NC-ND**
(`LICENSE.md:1`) — NoDerivatives forbids distributing the adapter that would make
it runnable. No engineering resolves the third.

---

## `CONCEPTUAL_LINEAGE_ONLY`

Scientifically close; a faithful common numerical task would require substantial
adaptation. **Prominent in related work. Never a numerical baseline built by us.**

Full ten-field schema records for MOG-DFM, AReUReDi, pCoMole and PepTune are in
`SAME_LAB_LINEAGE.md`. Summary of why each sits here:

| method | the blocking axis |
|---|---|
| Edit Flows | published state space is variable-length **sequences**; a graph port would mean inventing the chemical-support machinery COMPOSE contributes |
| Expanding Flow Maps | variable-size **de novo** generation, not source-conditioned legal rewriting |
| MOG-DFM | fixed-length token sequences, uniform initialization, local one-step guidance |
| PepTune | peptide SMILES tokens, de novo, and **Pareto dominance filtering rather than scalarization — so not preference-conditioned** |
| AReUReDi | fixed-length token strings, substitution-only, no supplied source — but see the conditional reassessment below |
| pCoMole | `UNVERIFIED`; the paper has not been read |

### The same-lab policy, applied

> **Scientific proximity determines comparison. Not shared authorship, in either
> direction.**

Avoiding the closest prior work *because* it shares a lab is riskier than a
careful comparison — a reviewer may reasonably suspect easier baselines were
chosen. This lane's placements were made on state space, source conditioning and
guidance horizon, and none of them turned on authorship.

**AReUReDi is the only same-lab method to seriously assess for direct numerical
comparison.** Its status may be revised from `CONCEPTUAL_LINEAGE_ONLY` **only if
all four hold**:

1. official code runs ordinary small-molecule SMILES;
2. objectives attach through a **thin wrapper**;
3. **no redesign** of its proposal, backbone or scalarization is needed;
4. **the original authors validate the configuration.**

On current evidence condition 1 already fails: its SMILES instantiation is
*peptide* SMILES over a 586-token vocabulary, handled as a **fixed-length token
string**, and single-token substitution at fixed length cannot express bond or
atom graph edits. Recorded as a conditional so the assessment is auditable rather
than merely asserted — and it is an assessment this lane cannot complete alone,
because condition 4 requires contacting authors.

**Framing, binding and non-negotiable.** Cumulative, never adversarial. Prior
work developed powerful guidance and control for discrete diffusion, flow and
edit processes; COMPOSE contributes a new executable molecular graph substrate,
and matched experiments determine what that substrate enables beyond prior
guidance formulations. **Never "COMPOSE defeats prior lab work."**

---

## What the roles changed, and what they did not

**Changed.** HN-GFN and InversionGNN were previously the two named external rows
this lane was organized around, on the reasoning that they are the purpose-built
preference-conditioned molecular Pareto generators a reviewer would ask about.
Under framework-first selection they are `TASK_COMPETENCE`: they optimize the
same properties, which is not the same as instantiating the same scientific
question. The nearest framework comparison is DDSBM and possibly GrIDDD.

**Not changed.** Every verified fact survives the reorganization — the licences,
the commit SHAs, the code defects, the empty-start finding, the oracle
mismatches, the budget arithmetic. A role is an assignment of evidentiary
purpose; it is not a re-reading of what the code does.

**Worth stating plainly:** the alignment audit found that Panel B carries no
numeric row, and the role amendment independently demotes Panel B's occupants
from primary evidence. Those two conclusions arrived from different directions —
one from protocol forensics, one from framework-first selection — and agree. The
multiobjective argument rests on `MATCHED_CAUSAL_CONTROL` arms and, if it
qualifies, one `FRAMEWORK_NEIGHBOR`.

---

## Cost summary

| item | cost | basis |
|---|---|---|
| Stage 0, Stage 1, alignment audit, this restructure | **0 compute** | local reads, five clones, one test file |
| HN-GFN, one held-in smoke | 10–13 GPU-hours, V100-class | authors' Appendix B.4 — **moot; we cite, not rerun** |
| InversionGNN, one held-in smoke | `UNVERIFIED`; CPU-capable, blocked by two defects | repo |
| OP-GFN | not applicable — excluded | licence |
| COMPOSE alone on a GSK3β/JNK3 benchmark | ~12 CPU core-hours, **but blocked** on a RandomForest oracle extraction (1–2 days, no GPU) that should not be built for a declined comparison | `BENCHMARK_ALIGNMENT_AUDIT.md` |
| GrIDDD qualification | `PENDING` | — |
