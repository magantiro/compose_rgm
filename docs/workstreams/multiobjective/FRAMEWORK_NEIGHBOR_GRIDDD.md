# GrIDDD — `FRAMEWORK_NEIGHBOR` qualification

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute.

**Task:** qualify GrIDDD as a `FRAMEWORK_NEIGHBOR` under
`docs/COMPARATOR_ROLES_CANONICAL.md` — *"discrete graph diffusion supporting node
insertion and deletion, closer to COMPOSE's variable-size dynamics than any RL
optimizer. Main row only if its native task and conditioning semantics align
without substantial adaptation."*

---

## The first thing to say: this is not a new method to this project

GrIDDD is **already embedded** in the repository, and a qualification that
ignored that would have been rewriting history it should have read.

| artifact | what it is |
|---|---|
| `src/compose_v4/experiments/griddd_conditional.py` | 93 KB, COMPOSE-native |
| `scripts/griddd_value_guided_smc_controller.py` | 22 KB |
| `configs/experiments/griddd_qed_frozen_residual_pilot_v5_canonical.json` | frozen pilot config |
| `modal_apps/train_qed_frozen_residual.py` | trains against that config |

**None of it is a port of GrIDDD.** `griddd_conditional.py` imports only
`numpy`, `rdkit` and `compose_v4.*` — no GrIDDD package, no GrIDDD checkpoint.
Its own docstring says it defines *"the engineering contract required before a
full GrIDDD-**style** benchmark can run"*, and its three arms (`direct`,
`controller`, `combined`) are COMPOSE arms built on `RewriteMarkSampler` and
`FixedMolecularStatePrior`.

**A naming hazard worth flagging.** A file called `griddd_conditional.py`
containing no GrIDDD code invites a later reader — or a reviewer given repository
access — to believe a GrIDDD baseline exists here. It does not. The module is
COMPOSE-on-a-GrIDDD-shaped-task. Renaming is not this lane's call and is not
proposed; the risk is recorded so it cannot be discovered late.

### What the project has already frozen

`GridDDProtocol` (frozen dataclass, `griddd_conditional.py:699-712`):

| field | value |
|---|---|
| `starting_qed_minimum` / `maximum` | 0.70 / 0.80 |
| `target_qed` | 0.90 |
| `minimum_tanimoto_similarity` | 0.40 |
| `candidates_per_start` | 20 |
| `guidance_oracle_calls_per_candidate` | 48 |
| fingerprint | ECFP4, 2048 bits |

That is the classic **similarity-constrained lead-optimization** setting: start
from a molecule, improve QED, stay within a Tanimoto ball. It is genuinely
source-conditioned, which is why it exists here at all.

### And what it has already blocked

`GridDDBenchmarkFairnessContract` (`griddd_conditional.py:785-803`), docstring
*"Keep native GrIDDD comparison separate from COMPOSE control studies"*:

```
788: official_griddd_lead_set_id: str = "griddd_release_qed_800_exact_unresolved"
789: exact_griddd_leads_available: bool = False
```

with a guard that **raises** if anyone marks an unresolved lead set as exact and
available, and a separate flag `query_matched_griddd_claim_authorized: False`
(`:755`).

> **The exact GrIDDD lead set has never been resolved, and the repository
> actively prevents claiming otherwise.** That is a pre-existing integrity guard,
> built before this lane existed, and it is doing its job.

### The committed non-claim

`README.md:43-44` already records, under *"What we do not claim"*:

> a head-to-head win on oracle-**light** property optimization (our SMC is
> oracle-hungry — **GrIDDD's home turf, which we do not contest**).

This converges with the alignment audit's independent finding that COMPOSE's
control is oracle-hungry by construction (2,187–53,977 evaluations per preference
trajectory against a benchmark class at 10³–10⁴). Two lanes reached the same
conclusion about the same property from different evidence. `README.md:122` adds
that *"'Matches GrIDDD' holds at the **element** level (CNOF), not the corpus."*

---

## The role question is a different question

This matters and it is easy to blur:

| question | axis | status |
|---|---|---|
| **GrIDDD as `TASK_COMPETENCE`** | can COMPOSE match it on oracle-light QED lead optimization? | already scaffolded; blocked on an unresolved lead set; and the README **already declines** to contest it |
| **GrIDDD as `FRAMEWORK_NEIGHBOR`** | is its insert/delete graph diffusion the nearest alternative *variable-size graph dynamics* to COMPOSE's fiber process? | **the new question**; turns on process semantics, not QED numbers |

The existing infrastructure answers the first. It says almost nothing about the
second, because a lead-set identifier and an oracle budget are not statements
about how the process changes graph size.

**So the internal evidence neither qualifies nor disqualifies GrIDDD as a
framework neighbor.** It does establish two things: the exact lead set is
unresolved, so a task-competence row is blocked regardless; and the project has
already committed to not contesting GrIDDD's home turf, so a task row was never
the plan.

---

## External qualification — RESOLVED

Paper: *Graph Diffusion that can Insert and Delete*, NeurIPS 2025. Official code
cloned at commit `cc3dc31ac341216a0bbcc62f63ae6831581ff40b` (2025-09-28, a single
commit). All citations below were read directly in that tree or in the paper text.

### The three axes

| axis | verdict | evidence |
|---|---|---|
| **(a) supplied source** | **YES** | `griddd/freegress.py:311` `original_smile = data.smiles[0]`; `:334` `z_t = dense_data.copy()`; `:337` `corrupt_data(z_t, corruption_step=self.corruption_step)`. An SDEdit-style corrupt-then-denoise loop starting from a real input molecule, and it is the paper's evaluated optimization mode — not an afterthought. |
| **(b) edit budget** | **NO** | the similarity constraint is a **post-hoc rejection filter**. `freegress.py:393-397` generates candidates, *then* computes `sim = similarity(sample_smiles[i], original_smile)` and applies `keep_mask[i] &= (sim >= self.cfg.guidance.similarity_threshold)`. `corruption_step` is a stochastic noise depth, not a count of edits, and nothing bounds how far a trajectory travels. |
| **(c) preference-conditioned** | **PARTIAL** | target property *values* are supplied at inference (`freegress.py:319-324`, `improvement_type` free/fixed), but the property **set** is baked in at training through classifier-free guidance. A new objective requires retraining. |

**(a) is a genuine first.** Every other external method this lane audited begins
from nothing — HN-GFN at `main.py:145` with an empty `BlockMoleculeDataExtended()`,
OP-GFN at `graph_sampling.py:79` with `self.env.new()`. GrIDDD actually starts
from a molecule you hand it. That is why it deserved a real look.

### The line that settles it — and it is the authors' own ablation

GrIDDD's insert/delete mechanism is the *entire* reason it was nominated as a
framework neighbor: variable-size graph dynamics closer to COMPOSE's fiber
process than any RL optimizer. The authors tested whether that mechanism matters,
and reported (paper §5.3, Ablation, line 561 of the text extraction):

> "we performed the same experiments with GrIDDD but disabling insertions and
> deletions. The results show that while **the success rate remains relatively
> unchanged in the LogP and DRD2 experiments**, it significantly drops to 33.8%
> when optimizing QED, likely because the QED score is a function of the
> molecular weight (and thus, it correlates with the number of atoms)."

So on the **potency-type task nearest COMPOSE's DRD2 objective, insert/delete
changes nothing measurable.** Its effect is confined to QED, where the objective
is itself a function of molecular weight, so size control is close to being the
objective.

This is careful science by the authors, and it is exactly the single source-level
fact that decides a role question. The variable-size dynamics that motivated the
`FRAMEWORK_NEIGHBOR` candidacy are, by the authors' own measurement, **not
load-bearing on the task closest to ours.** Nominating GrIDDD for that role and
then citing its DRD2 numbers would be citing a configuration whose distinguishing
mechanism its own paper reports as inert there.

### Reported numbers — Table 3, verified

Protocol (paper §5.2, text line 376): *"selecting 800 molecules from the test set
with DRD2 activity score ≤ 0.05. For each molecule, we sample 20 candidates from
different latents. We consider the optimization successful if at least one among
the candidates has a DRD2 score ≥ 0.5 and a fingerprint Tanimoto similarity with
the starting molecule ≥ 0.4."*

| method | LogP improv (sim ≥ 0.4) | QED succ (sim ≥ 0.4) | **DRD2 succ (sim ≥ 0.4)** |
|---|---|---|---|
| JT-VAE | 1.03 ± 1.39 | 8.8% | 3.4% |
| CG-VAE | 0.61 ± 1.09 | 4.8% | — |
| GCPN | 2.49 ± 1.30 | 9.4% | 4.4% |
| **GrIDDD** | **2.70 ± 0.94** | **45.1%** | **5.0%** |

Note the baseline set: **JT-VAE, CG-VAE, GCPN** — de novo and RL methods.

### A defect in one of our own committed artifacts

`artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json`, `provenance.lineage`,
states the pickle is

> "the oracle behind the classic similarity-constrained DRD2 benchmark and the
> **VJTNN/GrIDDD success numbers**"

**VJTNN and GrIDDD do not have joint success numbers.** VJTNN appears exactly
once in the entire GrIDDD paper — text line 208, a related-work paragraph that
explicitly sets it aside:

> "VJTNNs (Jin et al. 2018b), Seq2Seq models (He et al. 2021), and HierG2G (Jin
> et al. 2020) all work by translating between different types of molecular
> representations … they require a dataset of pairs of similar molecules with
> specific properties, which are of limited availability."

GrIDDD reports **no VJTNN number** and does not compare against it. Its DRD2
success is **5.0%**; a paired-translation model on this benchmark is a different
class of method with a very different success rate. Pairing the two as "the
VJTNN/GrIDDD success numbers" invites a reader to treat them as one comparable
figure.

**VJTNN's own DRD2 number is `UNVERIFIED` by this lane** and is deliberately not
quoted here — the point is the pairing, not either value.

**Not fixed, and deliberately.** That manifest is a frozen oracle artifact whose
`parameters_npz_sha256` and `pickle_sha256` are hash-bound and consumed by
claim-bearing code; editing its text changes the file digest. The defect is prose
provenance, not a number, and no measurement depends on it. **Reported for main
to correct in a future artifact version** — the same disposition the project gave
the `source_index_sha256` portability defect.

### Convergence worth recording

GrIDDD ships `griddd/metrics/drd2_scorer.py` with `gamma=0.015625`, `C=128`,
2048-dimensional features and `AllChem.GetMorganFingerprint(mol, 3,
useCounts=True, useFeatures=True)` folded to 2048 — and it repackages the
Python-3.6 pickle's parameters into `clf_py36_weights.npz`.

COMPOSE's manifest records `gamma: 0.015625`, `n_features: 2048` and the
identical fingerprint definition, and extracted the same pickle into an npz for
the same reason. **Two groups independently hit the same Python-3.6 sklearn
pickle problem and solved it the same way.** Byte-level agreement is `UNVERIFIED`,
but the oracle lineage is genuinely shared, which makes GrIDDD's published DRD2
numbers a meaningful *contextual* reference even at tier 4.

Its protocol also matches the frozen internal `GridDDProtocol` on the two fields
that matter — 20 candidates per start, Tanimoto ≥ 0.40 — which is presumably why
those constants are there.

### Blockers independent of the role question

| blocker | evidence |
|---|---|
| **no licence** | zero licence files anywhere in the tree. Default all-rights-reserved: **not vendorable**, same stop that excluded OP-GFN and blocks InversionGNN |
| **no checkpoints** | none released; the model would have to be trained |
| **single dormant commit** | `cc3dc31`, 2025-09-28, one commit |

### Adapter complexity — the fourth axis

**Substantial adaptation, not a thin adapter.** Concretely, we would have to
supply:

- **an edit budget** — none exists; the similarity filter is post-hoc and
  `corruption_step` is noise depth;
- **a successor-fiber notion** — GrIDDD draws each node and edge from a
  factorized mean-field categorical (its Eq. 8 assumes independence across
  components), so there is no enumerable one-step support to restrict or count;
- **a separable frozen reference law** — classifier-free guidance bakes the
  conditioning into the weights, so there is no frozen `R_θ` plus detachable
  control layer to compare against ours;
- **a checkpoint**, which the authors never released.

That is the disqualifying answer. Under the rule stated before the evidence
arrived, an adapter that supplies mechanism makes the comparison test our
engineering rather than their abstraction.

---

## The bar, as stated in advance

Recorded before the sweep returned, and reproduced unchanged:

| axis | what qualifies | what disqualifies |
|---|---|---|
| **supplied source** | a native inpainting / conditional / seeded mode accepting an input molecule | sampling from noise only |
| **insert/delete over chemical graphs** | node count genuinely varies along a trajectory over molecular graph states | fixed-size canvas with masked padding |
| **objective conditioning without retraining** | guidance attaches to a frozen base process | objective baked into training |
| **adapter supplies no mechanism** | we evaluate, canonicalize and count | we would have to supply source conditioning, an edit budget, or chemical support |

**If the fourth fails, GrIDDD drops to `CONCEPTUAL_LINEAGE_ONLY`.** That is the
same rule that bars a homemade Edit Flows port, and it applies here identically —
a framework neighbor we had to complete ourselves would be a competitor we
invented, and it would test our engineering rather than their abstraction.

The prior recorded in advance was that insert/delete brings the *size* dynamics
closer without necessarily bringing the *conditioning* closer, and that those are
separable. **That is what the evidence found**, in a sharper form than expected:
the authors' own ablation shows the size dynamics do not move the DRD2 result at
all.

---

## VERDICT — `CONCEPTUAL_LINEAGE_ONLY`, tier 4

**Not a `FRAMEWORK_NEIGHBOR` main row.** Four independent reasons, any one
sufficient:

1. **The fourth axis fails.** A working comparison would require us to supply an
   edit budget, a successor-fiber notion, a separable frozen reference law and a
   checkpoint. That is method invention, not an adapter.
2. **The authors' own ablation removes the motivation for the role.** Disabling
   insertions and deletions leaves DRD2 and LogP success *"relatively
   unchanged"*. The mechanism that made it a candidate is inert on the task
   nearest ours.
3. **No licence.** All-rights-reserved by default; not vendorable. The same stop
   that excluded OP-GFN.
4. **No checkpoints**, and one dormant commit.

### What it is instead, and this is not a consolation prize

**The closest external method to COMPOSE's task semantics that this lane has
found.** It genuinely starts from a supplied molecule, it is
similarity-constrained, it shares the DRD2 oracle's lineage and repackaged the
same pickle the same way, and its 20-candidates / Tanimoto-0.40 protocol is the
one already frozen in our own `GridDDProtocol`.

That makes it a **strong contextual citation** — tier 2 in citation terms even
though execution is tier 4 — and the natural reference point for the
similarity-constrained DRD2 setting. Cite its 5.0% DRD2 and 45.1% QED as
*reported*, with its protocol stated, and never as a head-to-head against a
COMPOSE number.

### Consequence for the package

**DDSBM remains the only `FRAMEWORK_NEIGHBOR` candidate**, and E1's framework
counterfactual is still unfilled. That is now the single largest gap in the
multiobjective evidence package, and it is a gap in *novelty* evidence rather
than competence evidence.

Two candidates have now been examined for that slot and both fell to the same
rule — the adapter would have had to supply the mechanism under test. That is
worth noticing: it is a pattern, not a coincidence, and it suggests the framework
counterfactual may be genuinely hard to source externally rather than merely
unsourced so far.

### Also settled, from the internal half

1. **No GrIDDD code exists in this repository**; the GrIDDD-named modules are
   COMPOSE arms on a GrIDDD-shaped task.
2. **A task-competence row is blocked** on an unresolved official lead set
   (`exact_griddd_leads_available = False`), and `README.md:43` already declines
   that comparison on record.
3. **One committed artifact carries a provenance defect** — the DRD2 manifest's
   "VJTNN/GrIDDD success numbers" pairing. Reported, not fixed.
