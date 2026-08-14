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

## External qualification — the four decisive axes

`PENDING` primary-source sweep. The bar, stated before the evidence arrives so it
cannot be adjusted to fit:

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

The prior on this is not neutral. A **discrete graph diffusion** model denoises
toward a data distribution; COMPOSE's process executes legal rewrites from an
exact realized state. Insert/delete brings the *size* dynamics closer without
necessarily bringing the *conditioning* closer, and those are separable. The
sweep is what settles it.

---

## Verdict

`PENDING` — external evidence not yet in. What is already settled:

1. **No GrIDDD code exists in this repository**, and the GrIDDD-named modules are
   COMPOSE arms on a GrIDDD-shaped task.
2. **A task-competence row is blocked** on an unresolved official lead set, and
   the README already declines that comparison on record.
3. **The framework-neighbor question is open** and is decided on process
   semantics — supplied source, variable-size chemical graph dynamics, frozen-base
   conditioning, and whether our adapter would have to supply mechanism.
