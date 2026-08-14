# External baseline task matrix — hard structural constraints

**Status: `DESIGN_ONLY`.** No baseline was run by this lane. No external
dependency was installed. Every capability cell cites a paper section, a repo
file, or a flag name; anything not confirmed against a primary source is marked
`UNVERIFIED` rather than guessed.

---

## Role assignment (canonical)

**Governed by `docs/COMPARATOR_ROLES_CANONICAL.md`.** Comparators have **roles,
not rankings**, and selection is **framework-first**: a method does not become a
primary baseline merely because it optimizes the same property.

**Roles are per-experiment.** Everything below is scoped to the **hard-constraint
block**. The same method may hold a different role elsewhere — GraphXForm is
`TASK_COMPETENCE` here and remains a source-conditioned editor in the general
editing table.

| method | **role (constraints block)** | **tier** | evidence mode | why it holds this role |
|---|---|:---:|---|---|
| **CDD** | **`FRAMEWORK_NEIGHBOR`** | **4** | **conceptual only** | Asks *how should a hard constraint be integrated into a pretrained generative process?* — the section's own question. Numerical comparison **tested and refused**: no native common protocol, no released code (placeholder README), de novo vs source-conditioned. **Its published `SA(y) ≤ τ` task is reusable — see §3a** |
| **PRODIGY** | **`FRAMEWORK_NEIGHBOR`** | **4** | **conceptual only** | Same axis, graph-native projection. Constraint class is aggregate scalars, so it cannot state our predicate; **repo carries no licence** |
| **ConStruct** | **`FRAMEWORK_NEIGHBOR`** | **4** | **conceptual only** | Closest lineage — guarantees structure *throughout the trajectory*. Appendix G.2 is the nearest published analogue to our setup. Still de novo, still unlabeled-structural |
| **post-hoc vs soft guidance vs exact support** | **`MATCHED_CAUSAL_CONTROL`** | — | numerical, internal | **The primary comparison.** Same `R_θ`, source, objective, controller, budget; the constraint mechanism is the only thing that varies. **Lane 2 builds it** — `LANE2_FIVE_QUESTIONS.md` |
| **GraphXForm** | **`TASK_COMPETENCE`** | **3** | numerical where native | Demoted from "core intellectual opponent". Our custom task + a native thin adapter exists (Lane 3). **Audit tier 1 first.** Hard-constraint cell is **`N/A`** — §0 |
| **Prompt-MolOpt^P** | **`TASK_COMPETENCE`** | **3** | numerical where native | Asks *can a specialized editor preserve structure while improving properties?* — a different question. **Must not drive the methods narrative.** MIT, released ^P checkpoint, thin adapter feasible. **Audit tier 1 first.** |
| **MolEditRL** | **`TASK_COMPETENCE`** | **4** | **none available** | Same demotion. `OFFICIAL_CODE_UNAVAILABLE`; preservation is soft (KL-to-prior) even if released |
| **InVirtuoGen** | **`TASK_COMPETENCE`** | **4** *(licence)* | conditional | Fragment-constrained generation. Token-level clamp, not graph-level. **CC-BY-NC-SA 4.0 non-commercial** — a licence decision for the lead, not for this lane |
| **DDSBM** | **`FRAMEWORK_NEIGHBOR`** — *general editing block, **not** here* | **4** | — | *Why COMPOSE's executable graph CTMC rather than another graph CTMC/bridge?* **Not a hard-constraint baseline.** **Lane 5 owns qualification — do not duplicate.** No LICENSE, no checkpoints |
| **GrIDDD** | **`FRAMEWORK_NEIGHBOR`** — *general editing block* | **UNVERIFIED** | — | Discrete graph diffusion with node insertion **and deletion**, closer to COMPOSE's variable-size dynamics than any RL optimizer. **Lane 5 owns qualification**; this lane did not audit it |
| **Edit Flows** | **`CONCEPTUAL_LINEAGE_ONLY`** | **4** | related work | Edit-based CTMC with insert/delete/substitute, but over variable-length **sequences**. A molecular-graph port would mean inventing the chemical-support machinery COMPOSE contributes — building a competitor for ourselves |

**Tiers** (`docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`): **1** exact alignment →
run COMPOSE only, cite baselines as *"reported"*; **2** partial alignment →
contextual literature numbers, never head-to-head; **3** rerun, smallest
necessary set, only when the question is on our custom task **and** a native thin
adapter exists; **4** do not reconstruct — cite and discuss.

> **No method in the constraints block is tier 1 or tier 2.** Every framework
> neighbour is tier 4, and the two competence comparators are tier 3. That is
> the concrete consequence of the audit: **the constraints section cannot be
> carried by published numbers**, so its causal weight falls entirely on the
> internal matched control.

### The tier-1 audit obligation, not yet discharged

> *"Before implementing or rerunning any external baseline, audit whether COMPOSE
> can instead be run under that baseline's published protocol."*

For the two tier-3 methods this audit has **not** been done and is **not** this
lane's to do. Whoever owns the competence table should check whether COMPOSE can
be run under GraphXForm's or Prompt-MolOpt^P's published evaluation protocol
before authorizing any rerun. If either aligns, it drops to tier 1 and no
external execution is needed at all.

### Two things this assignment fixes

**The section is no longer organized around editors.** The intellectual baseline
for exact hard-support control is CDD / PRODIGY / ConStruct. **A
scaffold-specific RL paper does not define this section**, and neither does a
fragment-preserving seq2seq model.

**"Framework neighbour" does not imply "numerical row".** All three neighbours
are conceptual-evidence-only, and that is a *finding*, not an omission: the
canonical policy makes their numerical role conditional on a native common
protocol, and `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §4 tested that condition and
found none. The framework evidence is delivered as the trilemma —
`CONSTRAINTS_SECTION_TRILEMMA.md` — plus a feature table, which is exactly the
"compare conceptually" branch the policy provides for.

> **Open question for the lead, flagged rather than decided.** Under a strict
> reading of the four definitions, a neighbour that can never supply a number is
> `CONCEPTUAL_LINEAGE_ONLY`. I have kept CDD / PRODIGY / ConStruct as
> `FRAMEWORK_NEIGHBOR` with `evidence mode: conceptual only`, because the
> canonical doc names them the constraints block's *methodological lineage* and
> lists them separately from Edit Flows and Expanding Flow Maps. If you prefer
> the strict reading, the label changes and nothing else in this lane does.

---

## Capability findings (unchanged, verified)

**Binding policy:** `docs/BASELINE_IMPLEMENTATION_POLICY.md` — published methods
run the **authors' native algorithm** with only a thin evaluation/accounting
adapter. `N/A` with a reason beats a distorted adaptation.

---

## 0. The headline, because it changes the paper's framing

> **GraphXForm does not have user-specified scaffold preservation.**

The charter's premise — *"GraphXForm already supports … preserving/excluding
specified moieties"* — does not survive contact with the source. Two different
mechanisms have been conflated in our own notes:

1. **`start_from_smiles`** — real and native. `config.py`:
   `self.start_from_smiles = None  # Give SMILES and set 'start_from_c_chains=False'.`
   Used in `core/gumbeldore_dataset.py`:
   `MoleculeDesign.from_smiles(self.config, self.config.start_from_smiles)`.
2. **`include_structural_constraints`** — **not** scaffold preservation. It is a
   hard-coded *solvent-chemistry feasibility filter*: ring sizes limited to five
   or six atoms; no N–N or O–O single bonds; restrictions on carbons bonded to
   two nitrogens unless forming urea (paper §3.3.2). Implemented as
   `molecule_evaluator.py::infeasible_by_special_constraints`, gated on
   `objective_type in ["IBA", "DMBA_TMB"]`. No `MolFromSmarts` /
   `HasSubstructMatch` appears in that path. **It takes no user SMARTS.**

What GraphXForm actually has is preservation **as a consequence of its action
space**: the actions are `DontChange`, `AddAtom`, `AddBond` (§2.1) — *there is no
deletion*. §3.3.3: *"Extending the action space to include atom and bond removal
is straightforward within our framework; however, we leave the exploration of
this possibility for future work."*

**Three consequences:**

- GraphXForm preserves the **entire** start molecule necessarily. It cannot
  express "preserve *this* core and freely edit the rest", because nothing can be
  edited away.
- It therefore cannot perform the edits our task requires. This matches Lane 3's
  independent finding that GraphXForm is `inappropriate` for the deletion arm of
  its conventional suite (task T4).
- The differentiator we may legitimately claim is **not** "COMPOSE can preserve a
  scaffold". It is that COMPOSE can preserve a *specified* core **while deleting
  and rewriting elsewhere**, with the constraint injected at inference time on an
  already-realized state. GraphXForm's preservation is total and structural;
  ours is partial and selectable.

`AddBond` also acts *"between existing, unbonded atoms"*, so new rings can form
across the start molecule's atoms. Atoms and existing bonds are preserved; the
**ring system is not** guaranteed.

---

## 1. Provenance

| method | venue / year | identifier | official code | checkpoints | license |
|---|---|---|---|---|---|
| **GraphXForm** | Digital Discovery (RSC) 2025 | arXiv:2411.01667; DOI 10.1039/D4DD00339J | github.com/grimmlab/graphxform | yes (`graphxform_pretrained.zip`) | MIT |
| **MolEditRL** | ICLR 2026 Poster (verified) | arXiv:2505.20131 | **none found** | no | n/a |
| **Prompt-MolOpt / ^P** | Nature Mach. Intell. 6(11):1359–1369, 2024 | DOI 10.1038/s42256-024-00916-5 | github.com/wzxxxx/Prompt-MolOpt | **yes, in-repo, incl. ^P** | MIT |
| **InVirtuoGen** | ICLR 2026 (verified) | arXiv:2509.26405 | github.com/invirtuolabs/InVirtuoGen_results | yes (Zenodo 16874868, HF) | **CC-BY-NC-SA 4.0** |
| **ConStruct** | NeurIPS 2024 Poster | arXiv:2406.17341 | github.com/manuelmlmadeira/ConStruct | **no** | MIT |
| **DDSBM** | ICLR 2025 | arXiv:2410.01500 | github.com/junhkim1226/DDSBM | **no** (Zenodo TODO unchecked) | **NONE — no LICENSE file** |

MolEditRL venue verified at `iclr.cc/virtual/2026/poster/10011588` (Poster,
2026-04-24). InVirtuoGen verified at `iclr.cc/virtual/2026/poster/10009569`.

## 2. Capability matrix

| method | native source molecule | **native HARD user-specified substructure preservation** | intermediate states | objective-specific retraining |
|---|---|---|---|---|
| **GraphXForm** | **YES** (`start_from_smiles`) | **NO** — preservation is total, from an add-only action space; `include_structural_constraints` is a fixed solvent filter | partial (`keep_intermediate_trajectories`, training-only) | **YES** (per `objective_type`) |
| **MolEditRL** | by design, **not runnable** | **NO — soft.** §3.4: `β·Σ D_KL(p_θ‖p_pre)` + RL reward. No frozen or masked nodes | `UNVERIFIED` | yes (RL stage) |
| **Prompt-MolOpt^P** | **YES** (`src_smi`, atom-map marked) | **YES — hard by construction** | per-iteration scripts; no per-step API | partial (new property ⇒ new SME/MGA pipeline) |
| **InVirtuoGen** | **YES** (fragment `prompt`) | **YES — hard clamp, but at fragmented-SMILES *token* positions, not graph atoms** | **YES** (`return_trajectory`) | fragment tasks **no**; property opt **yes** (GA+PPO) |
| **ConStruct** | **NO** | **NO** — projectors are `planar` / `tree` / `lobster` only | yes (`chains_to_save`) | yes |
| **DDSBM** | paired **dataset** only | **NO** | yes (`chains/`, `graphs/`) | yes (full SB training) |

### Evidence per cell

- **Prompt-MolOpt^P is hard by construction and is a separately released
  model, not an ablation.** Distinct training script `PromptP_molopt_train.sh`,
  distinct data `Prompt_MolOptP_train_val_ADMET.csv`, and a **committed 80 MB
  checkpoint** `checkpoints/fs_mga_remark_data/model.500000_99.pt`. The user
  marks editable atoms with atom-map numbers in `src_smi`;
  `mga_utils/mol_opt_data_remark_mga_generate_test.py:491` does
  `retain_sub = src_smi.split("[Opt Frag]")[0]`, with dedicated vocab tokens
  `[Pharmacophores]` / `[Opt Frag]`. The network **emits only the replacement
  fragment**, and `return_mark_opt_smi(...)` rebuilds from
  `Chem.RWMol(retain_sub)` — so the retained part is never generated and cannot
  be corrupted.
- **InVirtuoGen's clamp is real but string-level.** Paper §3.2: *"the
  corresponding positions are naively overwritten at every timestep of the
  simulation"*; §5 concedes this *"disrupts the learned flow dynamics"*. Code:
  `in_virtuo_gen/models/invirtuofm.py`, `if prompt is not None: if force_prompt:
  x_t[prompt_mask] = prompt_slice[...]`. **Token positions in a fragmented
  SMILES, not graph atoms** — worth flagging wherever we distinguish graph-level
  from string-level guarantees.
- **ConStruct is a general graph method.** `projector/projector_utils.py`
  defines only `PlanarProjector`, `TreeProjector`, `LobsterProjector`.
  Constraints must be edge-deletion invariant, which structurally rules out a
  "preserve this subgraph" projector in the released design. `configs/dataset/`
  ships no molecular config; molecules appear only in Appendix G.
  *Disambiguation:* the ICML 2026 poster "Hard-Constrained Graph Generation with
  Discrete-Projection Diffusion" is a **different paper** (NSPSG), not a ConStruct
  extension.
- **DDSBM takes a paired distribution, not a query molecule.** README requires a
  CSV with `REF-SMI` (`X_0`) and `PRB-SMI` (`X_T`) columns; a single user source
  cannot be dropped in without a bridge already trained for that transformation.

## 3. Runnability status (orthogonal to role)

**Role says what a comparator is *for*; this table says whether it *can be run*.**
A `FRAMEWORK_NEIGHBOR` may be unrunnable and still carry the section, and a
runnable method may be only `TASK_COMPETENCE`.

| method | role | runnability | rationale |
|---|---|---|---|
| **CDD** | `FRAMEWORK_NEIGHBOR` | **`NOT_RUNNABLE`** | Official repo is a 59-byte placeholder README; the differentiable surrogate is unreleased. Two unofficial third-party reimplementations exist and **must not** be used as "the published method". |
| **PRODIGY** | `FRAMEWORK_NEIGHBOR` | **`BLOCKED — no licence`** | Repo carries **no LICENSE file**; no licence is no grant of rights. Same blocker class as DDSBM. Also not a standalone library — integration is a manual per-model patch. |
| **ConStruct** | `FRAMEWORK_NEIGHBOR` | **`RUNNABLE_BUT_OFF_TASK`** | MIT, maintained, but **no checkpoints shipped** and no molecular Hydra config (`configs/dataset/` holds exactly `high_tls`, `lobster`, `low_tls`, `planar`, `tree`). De novo only. |
| **GraphXForm** | `TASK_COMPETENCE` | **`RUNNABLE`** | Native `start_from_smiles` is real. Hard-constraint cell is **`N/A`** — no user-specified preservation, add-only action space (§0). **Reuse Lane 3's artifacts; do not rebuild.** |
| **Prompt-MolOpt^P** | `TASK_COMPETENCE` | **`RUNNABLE`** | Official code, MIT, committed ^P checkpoint, hard-by-construction preservation, native source conditioning. |
| **InVirtuoGen** | `TASK_COMPETENCE` | **`CONDITIONAL`** | Token-level clamp, not graph-level; **CC-BY-NC-SA 4.0 (non-commercial)**. Coordinate before running — do not run it twice to add a row. |
| **MolEditRL** | `TASK_COMPETENCE` | **`OFFICIAL_CODE_UNAVAILABLE`** | No repo found; the release statement covers the MolEdit-Instruct dataset, not code. **No reimplementation.** Soft preserver regardless. |
| **DDSBM** | `FRAMEWORK_NEIGHBOR` *(general editing block)* | **`BLOCKED — no licence`** | No LICENSE file, no released checkpoints. **Lane 5 owns qualification.** |

## 3a. Tier 4 does not mean useless — reuse CDD's TASK, not CDD

**The distinction the amendment draws, applied here:** a method can be
unreconstructable while its **published task definition** remains valuable.

**Reusable:** CDD's constraint predicate `SA(y) ≤ τ` with its published
thresholds **τ ∈ {3.0, 3.5, 4.0, 4.5}** (§5.2, confirmed verbatim), as an
**externally defined task**.

**Why this is worth having, and it is not a consolation prize.** The constraint
definition is then **independent of our corpus**. We are not shopping our own
data for a constraint that happens to leave enough room to act — which is
*exactly* what killed the Bemis–Murcko branch. An externally fixed threshold
cannot be selected on our outcome, because it was fixed by someone else, for
their reasons, before they had ever heard of us.

**Reproduce it verbatim or not at all.** Three conditions:

1. `SA ≤ τ` in that direction, at those four thresholds, no others;
2. RDKit's `sascorer` (`molecular_quality.py:13` already exposes it) — **and we
   pin our own RDKit version and say so**, because CDD pins none and SA values
   shift with the fragment-contribution tables, so a τ = 3.0 boundary is not
   version-portable;
3. no re-tuning of τ to suit our fiber. If none of the four thresholds is
   workable on our sources, that is a reportable result, not a reason to pick a
   fifth.

**CDD itself remains contextual, non-head-to-head.** Reusing the task is not
reusing the comparison: their numbers are de novo QM9 under an unreleased
surrogate, and ours would be source-conditioned editing under the true scorer.
**The task travels; the numbers do not.**

## 4. The minimum defensible constraint table

**The framework layer carries no numerical rows.** That is the audit's finding,
not a gap: all three neighbours are de novo, none is source-conditioned, and none
shares a protocol with source-conditioned editing.

**Layer 1 — framework novelty (`FRAMEWORK_NEIGHBOR`, conceptual):** the trilemma
plus a feature table. CDD, PRODIGY, ConStruct. See
`CONSTRAINTS_SECTION_TRILEMMA.md`. **No numbers.**

**Layer 2 — causality (`MATCHED_CAUSAL_CONTROL`) — the primary comparison:**

| row | what it establishes |
|---|---|
| **post-hoc filtering vs soft guidance vs exact support restriction** | whether keeping the process inside the admissible state space beats generating and discarding, or penalizing |

Same `R_θ`, source, objective, controller class and budget; the constraint
mechanism is the only thing that varies. **Lane 2 owns building it** —
`LANE2_FIVE_QUESTIONS.md`. This lane does not design or run it.

**Layer 3 — competence (`TASK_COMPETENCE`), at most one or two rows:**

| row | what it establishes | hard-constraint cell |
|---|---|---|
| **GraphXForm** | a strong source-conditioned graph editor with native fine-tuning + search | **`N/A` — no user-specified preservation; add-only action space** |
| **Prompt-MolOpt^P** | a genuine hard, user-specified fragment/pharmacophore preserver | **YES — hard by construction** |

**Do not add a fourth competence row.** InVirtuoGen would make the same point as
Prompt-MolOpt^P at the cost of a non-commercial licence and a
graph-vs-string-granularity caveat.

> **The layers are reported separately and never merged into one leaderboard.**
> Merging them is exactly the failure the canonical policy exists to prevent: it
> lets a task-SOTA number stand in for the framework comparison.

## 5. What must NOT be claimed

- ❌ COMPOSE invented scaffold preservation or structure-preserving editing.
  **Prompt-MolOpt^P** (NMI 2024) and **InVirtuoGen** (ICLR 2026) both do hard
  user-specified preservation, and **MolEditRL** (ICLR 2026) does soft
  structure-preserving editing.
- ❌ COMPOSE invented hard-constrained graph generation. **ConStruct**
  (NeurIPS 2024) guarantees structural properties throughout the trajectory.
- ❌ That GraphXForm "supports preserving specified moieties". It does not; see §0.
- ❌ "without retraining" — **barred project-wide** by Lane 3's REINVENT finding
  (Adam optimizer state carries unbroken across an objective switch). Say
  *"with no objective-specific update to `R_theta`"*, which is what we mean and
  can defend.

**The defensible differentiator**, and the only one:

> Hard constraints compose with the same reusable executable molecular process,
> and can be imposed or changed **at inference time on the exact molecular state
> already reached**, restricting the canonical successor support without any
> objective-specific update to `R_theta`.

Every comparator above either bakes its constraint into training, fixes it at
initialization, or cannot delete. None of them injects a *new* hard constraint
onto a *realized intermediate state* of an ongoing process.

## 6. Reuse obligations

Lane 3 (`codex/compose-baseline-qualification`) owns the comparator registry and
the GraphXForm adapter. **This lane rebuilt none of it.** Reuse, do not fork:

- `docs/workstreams/baseline-qualification/comparator_registry_v3.json` — authoritative
- `src/compose_v4/experiments/graphxform_applicability.py` — frozen applicability domain
- `src/compose_v4/experiments/graphxform_adapter.py` — `verify_upstream_unmodified`, `run_graphxform_from_source`
- `src/compose_v4/experiments/oracle_accounting.py` — the three counters
- `src/compose_v4/experiments/shared_evaluator.py` + `scripts/compose_shared_evaluator_server.py` — the pinned RDKit boundary

Two Lane 3 conditions carry over to us:

1. **The frozen GraphXForm route** is `official checkpoint → native
   objective-specific fine-tuning / self-improvement → native beam / TASAR search
   → thin common evaluation adapter`. The hand-driven greedy action loop was
   withdrawn and **must not be revived**; its diagnostic finding (greedy argmax
   from the un-fine-tuned checkpoint selects `TERMINATE` immediately on every
   held-in source) must not travel as a result.
2. **`FAIRNESS_CONTRACT.md:64` is stale** — it still records the GraphXForm heavy-atom
   ceiling as "42 = 50 − 8" with `headroom_basis: CHOSEN_SAFETY_MARGIN`. The code at
   Lane 3's tip has `REQUIRED_HEADROOM = 0`, `HEAVY_ATOM_CEILING = 50`,
   `NONE_NATIVE_CEILING_ONLY` (the margin was withdrawn in `7265df6`). **The code is
   authoritative.** Flagged to Lane 3.

## 7. Counter-name discrepancy — resolve before any shared table

The charter names the three counters `raw_instrument_oracle_requests`,
`algorithmic_oracle_requests`, `benchmark_eval_requests`. The **implemented and
frozen** names in `src/compose_v4/experiments/oracle_accounting.py` are:

| implemented | meaning |
|---|---|
| `unique_valid_canonical_evaluations` | benchmark-native; the only PMO-comparable quantity |
| `oracle_requests` | every request incl. duplicates/rejects/invalids — algorithmic demand, **not** CPU |
| `evaluator_calls` | expensive executions after caching — real work |

**Use the implemented names.** They are already wired, tested, and carry four
machine-checked invariants. Flagging the mismatch so a shared table does not end
up with two vocabularies for one set of counters.
