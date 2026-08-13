# External baseline task matrix — hard structural constraints

**Status: `DESIGN_ONLY`.** No baseline was run by this lane. No external
dependency was installed. Every capability cell cites a paper section, a repo
file, or a flag name; anything not confirmed against a primary source is marked
`UNVERIFIED` rather than guessed.

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

## 3. Status decisions

| method | status | rationale |
|---|---|---|
| **GraphXForm** | **`MUST_RUN` as a source-conditioned editor; `N/A` for the hard-constraint cell** | Native start-from-SMILES is real, so it belongs in the source-conditioned comparison. It has no user-specified preservation and no deletion, so its hard-constraint cell is `N/A — not natively supported`, with the §0 reason stated. **Reuse Lane 3's artifacts; do not rebuild.** |
| **MolEditRL** | **`OFFICIAL_CODE_UNAVAILABLE`** | GitHub search for `MolEditRL` returns 0 repos; the paper's only release statement covers the MolEdit-Instruct dataset, not code; the ICLR poster page carries no code link. **No reimplementation** (charter). Note it would be a *soft* preserver even if released, so it is not a hard-constraint comparator. Related work. |
| **Prompt-MolOpt^P** | **`MUST_RUN` — the one direct structure-preserving editor** | Official code, MIT, committed ^P checkpoint, hard-by-construction user-specified preservation, native source conditioning. This is the correct third row. |
| **InVirtuoGen** | **`CONDITIONAL` — coordinate before running** | Task semantics align for scaffold decoration / motif extension. Two blockers: the clamp is token-level not atom-level, and the licence is **CC-BY-NC-SA 4.0 (non-commercial)**. Charter: do not run it twice to add a row — check with whichever lane evaluates its native fragment-constrained protocol first. |
| **ConStruct** | **`CONTEXT_ONLY` — cite, do not port** | General graph method, no source conditioning, no released molecular config. Charter is explicit. |
| **DDSBM** | **`CONTEXT_ONLY` — blocked** | **No LICENSE file at all** (default all-rights-reserved) and no released checkpoints. Legal blocker before technical. Matches Lane 3's independent `blocked` / `CONTEXT_ONLY`. |

## 4. The minimum defensible constraint table

Three rows. Two groups, reported separately per the baseline policy.

**Group 1 — external published methods (native algorithm, thin adapter):**

| row | what it establishes | hard-constraint cell |
|---|---|---|
| **GraphXForm** | a strong source-conditioned graph editor with native fine-tuning + search | **`N/A` — no user-specified preservation; add-only action space** |
| **Prompt-MolOpt^P** | a genuine hard, user-specified fragment/pharmacophore preserver | **YES — hard by construction** |

**Group 2 — controlled internal comparator (we implement it, because it is
defined relative to COMPOSE):**

| row | what it establishes |
|---|---|
| **COMPOSE `posthoc_scaffold_filter_verified` vs `hard_scaffold_mask_verified`** | whether keeping the process inside the admissible state space beats generating and discarding |

**Do not add a fourth row.** InVirtuoGen would be a fourth structure-preserving
generator making the same point as Prompt-MolOpt^P, at the cost of a
non-commercial licence and a graph-vs-string-granularity caveat.

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
