# Learnings

Durable, dated gotchas + design calls. Append; don't rewrite history.

## 2026-07-21
- **macOS fork deadlock:** the multiprocessing rollout pool (`tracelet_sampling_worker`) deadlocks
  at 0% CPU on this Mac at `workers>=2`. Use `workers=1` (serial) locally; Modal for scale. Killing
  the parent orphans the workers — `pkill -9 -f tracelet_sampling_worker` too, or they leak CPU/RAM.
- **Base corpus = GuacaMol, not ZINC.** `train_tracelet_gm.py` loads `guacamol_subset_500000_…smiles`;
  the earlier handoff's "ZINC-250k" claim was wrong. Constrained-design *leads* are ZINC-derived
  (Jin QED set). "Matches GrIDDD" holds at the element level (CNOF), not the corpus.
- **Base = Lineage B is a step-1000 preview** and overproduces small rings (aziridine/epoxide) — this
  inflates the pathwise-safety *magnitude* (not the guarantee/free-cost). See lineage-correction V2.
- **Conditional results are mechanism-driven** (fiber + SMC), so base-independent; they carry to any
  backbone. Only E1 (unconditional quality) and pathwise-safety magnitude depend on the base.
- **Framing (Paper 1):** pathwise-constrained generation is the spine/spotlight; structural control +
  oracle efficiency + anytime are supporting; exactness (E0 + guidance-ground-truth) is the rigor
  foundation. QED optimization is NOT a GrIDDD beat (oracle-hungry); FCD is deferred, not a headline.
- **Fairness:** never compare our *selected* top-k to a baseline's *all-sample* rate — use per-sample.

## 2026-07-22
- **Full-suite segfault = dual OpenMP, not a code bug.** `pytest tests/` segfaults (exit 139) at
  ~40% on this Mac: torch 2.11 (`~/Library/Python/3.14`, bundles its own libomp) + rdkit/numpy/
  scipy/sklearn (`/opt/homebrew`, brew's libomp) load two OpenMP runtimes in one process on
  Python 3.14.2; the `.venv` pip-manages none of them (only `_virtualenv.pth`). `KMP_DUPLICATE_LIB_OK=TRUE`
  silences the abort but not the corruption — after enough thread-pool churn it faults (tips at the
  sklearn-based `test_pan_lung_filtering` lipid test, whichever test crosses the threshold). Every
  test passes in isolation. **Fix (env, not code): `OMP_NUM_THREADS=1`** → segfault gone. The one
  remaining fail (`test_parallel_tracelet_sampling`) is a subprocess `PYTHONPATH` gap — the worker
  spawns `python -m compose_v4...` which needs `src` on the path; **`PYTHONPATH=src`** fixes it.
  Green invocation: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src pytest tests/` → 427/427.
  Real root cause is that `.venv` isn't a clean single-source install (`uv sync`/`pip install -e .`
  would put one OpenMP + compose_v4 in the venv); don't reinstall reactively — document + move on.

- **Rigor pass on the Pareto-reachability spotlight (control-substrate reframe).** Before trusting a
  "greedy vs optimal" gap, stress it three ways: fair accounting, sensitivity, evaluation-lookahead.
  (1) The frontier-**recovery** "matched-budget negative" (unguided detours lose to greedy) did NOT
  survive: it was confounded (asymmetric archiving; the walk never cooled) and, once fixed, purely a
  temperature dial (t0=0.3→0.21, t0=0.02→0.54==greedy). **Retracted.** (2) The **barrier diagnostic**
  DID survive: its monotone comparator is a generous upper bound on greedy, and giving greedy a full
  edit of *evaluation* lookahead changes nothing — provably, since every exterior neighbour of the
  monotone-reachable set has strictly lower U (`frac_dirs_suboptimal_lookahead1 == lookahead0`). The
  barrier only erodes with genuine ≥2-edit lookahead (cap4 2-obj: 18.6%→16.0%@d2→8.9%@d3) and the
  cap5 worst-lead 91% is unchanged at d1. So: greedy has a real structural ceiling; efficient
  *recovery* of the barrier-gated frontier is undecided by the toy (needs the learned model on real
  leads). Lesson: a scalar-U movement gap ≠ Pareto-frontier coverage — state which metric, and test
  evaluation-fairness before claiming a controller "can't reach" something.

## 2026-07-24
- **B's REAL editing vocabulary on a real (cyclic, unsaturated) molecule — traced, not assumed.** For
  the corrupted-source-prior editing model (B-edit), what B can actually apply to a complete drug-like
  molecule is narrower than the 10 mark families suggest:
  - **Graft (`bond_reroute`) is gated to `all_single_tree`** — `factorized_tracelet_rate_model.py:1073`
    (`real and nx.is_tree(graph) and all bonds single`). The `graft_mask` starts all-zeros and is only
    populated inside `if all_single_tree:`. So B graft fires only in the carbon-skeleton PHASE (early,
    pre-ring/pre-double-bond); it CANNOT propose graft on a real lead. (Not "carbon-tree at t=0 only" —
    any tree-like state, but a real molecule is never one.)
  - **Aromaticity is ONE-WAY.** B can aromatize (saturated→aromatic): `ring_system_grow` places
    aromatic rings via `aromatic_edges`; the only aromatic restate enumerator filters to
    `perceived_aromatic_ring_count(successor) > before` (`tracelet_fiber.py` `_validated_aromatic_restates`).
    B CANNOT de-aromatize: `ring_system_restate` is dead at inference (candidates hardcoded to `()` in
    `prepare_factorized_mark_batch` ~`factorized_tracelet_rate_model.py:960`) AND `is_valid_bond_reorder`
    hard-rejects aromatic bonds (`operators.py:168`: `old_order==BOND_AROMATIC → False`). Aromaticity is
    ring-level (delocalized), so single-bond `bond_reorder` correctly can't touch it — coordinated
    `ring_system_restate` is the only mechanism, and only its aromatizing direction was wired (a de-novo
    choice: build toward drug-like).
  - **Ring building IS fully parametric** (so the user's memory was right): `RingSystemGrow` carries
    `atom_payloads` (per-atom identity), `scaffold_bonds`+`system_atoms` (ring size/topology, incl.
    fused/spiro via the template structure — `topology_class` itself is only a coarse "cyclic" label),
    `aromatic_edges` (aromatic vs saturated).
  - **The structured ring delete is decoration-LOSSY — wrong tool.** `enumerate_structured_ring_system_deletes`
    → `_instantiate_structured_delete` (`ring_system_fiber.py`) opens the cycle bonds and **retypes every
    ring atom to carbon** (`atom_type=carbon`, `atom_deletions=()`, `retained_system_atoms=()`). Topology-
    only; destroys heteroatoms/aromaticity, so its inverse `ring_system_grow` can't cleanly round-trip
    (empirically: `ring_system_delete` fires in trim, `ring_system_grow` never survives the grow-replay).
    DROPPED from the corrupted prior. Clean whole-ring editing needs the exact inverse relationship
    (`inverse_ring_system_grow` ↔ delete), which preserves decoration; whole-ring *removal* is rare and
    lower-priority than de-aromatization + graft.
- **Design call (locked with user): full B-edit editing vocabulary in ONE retrain (v2), not v1-clean-then-enrich.**
  Vocabulary = micro (peripheral `atom_insert/delete`, bioisostere `atom_restate` incl. ring atoms,
  non-aromatic `bond_reorder`) [done, data-only] + **de-aromatization** (wire `ring_system_restate` at
  inference + a de-aromatizing enumerator) + **graft-for-editing** (ungate graft to cyclic bridge-pendant
  subtrees) + retained `ring_system_grow` (ring-adding); drop the carbon-izing ring-delete. The two
  additions are MODEL-INTERNAL changes. **Gating constraint:** they must be enumeration/mask/quotient
  changes that REUSE B's existing heads (`bond_reroute`, `ring_system_restate` heads already exist),
  preserving parameter shapes so the warm-start (`--initialize-from-source-checkpoint`, strict) still
  loads B — verify architecture-preserving before writing model code.
- **De-aromatization BUILT (native, architecture-preserving, verified).** Key fact: executable states
  are **Kekule** (integer bond orders; "aromatic" = RDKit perception on the round-tripped SMILES, never
  a stored class-4 bond — `chem/molecular_graph.py:509` kekulizes on load), so aromatic->saturated is a
  validity-checked double->single lowering that adds implicit H. Implementation: (1) new
  `enumerate_ring_system_restate_actions` (tracelet_fiber.py) returns restates that CHANGE perceived
  aromaticity — BOTH directions; the de-aromatizing all-single successor was already produced by
  `_ring_system_restate_candidates` (:386-387), only discarded by the `>` filter in
  `_validated_aromatic_restates` (:559). Refactored to `_validated_restates(..., keep)`; aromatize keeps
  `after>before`, the new enumerator keeps `after!=before`. (2) collator `prepare_factorized_mark_batch`
  gained `compute_ring_restates: bool = False` (default off preserves B perf + behavior); on -> enumerate
  restates, reviving the ring_system_restate head that was dead (`restate_actions.append(())`). Verified:
  flag off -> [], on -> toluene/cyclohexane populate. (3) corruption samples restates (`_RESTATE_WEIGHT`,
  `restate_prob=1.0`). inverse_step handles ring_system_restate -> both directions replay (28/28). Reuses
  B's existing `ring_restate_head`/`restate_order_embedding` -> strict warm-start safe. Config plumbing
  (`self.enable_ring_restates` -> the `:2984` sampling call + the training collator) is deferred to the
  recipe-wiring phase; the CAPABILITY is complete.
- **Corruption selection must be family-first** (draw a mark family by weight, THEN try its instances),
  not a weighted permutation over all instances: the latter over-picks whichever family is most reliably
  valid (bond_reorder ballooned to 40%, restate to 52%). Family-first tracks the weights — realized mix
  ~ peripheral + bioisostere dominant, bond-order + aromaticity-flip smaller. All edits are
  ring-count-preserving (ring topology add is left to B's retained ring_system_grow).
- **Graft-for-editing BUILT (cyclic pendant relocation, architecture-preserving, verified).** B gates its
  dense graft mask to `all_single_tree` -- its colored-tree canonicalizer peels leaves and asserts a tree,
  so it CRASHES on a cycle -- hence B cannot graft a real molecule. The editing model adds a
  `compute_cyclic_graft` branch (default OFF -> B byte-identical; also keyed into the feature cache) that
  enumerates the well-defined subset: relocate a PENDANT tree fragment across a SINGLE-bond bridge (ring
  core fixed, no ring travels). Facts nailed from the code, not assumed: the model graft is a RESTRICTED
  reroute -- `BondReroute(a=moved, b=removed_neighbor, u=moved, v=target)` so u must be a cut endpoint
  (teacher `_teacher_action_score` rejects `u not in {a,b}`; sampler rebuilds the identical form); default
  `new_order=BOND_SINGLE` matches the 2D (moved,target) mask (no bond-order axis). The successor quotient
  uses the GENERAL canonical key (`canonical_state_key` of the relocated state), never the tree
  canonicalizer. Shared enumerator `pendant_graft_candidates` / `enumerate_pendant_graft_actions`
  (`rewrite/factorized_fiber.py`) is used by BOTH the model mask and the corruption -> a teacher graft
  always lands in the mask it is scored against (DRY = the correctness guarantee). Reuses B's `graft_head`
  (`3*hidden->hidden->1`, shape depends only on hidden_dim) -> strict warm-start safe. Verified: propyl-
  benzene 11 grafts (off->0), benzene 0, benzene|pyridine-linker 0 (ring fragment = deferred v3), 11/11
  execute through the real executor AND match the quotient successor, group ids partition grafts exactly
  by molecule (`tests/test_cyclic_graft.py`, 4 tests). Corruption both directions replay 73/73
  (inverse_step handles bond_reroute). Realized v2 edit mix: bioisostere 26 / bond-order 20 / graft 17 /
  peripheral 26 / aromaticity-flip 9 %. **Full suite 440 passed.** Still TODO (recipe phase): a model-config
  flag driving `compute_ring_restates` + `compute_cyclic_graft` at the `:2984` sampling call and the
  training collator, so both light up for B-edit but stay off for de-novo B.
- **A corrupted-prior/edit teacher mark must come from the fiber the model's DENSE MASK represents,
  restricted to the sites that mask supports -- executor validity is a strict SUPERSET of dense-mask
  legality.** The corrupted-prior training loss was `+inf` until two corruption<->mask inconsistencies
  were fixed (found only because the corruption had never been *training*-smoked -- the dataset histogram
  + replay checks pass regardless). (1) It enumerated micro edits with `fiber._candidate_actions` (the
  GENERAL fiber: every atom x every `spec.atom_state`, executor-valid but a superset of B's mask). B is
  trained on the FACTORIZED fiber -> use `factorized_fiber._factorized_candidates(node,
  allow_bond_reroute=False)`, whose `atom_restate` uses the canonical derived h-count
  (`CNOF_VALENCE[type] - row_sum`) the dense head scores. (2) B's `atom_restate`/`bond_reorder` masks are
  **PERIPHERAL-ONLY** -- they exclude RING atoms and RING (cycle) bonds, because tree-transport teachers
  never restate/reorder a ring site (ring-atom identity is `ring_system_grow`'s job, ring bond orders are
  `ring_system_restate`'s), so the derived mask has no support there. A ring-site micro teacher mark is
  outside the mask -> grouped logsumexp `-inf` -> rate 0 -> bregman loss `+inf` (NOT an exception; it
  survives forward+backward as inf/nan). Fix: restrict the corruption's `atom_restate` to non-ring atoms
  and `bond_reorder` to non-cycle-edge bonds (`_ring_atoms_and_cycle_edges` in source_corruption.py).
  **CORRECTS the earlier "atom_restate covers ring-atom bioisostere" claim -- it does NOT**; pyridine<->
  benzene ring-atom swaps are OUTSIDE B's micro vocabulary (only aromaticity flips via ring_system_restate
  touch a ring). After both fixes all 6 marks score FINITE, the mixed batch backprops, 440 tests pass.
  **LESSON: always training-smoke a new teacher source (forward_mark_batch + bregman_loss FINITE) before a
  Modal launch** -- histogram/replay checks do not exercise the dense-mask scoring path.
- **B-edit recipe WIRED + training verified.** `--corrupted-prior-mix` in train_tracelet_cnof_gate.py
  constructs the factorized model with `enable_ring_restates` + `enable_cyclic_graft`, appends
  corrupted-prior records to `train_records` IN MEMORY right after the empty-partition check (never
  persisted, so B's carbon path cache stays reusable under `--train-only`), and adds a
  `corrupted_prior_mix` discriminator to `_training_support_cache_signature` so the mixed tuple recompiles
  its support. The collator flags thread via the MODEL (`train_factorized_mark_model` reads `model.enable_*`
  -> `factorized_mark_loader` -> `FactorizedMarkCollator`), so the gate only sets them once at model
  construction. Warm-start is SAFE: `--initialize-from-source-checkpoint` does `load_state_dict` (weights)
  into the flag-constructed model, so `enable_*` is preserved. Verified: `train_factorized_mark_model`
  runs on mixed de-novo+edit records with FINITE loss + backprop, collator picks up True/True. REMAINING
  before Modal: modal-app passthrough (add `corrupted_prior_mix` to `modal_apps/train_tracelet_gm.py`
  `main` -> `recipe["arguments"]`, check `build_tracelet_recipe_argv` bool handling), and the inference
  LOAD side (read the `corrupted_prior_mix` checkpoint-metadata flag when reconstructing B-edit for the
  editing experiments so `enable_*` turns on at sampling) -- both are Modal-launch / Phase-3 prep.

## 2026-07-25
- **Ring-atom heteroatom scanning ENABLED for B-edit** (updates the 07-24 "ring-atom bioisostere is
  outside B's vocabulary" note -- true for de-novo B, but the edit regime ungates it). `enable_heteroatom_
  scan` drops the `atom_topology==0` peripheral gate on the `atom_restate` dense mask, so pyridine<->
  benzene (N<->C) ring-atom swaps score FINITE. Off for de-novo B (byte-identical). Verified
  `Cc1ccncc1->Cc1ccccc1` finite with the flag, inf without.
- **Clean (decoration-preserving) ring editing for B-edit.** B's structured `ring_system_delete`
  carbon-izes (retypes every ring atom->C, destroying decoration); its inverse `ring_system_grow`
  cyclizes CARBON scaffolds (retyping). Added `enumerate_clean_ring_system_deletes` (open a ring keeping
  each atom's element, H re-derived per atom; Kekule states so it covers saturated/aromatic/any-size/
  fused/spiro/bridged) + `enable_ring_opening` to swap it into the model's sampling AND teacher-candidate
  enumeration. Verified: keeps N/O where structured->all-C (pyridine: `CCCCCC` vs `CC=CC=NC`), round-trips
  5/5 via inverse_step.
- **The clean delete is taught TRIM-ONLY.** Its exact inverse is a `ring_system_grow` on a HETEROATOM
  scaffold, and B's grow vocabulary only cyclizes CARBON scaffolds -> that grow teacher mark is
  mask-illegal (loss split: DELETE-trim FINITE 2.30, GROW-inverse inf). So ring-OPENING (de-cyclize
  keeping heteroatoms) is taught; ring-CLOSING stays retained from B's de-novo grow. B-edit thus reaches
  parity with B's ring vocabulary (grow retained + clean delete + restate), the delete upgraded to keep
  decoration. Full mixed corruption batch (all 7 families incl. `ring_system_delete`) backprops FINITE
  (4.63); 440 tests pass. Heteroatom re-cyclization would need a NEW grow mode (grows on heteroatom
  scaffolds) -- deferred, not a regression vs B (B never cyclized heteroatom scaffolds either).
- **`inverse_ring_system_delete` latent scaffold bug fixed:** it read `scaffold_bonds` off the pre-delete
  ring, so the closing bonds double-counted (scaffold + `bond_insertions`), yielding an invalid grow;
  exclude the reinserted bonds. Only reachable via `inverse_step` on a `ring_system_delete` (untested) ->
  latent. **LESSON reaffirmed:** split the finite-loss smoke by direction -- the mixed batch hid that
  DELETE was fine and only GROW was inf.

## 2026-07-26
- **Pre-launch B-edit audit (3 independent code sweeps + empirical model runs) found + fixed 2 real
  blockers; everything else verified clean.** Local gate green throughout (440 + ruff).
- **CHARGE was a training blocker (fixed).** ~6% of GuacaMol and 29% of the Jin-800 leads are charged
  (nitro/N-oxide/ammonium). Charged atoms broke teacher-scoring TWO ways: (1) `class_index` was called
  WITHOUT the atom's `formal_charge` at the two teacher sites (`factorized_tracelet_rate_model.py:3812,
  :3836`), so for a charged restate/insert (an inverse mark reconstructing the lead's aromatic N+) it
  returned `None` → `logits[...,None]` = a vocab-wide `[15]` tensor → `RuntimeError` in
  `_selected_mark_log_probability`; (2) `ring_system_restate` on a charged ring → "outside exact dynamic
  candidates". The dense edit heads encode only NEUTRAL `(element,valence)` classes, so charged atoms are
  unrepresentable. **Fix = (a) pass `formal_charge` to `class_index` (neutral atoms byte-identical), and
  (b) PROTECT charged atoms in the corruption** (`source_corruption.py` `_touches_charged`: no edit may
  change a charged atom's element/charge/H/bonds — a uniform successor-level guard, slot-stable states).
  Charges are preserved (verified 219/219), edits land on the neutral scaffold. This is a scope, not a
  limitation — charge/protonation edits are ~out-of-scope for (QED, similarity). **LESSON: the dev
  finite-loss smoke used small NEUTRAL molecules and never hit a charged restate target; always smoke on
  a charged, drug-like lead.**
- **Inference LOAD side wired (experiment blocker, fixed).** `load_factorized_rollout_checkpoint`
  (`evaluate_tracelet_rollouts.py`) reconstructed the model with default CNOF vocab + `enable_*` off, so a
  B-edit organic checkpoint CRASHED on load (15-wide vs 4-wide heads) and a CNOF-corrupted checkpoint
  silently sampled like de-novo B. Fix: persist `organic_vocabulary` in `checkpoint_metadata`
  (`train_tracelet_cnof_gate.py`) and, at load, set `atom_vocabulary=ORGANIC_VOCABULARY` +
  `enable_*=corrupted_prior_mix` (absent keys → CNOF + off, so de-novo B loads byte-identical). Verified:
  organic ckpt → vocab 15 + all flags on; de-novo ckpt → vocab 4 + off.
- **Ring open/close, corrected (I had it wrong first).** OPEN (`ring_system_delete`, clean) covers all
  sizes 3–7, N/O/S (heteroatoms PRESERVED), fused/spiro/bridged. CLOSE works: the MODEL SAMPLER proposes
  `ring_system_grow` on SATURATED chains for EVERY element (C/N/O/S/P, ~15–20%). **Do NOT test grow with
  `enumerate_ring_system_grows` — it returns 0 while the model samples grow 74× (it is NOT the model's
  grow enumerator).** The open form is Kekulé (unsaturated) but grow needs a SATURATED chain, so the
  round-trip is open → `bond_reorder`(saturate) → grow → `restate`(re-aromatize). Only genuine gap: S/P
  *aromatic* ring BUILDING (electronic model is CNOF; `factorized_tracelet_rate_model.py:1665` guard) —
  deferred, doesn't bite editing (leads' S/P rings are read/opened/restated/grafted, not built).
- **Hypervalent = no sink (verified).** S₂/S₄/S₆, P₃/P₅ all resolve their `(element,valence)` class and
  round-trip both directions (2/2) → every valence-change edit has a working inverse. `cycle_insert`/
  `cycle_attach`/`ring_ear_insert` are legacy null-prior ops, dead for B and B-edit (correct — ring adds
  go through `ring_system_grow`).

## 2026-07-27
- **B-edit corpus scope LOCKED = broad-organic + retained charged contexts, charge-PRESERVING (owner
  decision).** The de-novo loader `load_cnof_corpus_split` filters to C/N/O/F **and neutral** — only ~48%
  of GuacaMol (measured 237,945 of 500k). But `ORGANIC_VOCABULARY` has S/P/Cl/Br/I heads and the benchmark
  leads are majority S/Cl/charged, so a CNOF-only B-edit couldn't even *represent* them. New shared
  `src/compose_v4/data/organic_corpus.py` (`CorpusScope`/`BROAD_ORGANIC_V1`, hash `e59fb09801459470`)
  replaces the CNOF loader in EVERY production edit path (miner + trainer under `--organic-vocabulary`);
  membership = every atom's element in the ACTIVE vocab AND `class_index(elem,val,charge) is not None` (the
  same representability the teacher path uses — admits representable charges, no restated element list).
  See [[broad-organic-corpus-scope]].
- **Measured 500k broad census (Modal):** retained **490,466 / 500,000 = 98.1%** (2.05× the CNOF-neutral
  ~240k); S in 160,260 mols (33%), Cl 86,033 (18%), Br/P/I smaller; charge categories neutral 461,296 /
  **zwitterion-net-zero 23,870** / nonzero-net 5,300 (always report these THREE separately, never one
  "charged"). Benchmark coverage: broad accepts **800/800** Jin-QED leads vs **33%** CNOF-neutral (29% of
  leads are charged). The local 5k census proxy matched the 500k at 98.0% — trust it for quick estimates.
- **Charge is PRESERVED not optimized — net-charge invariance guard + a slot-stable measurement trap.**
  `source_corruption` now rejects any candidate whose successor changes the net formal charge (universal:
  also blocks a neutral molecule silently GAINING charge). The per-atom `_touches_charged` guard was
  INSUFFICIENT: deleting a *neutral* O of a delocalized carboxylate lets H re-derivation protonate the O⁻
  (`CC(=O)[O-]` → net −1→0). **TRAP:** states are SLOT-STABLE (a delete leaves a null mid-array), so
  `formal_charges[:n_real_atoms]` DROPS trailing real atoms and mis-measures net charge — always sum over
  the `is_element(atom_types)` mask. This bug first masqueraded as a corruption bug.
- **Cold-vocab audit (`scripts/cold_vocab_audit.py`): broad corruption supervises EVERY non-CNOF class.**
  A molecule merely *containing* sulfur does not warm the S output slots — measure positive edit TARGETS
  (`atom_insert`/`atom_restate` marks producing each `(element,valence)` class). Result: all 11 non-CNOF
  classes (S v2/4/6, P v3/5, Cl, Br, I v1/3/5, B) get positive targets (chiefly via `atom_restate` /
  heteroatom scan) → BEDIT_SUPERVISED, cold set empty. So B→B-edit head-widening + broad training warms
  the widened slots.
- **Full mining = map → global-group → PARALLEL-compile → reduce fan-out (owner: "just use 20 containers",
  NOT a cap).** Global grouping over the full 400k TRAIN partition yields **206,715 MMP one-cut pairs** (vs
  841 in a 20k shard — super-linear; global grouping >> per-shard). Compiling all serially in one container
  ≈ 6 h (0.107 s/pair). The Modal app (`mine_edit_traces_app.py`) is now `mine_pairs` (1 container:
  scan+map+group+corruption → writes `pairs.jsonl`) → `compile_shard` (N parallel, each a pair-stride) →
  `reduce_pool` (dedup+cap). `mine_mmp_pairs`/`compile_mmp`/`dedup_and_cap` are the reusable phase fns.
- **Two Modal-container gotchas that DISCARD a finished run at the very end:** (1) the debian_slim image has
  **no `git` binary** → the miner's `_provenance` `subprocess.run(["git",...])` raised `FileNotFoundError`
  AFTER all mining but BEFORE the summary/pool write, losing everything — catch `FileNotFoundError/OSError`,
  the launcher supplies the commit. (2) the `mine_shard` function prints nothing interior → add per-phase
  `print(..., flush=True)`; a black-box container hides both slowness and hangs. Corruption char is the
  serial long pole on Modal (~10× local; ~0.5 s/sample), so keep the validation-shard sample small.
- **Modal launch discipline that worked all session:** every launch from a CLEAN detached `git worktree`
  at the committed launch tag (never the dirty dev tree), gated by `prelaunch_gate.py`; a `_provenance`
  git call + `corpus_scope_hash` in every checkpoint/manifest so a cross-scope load fails loudly
  (`load_factorized_rollout_checkpoint(expected_scope_hash=...)`, `broad_preflight_gate.py`).
- **Compositional ring support RESOLVES the §9 "ring_system_grow fresh AND unsupervised" finding at the
  SUPPORT level (not by fixing the macro).** The redesign (owner-authorized, overrides "don't touch audited
  operator semantics"): ring generation SUPPORT = compositional `cycle_close`/`cycle_open` (add/remove a
  single non-bridge ring bond = `BondInsert`/`BondDelete`), which round-trip 400/400 and are densely
  supervised by `build_cycle_op_records` (real molecule → per-ring-bond open/close, both directions). The
  finite whole-ring catalog / `ring_system_grow` macro is ACCELERATION only and must NOT define support.
  Wired as `enable_cycle_ops` (default OFF → B byte-identical) overriding the dead `cycle_insert`/
  `cycle_attach` family slots 5/6, so `family_head` width stays 10 and the warm-start is strict-safe. The
  `--cycle-op-mix` chain is end-to-end: gate → Modal recipe (`build_tracelet_recipe_argv` bool → bare flag)
  → inference load (`enable_cycle_ops=bool(payload.get(...))`, absent → off → byte-identical). Subtype-gate
  run on 150 held-out broad-organic leads: cycle_close/cycle_open **588 targets each** + all 7 corruption
  families ≥44 → `GO_SUBTYPE_SUPERVISION`. See [[paper-master-plan]].
- **GOTCHA — a subtype-supervision audit MUST alias executor rule_names to their scoring family.** The
  cycle-op teacher marks are recorded under EXECUTOR names (`bond_insert`/`bond_delete`) but the dense head
  scores them under FAMILIES `cycle_insert`/`cycle_attach` (via `_CYCLE_OP_EXECUTOR_TO_FAMILY` in the rate
  model). Counting raw rule_names reports the cycle ops as unsupervised even when they are the most-supervised
  families — always pass `family_aliases=_CYCLE_OP_EXECUTOR_TO_FAMILY` to
  `operator_subtype_supervision_gate.check_operator_subtype_supervision`. The gate deliberately NO_GOs if
  `ring_system_grow` is enabled before its P4 K-macro supervision lands (proves it catches the macro gap,
  not silently passes — the exact false-positive `cold_vocab_audit` had).

## 2026-07-28
- **RingCore-V1 CPU dry-launch found a real EVAL-PATH bug: `sample_factorized_mark_batch` didn't enable the
  editing-family enumeration flags.** The eval/validation/test batch builder constructed
  `FactorizedMarkCollator(view, catalog)` with DEFAULT flags (`compute_ring_restates/opening/cyclic_graft=False`),
  while the TRAINING loader threads them from `model.enable_*`. So any `ring_system_restate` / `ring_system_delete`
  / `bond_reroute` teacher landed outside the eval batch's dynamic candidates → `RuntimeError: teacher ring
  restate is outside exact dynamic candidates` (factorized_tracelet_rate_model.py:4159) on the FIRST validation
  forward. Fails on NEUTRAL and CHARGED molecules alike — **charge is a RED HERRING** (a first `if not charged`
  guard was DISPROVEN: 5/13 neutral+charged fail without flags, 0/13 with). `cycle_close/cycle_open` never failed
  (per-coordinate scoring, no candidate list). **Fix = one shared representability contract:**
  `sample_factorized_mark_batch` gained the 4 `compute_*` flags; the gate passes them (from `corrupted_prior_mix`
  / `disable_ring_grow_macro`) at both validation+test sites; eval-batch-cache FORMAT_VERSION 1→2 (v1 caches built
  without flags are invalid). Teachers are DYNAMIC-only (stored MMP pool has 0 editing-ring teachers) → no pool
  regen. Regression `tests/test_teacher_in_candidates.py` reproduces the old failure + proves 0 mismatches across
  neutral/cation/anion/zwitterion/S/Cl/fused strata. **LESSON: any batch builder that scores editing teachers
  MUST enable the same editing-family enumeration flags as training — reaffirms the "teacher must be in the dense
  mask" rule, now for the EVAL path (train worked, eval didn't).**
- **Zero-mixture (`--scaled-manifest`) training path (RingCore-V1, no de-novo).** Under `--scaled-manifest`,
  `denovo_keep=0`, so the gate must NOT compile de-novo paths / open B's path cache / build a carbon-tree dataset.
  Coupled fixes: (a) Modal `train_stage` sets `require_path_cache=False` + `path_cache_source_run=None`; (b) gate
  skips `tree_source_prior` construction + the de-novo compile branch + reconstructs the ring catalog from the 5
  fixed production seeds (`_build_ring_core_seed_ring_catalog`, fingerprint `639ff6078c32d43c`); (c) skips the
  empty-partition guard (the de-novo partitions are intentionally empty); (d) builds edit VALIDATION from
  `split.validation` (disjoint from the `split.train` that sourced training edits) since the de-novo val partition
  is empty. `--dry-launch` = early exit from the REAL trainer after the first forward + one edit-validation forward
  (before backward/optimizer); CPU Modal fn `dry_launch_stage`; instrumentation `zero_mixture_instrumentation`
  (denovo_*=0, edit_*/forward>0, optimizer_steps=0). **`main()` uses `.spawn()` → MUST `modal run --detach`** or
  the app stop kills the spawned fn. Result → `/artifacts/<run-label>/dry_launch.json` (poll the volume).
- **The teacher-in-candidate invariant surfaced a SECOND, deeper corruption bug: inverse (GROW-direction)
  `ring_system_restate` on FUSED ring systems is not in the model's dense enumeration.** After fixing the
  eval-collator flags, the `assert_teachers_in_exact_candidates` safeguard fired on a complex fused molecule
  (`O=C1NC(O)C2CCCCC12`): the grow trace's inverse restate teacher wants a 3-bond re-aromatization of ONE
  6-ring of a fused system, but `enumerate_ring_system_restate_actions` on the saturated fused state offers a
  DIFFERENT 4-bond fused-system pattern — so the inverse teacher is outside the dense candidates (same
  inverse-mask-legality class as the 2026-07-25 "clean delete taught trim-only"). Only ~1-3% of traces, all
  GROW-direction restates on fused systems. **Fix = enforce the invariant at the DATA SOURCE:**
  `source_corruption.trace_teachers_representable(trace, system, catalog)` replays each trace and checks every
  ring_system_restate / ring_system_delete / bond_reroute step's action IS in its dense enumeration;
  `build_corrupted_prior_records` DROPS any trace that fails (logs `dropped_unrepresentable_traces`). Guarantees
  `teacher ∈ A_exact(x)` by construction. Verified: 154 ring/graft teachers, 0 mismatches; a 256-example
  validation batch on fused/bridged/charged leads scores finite. **LESSON: generating a trace via forward
  enumeration does NOT guarantee its INVERSE steps are in the dense mask — validate BOTH directions against the
  model's exact candidates before recording (construct the teacher from the representable set, per the owner's
  §3).** Micro families (atom_*/bond_reorder) + cycle ops are per-coordinate (mask-derived) so they need no
  list check.

## 2026-07-29
- **The corpus-load bottleneck is DOUBLE executor replay, not validation.** Loading the precompiled edit
  corpus cost ~21 ms/corruption-trace. `decode_trace_record` replays every step through the executor
  **unconditionally** -- `validate=False` gates only the canonical-key COMPARISONS, not the replay, because
  the final state is needed as `target` (measured speedup from `validate=False`: **1.1x**, not the large win
  assumed). `TraceProgressCTMC.__init__` then replays a SECOND time. Each `apply()` runs `is_valid_state` ->
  `is_rdkit_valid` -> `molecular_graph_to_smiles`: ~32 RDKit SMILES round-trips per record, per replay,
  re-proving validity the build already certified. **Fix = `packed_trace_store`**: materialize all K+1 states
  offline; `decode_packed_trace` rebuilds the `RewriteTrace` dataclass with ZERO executor calls (source =
  states[0], target = states[-1], steps = pure `decode_action`), and `PackedTraceProgress` subclasses
  `TraceProgressCTMC` overriding ONLY the replaying ctor so every other method is *inherited, not
  reimplemented*. Measured end-to-end on a real shard: **21.36 -> 0.217 ms/trace (98x)**, 7854/7854 states
  exact, full train partition 3.9 h -> 2.4 min, store 0.13 GB. Trade: executor drift is no longer caught by
  re-execution at load -- it is caught by refusing a provenance (`operator_registry_hash`/
  `codec_implementation_hash`) mismatch, plus an on-demand `verify_fraction` replay audit (2% = 0.47 ms/trace).
- **LESSON -- micro-benchmark a COMPONENT, measure the PATH.** Timing `PackedMolecularGraph.from_graph/unpack`
  in isolation predicted 630x; the first true end-to-end gave **2x**, because `read_packed_shard` still called
  `decode_trace_record`, which still replayed. Only after removing that second replay did it reach 98x. A
  component speedup is not a path speedup until the whole path is timed.
- **LESSON -- an equivalence test whose reference is a COPY cannot fail usefully.** The first packed-sampler
  panel compared against a local transcription of `_sample_tracelet_progress`; if the production function
  drifted, the panel stayed green. `tests/test_packed_vs_live_sampler.py` now drives the LIVE function over
  real `TraceProgressCTMC` records, and pins `PowerSurvivalScheduler.power == 1` (the packed path is
  parameterized by `alpha` and treats `alpha == t`; a non-unit power silently breaks that identity).
- **Sampling unit (measured, `scripts/sampling_unit_audit.py`):** one draw = one TRACE, then exactly ONE
  progress position -- not all its transitions. So a K-step trace gives each of its steps ~1/K of the trace's
  probability, and flattening transitions into a uniform urn would upweight long traces by ~K (a DIFFERENT
  objective). `path_length` is the only trace property the progress law needs: `N_t ~ Binomial(K, alpha(t))`.
  The terminal (no-jump) position is a legal landing and takes **48.9% of all draws** (cycle 61.0% since K=1
  always, corruption 47.0%, MMP 42.5% exact over the full 363,456-row pool) -- it supervises the exit rate.
  Family stratification (fraction 0.5) oversamples rare families then divides it back out via the importance
  weight, so it buys COVERAGE, not expected family mass. Consequence: a layer weight is a weight over trace
  draws, so configured 0.40/0.25/0.35 realizes as **41.5/19.1/39.4%** of teacher transitions.
- **The MMP analogue pool bypasses the ringcore-v1 scaffold partition** (mined under `split_seed 20260714`).
  Measured: **8.09%** of its rows fall in the held-out partitions (~29k records) -- they would train on
  validation/test scaffolds and inflate the very learning curve used to select checkpoints.
  `production_edit_corpus.load_mmp_records` routes every row through the same pure `partition_for_scaffold`,
  and applies any `limit` AFTER filtering so it cannot regress into partition-blind prefix truncation.
  GOTCHA: an empty SMILES **parses** to a zero-atom Mol, so it gets a partition bucket unless emptiness is
  rejected explicitly -- `murcko_scaffold`'s None contract only covers UNPARSEABLE input.

## 2026-07-29 (packed corpus + MMP partitioning)
- **A PAIR artifact cannot be partitioned by one endpoint.** The MMP packed build FAILED its reducer with
  `1325 sources span partitions`. Cause: partitioning used `murcko_scaffold(target_smiles)` alone, but an
  MMP one-cut pair has TWO endpoints and one source is paired with up to 8 targets
  (`max_pairs_per_source=8`) whose scaffolds can fall in different partitions -- so the same molecule
  appeared in train AND validation. That is molecule-level contamination of the very held-out set used for
  checkpoint selection, and it is invisible to any count-based census. **Fix = require BOTH endpoints to
  map to the same partition** (`pair_partition()`); drop the straddlers, counted as
  `endpoints_straddle_partitions`. Measured cost: **0.56% of pairs (~2,044 of 363,456)**; 99.44% already
  agreed, and 0/15,052 molecules mapped to >1 partition by their own scaffold (the partitioner is sound).
  Distinguish this from the earlier 8.09% pool leak: that was a MISSING filter, this was a filter on the
  WRONG KEY.
- **A reuse cache must key on the RULE, not just the inputs.** Partition reuse keyed on the pool content
  hash + `partitioner_provenance()`. Neither changes when the APP's own pairing rule changes, so a rebuild
  would have silently reused the leaking v1 partitioning. Added `_MMP_PARTITION_RULE_VERSION` to the reuse
  key and the manifest. Any derived-artifact cache needs the derivation logic's version in its key.
- **Two schemas in one store is a divergence path.** The MMP pool is V1 (`steps[].rule`), the packed loader
  decodes V2 (`steps[].action` via the codec), so storing raw pool rows would have failed at read time on
  every shard. Normalize V1->V2 ONCE at pack time (`encode_trace_record`, pool identity carried in `extra`)
  so the loader has one path. The pre-existing round-trip test had passed only because it converted to V2
  first -- a fixture shaped to the CODE rather than to the ARTIFACT.
- **Sidecar manifest naming: `.jsonl.gz` -> `.jsonl.manifest.json`** via `Path.with_suffix`, NOT string
  concatenation (which yields `.jsonl.gz.manifest.json` and finds nothing). This bug had TWO homes -- the
  library helper and a duplicate copy inside the Modal reducer -- so fixing only the library left the
  reducer reporting all 45 packed shards missing. One `manifest_path_for()` helper now, plus a test that
  scans `src/` and `modal_apps/` and FAILS if any module rebuilds the path by concatenation. NB chained
  `with_suffix` also fails when reversing (`shard.jsonl.manifest.json` -> `shard.jsonl.jsonl.gz`).
- **A clean-worktree gate can PASS while testing less than it claims.** The authoritative launch gate ran
  743 passed + **29 SKIPPED**, because the new tests' fixture (`diagnostics/composition/analogue_trace_pool
  .jsonl`) was UNTRACKED and therefore absent from any clean checkout. Committed a 40-record real-molecule
  fixture (`tests/fixtures/`, 47 KB, K 2-16) -> 0 skips. Always read the gate's SKIP count, not just PASS.
- **Micro-benchmarks lie about paths.** See the 630x -> 2x -> 98x sequence in the entry above; and here the
  packed-store projection ("2.4 min full train load") covered only the two packed layers while MMP still
  replayed at 13.74 ms/row -> 83 min PER PARTITION, rescanned per partition. Always state which components
  a projection covers.
