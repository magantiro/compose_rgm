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

## 2026-07-29
- **A preempted Modal container restarts with the SAME input, so a warm-started run carries its
  initialization flag into the retry -- and that collides with resume.** The first RingCore-V1 scientific
  16k run died this way at step 1500: `Container terminated due to preemption`, then the launcher added
  `--resume-checkpoint` (a recovery checkpoint now existed) on top of the still-present
  `--initialize-compatible-checkpoint`, and the gate rejected the pair as mutually exclusive
  (`train_tracelet_cnof_gate.py:1687`). Two retries died identically, then the app stopped. The recovery
  checkpoint on the volume was VALID the whole time -- the run was killed by argument parsing, not by data,
  model, or chemistry. **Fix:** `_supersede_initialization_for_resume` drops
  `initialize_compatible_checkpoint`/`initialize_checkpoint`/`load_checkpoint` when pointing at the
  recovery state, which is also the semantically correct resolution (the recovery state already holds the
  warm-started weights PLUS every completed step, so re-initializing would DISCARD progress). The
  regression test extracts the excluded-argument set from the GATE'S OWN source rather than duplicating a
  flag list, so adding a future checkpoint flag cannot silently reintroduce the collision.
- **`retries=modal.Retries(max_retries=1)` cannot carry a multi-hour GPU run.** Preemption is an expected
  event, not a code failure; one preemption exhausted a budget of 1. Raised to 5 for `train_stage`. This is
  safe here precisely BECAUSE each retry resumes from the recovery checkpoint (`recovery_every=500`), so
  retries make forward progress instead of repeating work -- a retry budget without a working resume path
  just burns the same steps repeatedly.
- **Changing the launcher necessarily changes the run identity -- you cannot fix the launcher and resume the
  same run.** `_source_fingerprint()` hashes `modal_apps/train_tracelet_gm.py` ITSELF (line 211) alongside
  `src/`, `scripts/`, `recipes/`, and `run_identity` includes that fingerprint; the immutable stage manifest
  then refuses a run label whose identity changed. So a launcher bugfix forces a FRESH run label. That is
  the guard working as designed (it stops silent mid-run code changes), not an obstacle to route around.
  Corollary: `configs/`, `diagnostics/`, `tests/`, and `.claude/` are NOT in the fingerprint, so committing
  there does not disturb a pinned launch worktree.
- **To reproduce a launch exactly, diff the recipe FILE against the dead run's
  `manifest.training.json` -> `run_identity.recipe.arguments`.** Every ADDED or CHANGED argument came from a
  CLI flag or a launcher-computed path, which recovers the exact flag list even when the launch script is
  lost or outdated. This caught that `early_stopping_patience 6 -> 0` came from `--disable-early-stopping`
  (so early stopping was DISABLED and the rising loss could not have aborted the run) and that
  `steps 30000 -> 16000` came from `--training-steps`.
- **`nohup cmd &` inside a tool call reports the WRAPPER's exit, not the job's** -- the harness said "exit
  code 0" seconds after launching a 10-minute gate whose log was still 0 bytes. Wait on the pid (or use a
  real background task) before believing a gate result, and never read an empty log as success.
- **The prelaunch gate needs `ruff` on PATH, which a fresh `git worktree` does not have** (ruff lives in the
  main tree's `.venv/bin`, and the gate shells out to a bare `ruff`). A launch worktree therefore needs
  `PATH="<main-repo>/.venv/bin:$PATH"`, or the gate dies with `FileNotFoundError: 'ruff'` AFTER the tests.

## 2026-08-02
- **A content-addressed identity move has a transitive blast radius; re-pinning "the configs that
  pin it" is not the same as re-pinning the chain.** Process V2 edits
  `factorized_tracelet_rate_model.py`, which `editing_v2_process_identity()` hashes, so the V1
  process identity moved (`6b98ee21…` -> `9874a69a…`). Two configs pin that value directly, but they
  are themselves content-addressed by others: cells registry -> cell-role policy -> {gate-0
  structural, T1 panel policy} -> T1 capacity policy -> {P50 recipe policy, T1 decision, training
  gate}. The full set is **nine artifacts plus four source constants**. Re-pinning only the first
  layer left 41 tests failing. **Method that works: sweep for every hash value your edits obsoleted**
  (old physical hash of each changed file, plus every 64-hex literal present in its old text and
  absent from its new text) and grep the whole tree until the sweep returns nothing; iterate, because
  fixing one layer changes the next layer's input. Re-pin only hash pointers, and assert
  programmatically that no policy, threshold, count, or cell definition moved.
- **A stale binding that VALIDATES is worse than one that raises, and both existed here.** The P50
  chain compares its pinned cell-role hash against `load_semantic_development_cell_roles()` (live) and
  failed loudly. The T1 chain (`editing_v2_semantic_t1_capacity_policy.py:112`) compares the config
  against a **source constant**, so config and constant were both stale, agreed with each other, and
  the policy loaded clean while binding a cell-role policy that no longer existed. Prefer live-loader
  comparison over a source constant for any cross-artifact binding; a constant-vs-constant check is
  self-consistent by construction.
- **Precedent is checkable: `git log -p` the artifact before deciding whether re-pinning is
  maintenance or gate-relaxation.** `d5d3016` (an analogous process change) re-pinned exactly the
  gate-0 model/process contract, capability-cell registry, Active8 decision runtime, gate-0 structural
  contract, and the `FROZEN_CONTRACT_SHA256` source constant, and did NOT touch P50. That settled a
  decision I had gone back and forth on twice. Re-pinning a non-authorizing binding contract is not a
  Gate-0 run: measured evidence produced under the superseded identity stays invalid either way.
- **Mutation-test the suite, not just the code.** Three production mutations survived the entire
  focused suite: bypassing `is_valid_atom_delete`, bypassing the declared-support check, and zeroing
  the V1 dense delete mask. The first two survived because no fixture reaches them on drug-like
  chemistry (over 3,617 candidate slots those codes fired 0 times); the third because both sides of
  the comparison were built from `_graph_application_masks`, so the expectation moved with the
  observation. **A comparison whose expectation is recomputed from the code under test cannot fail.**
  Reachable witnesses: `C1CC[SH4]CC1` slot 2 or 4 (valid source, connected + canonicalizable +
  in-support successor `CCCC[SH5]`, executor is the only refusal) and a 42-membered carbocycle
  (successor 41 atoms > `MAX_ACTIVE_ATOMS`).
- **Micro-benchmark vs path, again (cf. the 630x -> 2x -> 98x entry).** The V2 delete mask is ~12x the
  isolated `_graph_application_masks` call, but only **+0.29% to +0.35%** of the real
  `prepare_factorized_mark_batch` path, which is dominated by `_semantic_cycle_close_admission_mask`
  (54.8%) and `_semantic_atom_restate_admission_mask` (36.7%). A measured 2.52x optimization of the
  new mask was therefore **not** applied: profiling said it buys 0.2% of a batch. Separately measured
  and worth fixing someday: `enumerate_ring_restate_semantic_groups` sits outside the
  `if features is None:` branch, so it recomputes on every cache hit (86 ms/state, 13x the entire V2
  mask).
- **Non-articulation already implies successor connectivity.** `is_valid_atom_delete` is not a
  connectivity predicate (it returns True for cut-vertex deletions), which is true and load-bearing,
  but it does not make the explicit successor-connectivity check independent: removing a non-cut
  vertex from a connected graph leaves it connected. Measured over 3,617 candidates, only aromaticity,
  articulation, and the charge policy (7.6%) ever rejected anything. Corrected the claim rather than
  the code; the check stays as defence in depth.

## 2026-08-02 (Process-V2 correction round; supersedes parts of the entry above)
- **"Preserve the existing capability" is not "preserve the existing admission SET".** The first
  Process-V2 reading kept root/singleton/leaf deletions admitted exactly as the legacy dense rule had
  them, i.e. exempt from the authoritative charge policy, and expressed V2 as the union
  `V1_dense | connected_nonleaf`. That preserves a legacy DEFECT: for `C[N+](C)(C)CC(=O)[O-]` the
  legacy mask admits slots `[0,2,3,6,7]` and the production runtime refuses **4 of those 5** with
  `InvalidRewrite`, so the model was learning a rate for transitions the executor cannot perform.
  Measured breadth: 125 / 3,319 candidate slots (3.77%) across 105 / 800 leads. The correction is ONE
  admission authority with uniform gates for both candidate sources; the two sources survive only as
  diagnostic labels selecting which ADDITIONAL gates apply. Reviewer's framing, worth keeping: the
  evidence was already in hand and had been used to JUSTIFY the exemption rather than to refute it.
- **A union is the wrong shape for an expansion whose base is untrusted.** The give-away is that the
  forward guard could only assert a SUBSET relation between the scored mask and the admission mask. It
  now asserts EQUALITY, which is what "the learnable fiber is exactly the admitted fiber" means, and
  the teacher test asserts both directions: an admitted deletion scores finite, and a charge-violating
  inherited deletion scores `-inf` under V2.
- **The round-one re-pin method recorded above is WRONG and produced a silent stale pin.** "Sweep the
  obsoleted values and grep the tree" skipped files already touched in the same pass, which left
  `configs/editing_training_v2_gate.json` and `editing_training_gate.py` pinning a dead T1 capacity
  policy hash. It VALIDATED (constant vs constant) and only failed 19 tests later. Replacement that
  works: a fixed-point driver over the verifier's OWN structural pin discovery, iterated until a round
  writes nothing. Never exclude an already-touched file; exclude only the self-occurrence.
- **Three ways an automated re-pin corrupts an artifact, all found by tests the chain verifier passed.**
  (1) `process_identity_sha256` is a top-level `*_sha256` field that is NOT a self-hash, so
  recomputing "every `*_sha256`" destroys the pin -- identify the self-hash field on the PRE-mutation
  payload, where the equation still holds. (2) `*file_sha256` means the target's PHYSICAL bytes hash
  and `*semantic_sha256` means its SELF-hash; falling back to whichever role exists writes the
  self-hash into `file_sha256` and every loader that checks both rejects it. (3) A value produced and
  superseded inside one run was never committed, so `git show` cannot see it and the pin that held it
  addresses nothing. **Chain-verifies-green is necessary, not sufficient: run the focused suites.**
- **Do not re-serialize a config to re-pin it.** 16 of 66 configs here are stored in canonical
  single-line form; a pretty re-dump rewrites every byte and moves their physical hashes for nothing.
  Substitute `"<key>": "<value>"` textually, allowing the parenthesised continuation ruff's 100-column
  limit forces on a 64-character literal in Python.
- **A pin does not stop being a pin because it is written in Python, and an unresolvable pin is the
  WORST case, not a benign one.** The chain verifier scanned only JSON for pointer edges, so a
  module-level dict constant of the same path-plus-sibling-hash shape was invisible; and it derived
  chain membership from values matching a live or base index, so an artifact whose pin matched
  NOTHING fell out of the chain and had its finding downgraded from failure to warning. Both are now
  closed (`_python_pointer_edges`, plus edge-derived membership) with
  `tests/test_verify_process_v2_hash_chain.py` carrying a built-in negative control.
- **A builder that cannot construct the new process is an incomplete implementation, not a detail.**
  `editing_gate_zero_semantic_contract.py` called `semantic_model_identity()` unconditionally, so no
  Gate-0-shaped path could build a Process-V2 model at all, while every focused test still passed.
  Verified by construction, not by inspection: `build_semantic_scratch_runtime` builds V1 at
  capability fingerprint `d246bc88d8440d31` and V2 at `d79ffe8ef65f3fb3`.
- **`build_production_ringcore_catalog` drifts on this Mac** (`82fd910cafe2eeb7` vs the frozen
  `639ff6078c32d43c`), on the branch AND on its base with the catalog inputs untouched -- an
  environment fact, not a code defect. Neutralize that ONE constant when proving a builder path, and
  say so; do not let it stand in for a real failure or hide one.

## 2026-08-03
- **A file-inventory constant can be tautologically "verified" the same way a mask can.** The
  2026-08-02 lesson ("a comparison whose expectation is recomputed from the code under test cannot
  fail") recurred one level up, on the LIST of files rather than on a computed value.
  `CACHE_IMPLEMENTATION_FILES` names the modules hashed into `cache_implementation_sha256`, and its
  guard asserted `set(revision["implementation_files"]) == set(CACHE_IMPLEMENTATION_FILES)` -- but
  `build_cache_implementation_revision` produces that keyset FROM the constant, so the two sides move
  together. Proof of how empty it was: deleting `editing_v2_process_v2_chunk_cache.py` from its OWN
  file list left all 217 tests green. **Measured escape:** two trees differing only in
  `rewrite/progress.py`, reading the same committed cache, produced the identical
  `cache_implementation_sha256` and DIFFERENT decoded rows. The list covered what the cache *stores*,
  never what it *reads*, while the docstring claimed "stores, reads or validates".
- **Derive a closure by EXECUTION, not by reading imports.** The fix unions three sources: every
  `compose_v4` module whose code runs under `sys.settrace` during a real decode; every module defining
  a type (MRO included) reachable in the decoded object graph; and every module defining a payload
  class the frozen codec surface can construct (`action_codec_v4.supported_executor_rules()` ->
  ontology -> `type.__module__`), which is what catches a family the fixture traces never exercise.
  A hand-written list was wrong in BOTH directions: it named `action_codec_v2.py` (does not exist --
  `action_codec.py` IS v2) and `provenance_overlay.py` (imported by `packed_trace_store` only for the
  raw addressed-shard readers the chunk cache never calls), and it MISSED `molecular_graph`, `state`,
  `operators`, `tracelets` and `editing_v2_process_identity`. 9 -> 18 modules.
- **The test that replaces a tautology must not read the constant at all.** The new guard loops over
  the DERIVED closure, edits each module on a tree copy, and requires `cache_implementation_sha256` to
  move. A module the revision does not hash cannot have that property, and shortening the constant
  fails the loop -- because the loop never consults it.
- **Parallelism is free during implementation and is NOT free during measurement.** A full suite run
  concurrently with two agent suites took 25:33 against 17:27 and manufactured a phantom regression in
  a test with a 1.0 s timeout; the same code on an unloaded machine ran 19:23 with the failure gone.
  Distinguish it from a real regression by import closure, not by re-running: the failing test's
  33-module closure had ZERO overlap with the files the branch changed, which is structural proof.
  This is the LOAD half of the hazard whose WRITE half is the 2026-08-02 P50 working-tree race.

## 2026-08-06 (P50 compile cost: measured, and where it actually goes)

- **The teacher-fiber compile costs ~2.12 s/entry on real drug-like leads at 40 slots** (measured,
  `compile_prepared_entries`, n=24 Jin leads). Full train corpus of 1,803,032 transitions = **13.3 h on
  80 CPU workers**. Stored form is **2,503 B/entry** (independently confirmed against a real production
  leaf at 2,585 B/entry), so full-corpus leaves are ~4.5 GiB; the COLLATED tensors are 92 KiB/entry
  (~158 GiB) because collation pads to fixed shapes.
- **Where the time goes (cProfile, 8 entries):** `prepare_factorized_mark_batch` is 88% of it, and inside
  it `_semantic_cycle_close_admission_mask` (1.48 s/state) plus `_semantic_atom_restate_admission_mask`
  (1.06 s/state) are ~90%. The leaf cost is **55,210 `molecular_graph_to_smiles` calls for 8 entries --
  ~6,900 RDKit round-trips per entry.** RDKit sanitization is the authoritative validity gate, is C++
  per-molecule, and does not vectorise.
- **THE BIG ONE -- the expensive chemistry is computed TWICE on the same states.** The compile calls
  `prepare_factorized_mark_batch` (via `enumerate_factorized_legal_support_many`), extracts the teacher
  and self-event aliases, and **discards the masks**; then `collate_process_v2_p50_entries` calls
  `FactorizedMarkCollator`, which calls `prepare_factorized_mark_batch` **again** on the same states to
  build the tensors. Fusing the two into one pass per state is ~2x on the whole prep.
- **Do NOT "store prepared, collate live".** Collation is not chemistry-free despite its docstring: it
  runs the admission masks. Collating per batch would re-run ~13 h of chemistry PER EPOCH. Store the
  collated tensors sharded (~600 MiB/shard over 270 chunks) so training is chemistry-free.
- **`resolve_semantic_cycle_close` re-does source work per candidate.** `is_valid_state(state)`,
  `canonical_state_key(state)` and `_exact_state_identity(state)` are recomputed for each of ~698
  candidates on an unchanging source, even though the hoisted `context` already carries `source_key` and
  `exact_source_identity`. Profile arithmetic puts this at ~0.69 ms x 698 = ~482 ms of the 1.48 s
  cycle-close mask (INFERRED, not patched).
- **`included_families` exists and is computed by the caller, but is never passed to
  `prepare_factorized_mark_batch`** (`score_free_successor_support.py`: `requested` is used only in the
  decode loop at line 146). So the family restriction narrows decoding, not the dominant mask work.
- **Levers that did NOT pay (all measured, so do not re-try):** size-bucketing 40->26 slots = 13%;
  `chemistry_feature_cache` = **8.8% ceiling** (13,732 of 14,951 source states in a real chunk appear
  exactly once); vectorising the atom/bond loops in `molecular_graph_to_smiles` = **1.24x** on that
  function (120/120 identical output) = ~7.5% of compile; reusing Active8's work = **does not apply**,
  its `candidate_evidence` carries only `teacher_coordinate_legal` /
  `teacher_executes_to_exact_successor` and the canonical keys, not a mark census or alias coordinates.
- **METHOD WARNING, paid for twice today.** A standalone microbenchmark of the bond scan measured it at
  **105% of the enclosing function** -- impossible -- because the real loop short-circuits on null atoms
  (~300 cells) while the benchmark walked all 780. Always time the real function head-to-head with an
  equivalence check, never a transcription of its inner loop.

## 2026-08-06 (corpus shape + the cache measurement I got wrong)

- **CORRECTION to the 8.8% chemistry-cache ceiling recorded above.** That was measured on a
  `controller_validation` chunk (91% distinct sources) -- a SEALED role I opened without checking the
  role first. On a real TRAIN chunk (`real_endpoint_multistep_path`) source states are only **49.6%
  unique: 6,826 distinct of 13,774 transitions, mean 2.02 transitions per source.** So a within-chunk
  chemistry cache can avoid roughly **half** the compiles on train, not 8.8%. The 2.02 is almost
  certainly because both directions are taught, so every intermediate state is the source of one forward
  and one inverse transition. LESSON: check the partition role before measuring anything, and never let
  a number from one role stand in for another.
- **The P50 selection can never exploit that reuse.** It selects 3,200 entries spread across 275 tasks,
  ~12 per task, so repeated sources almost never land in the same call. A chunk-level compile holding one
  cache across all ~13,774 entries does. This is an argument for the chunk unit independent of the
  compile/collate fusion, and the two stack.
- **Corpus shape (from `PROCESS_V2_ACTIVE8_PLAN.json`, 328 planned tasks):** train 270,
  controller_validation 22, final_test 20, validation 16. Train data lanes:
  `reversible_synthetic_walk` **130**, `real_endpoint_multistep_path` 106,
  `linker_positional_topology_analogue` 12, `observed_local_analogue` 11,
  `operator_aware_real_endpoint` 11. So **48% of train chunks are synthetic corruption walks** -- the mix
  the master plan asks for does exist, but corruption is the single largest lane and that is worth
  stating before any training claim about learned rates.
- **Chunks are lane-homogeneous.** Both chunks inspected were ~50/50 `atom_insert`/`atom_delete` and
  contained no other family, so a single chunk is NOT a miniature of the corpus. Do not infer
  corpus-level family balance from one chunk.
- **The production row->entry mapping is exact on real data.** `_candidate_from_transition` mapped
  13,774/13,774 real train rows with zero refusals and zero identity collisions, and 16,385/16,385 on the
  other chunk. Every published row satisfies the frozen evidence contract.

## 2026-08-06 (measured on Modal: the real prep cost, and a projection I got wrong)

- **MEASURED on the real path, one train chunk, 256 entries:** `ms_per_entry` **2046.09**,
  `elapsed_seconds` 523.8, `chemistry_states_cached` **149 of 256** (58.2% distinct). Full train corpus
  of 1,803,032 transitions is therefore **~1,025 core-hours: 12.8 h on 80 workers, 3.4 h on 300,
  ~$48** either way since Modal bills core-seconds. Wall clock past this point is a worker-count
  choice, not an optimisation problem.
- **The compile/collate fusion is confirmed remotely.** 2,046 ms/entry against a local FUSED benchmark
  of 2,148 and an unfused 3,909 -- about 1.9x, consistent with the 1.82x measured locally with
  byte-identical output.
- **CORRECTION: within-chunk source reuse is NOT a second multiplier.** I projected ~7 h by stacking it
  on top of the fusion. The pilot's 2,046 ms/entry already contains 42% cache hits and they bought only
  ~5%, not 2x. The cache saves the ADMISSION MASKS; the alias search, executor calls and encoding are
  per-ENTRY not per-STATE and do not cache. Do not multiply a per-state saving by a per-entry rate.
- **O(n^2) I/O caught before it scaled.** `chunk_transition_rows` first streamed the whole role and
  filtered by task, so every worker in a per-chunk fan-out would read all 270 train shards (~600 MB
  each). Worse, that read sat INSIDE the region the pilot timed, so it would have corrupted the
  measurement as well as the runtime. `iter_process_v2_role_shards` now takes an optional
  `task_identity_sha256` so the existing digest/count/row authentication stays the single authority for
  whatever is opened.
- **The volume Gate-0 decision was on v5-era contracts while the repo is on v6**, so NO stage could open
  the Active8 source -- T1, P50 and the training prep alike. Re-running Gate 0 took **~2 minutes** and
  reproduced `decision_sha256` **613259c4...** byte-identically to the repo fixture, which is a clean
  demonstration that the content-addressed chain is deterministic. Check
  `load_gate_zero_contracts(repo_root=...)` against a decision BEFORE concluding a branch caused drift;
  here it had not.

## 2026-08-06 (lane -> family map: ALL ring editing is synthetic)

- **Scanned all 270 train `TASK_SUMMARY.json` (metadata only, no molecular content).** Lanes are nearly
  family-PURE, and chunks are lane-homogeneous, so a contiguous "slice of the corpus" is NOT a miniature
  of it:
  - `real_endpoint_multistep_path` (106 chunks, 1.36M transitions = 75% of train): **insert/delete ONLY**
  - `reversible_synthetic_walk` (130): the ONLY lane with all 8 families
  - `linker_positional_topology_analogue` (12): **bond_reroute only**, 23,349
  - `operator_aware_real_endpoint` (11): atom_restate 19,615 + bond_reorder 919 only
  - `observed_local_analogue` (11): insert/delete only
- **EVERY ring-topology transition in the train corpus comes from the SYNTHETIC corruption lane.**
  `cycle_insert` 122,183 in 102 chunks, `cycle_attach` 84,940 in 102, `ring_system_restate` 16,380 in 29 (CORRECTED 2026-08-07: I first wrote 17,067; the frozen Gate-0 decision says 16,380, and the eight family counts sum exactly to 1,803,032. Take family counts from the decision, not from a re-scan)
  -- all `reversible_synthetic_walk`, zero in any real-chemistry lane. Experiment D claims topological
  adaptation; on this corpus it would be trained entirely on corrupted-molecule walks. That is a
  claim-scoping decision to make deliberately, not to discover in review.
  `bond_reroute` and `atom_restate` each appear in two lanes, so they at least have a real-chemistry
  source; the ring operators do not.
- **Consequence for any subset:** family coverage requires `reversible_synthetic_walk`. Selecting chunks
  at random, or taking a contiguous fifth, would very likely yield insert/delete only and tell you
  nothing about whether graft, restate or the ring ops can learn.

## 2026-08-08 (rare-family regression: measured, and the first hypothesis falsified)

- **The 16,000-step failure reproduces at step 400 and the capability gate catches it.** Production
  6.2M model (`hidden_dim=256`, `message_passing_steps=6`), 54,043-entry set, balanced 32-per-family
  panel, loose 1.5-nat ceiling. Aggregate loss improved 39% (6.516 -> 3.986) while
  `ring_system_restate` regressed 1.509 nats. A run watching only aggregate loss calls this healthy.
- **Exactly 2 of 8 families regressed, and they are the two rarest.** Dose-response with data share:
  `ring_system_restate` 1.2% (+1.509), `bond_reorder` 2.2% (+1.433), `bond_reroute` 7.1% (-0.038,
  the break-even point), everything above 10% improved. `cycle_insert` is structurally similar to
  ring and improved MOST (-3.121) on 16x the data, so the operator is not intrinsically unlearnable.
- **HYPOTHESIS FALSIFIED: it is not trunk drift.** Resuming the step-400 checkpoint with
  `heads_plus_local_adapter` (80% of the model frozen, 1,210,866 trainable) left ring and
  bond_reorder degrading at the SAME rate (+0.568, +0.569). Freezing the trunk changed nothing for
  them while the abundant families improved further.
- **What the falsification narrowed it to.** With the trunk frozen the only shared trainable surface
  is `pair_project.`, contested by exactly four families, and phase-2 deltas are monotonic in their
  data share: cycle_insert 18.2% -1.449, cycle_attach 12.6% -0.177, bond_reorder 2.2% +0.569,
  ring_system_restate 1.2% +0.568. Next test is `heads_only`, which freezes `pair_project.`
- **Do not cite the T1 `heads_plus_local_adapter` pass as evidence a family trains.** That 0.947
  minimum teacher probability is a MICRO-OVERFIT probe -- memorising a 64-example panel. It
  establishes a mechanism worth testing, not an outcome. I quoted it as a training result and the
  phase-2 run contradicted it within an hour.
- **AdamW state cannot cross a parameter-scope change** (119 trainable tensors vs 33), so a phase
  transition needs `resume_weights_only`: restore weights, RNG and step, start fresh moments. It is
  opt-in so it can never happen silently -- a quiet fresh optimizer looks like a resume and behaves
  like a restart.

## 2026-08-08 (the regression was the METRIC, not the model)

- **`selected_productive_successor_log_probability` is a JOINT probability and must not be
  used as a capability gate.** It folds "how often does this family occur" together with "can the
  model execute it". A model that correctly learns ring edits are 1.2% of the data necessarily
  scores ring examples worse on it, which is indistinguishable from forgetting the skill.
  Decomposed at step 400: `ring_system_restate` JOINT +1.509 = WITHIN -0.063 (capability improved)
  plus FAMILY-PRIOR +1.571 (calibration). Same for `bond_reorder`: WITHIN -0.006.
  Use `selected_within_teacher_family_log_probability` -- and it is the right target anyway,
  because the controller picks the family at inference, so the base model's family prior is
  exactly what gets overridden.
- **A 32-example panel has a standard error of +-0.03 to +-0.27 nats per family.** `atom_delete`
  +0.259 sits at 1.0 sigma; `cycle_attach` +0.062 likewise. Both were read as regressions before
  the noise floor was measured. Compute SE before calling any per-family delta a finding.
- **Judge a family by its margin over uniform-within-family, not by its delta.** All eight sit
  2.66-6.67 nats better than guessing over their own legal successors. The three "flat" families
  were flat because they started near the floor (`atom_delete` 1.61, `bond_reorder` 1.92); the
  families that improved most started worst (`atom_insert` 5.79->3.62). `ring_system_restate` at
  0.227 (+6.67) is the BEST family in the model -- the one that looked like it was collapsing.
- **The panel was 100% inside the training set.** Every number above is fit, not generalisation.
  A random holdout of the train role does not fix it either: 29.3% of train entries share a source
  molecule with another entry, so the same molecule lands on both sides. The corpus `validation`
  role is the only clean split; sealing means the GATE does not read it, not that training may not
  validate against it. Verified disjoint five ways including "val source also a train TARGET".
- **All six hard families are exhausted at 94-100% of what the corpus holds.** Only
  `atom_insert`/`atom_delete` have data left, and `atom_insert` is the weakest family, so further
  compilation helps where the model is worst. The resulting dilution is a TRAINING-time concern:
  compiling is one-time and irreversible, sampling is free and per-run.
- **6 compile workers is the measured optimum on a 12-core machine** (589% CPU, 0 swapouts/s)
  against 8 (536%, 1,409) and 10 (6,339, degrading). Each worker holds its own chemistry cache, so
  the constraint is memory. Oversubscription took a build from 5,507 entries/h to 250.
- **Top-1 teacher-match is the WRONG metric for the B-edit model, by design.** `PAPER_MASTER_PLAN.md`
  §0 (LOCKED 2026-07-26) fixes the base as a source-AGNOSTIC prior `Q_theta(y|x,t)`: it never sees
  `x_src`, because source-conditioning lives in the Doob controller `h_phi` applied at inference.
  A correct universal prior therefore SPREADS mass over every plausible edit; scoring high on
  teacher-match would mean memorising one chemist's arbitrary choice. Cross-entropy against a single
  sampled teacher action is still a valid generator-matching loss (a Bregman divergence, so its
  minimiser is the marginal jump law) -- which is exactly why a perfectly trained model would STILL
  show high NLL here. Do NOT propose source-conditioning the base to "fix" accuracy; that is arm B
  of the mandatory A/B/C ablation, never the main model.
- **The Phase-A gate is five criteria, none of them accuracy** (plan §2): low identity collapse, low
  edit cycling, held-out-scaffold generalization, reasonable operator coverage, useful local
  exploration from unseen sources. Measure these before judging the model.
- **Score families against uniform-over-legal-marks, not against the untrained model.** The untrained
  net is not uniform. Correcting the baseline reversed two readings at step 3000: `cycle_insert` is
  the strongest family (28.3x chance over a 152-mark menu), while `ring_system_restate`'s 58%
  "accuracy" is a 1-of-4 coin flip at 1.5x. `bond_reroute` at 0.8x is genuinely BELOW uniform out of
  sample -- capacity fitting noise, and the reason plan §3 item 5 floors it with
  `Q_ref = (1-eps)*Q_theta + eps*Q_legal`.
- **Menu size does not explain family difficulty; determinism does.** `cycle_insert` (152 legal
  marks) reaches 28.3x while `bond_reroute` (154) sits at 0.8x. Where the source's chemistry forces
  the answer the model learns it; where the answer is a design choice living in the destination, it
  cannot.
- **Corpus health, measured on the 70,301-entry train role:** lane mix 33.8% observed_local_analogue
  / 32.0% reversible_synthetic_walk / 24.2% real_endpoint_multistep_path / 5.9% / 4.1%, matching the
  §3 mixture rather than being corruption-dominated; identity edits 0.00%, which is STRUCTURAL --
  `factorized_successor_training.py:112` raises on `source_key == target_key`. Three further
  per-entry invariants are enforced at compile time (executor-executable action, action provably
  reaches the recorded target, family consistent with the atom/edge delta). The plan's §2 PATH-level
  checks (round-trip `A ->pi B ->pi^-1 A`, similarity departure, protected core) do NOT run at
  chunk-compile time.
- **Do not estimate the irreducible entropy floor from repeat source states in this corpus.**
  `atom_delete` carries a 9.1% duplicate `(source, teacher_action)` rate (`atom_insert` 2.7%, others
  ~0); its apparent "chemists agree" concentration was ~1,960 collisions against ~1,940 duplicate
  records. Dedupe before any retrain, and note that after deduping there are too few genuine repeats
  left for the estimator to say anything.
- **`smiles_to_molecular_graph` returns a TIGHT graph and silently deletes the whole
  `atom_insert` family from the legal support.** Insertion is a birth operation that needs a free
  slot; corpus states carry `n_slots=40` (padded to `max_atoms`), a SMILES round-trip carries
  exactly `n_real_atoms`. Measured effect: 1267 -> 1930 marks at 36 heavy atoms, 179 -> 358 at 15
  (support understated by 35-100%). Fix: `pad_molecular_graph(g, 40)` from `compose_v4.chem.state`,
  or decode the corpus payload with `decode_state` (`rewrite/trace_shard.py:86`) instead of
  re-parsing SMILES. This invalidated a rollout gate, a safe-mass probe, and a baseline comparison
  before it was caught. The tell was that `atom_insert` appeared at 10% in ROLLOUTS but 0% in
  single-state enumeration -- one deletion frees a slot, so the family reappears mid-trajectory.
  Any "operator coverage" claim from an unpadded rollout is an artifact.
- **Slice iteration order is lane- and family-sorted; never sample by taking the first N.** This
  produced three separate bad measurements in one session: held-out sources that were 100%
  `reversible_synthetic_walk` vs train 100% `real_endpoint_multistep_path` (making a
  "generalization" gap that was pure lane), a teacher-drift sample from one lane compared against a
  model sample from another, and a baseline comparison whose 240 rows excluded `cycle_insert`,
  `cycle_attach` and `atom_insert` entirely -- omitting the model's strongest family and inverting
  the conclusion. Always stratify by lane and family, and print the realized strata.
- **LEARNING MATTERS: on the correct padded support the prior beats both free baselines.** Held-out,
  320 transitions stratified 40/family, scored by canonical-successor NLL on identical support:
  model 4.943 vs uniform-over-distinct-successors 6.414 (**+1.472 nats, 4.4x**) vs
  state-independent empirical-family 5.912 (**+0.969 nats, 2.6x**). 6 of 8 families beat uniform,
  five by >1.5 nats. Robustness: excluding `cycle_insert` (the outlier at +5.44) the remaining seven
  still average +0.905 vs uniform and +0.297 vs empirical-family; corpus-weighted rather than
  stratified gives +1.445 and +0.530. Losers vs uniform are `atom_restate` (-0.65) and `atom_insert`
  (-0.29); vs empirical-family, `atom_delete` (-1.27, tiny menu plus a 30% corpus prior makes the
  frequency law very strong there). Both margins GREW after padding even though the support nearly
  doubled (432 -> 718 successors), and dropped rows went 40/320 -> 0/320.
- **`scratchpad/eval_invariant.py` is the preflight every COMPOSE evaluation must pass.** It compares
  the probe's realized legal-family census against the census the production compiler recorded for
  the same source state -- stronger than asserting `n_slots == 40`, because it also catches a wrong
  rewrite system, catalog, or family gating. Verified to pass on padded states and to fire on tight
  ones with the exact diagnosis (`disagreeing: {'atom_insert': 663}`). Run it on a sample of an
  experiment's OWN sources before the experiment does any work.
- **Only ONE lane has full operator coverage, and it is the SYNTHETIC one.** Family coverage per lane
  over the 70,301-entry train role: `observed_local_analogue` 2 families (insert/delete 50/50),
  `reversible_synthetic_walk` **all 8**, `real_endpoint_multistep_path` 2 (insert/delete 50/50),
  `operator_aware_real_endpoint` 2 (restate 94.5%, reorder 5.5%), `linker_positional_topology_analogue`
  1 (reroute 100%). Consequence: `cycle_insert` (9,827), `cycle_attach` (6,832) and
  `ring_system_restate` (623) are **100% synthetic with zero real-data support**, and `bond_reorder`
  is 81% synthetic. `cycle_insert` is simultaneously the model's strongest family and its least real
  one. For a paper claiming native topology change this is a DATA-GENERATION gap, not a modelling one.
- **Progress conditioning (`tau`) is a compiler artifact, not chemistry -- do not train on it as-is.**
  In `real_endpoint_multistep_path`, family share moves monotonically from atom_delete 91.4% at
  tau<0.34 to atom_insert 94.2% at tau>0.67, over ONLY those two families: a demolish-then-rebuild
  decomposition of A->B. In `reversible_synthetic_walk`, where operators are applied by a real
  stochastic process, the family law is FLAT across tau (all six families constant to within noise).
  So a progress-conditioned model would beat the no-time model on held-out NLL by learning the path
  compiler's tie-breaking convention, and would bake a shrink-early/grow-late bias into the
  generative dynamics that a budget-driven controller inherits. This also blocks proper Generator
  Matching: GM needs a probability path, and the only paths available come from that same compiler.
  The blocker on GM is NOT cost (metadata joins 100%) -- it is the absence of chemically meaningful
  multi-step paths.
- **Validation is 90.1% synthetic corruption (12,743 / 14,140), but the learning result is
  CONSERVATIVE, not inflated.** Controlled within family and normalised by log(menu), the model does
  BETTER on real than synthetic rows: `bond_reroute` +1.515 real vs +0.201 synthetic,
  `atom_restate` +2.350 vs +1.215, `bond_reorder` +0.552 vs +0.338. Random legal corruption is
  genuinely less predictable than an edit a chemist chose for a reason, so the model is not learning
  the corruption generator's bias.
- **Train and validation family mixes differ by up to 10x** (`atom_insert` 30.0% train -> 3.1% valid;
  `atom_delete` 30.3% -> 4.5%; cycle families ~2.7x enriched in valid), because the split is by
  lane/task and family composition rides along with lane. NEVER compare a held-out family
  calibration against the TRAIN corpus prior -- doing so produced a confident but wrong claim that
  the model under-weights the two largest families. Fitted family biases (8 scalars, LBFGS, source-
  disjoint halves of validation) are worth a real but modest **+0.128 nats** on the joint successor
  NLL, and are tuned to validation's mix so they must be refit before use on train-like data. The
  family head emits `MARK_RULE_NAMES` order (10 slots) not `ACTIVE8_FAMILIES` order (8); they agree
  on slots 0-6 and diverge at 7, so the contract order silently maps ring_system_restate onto
  disabled ring_system_grow.
- **Exact finite-horizon bridge control VERIFIED to machine precision** (`scratchpad/exact_bridge.py`).
  On the closed enumerable graph (197 states / 2,140 edges from a 5-slot carbon seed), with the learned
  successor law as reference R and backward recursion `h_b(x) = sum_y R(y|x) e^{-c} h_{b-1}(y)`:
  terminal-tilt TV 1.4e-16 to 2.4e-16, `|h_B(x0) - sum_y R^B(x0,y)g(y)|` ~1e-16, max |row sum - 1|
  4.4e-16, **zero** edges with P*>0 where R==0, backward-equation residual exactly 0, and
  mid-trajectory objective switching reproduces the exact bridge from the realized state (TV 1.2e-16).
  Reach probabilities: ring 0.561 -> 1.000000000000, bicyclic 0.104 -> 1.000000000000, triple bond
  0.214 -> 1.000000000000. The h-ratios telescope so the controlled B-step endpoint law equals the
  reference law tilted by g -- this is what makes the controller a BRIDGE rather than a heuristic.
  **Caveat: `mean_mass_inside_graph` is only 0.648 (min 0.013)** because the graph is built by the
  DE-NOVO enumerator while the model is scored through the EDITING kernel, so R is a renormalised
  restriction of the learned law, not the full molecular law. Rebuild the enumerable graph under the
  editing rewrite system before quoting these as statements about the production process.
- **Exact finite-horizon Doob/KL control, verified on a slice CLOSED UNDER THE EDITING KERNEL**
  (`scratchpad/exact_bridge_v2.py`; v1 was closed under the de-novo enumerator instead and leaked
  35% of successor mass through silent renormalisation). Escaped mass now goes to an explicit
  absorbing CEMETERY with g=0, so every row sums to 1 to 3.3e-16 with no rescaling. Results at
  budget 6 on a 196-state carbon-only closure: common target (P=0.127) TV 9.9e-17; **rarest
  reachable target (reference P=1.94e-12) TV exactly 0.0** with partition agreement 4e-28;
  unreachable target gives h_B(x0)=0 and an explicitly UNDEFINED controlled row rather than an
  epsilon patch. Zero support violations and zero cemetery leak under the bridge in every reachable
  case. Call this **exact finite-horizon Doob/KL control**, NOT a two-marginal Schrodinger bridge --
  and note that `P(target)=1` under an indicator desirability is definitional whenever the event is
  reachable, so it is a sanity check, never a performance result.
- **The editing kernel inserts heteroatoms, so a carbon-only slice is acyclic and leaks.** From
  `CCCC` at 5 slots: 181 marks, of which 118 are `atom_insert` producing boron/nitrogen/etc; only
  16 of 93 successors are carbon-only (74.5% of mass). The closure therefore contains ZERO ring
  states, which is why ring targets read as unreachable rather than rare. The de-novo enumerator
  gets 149 ring states of 197 only because it restricts elements at the enumerator level. Choose
  bounded slices by what the kernel actually proposes, and remember the slice (<=5 heavy atoms) is
  far off-distribution for a model trained on ~26-heavy-atom drug-like molecules.
- **Step 5 first result: h_phi fits the exact control law on TRAINED goals, fails to generalise to
  UNSEEN ones.** On the cached 197-key closure with exact h as oracle (`scratchpad/learned_value.py`,
  budget 6, chemical-descriptor features, no per-state embedding): fit goals reach zero-accuracy
  ~1.00, log-MAE 0.21-0.28, per-transition TV 0.057-0.069, terminal TV 0.049-0.094. Held-out goals
  degrade to terminal TV 0.37-0.49. So the weak axis is GOAL generalisation, not state
  generalisation or value regression -- which bears directly on the "same frozen prior, swap z, no
  retraining" modularity claim: the PRIOR is reusable, but h_phi currently is not.
- **Never condition a value on a one-hot goal id when testing goal generalisation.** A held-out
  one-hot slot is untrained, so its weights are still at initialisation and the test measures the
  encoding rather than the value function. Encoding the goal by a descriptor of the target SET
  (centroid of member state features + set size) improved held-out `branched` terminal TV from
  0.604 to 0.370. Raw-value MSE is also the wrong loss here: exact values span 0 to ~1e-12 to ~1, so
  split the head into a reachability CLASSIFIER (is h_b exactly zero) plus a log-value REGRESSION,
  and judge by TV of the induced controlled kernel and of the propagated terminal law, never by
  pointwise value error.
- **Heteroatom ring formation works; the acyclic exact slice was a SLICE property.** From N/O/S
  acyclic sources at 8 slots, `cycle_close` enumerates, executes and sanitises heterocycles with the
  heteroatom verified inside the ring (`NumAtomRings > 0`): pyrrolidine C1CCNC1, THF C1CCOC1,
  tetrahydrothiophene C1CCSC1, piperidine C1CCNCC1, pyrazolidine C1CCNNC1, plus unsaturated
  C1=NCCC1 / C1#CCNC1. Ring formation comes from the primitive cycle operator, not a ring-template
  mechanism. The carbon-only bounded slice contains 149 ring states of 196 and has ring mass 0.126
  at budget 6 -- an earlier "zero rings / unreachable" reading was a bug in a throwaway probe.
- **A universal goal-conditioned value DOES generalise -- over a goal LANGUAGE, not over named
  predicates** (`scratchpad/universal_value.py`). Training on 750 procedurally generated goals
  (box / soft-Boltzmann / conjunction over interpretable state features), encoded by SEMANTIC
  PARAMETERS (active features, bounds, weights) and fit with a four-term loss (reachability BCE +
  log-value + log-domain backward residual + controlled-row KL), against exact Doob values on the
  197-state closure at budget 6:
      split            zero-acc  transTV  termTV mean  termTV median
      train             0.9921   0.0768     0.0992        0.0484
      interpolation     0.9887   0.0948     0.1327        0.0655
      composition       0.9658   0.1483     0.1893        0.1713
      extrapolation     0.9625   0.1556     0.1593        0.0633
  The earlier round's held-out failure (termTV 0.37-0.49) was the ENCODING plus only four training
  predicates, not the value function. Mean and median diverge on every split, so a tail of hard
  goals persists even where typical performance is good, and the late-training loss is unstable
  (0.76 -> 2.20 -> 0.95), so these are not converged numbers.
- **Corpus verdict: targeted V2 repair, NOT a redo.** Three cautions beyond the obvious plan.
  (a) Ring closure/opening pairs are precisely what MMP mining handles worst -- they change the
  scaffold -- so scope a bounded feasibility probe before committing to a V2 timeline.
  (b) Downweighting the compiled-path lane does NOT fix the two-family concentration:
  `real_endpoint_multistep_path` (24.2%) AND `observed_local_analogue` (33.8%) are BOTH
  insert/delete-only, so 58% of the corpus sits in two families and only one of those lanes is a
  path lane. (c) Real-data volume and NLL are currently ANTI-correlated -- the two families losing
  to uniform (`atom_restate` -0.65, `atom_insert` -0.29) are among the best supplied with real data,
  while the strongest (`cycle_insert` +5.44) has none. Mining real ring data is for CLAIM validity,
  not for scores; say so in advance so a flat result is not misread as failure.
- **The topology-provenance gap is a PAIR-SELECTION problem, not a compiler or compute problem.**
  Measured over the Active8 output: of 61,424 real-lane traces, exactly **1** has endpoints that
  differ in ring count (`real_endpoint_multistep_path` 0/47,863; `observed_local_analogue` 1/6,555;
  `linker_positional_topology_analogue` 0/4,958; `operator_aware_real_endpoint` 0/2,048). Analogue/
  MMP mining requires a shared scaffold with a large constant core, and a shared scaffold
  mathematically EXCLUDES a ring-count change -- so the corpus cannot contain real ring-forming
  chemistry by construction. Consequently: compiling more tasks will not help (zero ring-touching
  accepted rows in ANY real lane, compiled or not, across ~299k uncompiled real-lane rows), and
  extending the path compiler will not help either (the endpoints themselves do not differ). Closing
  this needs a DIFFERENT pair source -- ring formation/opening, ring expansion/contraction, scaffold
  hops -- which is exactly what MMP is designed to exclude. The alternative is to scope the topology
  claim to synthetic-validated CAPABILITY rather than learned chemistry.
- **The corpus can grow ~10x for free on every other family.** 328 Active8 tasks exist on disk; only
  34 are in `configs/process_v2_prep_subset.json`. Accepted uncompiled rows: 254,576
  `real_endpoint_multistep_path`, 23,075 `linker_positional_topology_analogue`, 18,667
  `operator_aware_real_endpoint`, 2,459 `observed_local_analogue`, 282,051
  `reversible_synthetic_walk`. Pure compute, no mining risk, and it targets exactly the two families
  that currently lose to uniform (`atom_restate`, `atom_insert`).
- **h_phi's failure mode is RARE TARGETS, and that is exactly the lead-optimization regime.** With a
  stabilised run (16 goals/step, grad clipping, cosine decay) on the 197-state closure at budget 6:
      split          zero-acc  transTV  termTV mean / med / p90 / max   >0.3   corr(log tgtfrac, tv)
      train           0.9946    0.0684   0.0796 / 0.045 / 0.181 / 0.363    3%       -0.31
      interpolation   0.9890    0.0858   0.1204 / 0.066 / 0.292 / 0.997    8%       -0.58
      composition     0.9671    0.1471   0.1877 / 0.168 / 0.356 / 0.502   17%       -0.29
      extrapolation   0.9625    0.1560   0.1605 / 0.069 / 0.430 / 0.499   27%       -0.77
  The correlation between log target-fraction and terminal TV is negative on EVERY split: the rarer
  the target, the worse the value. By goal kind, conjunctions (smallest target sets) are hardest
  (`and` 0.233 interp / 0.331 extrap) and dense Boltzmann goals easiest (`soft` 0.110 / 0.072).
  Since real multi-objective lead-op goals ARE narrow conjunctions of property constraints, the
  amortised value is weakest precisely where the flagship application needs it -- design Step 6
  around a fallback tier (SMC/Feynman-Kac correction, or exact h on enumerable slices) rather than
  assuming one learned value covers every objective. Stabilisation only moved train 0.0992 -> 0.0796
  and the loss still oscillates, so the variance is inherent across goals of very different
  target-set size; importance-weighting toward rare targets is the untried lever.
- **Step 6: local guidance samples the WRONG distribution; SMC amplifies a good value but cannot
  rescue a bad one.** Difficulty-stratified against the exact bridge on the 197-state closure
  (budget 6, goals binned by q_z = P_R{X_B in G_z} computed BEFORE any control; median TV):
      band                n   greedy  boltzmann  h_phi  h_phi+SMC  unguidedSMC  ESS_tw
      broad q>1e-2       87   0.1731   0.1074    0.0349   0.0368     0.0428      0.778
      mid 1e-2..1e-5     26   0.9430   0.9511    0.2063   0.5000     0.4220      0.350
      rare q<1e-5         7   0.9867   0.9998    0.5304   0.5000     0.5000      0.197
  Greedy and Boltzmann collapse to TV 0.94-1.00 outside the broad band -- the plan's headline
  (high reward, wrong distribution) is demonstrated. Reachability precision 0.9931 / recall 0.9917.
  A TV of exactly 0.5000 is the signature of ZERO total particle weight (no particle reached the
  target), not graceful degradation.
- **The SMC is correct; the value is the bottleneck.** Re-running twisted SMC with the EXACT h
  returns full mass on every goal and achieves **TV = 0.0000 at q = 1.94e-12 with 2,000 particles**,
  while unguided SMC returns zero mass on rare goals. So twisted Feynman-Kac reaches a rare target
  only if the potential steers particles there: it gives variance reduction proportional to h's
  quality and cannot manufacture reachability from a bad h. **Consequence: improving h_phi in the
  rare regime is a PREREQUISITE, not a fallback** -- importance-weighting the training goal
  distribution toward rare targets moves onto the critical path. Also: even with exact h, SMC TV
  ranges 0.03-0.32 across goals purely from 2,000-particle Monte Carlo error, so SMC arms are not
  comparable to analytic arms without matched budgets.
- **Rarity-balanced h_phi training is NET NEGATIVE -- the rare tail is not a sampling problem.**
  Matched ablation (same 4,000-goal pool, same eval goals, same seed; only the training
  distribution differs; alpha=0.35, weight cap 8x, strata-balanced across band x kind, row-KL
  upweighted 4x):
      band   median targets   control h_phi   balanced h_phi   delta
      broad        72             0.0639          0.1344      -0.0705
      mid          18             0.2368          0.2534      -0.0165
      rare          7             0.5269          0.4935      +0.0334
  It doubles broad-regime error to buy 6% in rare, and 0.494 is still a half-wrong distribution.
  The control reproduces the earlier Step 6 numbers on this harder conjunction-biased pool
  (0.064/0.237/0.527 vs 0.035/0.206/0.530), which confirms the pool -- not a regression -- explained
  the first apparent collapse. INTERPRETATION: rare goals are not hard because they are
  under-sampled; they are hard because h spans ~12 orders of magnitude and the successors that
  matter carry vanishing mass. That is a representation/precision problem, so reweighting the goal
  distribution cannot fix it. NEXT: nested / annealed bridge control (progressively tightening
  regions, or g_beta1 -> ... -> g_betaL), which converts one rare-event problem into a sequence of
  easy ones -- do NOT keep enlarging the value network.
- **Always run the matched control before reading a training-distribution result.** The first
  rarity-balanced run looked like a catastrophic failure against the Step 6 baseline, but that
  baseline used a DIFFERENT goal pool (unbiased vs 45% conjunction-biased) with easier goals in
  every band and median rare target-set size 1 vs 7. Only the same-pool ablation separated "the
  weighting hurt" from "these goals are harder".
- **Rare MULTI-STATE targets barely exist in a 197-state closure.** Sorting rare goals by
  target-set size still gives median 1 in an unbiased pool; a conjunction-biased pool reaches median
  7. Since a single-terminal-state rare goal has a degenerate conditioned law (TV=0 is trivial
  there), the meaningful finite-particle SMC benchmark needs either a larger closure or a longer
  budget.
- **Split the exact-slice roles; do not make the small closure carry rare-event inference.** The
  197-state closure is the right testbed for exact terminal tilt, unreachable rows, dynamic
  retargeting, exact-h supervision, h_phi interpolation/composition, and the broad->mid->rare
  degradation curve -- all settled there at machine precision or with adequate power. It is the
  WRONG testbed for finite-particle rare-event inference, because rare multi-state targets barely
  exist in it. CAVEAT ON AN EARLIER CLAIM: the `q = 1.94e-12 -> TV = 0.0000` twisted-SMC result used
  a SINGLE terminal state, so the conditioned law is degenerate and TV=0 is the easy case; it is a
  numerical stress test that log-weights survive ~12 orders of magnitude, NOT evidence that SMC
  solves multimodal rare conditioning. A dedicated larger slice (6 slots, ~967 states; measured
  scaling is ~5x states per added slot: 4->51, 5->197, 6->967) gives targets with |G_z| ~ 5-50 at
  q_z ~ 1e-3/1e-5/1e-7 while keeping exact h computable (dense R ~7.5 MB, R^B is six 967^3 matmuls).
  7 slots (~4,800 states) makes per-goal matrix powers too slow to iterate on.
- **Rare-event benchmark on the 966-state closure (multi-state targets, |G| 5-50, 2,000 particles).**
  Median TV against the exact bridge; `exactH+SMC` is the Monte-Carlo FLOOR at this budget and is
  what every other SMC arm must be read against:
      band          n  |G|   h_phi  h_phi+SMC  unguided  exactH+SMC   mass    ESS
      1e-2..1e-4   25   19  0.3076    0.5000    0.5000     0.0774    0.240  0.365
      1e-4..1e-6   25   28  0.4633    0.5000    0.5000     0.2701    0.240  0.156
      q<1e-6       25    8  0.4985    0.5000    0.5000     0.0873    0.000  0.158
  (1) Twisted SMC DOES solve multimodal rare conditioning given a good potential -- TV 0.087 at
  q<1e-6 on 8-state targets, non-degenerate, which the earlier single-state 1.94e-12 result could
  not show. (2) A BAD potential makes twisted SMC WORSE THAN NO SMC: h_phi+SMC returns zero median
  mass in every band including 1e-2..1e-4 where h_phi alone scores 0.308, because the twist steers
  particles away and the weights cannot recover. SMC amplifies the value's errors as well as its
  signal. (3) Monte-Carlo error tracks TARGET SPREAD, not rarity -- the floor is worst at
  1e-4..1e-6 (|G|=28, 0.270) and best at q<1e-6 (|G|=8, 0.087) -- so an SMC number quoted without
  its per-band floor is uninterpretable. CONSEQUENCE: "amortise broadly, correct the tail with SMC"
  does not work, because correction needs a potential good enough to steer, which is exactly what is
  missing in the tail. Nested/annealed bridge control is the only remaining route, and its
  justification is that it MANUFACTURES usable potentials by never asking for a vanishing event in
  one shot -- not that it works around any limitation of SMC itself.
- **Hard staging is BIASED; anneal as a twisting schedule inside SMC instead.** First annealed-bridge
  probe on the 966-state closure (rare conjunctive goals, q 1.2e-07..9.9e-04, |G| 5-50, 3 stages
  with beta chosen by bisection so each intermediate target's reference mass hits q^(j/L)):
      arm                    median TV   median target mass
      exact one-shot            0.0000        1.0000
      EXACT staged (oracle)     0.4608        0.6005     <- perfect values, still biased
      h_phi one-shot            0.4872        0.0256
      h_phi+SMC one-shot        0.6953        1.0000
      LEARNED staged            0.4854        0.0293
  The staged arm ran a bridge to each intermediate set AT INTERMEDIATE TIMES, which conditions on
  strictly more than the terminal event -- hence a large bias that no amount of value accuracy can
  remove. Annealing must therefore enter as a twisting/PROPOSAL schedule inside SMC, where
  importance weights correct it (unbiased for any schedule), not as hard intermediate conditioning.
  Including the EXACT-staged oracle is what caught this: without it, `LEARNED staged ~= h_phi
  one-shot` would have read as "annealing does not help" rather than "this annealing is the wrong
  kind". Staging does buy target attainment (mass 0.60 vs 0.026, ~23x) at the cost of distributional
  fidelity -- relevant if attainment ever matters more than exactness.
- **Annealed-twisting GATE FAILED, structurally: one beta per transition cannot span the range.**
  Exact h_beta, no training, 966-state closure, rare multi-state goals (q 2.6e-07..7.2e-05,
  |G| 5-50, 4,000 particles). With an ESS-respecting adaptive ladder beta only climbs to ~1-2.4
  over the 6 transitions, ESS stays near target (0.52-0.57), and TARGET MASS IS 0.000 on 11 of 12
  goals -- the proposal is too weak to steer into the event. Forcing beta=200 instead reaches the
  target (mass 1.0, TV 0.127 vs a 0.102 floor) but is not annealing at all and runs ESS 0.09-0.32.
  So ESS-respecting beta is too weak to reach; reach-capable beta destroys ESS; and SIX TRANSITIONS
  is not enough room to traverse between them. Fix: decouple tempering from state transitions --
  standard SMC-samplers use many tempering steps at a FIXED time index with MCMC rejuvenation
  between them -- or lengthen the horizon. Do not tie one temperature to each transition.
- **Two adaptive-ESS traps, both worth remembering.** (1) Scoring the spread of Z(x) across CURRENT
  particles is the wrong criterion: degeneracy comes from the 1/h(y) factor AFTER proposing, and at
  step 1 all particles sit on one state so the proxy is flat and beta pins to beta_max -- producing
  a vacuous "PASS" with no annealing. (2) The right criterion is available in CLOSED FORM: for
  y ~ P(.|x) ∝ R(x,y)h(y), the incremental weight Z(x)/h(y) has E[w|x]=1 and
  E[w^2|x] = Z(x) * sum_{y:h(y)>0} R(x,y)/h(y), so post-step ESS is predictable before sampling.
  Also: for hard conjunctions anneal a CONTINUOUS violation magnitude (distance outside the boxes);
  an indicator raised to any power is still an indicator, and a count of satisfied constraints gives
  only k+1 coarse levels.
- **Terminology: annealed twisted SMC is CONSISTENT / asymptotically exact, not "unbiased".** The
  self-normalised finite-particle estimator carries Monte Carlo bias. Pass criteria must therefore
  be "matches the one-shot bridge to the same finite-particle floor plain exact twisting achieves at
  this budget", never "matches exactly".
- **PATH-SPACE SMC SAMPLER GATE: PASS.** Keeping the six molecular edits fixed as task semantics and
  moving tempering into the inference layer -- targets `pi_j(path) ∝ prod_b R(x_{b-1},x_b) *
  g_{beta_j}(x_B)` with endpoint-only reweighting, ESS-adaptive beta, resampling, and suffix
  regeneration between levels -- recovers the one-shot exact bridge on rare multi-state goals
  (q 1.2e-07..4.5e-05, |G| 5-50, 4,000 particles, 966-state closure):
      median TV 0.0256   median target mass 1.000   ESS 0.61-0.65   only 7-10 tempering levels
  The rejuvenation is a GIBBS move accepted with probability 1, because with exact h_beta the
  twisted kernel generates precisely pi_j(suffix | prefix). **With a learned h_phi that is no longer
  true and an MH ratio is required** -- not a drop-in swap, and the reason to gate the exact case
  first.
- **CORRECTION: plain exact-h twisted SMC is NOT the Monte-Carlo floor.** It resamples WITHOUT
  rejuvenation, so resampling duplicates particles and destroys diversity. At the same 4,000-particle
  budget the path sampler is ~7x more accurate (median TV 0.0256 vs 0.1771). Earlier statements that
  ~0.09 was "the floor" / "known achievable" understated the reachable accuracy; future arms should
  be measured against ~0.026. Resample-then-rejuvenate, and never quote a resampling-only estimator
  as a floor.
- **Durable machinery now lives in the REPO, not scratchpad.** Promoted with tests
  (`tests/test_editing_v2_bridge_control.py`, 6 passing):
    `src/compose_v4/experiments/editing_v2_evaluation_semantics.py` -- the production-semantics
      preflight (`assert_production_state_semantics`, `production_state_from_smiles`). Run it on a
      sample of an experiment's OWN sources before the experiment does any work.
    `src/compose_v4/experiments/editing_v2_bridge_control.py` -- editing-kernel closure with an
      explicit CEMETERY, exact backward values, controlled kernel (unreachable rows returned as
      all-zero/UNDEFINED rather than epsilon-patched), `terminal_tilt_residual` evidence, and the
      path-space annealed SMC sampler.
  Verified against the real 966-state closure after promotion: terminal-tilt TV 1.5e-16, zero
  support violations, path-sampler TV 0.0303 at mass 1.000 / ESS 0.639 / 8 levels -- matching the
  scratchpad run. Everything else built this session (~90 files) is genuinely throwaway probing and
  correctly stays in scratchpad.
- **MH-corrected rejuvenation with a learned h_phi: derivation looks right, empirics INCONCLUSIVE.**
  Demoting the value to a proposal and restoring the target by Metropolis-Hastings is computable
  from h_phi alone -- every R factor cancels between target and proposal, leaving
      alpha = min(1, [g_beta(x'_B)/g_beta(x_B)] * [L(S)/L(S')]),
      L(S)  = prod_{k>m} h^phi_{B-k}(x_k) / Z^phi_{B-k+1}(x_{k-1}).
  With deliberately corrupted h_phi (lognormal sigma) on the 966-state closure, acceptance falls as
  predicted (0.994 / 0.746 / 0.555 / 0.375 at sigma 0 / 0.5 / 1 / 2) but terminal TV also rises
  (0.027 / 0.053 / 0.122 / 0.353). That is consistent with EITHER bias (derivation wrong) or
  variance (poor mixing leaves post-resampling duplicates, shrinking the count of distinct paths).
  A particle-scaling check at sigma=2 was under-powered and confounded: TV fell 0.367 -> 0.239 ->
  0.126 at N = 1k/4k/16k (roughly 1/sqrt(N)) then rose to 0.213 at 48k, because the beta ladder is
  ESS-ADAPTIVE and therefore a different algorithm at each N, and the median was over only 5 goals.
  CLEAN REDESIGN before concluding: freeze the beta schedule from the sigma=0 run and reuse it
  across all N and sigma, use >=30 goals, average several seeds per cell. Do not record a verdict
  until that runs -- a FAIL here would itself be an over-claim.
- **V2 CENSUS (`scripts/editing_v2_corpus_census.py`, `diagnostics/editing_v2_corpus_census.json`):
  compile selectively, not wholesale.** 328 Active8 tasks, 34 compiled, 2,311,080 accepted
  transitions scanned (an earlier 299k figure came from `rows.jsonl.gz`, a coarser granularity).
  Uncompiled by lane:
      real_endpoint_multistep_path      1,732,637 rows   0 cycle   ~100% insert/delete
      reversible_synthetic_walk           386,603 rows   239,032 cycle  (the ONLY cycle source)
      linker_positional_topology_analogue  23,075 rows   0 cycle   100% bond_reroute
      operator_aware_real_endpoint         18,667 rows   0 cycle   95% atom_restate
      observed_local_analogue               4,918 rows   0 cycle   insert/delete
  (1) ZERO cycle-family rows in any real lane across all 2.31M rows -- the topology gap is confirmed
  at full scale and is a pair-selection problem, not a compute one. (2) Compiling everything would
  push insert/delete from 60.3% to 82.3%: `real_endpoint_multistep_path` is 1.73M rows of pure
  insert/delete and would make the imbalance WORSE. (3) The two SMALL lanes are the valuable ones
  and are invisible if you reason from row counts: `operator_aware_real_endpoint` adds 17,765 real
  `atom_restate` (5.5x the ~3,900 currently held, and atom_restate is the family that LOSES to
  uniform at -0.65), and `linker_positional_topology_analogue` adds 23,075 real `bond_reroute` (9x
  the ~2,870 held, for the family marginal at +0.20). PLAN: take those two lanes whole (~42k rows),
  subsample `real_endpoint_multistep_path` for scaffold diversity only, subsample the synthetic walk
  for cycle coverage. New Bemis-Murcko scaffolds 23,930 vs 18,788 compiled-sampled (~2x diversity),
  far less than the ~10x row count implies -- many new sources are analogues of molecules already
  present.
- **UNCAPPED scaffold census: 99,380 new scaffolds, not 23,930.** The capped figure was a 4x
  understatement quoted as if measured; the report now carries `scaffold_count_is_exact` and prints
  [EXACT] vs [FLOOR] so it cannot recur. Compiled corpus holds 21,872 scaffolds. Per-lane
  concentration (uncompiled sources; med/p90/max = sources per scaffold):
      reversible_synthetic_walk             94,852 scaffolds  88,030 new   med 1  p90 3   max 18,527
      real_endpoint_multistep_path          23,524            12,519 new   med 7  p90 23  max 31,569
      linker_positional_topology_analogue    7,573             3,121 new   med 2  p90 4   max 801
      operator_aware_real_endpoint           6,600             2,092 new   med 2  p90 4   max 670
      observed_local_analogue                   55                19 new   med 17 p90 90  max 952
  SUBSAMPLING RULE this implies: take `operator_aware_real_endpoint` and
  `linker_positional_topology_analogue` WHOLE (already diverse, med 2 / p90 4, nothing wasted);
  cap `real_endpoint_multistep_path` at 4 sources per scaffold, retaining 91,545 of 375,894 sources
  (24%) while keeping essentially all 12,519 new scaffolds -- it is row-heavy and diversity-light,
  with a single scaffold carrying 31,569 sources; SKIP uncompiled `observed_local_analogue`
  (4,918 rows for 19 new scaffolds).
- **89% of the scaffold-diversity gain is SYNTHETIC.** The reversible walk supplies 88,030 of the
  99,380 new scaffolds at median 1 source per scaffold; all real lanes together contribute ~17,750.
  So real-source scaffold diversity goes 21,872 -> ~39,600 (about 1.8x) while the headline including
  synthetic is ~5.5x. Do not let the aggregate imply real-chemistry breadth the corpus does not
  have -- state the real-only figure alongside it.
- **V2 MIXTURE TABLES (`scripts/editing_v2_v2_mixture_plan.py`,
  `diagnostics/editing_v2_v2_mixture_plan.json`).** (1) The per-scaffold cap retention curve for
  `real_endpoint_multistep_path` is FLAT on every diversity axis:
      cap 1 -> 159,967 rows (9.2%), 23,525 scaffolds, 12,520 new (100%), ins/del 100%
      cap 2 -> 289,218 (16.7%)   cap 4 -> 484,055 (27.9%)   cap 16 -> 925,830 (53.4%)
  all retaining 100% of scaffolds and 100% of new scaffolds, because a per-scaffold cap of 1 already
  touches every scaffold. So **cap=1 (or 2) is correct and cap=4 would have taken 3x the rows for
  zero additional chemistry** -- the earlier cap=4 suggestion was an artefact of which column the
  census happened to instrument. The lane is 100% insert/delete at every cap, so no cap changes its
  family character; it is worth taking ONLY for scaffold breadth.
  (2) Capability-driven synthetic sampling at 12,000/family selects 56,459 sources / 148,030 rows
  (38% of the lane) and satisfies all eight families with none short (cycle_attach 51,926,
  atom_restate 19,152, ring_system_restate 13,152, cycle_insert 12,000 = the binding constraint,
  since synthetic is the ONLY source of cycle families).
  (3) MIXTURE WARNING: at that target V2 lands near 40% synthetic (new real 201,709 vs new synthetic
  148,030, plus the existing 70,301 at ~32% synthetic), which is not "real dominates, synthetic
  regularizes". Lower per-family targets trade cycle coverage for synthetic share; a LINEAR
  EXTRAPOLATION (estimate, not measured) puts target 6,000 near 28% and target 3,000 near 19%.
  Measure rather than extrapolate before freezing the recipe.
- **V2 RECIPE FROZEN.** Corrected incremental synthetic sweep (floor on every family's combined
  old+new count, ranking aligned to the stop condition):
      floor  sources    rows  synth%  cells uncov  rarest fam
      1,000      125     556    6.6%     21     0       1,002
      2,000    1,222   3,394    7.4%     21     0       2,000
      3,000    2,382   5,714    8.0%     21     0       3,000
      5,000    7,687  16,767   10.8%     21     0       5,000
      8,000   36,020 127,328   31.6%     21     0       8,000
  Sharp knee between 5,000 and 8,000 (7.6x the rows for a 1.6x higher floor, as the scarce families
  exhaust dense sources). CHOSEN: floor 5,000.
      real_endpoint_multistep_path   cap 1/scaffold   159,967
      linker_positional_topology     whole             23,075
      operator_aware_real_endpoint   whole             18,667
      reversible_synthetic_walk      floor 5,000       16,767
      observed_local_analogue        SKIP                   -
      new 218,476 + existing 145,180 = 363,656 rows, 10.8% synthetic, all 21 cells,
      ring_system_restate 623 -> 5,000.
  THREE ERRORS CORRECTED ALONG THE WAY, all mine: (a) "~40% synthetic" used the wrong denominator,
  omitting the 122,656 existing compiled real rows -- the true range was always 26-32% and is now
  10.8%; (b) capability cells were never the binding criterion -- the compiled corpus already covers
  19 of 21 and synthetic adds only 2, saturating below the smallest target swept, so the real
  constraint is rare-FAMILY counts; (c) the greedy ranked sources by total cycle-family density
  while stopping on `cycle_insert` alone, inflating row counts by up to 170x (556 vs 95,678 at floor
  1,000). Rank by the same quantity the stop condition uses.
- **Cache an expensive corpus scan before sweeping over it.** The mixture script re-scanned 2.31M
  rows and recomputed ~476k Murcko scaffolds on every run -- ~13 of each 15 minutes was identical
  work, three runs running. Only the sweep parameters changed.
- **V2 SELECTION FINAL: source/scaffold diversity picks WHICH x; distinct canonical successors
  decide how much supervision each x keeps.** Two guards, both measured, both load-bearing.
  (1) MULTI-SUCCESSOR PRESERVATION (`scripts/editing_v2_multisuccessor_audit.py`): selecting one
  `(task, source)` pair per scaffold discarded 8,145 distinct successors (17% of pairs) and dropped
  the multi-successor rate 67.7% -> 57.2%. Taking the selected SOURCE across every task it appears
  in restores it. The selection is 3.2x ENRICHED in multi-successor states vs the candidate pool
  (57.2% vs 17.8% even before the fix), because ranking by `-rows` picks the most productive source
  per scaffold. Median distinct successors per selected source is 2.
  (2) CANONICAL (x,y) DEDUP IS MANDATORY: 45.1% of selected records are duplicate canonical pairs
  (multistep 63.6%, synthetic 10.7%, the two valuable real lanes 0.0%), and ONE (x,y) appears 129
  times. Uncollapsed, that successor carries 129x its true mass from compiler serialisation alone --
  a serialisation artefact learned as chemistry. Duplicate serialisation is NOT repeated independent
  observation; keep provenance as metadata only.
  NET: 186,571 raw -> **102,430 deduped rows**, versus 109,055 for the one-task-per-source variant --
  same size, but 67.7% multi-successor instead of 57.2%. Duplicate serialisations traded for genuine
  conditional-distribution supervision at no cost in corpus size.
  (3) SPLIT BY SOURCE, NEVER BY ROW. Because 67.7% of selected sources carry several distinct
  successors, splitting rows independently puts y1 in train and y2 in validation for the same x.
  The manifest carries `split_constraint` requiring all records sharing a source_canonical_key to
  land in one component -- the leakage route is created by the very structure that makes the corpus
  useful.
- **POST-DEDUP CENSUS CAUGHT A SILENT FLOOR FAILURE; V2 NOW FROZEN.**
  (`scripts/editing_v2_postdedup_census.py`.) The synthetic family floor is applied to RAW rows, but
  the corpus is deduplicated to distinct canonical (x,y) afterwards, so a family whose synthetic
  sources duplicate heavily lands UNDER its floor. Measured: at floor 5,000 `ring_system_restate`
  reached only 3,379 new + 623 compiled = **4,002, a 20% shortfall**, and nothing in the pre-dedup
  numbers showed it. Left alone it would have surfaced months later as "the rare family still
  underperforms" after a multi-day compile and a full training run.
  FIXING IT GLOBALLY IS AN OVER-CORRECTION: raising the floor 5,000 -> 6,500 recovered the 998-row
  shortfall but lifted every other family too (atom_restate 18,100 -> 33,371), adding ~22k synthetic
  rows and moving the mixture 14.5% -> 21.6%. PER-FAMILY floors (5,000 default,
  ring_system_restate=6,500) meet the guarantee at **16.4% synthetic** with 17,000 fewer synthetic
  rows.
  FROZEN V2: 107,872 rows post-dedup; 16.4% combined synthetic; ring_system_restate 623 -> 5,307;
  43.0% multi-successor sources (mean 1.574); 18 cross-lane duplicate pairs (per-lane dedup alone
  would have missed these); selection sha256 b1a38c2f4ba8401b, census sha256 56cf4a717970f02a.
  STALE FIGURES, do not reuse: "10.8% synthetic" and "102,430 rows" were both pre-dedup or
  pre-fix. ALWAYS recompute headline counts after the transformation that actually produces the
  training corpus.
- **THE CHEMISTRY KERNEL CANNOT BE OPTIMIZED WITHOUT RE-SEALING PROCESS V2.**
  Profiling one compile slice found the dominant cost is not what anyone assumed: 5 entries issued
  **80,448 `molecular_graph_to_smiles` calls over 6,391 distinct graphs -- 92.1% redundant**, one
  graph converted **11,141 times**. Support enumeration re-validates the same intermediate graphs,
  and `canonical_state_key` rebuilds the exact string `is_rdkit_valid` already computed and threw
  away (`is_rdkit_valid` = graph -> SMILES -> re-parse, i.e. two full RDKit sanitizations per call).
  A content-addressed memo on those two pure functions is trivial and exact -- and it is NOT
  SHIPPABLE. `src/compose_v4/chem/molecular_graph.py` is the first of the 18 files in
  `_PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS`, hashed into `implementation_source_sha256`, which IS
  the process identity. Adding the cache moved it `0c938177 -> a31040f6`, and every contract then
  refused to load: "not the live Process-V2 identity". Every hot function in the profile
  (`kernel.py`, `operators.py`, `aromatic_kekule.py`, `semantic_atom_restate.py`,
  `factorized_tracelet_rate_model.py`) is in that same set, so there is no identity-free seam to
  hide a cache behind. That is the design working, not a bug: the corpus is reproducible only by
  exactly these bytes, and slices compiled under two identities would be a split-brain corpus.
  CONSEQUENCE: kernel performance work is a deliberate PROCESS REVISION -- re-seal the chain,
  re-run Gate 0, recompile every row -- never an incidental speedup mid-corpus. Patch and its 13
  equivalence tests are preserved for that revision. Do not benchmark-and-commit chemistry.
  MEASURED PAYOFF, so the revision can be costed rather than guessed: unit costs are SMILES
  240us, re-parse 91us, content-key 13.4us. Chemistry is 23.3s of a 39.4s five-entry compile
  (59%); the memo takes it to 3.2s (**7.3x on chemistry**) including a 1.08s key-hashing tax,
  giving **2.05x END-TO-END**, 7.88 -> 3.85 s/entry. A full 36,864-record compile goes 117.8
  -> 57.6 Modal core-hours. Beware two traps that inflate this: replaying `is_rdkit_valid` and
  `molecular_graph_to_smiles` as independent workloads double-counts (valid CALLS smiles) and
  reports ~3.5x; and `is_rdkit_valid` cannot be intercepted by patching the module attribute,
  because callers such as `chem/state.py` bind it directly at import -- the memo therefore has
  to live INSIDE the function, which is also why it cannot dodge the identity hash.
- **/private/tmp IS REAPED MID-SESSION; NEVER LET IT HOLD THE ONLY COPY.**
  Three near-misses in one day, each caught by luck rather than by a check: the
  70,301-entry `train_65k` corpus lived only in a scratchpad directory with a
  partial (31,543-row) backup and was briefly believed not to exist at all;
  21,778 compiled entries lived only on a Modal volume; and then the git
  WORKTREE holding the day's work was reaped while a test run was in flight --
  `.git` vanished and `src/` went from 118 modules to 18. That last one first
  presented as `ModuleNotFoundError: editing_p50_gate` and read exactly like a
  pytest path problem.
  NOTHING WAS LOST, for two reasons worth knowing. A git worktree keeps its
  objects in the PARENT repository, so all 25 commits survived the deletion of
  the checkout and were recovered with `git worktree prune` + `git worktree add`
  against a durable path. And the corpora had already been replicated to
  `~/compose_trainset_backup` after the first scare.
  THE FIX IS A GATE, NOT A HABIT: `compose_v4.data.durable_path` refuses a
  source-of-truth path under `/private/tmp` or `/var/folders`, with
  `COMPOSE_ALLOW_REAPABLE_PATH=1` as an explicit per-invocation opt-out for
  genuinely disposable probes. A rule you have to remember is a rule that fails
  at 00:30 after twenty hours of work -- which is precisely when it did.
  STILL OUTSTANDING after all that: 25 commits existed on NO remote branch, so
  the code was single-copy on one disk even after the corpora were safe. Push
  the branch; replication is the only real protection, the gate just removes
  the failure mode already paid for.

## 2026-09-20 (T4 held-target `candidate_exhaustion`: it is the JOINT gate, and the fiber is not empty)

- **All 15 held-target cells reproduce their live terminal status offline, zero oracle calls**
  (`scripts/t4_support_stage_audit.py`, `diagnostics/t4_support_stage_audit_v1.json`). Round one is
  deterministic given the cell: one parent (the docked root), three experts, seed
  `controller_seed + 1_000_003*round + 10_007*parent + 101*expert`. Re-running the pinned library
  functions gives selected = 0 for the five `candidate_exhaustion` cells and 3/7/8/8/8/8/8/8/8/8 for
  the ten that search. `parent_score` is unobtainable offline and provably cannot change the count:
  `Fiber.check` never sees it and `ProgramValue` is unfitted at round one, so every pick is random.
- **METHOD: instrument a gate by WRAPPING it, never by transcribing it.** `expand` returns only
  endpoints that already passed `Fiber.check`, so the funnel interior is invisible from its return
  value. The audit monkey-patches the module globals `t4_fiber_campaign` resolves at call time, in
  its own process only; `Fiber.check`'s ORIGINAL method still decides, and the per-gate decomposition
  is recomputed beside it and compared on every call. **0 disagreements over 118,000 gate calls** is
  what licenses the decomposition; a transcription would have had nothing to check itself against.
- **THE BINDING CONSTRAINT IS THE INTERSECTION, NOT ANY SINGLE GATE.** For braf_0/braf_1/fa7_0/fa7_2
  each threshold individually admits hundreds to thousands of endpoints (braf_0: 1,638 pass
  similarity, 3,534 pass SA) while `similarity AND QED` admits **zero**. The discriminating scalar is
  `max similarity among QED>=0.6 endpoints`: braf_0 0.488, braf_1 0.566 (FAIL) vs braf_2 **0.650**
  (works). Reporting per-gate pass rates alone hides this completely -- always report the pairwise
  intersection.
- **5ht1b_2 fails on a DIFFERENT gate and must not be grouped with the other four.** 474 endpoints
  pass similarity AND QED; the minimum SA among them is **4.332** against a 4.0 ceiling, while the
  SOURCE itself sits at SA 4.687. The SA ceiling constrains RETURNED molecules, not the source, so
  this is a legitimate benchmark instance that happens to demand a substantial SA REPAIR (>=0.69)
  while holding Tanimoto >= 0.6 -- it is a hard cell, NOT an out-of-spec one, and calling the source
  out-of-spec would wrongly imply the instance is invalid. Relaxing SA to 4.5 admits 6, to 5.0 admits 163, while
  relaxing similarity or QED by 0.10 admits 0-1. One label, two mechanisms.
- **"Zero eligible" is NOT "empty fiber" -- 4 of the 5 failures have a demonstrated eligible witness.**
  A program-free acyclic-bond truncation beam finds, inside the delta=0.6 ball, braf_0 at
  sim 0.632/QED 0.635/SA 2.41 (-14 heavy atoms), braf_1 at 0.686/0.663/2.19, fa7_2 at
  0.600/0.663/2.08, and a single-edit beam finds 5ht1b_2 at 0.758/0.626/3.77. Only fa7_0 has no
  witness. Confirmed independently: braf_1 and braf_2 differ ONLY in the amide tail, and the exact cut
  that produced braf_2's three eligible endpoints, applied to braf_1, gives sim 0.657 / QED 0.790 /
  SA 2.17 -- BETTER margins than braf_2's actual winner. **So these are proposal-coverage failures at
  480+512 draws, not support failures.**
- **A single-edit beam is the wrong probe for a move class that deletes 13 atoms.** The first
  geometry probe (depth-5 single-atom edits) returned NEGATIVE on braf_2, a cell that searches --
  which is how the probe was caught being uninformative rather than being read as "empty fiber".
  Probe the move class that actually wins: every eligible endpoint in every control here is a PRUNE.
- **The held-target PRIOR is not the discriminator.** It enters only the `route_complete_region`
  lane, which returns **0 eligible in 8 of the 9 braf/fa7/5ht1b cells, controls included** (5ht1b_1
  gets 1), despite committing 7-64 complete programs each -- and it was MOST productive at the
  proposal level on the failing 5ht1b_2 (64 committed). Every control survives on the model-free
  `shallow` lane alone (3, 7, 27, 38 eligible). Attributing the failures to the prior does not
  survive the sibling comparison.
- **The failure is marginal, not categorical.** braf_2 survives round one on **3** eligible endpoints
  out of 7,420 and fa7_1 on **7** of 8,583. FAIL vs CONTROL here is 0 versus 3, so any change that
  moves the boundary slightly re-labels cells. fa7_0's best similarity-passing endpoint has
  QED **0.5983** -- it misses eligibility by 0.0017 QED.
- **DEFECT FOUND, live campaign: `configs/t4_held_target_distilled_jak2_d06_250.json` carries
  `delta = 0.4`, not 0.6.** Its own `claim_boundary` says "held-target jak2 panel at delta 0.6". The
  value was inherited from `frozen_from.controller_contract`
  (`t4_shared_retained_fiber_jak2_v2.json`, delta 0.4, 49 calls) while `charged_calls_per_cell` WAS
  updated to 250 -- a partial copy. All four sibling `jak2_d06_v1..v4` contracts carry 0.6, and the
  other four proteins' 250-call contracts carry 0.6. Measured effect: jak2 yields 183-1,224 eligible
  endpoints per cell against <=150 everywhere else, so no jak2 number from this run is a delta=0.6
  result. **`delta` is executable (`Fiber(cell["smiles"], contract["delta"], ...)`); `claim_boundary`
  is prose. Diff them.**
- **Environment parity is checkable cheaply and was checked.** The audit ran on rdkit 2025.09.6 while
  production pins 2024.3.5. On 8,183 real endpoints from a completed shard, similarity, QED, SA,
  heavy count and the `med_chem_gate` verdict are **byte-identical** across the two. Build the pinned
  env (`uv venv --python 3.11` + `rdkit==2024.3.5 numpy==1.26.4 scipy==1.13.1 networkx==3.3`, ~1 min,
  no torch needed for this path) and diff on REAL generated endpoints, not just on the seeds.
- **The proposal source is padded to 48 slots (`pad_molecular_graph(..., 48)`), so atom birth is
  expressible, but no endpoint anywhere exceeded 40 heavy atoms and the `heavy > 40` gate fired
  0 times in 118k calls.** The ceiling binds upstream in the executor, not at the fiber. Consequence:
  a source AT 40 heavy atoms has a proposal distribution with literally **0% grow** (braf_0), versus
  19.7% at 39 (braf_1) and 36.6% at 37 (braf_2). Headroom shapes the direction mix; it does not by
  itself decide feasibility, since 5ht1b_0 at 39 heavy atoms searches fine.

## 2026-09-20 (PMO scale-up prep: the budget a contract declares is not the budget that runs)

- **A contract budget is only real if the runtime READS it. `pmo_population_v1.execute_task` does
  not.** The corrected scored contract declares `charged_calls_per_task: 250` and
  `launch_pmo_population_v1_corrected.py` verifies that block byte-for-byte -- but `execute_task`
  builds its ledger as `ProgramQueryLedger(..., budget=QUERY_BUDGET)` with the MODULE CONSTANT
  `QUERY_BUDGET = 1000`, and computes the AUC with `budget=QUERY_BUDGET` too. The string
  `contract["budget"]` appears nowhere in the module, and `modal_apps/pmo_population_v1_app.py` has
  zero occurrences of `budget`. `run_program_campaign`'s only call-count authority is
  `ledger.remaining`; `stagnation_rounds=None` disables the stagnation break, leaving only a
  wall-clock guard. 64 rounds x 16 queries + 16 init = 1040 > 1000, so the ledger binds at 1000.
  **Consequence: the "3x250" pilot would charge up to ~3,000 oracle calls, 4x its declared 750, and
  report `auc_top10_development_1000` normalised by 1000 rather than 250.** The launcher's budget
  check is a contract-consistency check, not an enforcement. LESSON: when a re-budget moves a
  contract, grep the RUNTIME for the key that contract sets; a fail-closed chain that validates a
  number nothing consumes is fail-closed about the wrong thing.
- **`top_auc` trapezoids up from (0, 0), so a short-budget AUC is structurally DEPRESSED.** Closed
  form, verified exactly against the production `pmo_top_ten_auc`: a run holding a constant top-10
  level `c` scores `c * (1 - frequency / (2 * budget))`. At the official 10,000-call budget that
  removes 0.5% of the level; at 1,000 it removes 5%; **at 250 it removes 20%**. So comparing a
  250-call AUC against a published 10,000-call baseline does not merely compare different budgets,
  it understates COMPOSE by a fifth at identical performance -- the error points toward a false
  NEGATIVE. Also note the log grid: at budget 250 with frequency 100 the metric evaluates at only
  (100, 200, 250), three points against the official 100.
- **PyTDC does NOT define the PMO suite and ships NO oracle-direction metadata.** There is no `pmo`
  benchmark group in `tdc/benchmark_group/`, and no `direction`/`higher_is_better` field anywhere in
  `metadata.py`. PyTDC supplies the oracles; the 23-task membership comes from the PMO paper. Take
  suite membership from the published transcriptions (`docs/invirtuogen_pmo_targets.json` and
  `docs/genmol_pmo_targets.json` agree exactly, 23/23), and treat "maximize on [0,1]" as the
  PMO/mol_opt convention -- independently enforced by `pmo_top_ten_auc`, which rejects any reward
  outside [0,1].
- **Verify oracle names WITHOUT constructing an Oracle.** `Oracle.__init__` calls
  `fuzzy_search(name, oracle_names)`, which returns the name unchanged when it is an exact lowercased
  registry member and otherwise falls back to a Levenshtein match at **threshold 0.8** -- a near-miss
  name does not raise, it silently resolves to a DIFFERENT oracle, and `sitagliptin_mpo_prev` /
  `zaleplon_mpo_prev` sit one token from two PMO tasks. So exact membership in
  `tdc.metadata.oracle_names` is equivalent to resolution, and can be checked by loading
  `tdc/metadata.py` standalone (stub `pkg_resources`; it is imported but unused by the registry
  lists). Constructing the oracle would download predictor pickles for drd2/gsk3b/jnk3.
  `tools/verify_pmo_task_registry.py` does this offline from a cached wheel; all 23 resolve exactly
  against PyTDC 1.1.15, the version `pmo_population_v1_app.py` pins.
- **`gsk3b: 0.952` is an unsourced promotion target pinned in three contracts.** The other two IVG
  targets match `docs/invirtuogen_pmo_targets.json`'s no-prescreen table exactly (celecoxib 0.798,
  perindopril 0.645), but **gsk3b is ABSENT from that table** -- its prescreened value is 0.988 --
  and 0.952 appears nowhere in `docs/`. Promotion requires beating 2 of 3, so an unsourced number can
  decide it. Related: the published no-prescreen column sums to 16.676 over 23 tasks but only **7 of
  23** per-task values are transcribed, so no 23-task suite-sum comparison is possible yet. Record
  ABSENT rather than substituting the prescreened value; the two columns differ by a 250,000-call
  prescreen.

## 2026-09-20 (an asset-backed oracle can fail silently for a whole budget)

- **A TDC oracle that CONSTRUCTS is not an oracle that SCORES, and the gap cost a full 250-call
  budget.** `Oracle("gsk3b")` loads its RandomForest **lazily on the first call**, from the
  **relative** path `oracle/gsk3b_current.pkl`. `modal_apps/pmo_population_v1_app.py` chdirs into
  the assets directory only around CONSTRUCTION and restores cwd in a `finally`, so every call
  resolves that path against `/root`, raises `FileNotFoundError`, and TDC's `Oracle.__call__` has a
  **bare `except:`** returning `default_property` = **0.0**. The module global is never set, so the
  failing load repeats and is swallowed forever. Counterfactual on five known actives, same oracle
  object / image / asset, differing only in cwd at call time: production pattern `[0.0]*5`, cwd held
  during the call `[1.0]*5`. The asset was innocent -- sha256 matched the contract and the same
  pickle loaded directly reproduced the frozen reference exactly.
- **The zero-call smoke could not catch it, by construction.** The smoke set `oracle_called: False`
  deliberately to avoid spending budget, and the defect lives in the lazy load on FIRST CALL.
  "The oracle imports" was verified; "the oracle scores" never was. **FIX: any asset-backed oracle
  needs a positive-control assertion before the first charged call -- a known ACTIVE must score > 0.**
  A silent-by-construction defect is only caught by a nonzero expectation.
- **Do not read an all-zero ledger as self-evidently broken.** On the frozen 400-molecule panel
  **34% of random ZINC-like molecules genuinely score exactly 0.0** on this oracle (mean 0.028),
  while **0% of known actives do**. The ledger distribution is NOT the tell; the positive control is.
  Timing was a real signal though: 0.09-0.60 ms is parse + fingerprint + a FAILED `open()`, far too
  fast for a 100-tree depth-78 forest (perindopril_mpo's 1.1 ms is genuine descriptor work).
- **INFERRED blast radius:** `jnk3` and `drd2` share the relative-path lazy-load idiom and are
  exposed to the same construct-then-restore-cwd pattern. Not measured. This bites the 5-task rung
  (jnk3) and the 23-task rung. `perindopril_mpo` and `celecoxib_rediscovery` are pure-RDKit
  evaluators with no asset file and no cwd dependency -- measured unaffected.

## 2026-09-20 (PMO binder: the 96.7% failure decomposed; and a worker leak that corrupts timing)

- **The single reason string `"joint plan has no legal binding on this parent"` conflated three
  outcomes, and the beam-width/ranking hypothesis was the WRONG one.** Decomposed with an
  exhaustive referee (beam removed, deduped on the full determinant of a prefix's future --
  state + created-handle map + next ordinal -- so collapsing it preserves the EXISTENCE answer):
  **(a) 66.9% of all 2,185 (plan, parent) pairs are PROVEN unbindable before any search**, in 1.0 s,
  because role 0 demands operand descriptors no atom of the parent carries. That caps the achievable
  binding rate at 33.1% for any beam width, any ranking, any budget. Among step-0-feasible
  production pairs the referee still proves no-binding for 27/30, and widths 4->8->16->32->64 all
  return **0.000**. (b) DISCARDED_BY_SEARCH is real but marginal: 2 of 80 pairs -- and widening the
  beam did NOT recover them, the wider arms exhausted their wall budget first. (c) budget exhaustion
  dominates the teacher strata. **Do not size a search-side repair before measuring how much of the
  failure is reachable by search at all.**
- **Under a fixed wall budget a wider beam is strictly NEGATIVE here**: complete programs fall
  17 (w4) -> 12 (w8) -> 6 (w16/32/64). Breadth is paid for in the time the arm does not have.
- **The one intervention that moved the rate is not a fix, and the yield metric is what caught it.**
  Compatibility ranking (rank survivors by an exact one-step-lookahead operand-availability count
  instead of the content hash) gave +82% complete programs at w4 -- at identical median scale (16.0),
  identical max (23) and identical retained fraction (**1.000**, i.e. purely additive, removing
  nothing) against teachers at 29-40 primitives and 0.52-0.96 retained. **Zero programs at teacher
  scale in ANY arm.** Always report realized program scale and retained fraction beside a binding
  rate; a rate that rises while scale does not has not solved the problem.
- **The defect is OVER-SPECIFICATION in the plan representation, and it attributes to ONE field.**
  Projecting the operand descriptor: dropping `neighbor_element_histogram` lifts step-0 feasibility
  **33.1% -> 84.4% (2.55x)**, while dropping `implicit_hydrogens` or `bond_class_histogram` changes
  *nothing*. A v1 role pins each operand's complete elemental neighbourhood -- a whole-environment
  fingerprint -- and 63 of 95 plans additionally pin the teacher's step ORDER through
  `created_ordinal`/`creation_lag`. NB this projection is a DIAGNOSTIC: a match under a coarser view
  realizes a *different* transformation, so it locates the constraint, it does not license relaxing it.
- **`pkill -f <script.py>` does NOT kill ProcessPoolExecutor children under the spawn start method.**
  Their cmdline is `python -c from multiprocessing.spawn import spawn_main...`, which the pattern never
  matches, so they orphan to ppid=1 and keep burning ~30% CPU each. Ten orphans from two killed runs
  silently halved the throughput of the next run and contaminated its wall-clock timings. Kill by
  `ppid`, or `pkill -f "from multiprocessing"` as well -- and report `successors_enumerated` (or any
  load-independent work counter) as the cost metric, never seconds alone. Generalizes the 2026-07-21
  `tracelet_sampling_worker` note to every spawn pool.
- **A wait loop of the form `until ! pgrep -f "<name>"; do sleep; done` matches its OWN command line
  and never exits.** Poll for the artifact the job produces instead of for the absence of a process.

## 2026-09-20 (PMO oracle assets: a complete 250-call ledger of swallowed defaults)

- **A relative asset path plus a lazy load plus a bare `except` is a silent zero
  generator, and a constructor-scoped `chdir` does not fix it.** PyTDC's `gsk3b` and
  `drd2` evaluators open `oracle/<name>.pkl` **on the first CALL**, cached in a module
  global, and `tdc.Oracle.__call__` wraps the evaluator in a bare `except` returning
  `default_property == 0.0`. The PMO worker entered the asset directory only around
  `Oracle(name=...)` **construction** and restored cwd in a `finally`, so every call
  raised `FileNotFoundError` from `/root`, was swallowed, and — because the module
  global is only set on success — the failing load repeated forever. Result: a gsk3b
  task that charged 250/250 calls on 250 distinct molecules and recorded `best_score
  0.0`, with nothing in the artifact to distinguish it from a real result. MEASURED
  counterfactual, same oracle/image/asset, differing only in cwd at call time: `[0.0]*5`
  vs `[1.0]*5` on five known actives.
- **The invariant is over the oracle's LIFETIME, not its constructor.** Fix =
  `compose_v4.experiments.pmo_oracle_assets.AssetPinnedOracle`: pin the working
  directory around **every call**, and `prime()` the lazy load inside that window so a
  caching evaluator never touches the filesystem again. The two are deliberately
  redundant — priming covers the cached case, per-call pinning covers evaluators that
  do not cache — and the working directory is restored in a `finally` so a raising call
  cannot strand the process.
- **Construction success is not evidence, and `score > 0` is not evidence either.**
  `pmo_environment_smoke_app.py` records `oracle_called: False` **by design** and passed
  throughout the defect's life, because construction never triggers a lazy load. The
  replacement gate CALLS each asset-backed oracle against pinned reference molecules
  with **expected values and tolerances**, including graded intermediates — a
  constant-0.87 oracle passes every nonzero check and fails this one. Measured in the
  pinned image: gsk3b 13/13, jnk3 10/10, drd2 6/6, **max_abs_delta 0.0** against the
  frozen 400-molecule panel. The control runs BEFORE `started.json`, so a failure leaves
  the task re-runnable instead of burning its one no-retry attempt.
- **AUDIT, all 23 PMO tasks, MEASURED against PyTDC 1.1.15 in the pinned image**
  (`diagnostics/pmo_oracle_asset_audit_v1.json`; 20 of 23 are pure RDKit and open no
  file):
    gsk3b  `gsk3b_current`  LAZY relative at CALL   cwd-dependent TRUE   control 13/13
    drd2   `drd2_current`   LAZY relative at CALL   cwd-dependent TRUE   control 6/6
    jnk3   `jnk3_current`   EAGER relative at CONSTRUCTION  cwd-dep FALSE  control 10/10
  **The pre-diagnosis inference that jnk3 shared gsk3b's failure mode is FALSIFIED.**
  `class jnk3.__init__` loads the pickle eagerly and caches it on the instance, so it
  raises `FileNotFoundError` at construction rather than returning a swallowed score,
  and a correctly constructed jnk3 is not cwd-dependent at call time. Lazy and eager are
  different exposures: only the lazy form can manufacture a plausible ledger. drd2, by
  contrast, was confirmed — it was INFERRED and is now MEASURED.
- **To audit a dispatch, run the shipped dispatch — but stub the NETWORK, not the
  resolver.** The first pass stubbed `oracle_load` itself; it returns the *exact asset
  name* (`gsk3b` -> `gsk3b_current`) to `Oracle.__init__`, so stubbing it returned None
  and all three asset-backed tasks failed to construct — reported as `asset_backed: []`,
  i.e. exactly "nothing is affected". Stub `dataverse_download` instead. Second trap:
  TDC evaluators are a mix of plain functions and **instances** of callable classes;
  `inspect.getsource` on an instance raises, which silently produced an empty call graph
  and classified every class-backed oracle as asset-free. Resolve an instance to
  `type(obj)` before taking source.
- **Void per TASK, never per run.** The same run's `perindopril_mpo` (0.486) and
  `celecoxib_rediscovery` (0.196) use pure-RDKit evaluators with no asset and no cwd
  dependency, measured unaffected. Voiding the run would have discarded two sound
  measurements; voiding nothing would have kept a defect as a published number.
  `diagnostics/pmo_run_status_ledger.json` + `compose_v4.experiments.pmo_run_status`
  record the exception and `assert_admissible` **raises** — a warning beside a plausible
  score gets read as a caveat and the score gets quoted anyway.
- **Retrospective rescoring is a counterfactual, not a correction.** An independent
  parity-validated forest scored 196/250 of the voided endpoints nonzero (max 0.18, mean
  0.039; 11 of 16 initialization molecules at 0.01-0.08). That bounds the signal the
  defect hid. It is NOT the corrected trajectory: FiberControl never observed those
  values online, so every selection and credit assignment in the voided run was made
  against a constant, and the corrected run's policy diverges from the first decision.
- **Three latent defects in the PMO launch chain, all found by exercising it rather than
  reading it.** (1) `prepare_pmo_250_pilot.py` wrote the authorization receipt AFTER the
  capsule manifest that pins it, and `authorized_at_utc` moves every run, so the
  launcher's capsule check could never pass — write the manifest LAST and re-hash both
  files it pins. (2) It archived superseded receipts into a fixed
  `dead_1000call_attempt/`, so a second supersession would have overwritten the first
  attempt's record; archive under the payload hash the RECEIPT itself carries, not the
  contract's current hash (they differ whenever a re-seal ran first). (3) The capsule's
  own `contract_payload_sha256` was never updated and addressed a payload that no longer
  existed — an unresolvable pin that reads as verified.
- **Anything on the scoring path must be in the contract's `implementation_sha256`.**
  `pmo_oracle_assets.py` decides whether a number is the oracle's output or PyTDC's
  swallowed default, so it is pinned in both the contract and the source capsule. The
  chain caught the worker edit immediately and correctly: `load_contract` refused with
  `input identity mismatch` before the free-oracle gate could run.

## 2026-09-20 (campaign state: locks are authority, checkpoints are not)

- **`modal volume get <vol> <dir> <dest>` SILENTLY COLLAPSES a directory onto one path when
  `<dest>` does not already exist as a directory.** Twenty-one files land on the single path
  `<dest>`, the CLI prints `OK Finished downloading files to local!` and exits **0**. This
  produced two FALSE readings in one session -- a "no better score" verdict that was the exact
  opposite of the truth, and a `json.JSONDecodeError: Extra data: line 2` that looked like a
  schema problem. **Pre-create the destination directory**, and verify every file is present,
  non-empty and parseable BEFORE computing any verdict. Also never parse `modal volume get ... -`
  from stdout: the CLI mixes its success banner into the stream.
- **A checkpoint is overwritable; a round lock is not.** `run_cell` in the T4 apps reads
  `result.json` and `checkpoint.json` off the mounted volume with **no `volume.reload()`** (the
  only reload sits in the driver, and the ten `t4_integrated_route_fiber_held_*_app.py` wrappers
  have none at all). A Modal volume shows a snapshot from mount time, so a container that restarts
  after preemption reads a STALE view, resumes from an older checkpoint and overwrites the newer
  one. Measured: `parp1_0` d0.6 locks prove -12.6 at 145 charged calls while its checkpoint read
  -11.4 at 129 -- a verdict flip from a 0.90 loss to a 0.30 win. `parp1_2` sat **eleven rounds**
  behind with `rounds == [1, 7, 8]`. `_resume_state` already rebuilds the archive from locks
  correctly; it failed only because it globs them off the same stale mount.
  **Reconstruct final results from locks: a round lock carries, per query, the docked
  `parent_score` of the parent it came from, which is direct evidence the molecule was scored.**
- **`reconciled_charged_calls` is a LOWER BOUND in both directions.** Locks are written per round
  INDEX, so a round that rolled back and was redone rewrites its own lock with a smaller
  `charged_before`. Check ladder monotonicity per cell (0 of 45 were overwritten in the one sweep
  that checked); where a cell rolled back and advanced past it, true lifetime spend exceeds every
  surviving artifact.
- **`retries: 0` means a preempted container STOPS and waits for a human** -- it does not silently
  restart from its baked image. With detached ephemeral apps this is invisible: the app still reads
  as alive. Watch for `tasks == 0` on an ephemeral app, and never assume a campaign is progressing
  because its app exists.
- **Never cache `checkpoint.json` or `result.json`.** They are mutable committed state; a cached
  copy reported a stale best. Caching the mutable state is the same error class as the overwrite
  defect being measured.
- **`compose-t4-held-target-distilled-jak2-d06-250` is the delta=0.4 arm** despite its name -- its
  frozen contract declares `delta: 0.4`. Always read delta from the contract, never the volume name.

## 2026-09-20 (T4 failure decomposition: three mechanisms, one of them a vocabulary boundary)

- **The five T4 no-distill failures are THREE different problems, and only one is
  exhaustion.** Decomposed against each cell's own witness, so these are per-cell
  diagnoses and not an aggregate story:
  - `braf_0`, `braf_1`, `fa7_2` -- **exhaustion, repairable.** The witness is already
    inside the v1 proposal support at mass ranks 309/319, 46/157 and 145/247.
    `MAX_SEGMENT_LENGTH = 8` splits one chemically coherent 11-14 atom excision into 2-3
    independent UNIFORM UNCONDITIONED region draws, and mass decays ~10x per extra module.
    Minimal supported change: ONE draw over bridge-separated substituents of any size,
    weighted by the free gate margin of the child. Measured: braf_1 rank 46->1,
    braf_0 309->1, fa7_2 145->2, no control regressing. Raising the size cap ALONE
    degrades a working control (fa7_1 rank 2->6, mass -38%) and conditioning ALONE leaves
    braf_0 at 109/319, so it is the joint change or nothing.
  - `5ht1b_2` -- **outside the vocabulary, NOT repairable by budget.** Its witness needs a
    net formal-charge change, and across all 482 enumerated single-edit actions in all
    five families ZERO change net charge. Verified structurally at three load-bearing
    points, not inferred from the census: `factorized_fiber.py:397` RAISES on any charged
    state ("factorized fiber supports only neutral atom states"); every atom the
    enumerator creates is hardcoded `formal_charge=0` (:154,:171,:198); and
    `source_corruption.py:209` rejects any candidate whose successor changes net charge,
    universally, so a neutral molecule cannot silently gain charge either. This is the
    same charge-PRESERVING scope decision locked on 2026-07-26/27 -- the dense edit heads
    encode only neutral `(element,valence)` classes -- now surfacing as a T4 ceiling.
    SCOPE THE CLAIM, do not spend the ladder on it: the axis is absent by construction.
    CAREFUL: this says the axis THAT WITNESS needs is unavailable, not that no neutral
    candidate could ever pass; 5ht1b_2 is also SA-bound.
  - `fa7_0` -- **not a proposal failure at all.** Best QED margin over 1,537
    similarity-passing endpoints is -0.0017. The cell is QED-bound, so a better search
    finds nothing; only a different objective trade would.
- **A matched pair that looks like a mechanism can be Poisson noise.** `braf_1` and
  `braf_2` are identical on every categorical axis measured (binding gate, depth, min cut
  size, attachment degeneracy, single-edit eligibility) and differ 1.31x in expected yield
  (1.92 vs 2.51 per 480 draws). Observed 0 vs 3 has P(0) = 0.147 -- unremarkable. Do not
  build a mechanism on a 0-vs-3 contrast without the yield model.
- **Growth-only rescue experts cannot rescue a size-capped cell.** `braf_0_d06` allocates
  ZERO plans under the v4 composer because at 40 heavy atoms it has no growth headroom and
  every v4 mode is growth. That is consistent with its witness requiring an 11-14 atom
  EXCISION. Check headroom against the expert's modes before costing a rescue arm.

## 2026-09-20 (CORRECTION: the T4 failure decomposition was right about mechanism, wrong about scope)

- **CORRECTS the entry above ("5ht1b_2 -- outside the vocabulary, NOT repairable by budget").
  That claim was TOO STRONG and the mechanism was different.** Measured witness census over all
  five exhausted delta=0.6 cells: 5ht1b_2's source carries net charge **+1** and all FOUR of its
  clean eligible witnesses carry net charge **0**, at heavy-atom deltas -8 to -7. The net-charge
  change is real, but it is achieved by **EXCISING the [NH+]-bearing fragment** -- ordinary atom
  deletion, already in the vocabulary -- not by a charge-state or protonation edit. So the cell is
  reachable without any new action axis.
  **How the error was made, because the shape recurs:** three code facts were verified correctly
  (`factorized_fiber.py:397` raises on charged states; created atoms hardcoded `formal_charge=0`;
  `source_corruption.py:209` rejects net-charge-changing successors) and then generalized from the
  EDITING vocabulary and the corruption/training path to a claim about what the **T4 proposal path**
  can reach. Verifying a guard exists is not the same as establishing that the guard is the binding
  constraint on a different code path. A direct measurement on actual witnesses beat the inference.
  Reaffirms the standing rule: separate MEASURED from INFERRED, and never state an inference in the
  voice of a measurement.
- **The direction mismatch is 5 of 5, not 3 of 5 -- braf_0 was never a special case.** Every clean
  eligible witness in EVERY exhausted cell is a net EXCISION:
      braf_0   12 witnesses  -15..-13      fa7_2    12 witnesses  -13..-9
      braf_1   12 witnesses  -15..-12      5ht1b_2   4 witnesses   -8..-7
      fa7_0     5 witnesses   -9..-8
  Not one is a growth. `generic_feasibility_headroom_v4` is growth-only in every `macro_mode`
  (`direct_ring_growth`, `grow_then_close`, `grow_then_ring`, `replace_then_ring`,
  `double_ring_growth`, `grow_then_double_ring`), so a rescue built on it is directionally wrong for
  all five. It allocates 0 plans on braf_0 only because braf_0 sits at the 40-atom ceiling with zero
  headroom; the other four get 8-24 plans that point the wrong way, which is WORSE than zero because
  it looks like coverage. **A rescue arm must be checked for DIRECTION against the witnesses, not
  just for non-empty plan allocation.**
- **fa7_0's "QED-bound, not a proposal failure" reading also weakens.** It has 5 clean excision
  witnesses at -9..-8 heavy atoms. The -0.0017 QED margin was measured over similarity-passing
  endpoints the CURRENT proposal law reaches; it does not bound what a large-excision law reaches.
  A bound computed inside a restricted support is a statement about that support, not about the cell.

## 2026-09-20 (T4 objective refinement: the comparator is settled, and what it cost to settle)

- **The T4 comparators are NON-STRICT and correct as shipped, and chasing them recovers NOTHING.**
  `t4_fiber_campaign.py:153` is `if similarity < self.delta or quality < QED_MIN or access >
  SA_MAX: return None` with `QED_MIN, SA_MAX = 0.6, 4.0` (:58). The strict `>`/`<` form described
  in the `Fiber` docstring is HISTORICAL, not what runs. Re-scoring every stored endpoint of all
  15 held-target cells under both comparators gives `comparator_only_gain = 0` in EVERY cell.
  I hypothesised a free cell recovery here on the strength of a -0.0017 margin; it is a clean
  decisive negative. Do not re-open it.
- **BUT the similarity comparator IS load-bearing, and QED/SA are not.** 45 fa7_0 endpoints
  (6-129 across the panel) sit at similarity EXACTLY 0.600, because Tanimoto is a ratio of small
  integers and lands on rational bounds routinely. ZERO endpoints anywhere sit on the QED or SA
  bound -- those are continuous and never do. So changing the similarity comparison to strict
  would silently delete real endpoints in every cell, while the other two are inert. Asymmetric,
  and not guessable from the code.
- **fa7_0's QED ceiling is ONE functional group, and it is the pharmacophore.** `QED.properties`
  reports ALERTS=2 on both the source and its best endpoint, from two SMARTS
  (`[C&!R]=[N&!R]`, `N=[C&R0][N,n,O,S]`) that both fire on the SAME acyclic amidine -- the FA7
  S1-pocket binding group. ALERTS carries QED's largest weight (0.95 of 3.92), worth x1.41. Every
  alert-relieving edit LEAVES the delta=0.6 ball: unconstrained Pareto max QED 0.901, constrained
  to sim >= 0.6 it is 0.633. **Quote both numbers or the headroom is fiction.**
- **Padding is load-bearing on a REAL rescue, not just in principle** (sharpens the 2026-08-08
  entry). braf_0's only known one-edit rescue is an `atom_insert` -- exactly the family a tight
  graph deletes. Measured on the fa7_0 leader: tight 24 slots -> 428 marks, **0** `atom_insert`;
  padded 40 -> 733 marks, **305**. A tight-graph probe would have reported that cell unrescuable.
- **GENERALISING A SELECTOR FROM THE CELL THAT MOTIVATED IT IS THE TRAP, EVEN WHEN THE RULE IS
  PERFECTLY PREDICTIVE THERE.** "Endpoint fails only QED" is a flawless predictor on fa7_0 (5 fire
  -> 5 lift, 20 do not -> 0 lift). braf_1 refutes it: a violation-ranked panel lifts 5 where the
  QED-only panel lifts 1, and braf_0 fires the QED label ZERO times in its whole panel yet still
  yields a rescue. A rule validated on one cell is a description of that cell.
- **A cell can hold eligible molecules ONE legal rewrite outside its reachable support.** None of
  the four eligible successors of fa7_0's leader appear among the 9,527 distinct endpoints the cell
  generated, and `round_one_decision.selected = 0` means that set IS the campaign's entire output.
  Independent corroboration, from the small-edit direction, of the same support gap the witness
  census found from the large-excision direction -- and the two witness pools look DISJOINT, so a
  repair validated on one does not automatically cover the other.
- **Report benchmark eligibility and chemical plausibility as two numbers.** fa7_0's 15 eligible
  endpoints are 14 distinct, of which ~4 are conventionally drug-like; the rest are exocyclic
  quinoids and strained azirines that pass the thresholds, `med_chem_gate` AND `LEGACY_SCREENED`
  (15/15 survive). Eligibility is a floor, not a score.

- **"48" and "40" are DIFFERENT QUANTITIES in the T4 path and two agents stated them as competing
  rules.** `pad_molecular_graph(smiles_to_molecular_graph(parent), 48)`
  (`t4_fiber_campaign.py:257`) sets the SLOT CAPACITY of the proposal source array;
  `REPRESENTABLE_HEAVY_ATOMS = 40` (:62), enforced at :148 by `if heavy > REPRESENTABLE_HEAVY_ATOMS`,
  is the HEAVY-ATOM CEILING that `Fiber` applies to ENDPOINTS. Both hold at once: pad the source to
  48 slots, and the fiber refuses any endpoint above 40 heavy atoms. "Pad to 40, not 48" and "the
  executor requires 48" are each locally true and read as a contradiction, which is how a load-
  bearing detail gets propagated wrong. Say which quantity you mean every time.
- **A T4 endpoint can parse in RDKit and still be unbuildable in COMPOSE.** `[CH]c1ccc(...)` is a
  real radical: RDKit accepts it, the COMPOSE executor cannot construct it. RDKit-parseability is
  not a COMPOSE-validity check, so a probe that filters on `MolFromSmiles is not None` will admit
  endpoints the production path refuses -- and `t4_fiber_campaign` documents that the intervention
  layer still emits radicals, which is what `_rebalance` exists to prevent.

## 2026-09-20 (branch health on t4-objective-dynamic-reset: large, pre-existing, and worth stating)

- **`pytest tests/` on this branch is 292 failed / 5,806 passed / 101 errors**, and the failures
  are pre-existing rather than introduced. Attributed three ways rather than asserted: (1) every
  test file whose import or config-read closure touches the changed PMO modules -- 13 files, 122
  passed / 8 failed; (2) those same 8 reproduce IDENTICALLY at baseline in a clean detached
  worktree, same names and counts, and live in `test_pmo_dynamic_v21*.py` whose config was never
  resealed; (3) the bulk is the editing-V2 family cascading from one collection error, "semantic
  capability registry bindings have drifted", with the two largest contributors running 4 passed /
  41 errors at baseline too. Structural backing: `_PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS` is an
  explicit file list, not a glob, so new `experiments/` modules cannot move the process identity.
- **`scripts/prelaunch_gate.py` does NOT pass on this branch**: it lints `src/`, which carries ~314
  pre-existing ruff findings on the current ruff version (newer rules -- RUF022/UP035). This is NOT
  the gate for the T4 held-target launches; those are gated by `tools/preflight.py` (drift=0) plus
  the app's own `_local_task` runtime-input verification, which is what actually pins the 14 runtime
  inputs. Know which gate governs which launch path before treating a red gate as a blocker -- or
  before treating a green one as coverage.
- **Reporting a suite as "one known failure" when it is 292/101 is an understatement that changes
  the decision**, even when the attribution is correct. State the magnitude and the attribution
  together; the attribution is what makes it safe, not the size.
- **Two harness traps that cost a full-suite run each.** `cmd 2>&1 | tail -N` as a background task
  keeps only N lines, so 292 failures cannot be classified afterwards from a 30-line tail. And a
  buffered pipe loses EVERYTHING if the job is killed -- a re-run died at exit 144 after 9 minutes
  having written 0 bytes because `grep` buffers. Redirect raw output to a FILE and filter at read
  time. Generalizes the 2026-07-29 `nohup` lesson to piped background jobs.
- **PARKED, needs an owner decision: `live_parent_support_gate_sha256` is an unresolvable pin.**
  Both scored PMO contracts pin `510f6920...` for `configs/pmo_population_live_parent_gate_v2.json`,
  and that value matches neither the file's current bytes, nor its current payload hash, nor either
  of those at baseline -- it was already stale before this session's work, and NO Python reads it.
  Correctly left unguessed: the field name carries neither the `_file_sha256` (physical bytes) nor
  the `_semantic_sha256` (self-hash) convention, and writing the wrong role into a pin is its own
  failure mode, per the 2026-08-02 entry on exactly that corruption.

## 2026-09-20 (rdkit parity holds on real endpoints and BREAKS on states the search invents)

- **Two environment-parity checks in this session appeared to contradict each other; both are
  right, and the distinction is decision-critical.** The support-stage audit compared rdkit
  2025.09.6 against the pinned 2024.3.5 on **8,183 real endpoints from a completed shard** and got
  byte-identical similarity, QED, SA, heavy count and `med_chem_gate` verdict. The Modal fan-out
  agent compared 2026.03.6 against 2024.3.5 on the **T4-v2 feasibility grid** and found the two
  kernels write DIFFERENT canonical SMILES for the same molecule (same InChI) on a Kekule-
  degenerate hypervalent-sulfur ring **the search itself constructs**, diverging 105 of 600
  evaluations. RESOLUTION: parity on drug-like molecules that already exist says nothing about
  parity on the exotic intermediates a search builds. Validate the kernel on the states the
  PROCEDURE generates, not on the corpus it starts from.
- **Under the pinned kernel one T4-v2 cell cannot be computed AT ALL.** `parp1_2` dies in every arm
  inside `Chem.MolToSmiles` with `RuntimeError: Invariant Violation / could not find atom1 /
  Canon.cpp:222` (RDKIT 2024.03.5), raised from `t4_v2_feasibility_proposal::_canonical` via
  `replace_moves`. On rdkit 2026.03.6 the same run completes and reports `parp1_2` as an ORDINARY
  RESULT. So the newer kernel silently tolerates a state production refuses -- the dangerous
  direction, because it manufactures a plausible row rather than an error.
- **Canonical SMILES is a CACHE KEY here, which turns a cosmetic difference into a permanent
  divergence.** The search keys its endpoint cache on the canonical string, so two kernels that
  merely SPELL a molecule differently explore different trees from that point on. A "same molecule,
  different string" difference is not cosmetic in any procedure that dedupes on the string.
- **CONSEQUENCE, and it is a live claim-validity question, not a throughput one:** any T4-v2 row
  produced by a LOCAL run was computed under a kernel that diverges from production. The campaign
  docking results are unaffected -- those run in the pinned Modal image and are reconstructed from
  round locks -- but locally-produced gates, feasibility grids and proposal-rank tables are exposed
  and must be re-run pinned before they support a launch or a published number. Rebuilding the
  pinned env costs about a minute: `uv venv --python 3.11` plus rdkit==2024.3.5, numpy==1.26.4,
  scipy==1.13.1, networkx==3.3, and torch==2.4.0 (needed only because `t4_fiber_campaign` imports
  it transitively -- but omitting it forces you to TRANSCRIBE the gate instead of importing it, and
  a transcribed gate cannot fail usefully).
- **A metric that cannot vary is not a measurement.** A container census built on
  `os.uname().nodename` reported the literal string "modal" in every container. Check that a
  diagnostic's value is capable of differing before drawing an inference from its uniformity.
- **The T4 Modal workspace is container-starved and a sibling workspace is not.** A chemistry-free,
  volume-free scaling probe (40 tasks x 20 busy-seconds, same code, minutes apart) measured
  `nitya` at 3 containers / 507.7 s against `rahul-94866` at 40 containers / 34.2 s -- **14.8x**.
  The running campaigns hold `nitya`. Run zero-oracle side work on the other workspace, and re-run
  the probe (about two cents) rather than assuming capacity.
## 2026-09-20 (T4 region draw: the cap and the uniform weight are ONE joint defect)

- **`MAX_SEGMENT_LENGTH = 8` is not a size preference, it is an EXPRESSIBILITY bound, and the
  uniform weight beside it is the other half of the same defect.** `_delete_pendant_fragment`
  draws uniformly over bridge-separated fragments of at most 8 atoms, so a coherent 9-15 atom
  substituent is not drawable at all -- it must be spelt as 2-3 INDEPENDENT bounded cuts whose
  joint mass is a product of per-module terms. Every clean eligible witness of all five exhausted
  delta=0.6 cells is a net EXCISION of 7-15 heavy atoms (57 witnesses, production provenance),
  which is exactly the move class the cap cannot express in one draw.
- **MEASURED, both halves are individually insufficient and the joint change is not.** Pure-prune
  closure, horizon 3, best-eligible mass rank (the shipped `BridgeRegionLaw` drives the harness,
  nothing is transcribed):
      cell      role     v1        cap_only   conditioned_only   repair
      braf_0    failed   161/165   22/347     88/165             1/307
      braf_1    failed    49/101   17/272     10/101             1/264
      fa7_2     failed   189/276  110/546     23/276             9/492
      braf_2    control   35/83    14/203     24/83              1/192
      fa7_1     control    2/258   22/438      1/254             1/379
      5ht1b_0   control    1/15     3/16       1/15              1/16
  `fa7_1` is the decisive control: raising the cap ALONE demotes it 2 -> 22 (eligible mass share
  0.146 -> 0.052), reproducing the known cap-raising regression; the joint change instead promotes
  it to 1 at share 0.377. No control regressed under the joint change.
- **A single-draw delete-half measurement separates the law from the rest of the program.** For
  replace/grow-mode witnesses the region draw is only the delete half, so ask which rank a region
  whose child is a SUBSTRUCTURE of the witness gets. v1 covers **0 of 41 witnesses across all five
  cells** -- the cap makes every needed region undrawable in one draw -- while the joint law covers
  all of them for four cells at ranks 2-4. This is what rescues fa7_0, whose pure-prune closure has
  no eligible endpoint at all but whose five (complete-pool) witnesses all have their delete half at
  rank 2.
- **The T4 proposal path is 48 SLOTS, not the 40 of the editing corpus.** `whole_ring_plan`
  refuses anything else outright (`n_atoms != 48 or not 1 <= n_real_atoms <= 40`), and
  `t4_fiber_campaign` builds every parent as `pad_molecular_graph(smiles_to_molecular_graph(p), 48)`.
  So `assert_production_state_semantics` / `production_state_from_smiles` (40-slot, editing corpus)
  is the WRONG preflight for this path -- it constructs a state the T4 executor rejects. The
  equivalent evidence here is that production actually executes the states, which is cheap to check.
- **A pendant excision's endpoint has an exact closed form, and the executor is the only thing
  that may certify it.** Deleting a bridge-separated fragment leaf-by-leaf returns hydrogens to
  atoms that are themselves deleted, so the only surviving effect is that the retained anchor
  recovers the bridge bond's hydrogens. Measured over 5 real sources: **72 regions executed,
  0 endpoint mismatches**. Keep the closed form for weighting hundreds of candidates cheaply, and
  keep a test that drives the LIVE `_delete_pendant_fragment` -- a transcribed reference could not
  fail if the executor drifted.
- **5ht1b_2 is blocked by the EXECUTOR's charge policy, not by the region law, and needs a
  DIFFERENT region shape as well.** Two independent facts, both measured:
  (a) its witness `C1=CC2=NC=C(CCCc3ccccc3)[C@H]2C=C1n1cnnc1` is a single **8-atom interior
  (two-bridge) excision plus one reattachment bond** joining the two flanks -- reached exactly,
  once stereochemistry is stripped (MolecularGraph carries no stereo, so a stereo-bearing witness
  SMILES never matches a graph-derived one; the contract sets `stereochemistry_claim: false`).
  A one-bridge pendant draw cannot express it, and neither can any deletion-only search, because
  the remainder would be disconnected.
  (b) even given that region, `whole_ring_plan.execute_program` calls `charge_policy_preserved` at
  EVERY step, and `audit_charge_policy_transition` reports `formal_charge_deleted_slots=(12,)` plus
  charged-centre element/H/bond-row changes, so deleting the `[NH+]` is refused outright. Measured
  on the real source: **9 of 16 pendant regions are refused by the charge policy**; zero refusals on
  every neutral cell. Also `bond_insert` is OUTSIDE the frozen Active8 codec surface
  (`atom_delete, atom_insert, atom_restate_semantic, bond_reorder, bond_reroute, cycle_close,
  cycle_open, ring_system_restate`), so the reattachment primitive is `cycle_close`.
  Conclusion: a region law is the wrong layer for this cell; the repair is a charge-policy decision
  plus a two-boundary region shape, and both should be named rather than folded into a size claim.
- **Keep the tilt a RE-RANKING, never a filter.** Weights are `max(floor, exp(margin/T))` with
  `floor = 0.05`, so every region the executor could reach keeps positive probability and the
  uncapped support is a strict superset of v1's. A filter here would have deleted exactly the
  charge-changing cuts that one cell's only witnesses need.

## 2026-09-20 (the region repair was inert: a keyword nothing passed)

- **A validated repair with an opt-in keyword is INERT until some caller passes it, and
  "the module has tests" hides that completely.** `bridge_region_law` was measured,
  tested (20 passing) and merged; `synthesize_dynamic_program` accepted `region_law=`;
  `dynamic_program_synthesis.py` hashed to the post-repair `a1370685`. Every one of
  those facts is true and none of them made the repair reachable. **MEASURED: no
  production caller passed it**, so a rescue launched on that commit would have run
  `law=None` -- v1 verbatim -- and spent up to 1,241 oracle calls reproducing the same
  `candidate_exhaustion`. The tell is available statically and costs one grep: search
  for the keyword at CALL sites, not at definition sites.
- **The field is `proposal.shallow.region_law`, and ABSENT is the only byte-identical
  OFF.** `_delete_pendant_fragment` consumes `rng.permutation` when unlawed and
  `rng.random` (Efraimidis-Spirakis) under ANY law object, so `UNIFORM_BOUNDED_V1`
  reproduces v1's SUPPORT but not v1's DRAWS. An "off" implemented as a uniform law
  would have moved every existing run while reading as a no-op. No uniform law is
  registered in `region_law_contract` for exactly that reason -- a name that looks like
  "off" but is not is worse than no name.
- **A consumption check must RUN the path, not read the signature.** `inspect.signature`
  would have passed both of today's earlier defects (a receipt reading `SPAWNED_ALL`
  while nothing ran; a contract declaring `charged_calls_per_task: 250` beside a module
  constant of 1000), because in both the field existed and was dropped one hop later.
  `assert_region_law_is_consumed(draw)` takes the CALLER's own draw closure, installs a
  probe that raises from `order`, and requires the production path to reach it. The
  probe exception is deliberately not a `ValueError`/`RuntimeError`/`KeyError`/
  `IndexError`/`TypeError`: `synthesize_dynamic_program` catches `ValueError` per family
  and `expand` catches all five per draw, so any of those would be swallowed by the very
  path being observed. MEASURED cost on braf_0: consumed on attempt 1 in 0.03 s.
- **Bound the probe with FIXED seeds, not random ones.** `_weighted_module_order` is a
  permutation over thirteen families and only two route through the law, so a single
  draw can legitimately miss it. Sixteen deterministic seeds make the check reproducible:
  for a given code state it always passes or always fails, so a failure is a wiring
  defect rather than an unlucky draw.
- **MUTATION-PROVEN, four ways.** Dropping the keyword at `expand`, at
  `synthesize_dynamic_program`, at both delete modules, and a resolver that ignores the
  field each turn `tests/test_region_law_contract_wiring.py` red. NOTE the near-miss:
  dropping the law from `substituent_delete` ALONE leaves the consultation test GREEN,
  because `segment_replace` still threads it -- caught only by the separate behavioural
  test asserting the production module can excise past the eight-atom cap. One test per
  hop is not enough when two hops share a sink.
- **OFF vs ON on the campaign's own path, braf_0, 24 shallow draws, ZERO oracle calls:
  OFF 0 eligible (reproducing the live `candidate_exhaustion`), ON 4 eligible**, best at
  similarity 0.640 / QED 0.711 / SA 2.32 from a 15-heavy-atom excision. All four ON
  endpoints exceed v1's cap, which is the axis that made them undrawable in one module.
- **The rescue re-pin is exactly three files per arm, and TWO MORE MUST BE ADDED.**
  `modal_apps/t4_integrated_route_fiber_parp1_app.py`, `dynamic_program_synthesis.py`
  and `t4_fiber_campaign.py` move; `bridge_region_law.py` and `region_law_contract.py`
  are in NO existing `runtime_inputs_sha256`, so a rescue contract that does not add
  them leaves the module implementing the repair unpinned and free to drift.
- **Attribute a red test before blaming your own change.** Of three hash-pin failures at
  HEAD, two come from MERGING the repair and one (`t4_integrated_route_fiber_parp1_v1`)
  was ALREADY stale on four files at the branch base `4cfd398d` -- a superseded v1
  contract, unrelated to either change. Measured by re-running at the merge commit and
  by hashing the base revision's own blobs, not inferred from the diff.
- **SCOPING, MEASURED: the region law neither reaches nor SUPPRESSES the micro-edit
  witness pool.** Two disjoint witness pools are known for these cells -- large
  bridge-separated excisions (7-15 heavy atoms) and plus-or-minus-one-atom edits of an
  existing endpoint. Enumerating the law's support on the real sources: size-1 regions
  number **8 (braf_0), 5 (fa7_0), 9 (fa7_2) under BOTH v1 and the repair** -- identical,
  because v1's eight-atom cap only removes LARGE regions, and the repair's support floor
  keeps every drawable region strictly positive. So the micro shape was always drawable
  and still is. What the repair changes is RANK: braf_0's only two one-module eligible
  children are 14- and 15-atom excisions (sim 0.632/QED 0.635/SA 2.41 and
  0.616/0.719/2.22), and the conditioned tilt puts them at ranks **1 and 2 of 42**.
  **No micro excision is eligible in ONE module on any of the three cells** (0 of 42, 26,
  38), so the plus-or-minus-one-atom pool is reached -- if at all -- through families the
  law does not touch (`atom_insert`/`atom_delete`/`functionalize`/`restate`), which are
  byte-identical between arms. CONSEQUENCE: for a cell whose only in-ball rescue is a
  one-atom edit of a DESCENDANT, wiring the region law changes nothing in the region
  draw, and takes nothing away either. State that scope before the rescue, not after.
- **fa7_0 and fa7_2 have ZERO one-module eligible children even under the repair**, while
  braf_0 has two. A rank improvement measured over a multi-draw closure is not the same
  claim as a one-module witness; do not let the first stand in for the second.
- **THE LAPTOP `.venv` IS NOT THE PRODUCTION CHEMISTRY KERNEL, and a region-law rank is a
  statement about whichever kernel is loaded.** `/Users/rmaganti/compose_rgm_git/.venv`
  runs python 3.12 / **rdkit 2026.03.6** / numpy 2.5.3; the production image pins python
  3.11 / **rdkit 2024.3.5** / numpy 1.26.4 / scipy 1.13.1 / networkx 3.3 / torch 2.4.0.
  Rebuilding the pinned stack takes about a minute:
  `uv venv --python 3.11 ~/compose_region_pinned_env` then
  `uv pip install --python ~/compose_region_pinned_env/bin/python rdkit==2024.3.5
  numpy==1.26.4 scipy==1.13.1 networkx==3.3 torch==2.4.0 pytest`. torch is needed only
  because `t4_fiber_campaign` imports it transitively -- but it IS needed, because
  without it you must transcribe the gate instead of importing it, and a transcribed
  gate cannot fail usefully.
- **THE DANGEROUS PART: `expand` catches `RuntimeError` per draw**, and rdkit 2024.3.5
  raises `RuntimeError: Invariant Violation ... could not find atom1 (Canon.cpp:222)`
  from `Chem.MolToSmiles` on Kekule-degenerate hypervalent-sulfur rings that 2026.03.6
  canonicalizes happily. So a production-kernel refusal is SWALLOWED AS A DROPPED DRAW,
  never as a crash, and is invisible in every artifact. Counting refusals requires
  wrapping `molecular_graph_to_smiles` -- and rebinding it in each module that imported
  the symbol directly (`t4_fiber_campaign`, `bridge_region_law`), not just on its home
  module.
- **MEASURED, and it is PARITY: `scripts/t4_region_law_kernel_parity.py` over all six
  braf/fa7 delta=0.6 cells, 212 region-to-child states per kernel, zero oracle calls.**
  `max_abs_delta` similarity 0.0 / QED 0.0 / SA 0.0; 0 canonical-SMILES disagreements;
  0 verdict flips; 0 rank moves; **0 refusals under EITHER kernel**. Campaign path too:
  braf_0 at 24 shallow draws from one seed gives OFF 0 / ON 4 under both, same max
  excision, same shape census, same best QED to sixteen digits, same example SMILES.
  47 wiring+repair tests pass under the pinned stack. So these particular sources never
  construct the pathological ring -- which is a MEASUREMENT about these cells, not a
  guarantee about a cell whose search wanders into hypervalent sulfur. Re-run the parity
  script for any new cell set rather than citing this one.
- **Direct refusal census on the campaign path, production kernel: 344,831
  `molecular_graph_to_smiles` calls across braf_0/braf_1/braf_2 at 40 law-ON draws each,
  REFUSED = 0.** 900 calls returned `None`, which is the ordinary invalid-state
  rejection and not a kernel refusal -- do not conflate the two. This is the measurement
  that closes the swallowed-RuntimeError hole, because identical endpoint counts across
  kernels alone would not distinguish "neither kernel refused" from "both refused the
  same states".
- **`git worktree add` + `git merge FETCH_HEAD`:** `git merge origin/<branch>` fails with
  "not something we can merge" in a fresh worktree whose remote refs were not fetched
  into that name; `git fetch origin <branch>` then `git merge FETCH_HEAD` works. The only
  conflict was `learnings.md`, which is append-only -- resolve by keeping BOTH blocks in
  order, never by choosing a side.
## 2026-09-20 (PMO-v2: promoting the realizer, and budgeting a COMPLETE search)

- **Choose a search budget from the SUCCESS cost distribution, never from the mean.** The
  realizer replaced the jump lane's width-4 beam. A 12 s per-attempt cap looked comfortable on
  aggregates (mean 4.94 s/attempt, median 0.30, 9.1 attempts per 45 s lane) and **silently lost
  3 of the 6 measured realizations**, because every realization the matched comparison found
  needed **51-55 expanded nodes of a 64 budget** -- so the first one arrives at ~80% of the way
  through, and capping near the mean truncates exactly the productive tail. 20 s keeps 6 of 6 at
  6.88 s/attempt. The cut is safe only because the two distributions SEPARATE: every success
  finishes within 16.92 s while the unproductive searches run 18-64 s. **The expensive searches
  are the ones that produce nothing** -- the opposite of the intuition that would have set the
  cap by average cost.
- **Measure the production configuration, not the component.** The matched-comparison numbers
  were time-to-FIRST realization under 3 concurrent workers; the production adapter collects
  SEVERAL (the controller ranks bindings by retained fraction against a random target, so one row
  makes that target inert). Re-measuring at the real setting -- node_budget 64, collect_all,
  max_realizations 4, single process -- was what produced the cap decision, and it reproduced the
  headline exactly through the adapter: 5 teacher-scale pairs, median 31 primitives, median
  retained 0.630, 3 in the 0.52-0.96 band. Single process ran **1.3-2.1x faster** than the
  3-worker measurement, restating the standing rule that parallelism is free during
  implementation and not during measurement.
- **A wall cap checked between expansions is SOFT.** `over_budget()` runs between node
  expansions, so the executor replay and program extraction of an in-flight node complete past
  it: measured max 20.77 s against a 20.0 s cap (~4%). Budget the lane with that headroom rather
  than assuming the cap is hard.
- **Record the throughput trade in the contract instead of discovering it later.** The lane now
  makes ~6.5 attempts per 45 s batch instead of up to 32. That is the intended trade -- fewer,
  deeper attempts -- but a later run judged against the beam's attempt count would read it as a
  regression. Raising `wall_seconds` is the available lever and was deliberately NOT taken: it
  affects all three lanes.
- **A preflight can stop gating its own runtime the moment you swap the thing it measures.**
  `pmo_population_live_parent_gate.py` says "scoring is prohibited until the resulting receipt
  passes the non-root support floor" and binds with `bind_joint_plan` -- the BEAM. After the
  controller moved to `realize_plan_binding`, that gate measures a binder production no longer
  uses, so its sealed decision does not gate the new runtime. Found by grepping for remaining
  callers of the replaced function, not by any test. **After replacing a production function,
  grep its other callers and ask which of them are GATES.** Deliberately not rewired in the same
  pass: the gate could legitimately fail under the realizer's per-attempt cost, and that is a
  measurement the owner should sanction, not a side effect of an integration commit.
- **A moved contract identity SHOULD break the launcher, and re-pinning it would forge an
  authorization.** Re-sealing moved `pmo_population_controller_v1_scored_contract_corrected.json`,
  whose old hash is pinned in `tools/launch_pmo_population_v1_corrected.py` plus four receipts and
  a source capsule. The launcher now refuses with "corrected contract payload hash changed" --
  correct, because PMO-v2 is a different runtime from the one that authorization covered. The
  receipts are records of the SUPERSEDED identity and must stay addressing it. Distinguish this
  from the 2026-08-02 re-pin sweep: there the pins were maintenance on a non-authorizing binding
  contract; here the pin is an AUTHORIZATION and re-pointing it would manufacture consent.
- **Mutation is the only proof an adapter test is load-bearing.** Six production mutations --
  dropping the source state from `states`, pointing `endpoint_state` at the source, collapsing the
  returned rows to one, ignoring `max_realizations`, returning search order instead of the
  content-addressed order, and drifting the production predicate onto a relaxation -- each had to
  make a NAMED test fail. All six were killed. The strongest of these guards does not recompute
  anything from the adapter: it drives the real consumer (`_joint_program`), which replays the
  actions and asserts `trace["states"] == bound["states"]`, and it cross-checks
  `retained_fraction` computed by `finalize` from the executed graph against the controller's own
  recomputation from `decode_state(endpoint_state)` -- two independent paths to one number.

## 2026-09-20 (PMO-v2: the support gate was measuring the binder production had replaced)

- **A gate that declares its own authority is exactly the file to re-point when the runtime
  moves, and nothing in the suite noticed.** `pmo_population_live_parent_gate.py` -- whose
  docstring says "Scoring is prohibited until the resulting receipt passes the non-root support
  floor" -- still imported `bind_joint_plan` (the width-4 beam) at two call sites after the
  controller's jump lane moved to `realize_plan_binding`. Its sealed v2 decision therefore
  certified support for a binder production no longer runs. Found by grepping remaining callers
  of the replaced function, NOT by a test.
- **Re-pointed gate PASSES; the verdict is robust, the parent set moved.** Scheduler lane, inputs
  byte-identical to the beam-era receipt (initialization, checkpoint payload, plan ids), 16
  parents / 14 exact descendants / 128 plan attempts each = 1,792 attempts, 0 oracle calls:
  `supported_parent_count` **6** against a floor of 4 (beam: 7); route-scale bindings **73** vs
  37; unique route-scale endpoints **59** vs 30. Lost parents {4,11,13}, gained {7,15}.
- **Carrying the realizer's outcome up is what makes a FAIL readable.** Over the 1,792 attempts:
  `proven_incompatible` 1,704 (95.1%), `search_budget_exhausted` 52 (2.9%),
  `completed_realization` 36 (2.0%). The three lost parents each show comp=0 / exh=13 -- every
  one of their 13 step-0-feasible plans ran to the 20 s cap. `proven_incompatible` is
  LOAD-INDEPENDENT (constraint propagation, no clock); the exh/comp split is NOT, so the 7->6
  delta carries a machine-load caveat that the PASS verdict does not (6 >= 4 with margin).
- **The realized program SCALE did not move, and that is the thing to watch.** Bound-program
  lengths are {14,16,17,23} in BOTH arms -- median 16, max 23 -- while **45 of the 95 plan
  latents are >=29 primitives and ZERO of them bound in either arm**. That is the shape the
  predeclared falsifier names ("median ~16 primitives"), but its other conjunct is
  `retained_fraction ~1.000` and the gate does not record retained fraction, so the falsifier is
  **UNEVALUATED, not fired** -- and the gate's objective-blind descendants are a different parent
  population from the matched comparison that measured median 31 at retained 0.630. Settle it on
  the scored run's artifacts; do not read it as a verdict either way.
- **Derive a drift test from the CALL SITE, never from a name written twice.**
  `tests/test_pmo_gate_binder_parity.py` identifies a binder call STRUCTURALLY (a call whose
  second positional argument is `plan`), resolves the callee through the module's own
  `from ... import` bindings, and requires the gate's resolved target, its keyword-parameter set
  and the runtime function OBJECT to equal the controller's. Neither binder is named in the test,
  so it cannot agree with a stale constant. Proven by mutation BOTH directions: 3/3 red against
  the unfixed gate, 2/3 red when the controller is pointed at `bind_realized_plan` (the
  keyword-set check correctly stays green there -- same signature, different function).
- **The mechanical resealer would have buried this.** `reseal_pmo_population_contracts.py`
  re-hashes every key already present in `implementation_sha256`, so running it re-pins the gate
  SOURCE while `support_observed` -- ordinary semantics, guarded only against change -- keeps the
  beam-era numbers. The integration had already re-pinned the controller and added
  `pmo_realization.py` to that config while its support numbers stayed from the beam run: a
  config that VALIDATES while binding a stale measurement, the 2026-08-02 failure class again.
  `support_observed` must be RE-MEASURED from the new receipt, never re-pinned.

## 2026-09-21 (the PMO-v2 launch chain: three guards, three real defects, zero charged calls)

- **A re-pinned contract on disk does NOT reach a deployed Modal app until the app is redeployed.**
  `pmo_population_v1_app.py` bakes the contract with `add_local_file(..., copy=True)`, which fixes
  it at IMAGE BUILD time, and the launcher spawns against a DEPLOYED function via
  `Function.from_name`. So the container kept validating against the v1-era contract long after the
  file on disk had moved. The worker's own check
  (`spec["contract_payload_sha256"] != contract_envelope["payload_sha256"]`) caught it and raised
  `scored payload authorization identity changed` BEFORE charging anything. Fix is `modal deploy`,
  not a code change -- but the failure reads like a contract bug, so know the shape.
- **An edit to a pinned file that lands AFTER a re-seal leaves exactly one stale pin, and it fails
  at the far end of the chain.** `pmo_population_live_parent_gate.py` was re-pointed from the beam
  binder to the realizer after the integration re-sealed, so the base contract pinned `518045a7`
  while disk held `2b44294b`. The failure surfaced as `input identity mismatch` inside the
  container, not at seal time. **When an agent edits a pinned source file, re-pin in the SAME pass
  or the next launch pays for it.**
- **THE NEAR-MISS WORTH REMEMBERING: the capsule rebuild script would have silently DE-AUTHORIZED
  the run.** `prepare_pmo_250_pilot.py` sets `status =
  FROZEN_FAIL_CLOSED_PENDING_NEW_EXPLICIT_PAYLOAD_AUTHORIZATION`, forces `scored_launch_authorized`
  and `modal_launch_authorized` to False, overwrites `budget`, and RECOMPUTES the payload hash --
  moving the contract off the exact hash the owner authorized. It is the right tool for preparing a
  NEW pilot and the wrong tool for refreshing a capsule against an AUTHORIZED payload. Update the
  capsule's `source_files` hashes surgically instead and leave the contract payload untouched.
  Read what a "prepare" script mutates before running it on a sealed artifact.
- **Re-pointing an authorization pin is only legitimate once the authorization names the new
  value.** The launcher pinned `103d9d92` (v1, beam binder) and refused. An agent had deliberately
  left it refusing, correctly calling a pre-emptive re-pin "manufacturing consent". After the owner
  authorized `38afd1c4` explicitly, re-pointing became the correct resolution rather than a bypass.
  The v1 authorization receipt was ARCHIVED beside its completed run, never edited.
- **Archive a superseded launch receipt under the payload hash the RECEIPT carries, not the
  contract's current hash** -- they differ whenever a re-seal ran first, and a fixed archive
  directory lets a second supersession overwrite the first attempt's record.
- **A scripted rewrite of authorization contracts is correctly refused as self-modification.** The
  resolution is individual visible edits, not a workaround. Four edits (gate pin, base payload,
  scored `controller_contract_sha256`, scored payload) re-sealed the chain; the ORIGINAL
  authorization record was left unedited and a `reseal_supersession` block appended, so the history
  of what was first approved survives alongside what actually ran.
- **PARITY IS NOT THE GATE; THE PINNED KERNEL IS THE MEASUREMENT.** I proposed requiring
  `max_abs_delta = 0` between an rdkit 2026.03.6 gate and its 2024.3.5 re-run. Wrong criterion:
  the pinned production kernel's own result is authoritative, and a differing endpoint count
  (e.g. 12/44 -> 9/37) is a PASS provided each arm stays non-trivial, eligibility comes through the
  unmodified production `Fiber.check`, the predeclared settings perturbation still moves yield, and
  the abstention/reverse controls still behave. Demanding numerical parity would let a
  canonical-SMILES spelling difference block a valid launch. Record divergence; do not gate on it.

## 2026-09-21 (de novo: the published quality metric is ABOVE the training corpus, measured)

- **The GuacaMol training corpus itself scores quality 0.4236 on the published de-novo metric**
  (valid AND unique AND QED >= 0.6 AND SA <= 4), with a held-out reference at 0.4330. Both at
  validity 1.000, uniqueness 1.000, diversity ~0.888, n=5,000 each
  (`diagnostics/denovo_generation_v1/corpus_reference_v1.json`).
- **CONSEQUENCE, and it reframes the whole de-novo comparison: a model that PERFECTLY matched its
  training distribution would score ~42% quality, not ~90%.** So "quality" on this metric does not
  measure distribution fidelity -- it rewards deviation AWAY from the corpus toward drug-likeness.
  Any baseline reporting ~90% quality is therefore reporting a distribution SHIFTED from GuacaMol,
  not a faithful sample of it. State the corpus ceiling beside any de-novo quality claim, ours or
  a comparator's, or the number reads as fidelity when it is the opposite.
- **Diversity behaves the other way**: the corpus sits at 0.888, above the ~0.83 the baselines
  report, so on that axis matching the corpus is the strong outcome. Quality and diversity pull in
  opposite directions relative to the same reference, which is exactly why the IVG-style
  quality-diversity FRONTIER is the more honest comparison than either single number.

## 2026-09-21 (fragment verification: a pilot reproduced and was still not a result)

- **The BARICITINIB superstructure pilot reproduced within seed noise** (100.00 validity / 99.67
  uniqueness / 48.33 quality / 0.715 diversity over 3 seeds, 300/300 preservation) **and was not
  representative.** The full 10-drug task row is 93.53 / 97.58 / 37.67 / 0.726. One drug is not a
  task row; the pilot's apparent +8 uniqueness / +10 quality margin shrank to +13.98 / +2.87 once
  measured against the whole instance set. Never promote a single-instance pilot to a headline.
- **LINKER DESIGN IS STRUCTURALLY INVALID IN OUR HARNESS and must not be reported.**
  `build_prompt_context` joins the two cores with a DIRECT BOND, then `RegionLock` pins the bond
  order of every PAIR of locked slots -- which includes that join. The cores therefore stay
  directly bonded and the emitted "linker" has ZERO atoms. Measured: 9/9 ELIGLUSTAT emissions match
  `[NX3;r5]-[c]` where the real linker is 3 atoms; 15 locked slots carry 17 locked core-core bonds.
  The endpoint check does not catch it because the other fragment's atom satisfies the attachment
  requirement. Scaffold morphing shares the mechanism. Both withheld.
- **`abs(hash((drug, task, seed)))` IS NOT A SEED.** `str` hashing is `PYTHONHASHSEED`-salted, so
  the same key gave 704705714 / 159431064 / 4134730822 across three processes -- no row produced
  this way is reproducible by anyone, including us. Use BLAKE2b (or any stable digest) for any
  derived seed. This silently invalidated every fragment row until it was caught.
- **The fragment-preservation guarantee is narrower than "100%", and the precise statement matters.**
  GRAPH-LEVEL region preservation IS by construction: locked slots keep element, formal charge and
  every pairwise bond order, so the core's induced labelled subgraph is byte-identical in every
  committed state (100.0000% of committed endpoints chemically valid). RDKit SUBSTRUCTURE
  preservation is NOT by construction -- measured 99.89 / 99.30 / 99.89%. Mechanism localised to a
  single event class: `BondInsert` closing a ring THROUGH core atoms; states are Kekule so every
  locked bond order is unchanged and the lock correctly admits it, but aromaticity is whole-molecule
  RDKit perception, so core atoms flip aliphatic->aromatic and an aliphatic query atom stops
  matching. Claim graph-level by construction; claim substructure as endpoint-enforced and measured.
- **CONTAINMENT IS SOLVED, PLACEMENT IS NOT -- one mechanism gates three of five task rows.** Motif
  extension is 85.4% committed and 99.30% fragment-CONTAINING yet only 41.4% valid: 1,319 endpoints
  kept the motif and grew somewhere other than the one declared attachment site. Decoration is 93.5%
  committed, 99.89% containing, 3.5% valid with 2,699 site failures. The missing capability is
  attachment-site steering, not fragment preservation.
- **I relayed comparator numbers I had not sourced.** GenMol V2 figures (99.7 / 89.8 / 39.0,
  div 0.551, dist 0.769) appear in no pinned artifact; the upstream `reference_metrics.csv` says
  97.5 / 83.6 / 34.8 / 0.599 / 0.762. A baseline quoted from a README is not a baseline. Pin the
  comparator table by hash before it decides anything.

## 2026-09-21 (fragment "validity" was task success; the correction was worth +44 and +90 points)

- **A reported metric contradicted an architectural invariant, and the invariant was right.** The
  fragment harness reported motif-extension "validity" at 41.4% and scaffold decoration at 3.5%.
  COMPOSE's committed states are validity-closed BY CONSTRUCTION, so chemical validity that low is
  impossible unless something bypasses the executor. Per-sample audit of the row reporting
  "validity 8.00": 100 attempts -> 71 produced; of those 71, chemically valid 71, connected 71,
  graph-valid 71 (re-enters a production `MolecularGraph` and passes `is_valid_state`),
  fragment-containing 71, TASK-SUCCESSFUL 8. All 63 rejections were
  `task_only: missing_fragment_or_attachment`; ZERO were chemistry. The harness was labelling
  conditional-task failure as chemical invalidity.
- **CORRECTED, like-for-like against baselines whose evaluators never inspect the fragment:**
      motif          85.40 vs GenMol 82.90  -> COMPOSE AHEAD (+2.50); was reported as 41.43
      superstructure 93.63 vs 97.50         -> -3.87; the correction moved it UP from 93.53
      decoration     93.47 vs 96.60         -> -3.13; was reported as 3.50
  Worth +44 points on motif and +90 on decoration. **When a measured number contradicts a
  construction-level guarantee, audit the metric before believing the number.**
- **The two competing explanations were RULED OUT by measurement, not argument.** Adapter bypass:
  every emitted string is literally a committed endpoint. Serialization/kernel failure: chemical
  validity of produced molecules is 100.0000% on all three tasks. Naming the alternatives and
  killing them is what makes the conflation finding safe to act on.
- **DENOMINATORS: "100% valid" and "85.4% valid" are both true and mean different things.** 100% is
  the share of COMMITTED endpoints that are chemically valid -- the architectural claim. 85.4-93.6%
  is validity over ATTEMPTS, and the entire gap is trajectories that committed NOTHING (rejection
  budget exhausted or zero events). That is SAMPLER EFFICIENCY, not validity. State which
  denominator every validity number uses, always.
- **PERSIST THE MOLECULES, NOT JUST THE COUNTERS -- this cost us real data.** Shards stored counts
  plus five example SMILES. So the corrected primary metrics could be rebuilt from counters, but
  uniqueness/quality/diversity/distance for motif (~51% censored) and decoration (~96%) were scored
  over the self-censored emitted set and are NOT repairable -- they describe survivors, not the
  generator, and are flagged INVALID rather than quietly reported. Superstructure survived only
  because it censored 3 of 2,809. A run that stores aggregates cannot answer a question posed after
  the fact.
- **Guards now assert the invariant instead of assuming it:** across all 90 shards,
  `committed_chemically_valid == committed` and `task_success <= containment <= produced`.
- **The capability actually missing is CONDITIONAL CONTROL OF WHERE A VALID EDIT LANDS**, not
  validity and not fragment preservation. Containment is 84.8-93.4%; placement is 3.5-93.5%. Name it
  "conditional task success" everywhere so the diagnosis cannot drift back into a validity claim.

## 2026-09-21 (de novo: COMPOSE's QED beats its corpus; the whole gap is SA, and SA is the preview defect)

- **First real COMPOSE de-novo row** (N=50, one seed, frozen default sampler, rdkit 2024.3.5,
  `diagnostics/denovo_generation_v1/partial_seed20260920_n50_v1.json`):
      validity 1.000   uniqueness 1.000   quality 0.280 +- 0.063   diversity 0.8925
  Validity and uniqueness at 1.000 are the construction guarantee showing up empirically
  (`valid_state_fraction` and `connected_fraction` both 1.000). Diversity 0.8925 BEATS both
  published systems (GenMol V2 0.830, V1 0.818) and matches its own corpus (0.888).
- **THE DECOMPOSITION IS THE RESULT, not the headline number.** Against the corpus control:
                     mean QED  QED pass   mean SA  SA pass  quality
      GuacaMol train    0.553    44.7%      2.92    90.0%    42.4%
      COMPOSE (N=50)    0.591    50.0%      4.11    46.0%    28.0%
  **COMPOSE's drug-likeness EXCEEDS its own training corpus** (0.591 vs 0.553). The entire quality
  deficit is synthetic accessibility: SA pass collapses 90% -> 46%. Never report the 0.28 without
  the decomposition; the aggregate reads as "worse generator" when the model is better on one half
  and worse on the other for a known reason.
- **The SA failure is ATTRIBUTED, not guessed:** a strained-ring census measures 50% of generated
  molecules carrying a 3- or 4-membered ring (40% three-, 14% four-; SSSR 3:25 4:7 5:31 6:97),
  reproducing the documented 51% small-ring rate of the step-1,000 Lineage B preview almost exactly.
  The weak number is the known preview defect appearing in precisely the metric half it should.
- **CORRECTION to my own earlier framing: 0.4236 is NOT a "ceiling".** It is what corpus FIDELITY
  scores on the published metric. COMPOSE's QED already exceeds it, so the metric is not a cap --
  it is a reference point that a drug-likeness-seeking generator is expected to pass.
- **A STRONGER DE-NOVO GENERATOR CANNOT BE PROMOTED, ONLY TRAINED.** A full recursive scan of
  `compose-v4-artifacts` (79 dirs, 81,274 entries, 4,421 `.pt`) found ZERO de-novo training
  artifacts -- no `checkpoint.best_so_far`, no `manifest.training`, no `tree_source_prior`, and none
  of the three known run labels resolve. That volume was created 2026-08-09, AFTER the July
  generator runs, and every general model on it is EDITING (`frozen_process_sha256 = 0c938177`).
  The numerically better step-2,500 checkpoint (small rings 51% -> 40%, FCD 23.0 -> 19.2) was
  deliberately not promoted and its only homes were reaped `/private/tmp` paths.
- **Catalog drift does not apply to this path, verified by grep not assumption.** The de-novo model
  is constructed from `payload["ring_catalog"]` deserialized from the checkpoint and never calls
  `build_production_ringcore_catalog`. That is STRICTLY STRONGER than the fingerprint check --
  byte-identical to the training catalog -- so this Mac's local drift to `82fd910c` is irrelevant
  to these numbers.
- **Unconditional means the trained initial law, and it is checkable.** The prior is
  `DegreeBoundedCarbonTreePrior` serialized in the checkpoint; t=0 states are random degree-capped
  all-carbon alkane trees carrying no heteroatom, bond-order, ring or scaffold information, and are
  not corpus molecules. Slots are allocated directly at `n_slots=40`, so the tight-graph
  `atom_insert` hazard cannot bite -- this path never parses SMILES into a state.
- **COST, so nobody re-derives it: 72.65 s/trajectory measured on the real path.** 1,000 molecules
  is ~20 core-hours; the predeclared 6-point Pareto sweep would be ~265 core-hours. Not justifiable
  on a preview checkpoint, and the 40-container 3x1,000 run was stopped after producing 1 of 60
  shards in 80 minutes.

## 2026-09-21 (preemption killed a 60-shard fan-out; the lever is WORK-UNIT SIZE, not retry count)

- **SHARPENS the 2026-07-29 retry lesson into its general form.** That entry said a retry budget
  without a working resume path just burns the same steps repeatedly. The de-novo run shows the
  corollary that actually decides launches: **retries only make forward progress when each retry
  resumes from COMMITTED state. Absent a resume path, the lever is not `max_retries`, it is the
  SIZE OF THE WORK UNIT.**
- **The measured failure.** A 3x1,000 de-novo run fanned 60 shards of 50 trajectories each. At a
  measured 72.65 s/trajectory a shard is ~61 minutes with NO intra-shard checkpointing, so every
  preemption discarded up to an hour and the retry restarted from zero. 27 preemptions
  ("Container terminated due to preemption. Your Function will be restarted with the same input");
  `max_retries=3` could not carry a 61-minute unit on preemptible capacity. When one shard exhausted
  its retries the local `.map()` raised `RemoteError`, **which killed the whole fan-out including
  the 59 healthy shards**. Exactly 1 of 60 committed.
- **A one-flag fix, no code:** `--shards-per-seed 100` gives 10-trajectory shards of ~12 minutes --
  short enough to land between preemptions. 3,000 trajectories becomes 300 shards, ~90 min wall on
  40 containers. Shards are horizon-scoped and idempotent, so a relaunch reuses the committed shard
  and redoes nothing.
- **ONE `.map()` SHARD EXHAUSTING RETRIES CAN DESTROY EVERY HEALTHY SIBLING.** If partial results
  matter, do not let a fan-out's failure mode be a single raising map -- commit per shard (this one
  did, which is why the surviving shard was scorable) and catch per-shard failure rather than
  letting it propagate.
- **"NO OUTPUT YET" CANNOT DISCRIMINATE "still starting" FROM "being preempted to death" --
  both predict silence, and I misdiagnosed it as slow container ramp-up.** Two commands settle it:
  `grep -c preemption` on the log, and `modal app list` for the task count. A monitor armed only for
  the SUCCESS event (here `SEED_COMPLETE`) is structurally blind to this failure -- watch for the
  failure signature too, or the watch tells you nothing for an hour.
- **ALSO: verify an app is actually alive before claiming you freed its capacity.** I ran
  `modal app stop` on a listing that showed `tasks=40` and reported "40 containers freed"; the app
  had already died of preemption and those entries were stale or terminating. A task count in a
  listing is not proof of live capacity -- check the function-call state, as the same listing
  misled a phantom-run diagnosis earlier the same day.

## 2026-09-21 (a launcher repair FORCES a supersession; and a mutation battery that refused for the wrong reason)

- **A launcher cannot be repaired while keeping the run identity it pins, and the honest
  resolution is an explicit supersession with a reconstruction proof -- never a quiet re-pin.**
  The 5ht1b_2 v1 wrapper validated against the superseded v1 contract module AND required an
  authorized status INSIDE the scientific payload. Repairing it changes
  `modal_apps/t4_5ht1b2_protonation_rescue_<arm>_app.py`, which is pinned in
  `runtime_inputs_sha256`, which lives INSIDE the payload -- so the contract identity had to move
  (`d7d2fbc2 -> f4c00e06`, `4486c376 -> ea7c5c73`). This is the 2026-07-29 lesson ("changing the
  launcher necessarily changes the run identity") reaching an AUTHORIZED artifact, where it is
  sharper: the guard is working, and re-pointing the pin to keep the old authorization would
  manufacture consent. What makes the supersession safe is a `launch_path_revision` block carrying
  `supersedes_payload_sha256` plus a `reconstruct_superseded_payload()` that rebuilds the
  authorized payload EXACTLY, so the claim "only plumbing moved" is checkable rather than asserted.
- **Verify a supersession with a diff that does NOT import the reconstruction function.** A
  comparison whose expectation is recomputed from the code under test cannot fail (2026-08-02,
  again). Method that works: flatten both payloads to leaf paths, classify EVERY difference against
  an explicit allow-list of prefixes, and require the unexplained set to be EMPTY -- then run the
  branch's own reconstruction separately as a cross-check. Measured here: 7 differing leaves per
  arm (5 supersession-metadata, 1 repaired-wrapper hash, 1 added-validator hash), 0 removed,
  0 unexplained, every non-plumbing leaf and every other runtime-input hash byte-identical.
- **A MUTATION BATTERY RUN IN A TREE WITHOUT `.git` REFUSES FOR THE WRONG REASON.** My first run
  read a triumphant 20/20 negatives refused -- and 0/4 positive controls. The validator's
  `require_clean_runtime` shells out to `git diff --exit-code`, which fired first in a temp copy
  built with `ignore_patterns('.git', ...)`, so 10 of the 20 "refusals" were the missing repo, not
  the mutation, and proved nothing. **The COSMETIC RE-SERIALIZATION positive control is what caught
  it** -- a case that changes bytes and nothing else, which MUST still pass. Fix: copy `.git`,
  detach it into a standalone repo (a worktree's `.git` is a FILE pointing at the parent), and
  COMMIT each mutation so the tree is clean and only the SEMANTIC check can decide. Then 22/22
  refuse with the correct reason and 4/4 positives pass. **Always include a control that perturbs
  something irrelevant; without it, "everything refused" is indistinguishable from "the harness is
  broken".**
- **A crosswire test whose harness derives the contract path FROM the arm cannot test the
  crosswire.** My two reported "HOLES" were my own test: changing `arm` made the harness load that
  arm's own contract, which correctly validates. The real crosswire is arm X + contract Y, and the
  validator refuses it. Before recording a hole, check the test actually constructs the condition.
- **A shard can retain what its own diagnostic artifact threw away.** The de-novo N=50 artifact
  stored counters only (1,086 bytes), so "SA conditional on ring strain" read as unanswerable --
  but the committed shard on the volume kept per-molecule SMILES and the analysis was fully
  recoverable. Before declaring a post-hoc question dead, check the RAW artifact, not the reduction.
  (Does not weaken "persist the molecules": it was luck that the shard schema differed.)
- **The de-novo small-ring finding is ASSOCIATION, and it is CONFOUNDED WITH SIZE.** n=50 split
  25/25 on presence of a 3- or 4-ring, pinned kernel: SA 4.585 vs 3.639 (diff +0.946, permutation
  p=0.0034), QED 0.531 vs 0.651 (-0.120, p=0.0173), quality 0.16 vs 0.40 -- but strained molecules
  are also **+4.0 heavy atoms** (p=0.0402), and SA rises with size independently of strain. So the
  earlier "the ENTIRE quality gap is SA" is TOO STRONG twice over: QED is significantly worse too,
  and the SA effect is not cleanly attributable to strain at this sample size. Any checkpoint sweep
  must report SA and QED stratified by heavy-atom bin AND strain, or it cannot distinguish "training
  fixed pathological ring chemistry" from "training shrank the molecules".
- **Check a checkpoint for RESUME-CRITICAL keys before planning to resume it.** Lineage B's
  `checkpoint.best_so_far.pt` carries the complete recipe (seed 20260717, batch 64, lr 3e-4,
  adamw_decoupled_v1, warmup 500, eval every 250, hidden 256, mp 6, horizon 16.0), the serialized
  `TypedRingCatalog`, and `completed_steps: 1000` -- but **no optimizer, scheduler or RNG state**.
  So it cannot be resumed, only warm-started with a fresh optimizer, which is a restart wearing a
  resume's clothes. Re-running from scratch at the serialized seed is the only way to reproduce the
  trajectory -- and it buys a free correctness gate, because the reconstruction passes through
  step 1,000 and must match the stored checkpoint. Also note `training_steps: 3000`: the original
  de-novo run targeted 3,000 steps, not the 16,000 of the editing runs.

## 2026-09-21 (CORRECTION: the Lineage B training artifacts were never lost -- two volumes share one name)

- **CORRECTS the 2026-09-21 entry "A STRONGER DE-NOVO GENERATOR CANNOT BE PROMOTED, ONLY TRAINED"
  and its claim that a full scan found ZERO de-novo training artifacts.** That scan read the WRONG
  VOLUME. There are TWO distinct Modal volumes both named `compose-v4-artifacts`:
      profile `rahul`         created 2026-07-17 03:16 EDT   <- holds the July Lineage B run
      profile `rahul-94866`   created 2026-08-09 00:13 EDT   <- the one that was scanned, empty of them
  **Both print `rahul-94866` in the workspace column of `modal volume list`**, so the listing itself
  cannot distinguish them; only the profile the CLI is invoked under and the creation date can.
  The July volume holds the original run, its **recovery checkpoint with optimizer + RNG state**, and
  a **step-2500 checkpoint that is the genuine continuation of the Lineage B trajectory**.
  COST OF THE ERROR: I instructed a from-scratch retrain at seed 20260717 to "reproduce the
  trajectory", on the stated premise that no resumable state existed. The premise was false, the
  agent checked rather than complied, and resuming beats reconstruction because a from-scratch rerun
  can only APPROXIMATE a trajectory that a recovery checkpoint reproduces exactly.
  **LESSON: a volume NAME is not a volume identity. Pin the profile and the creation date before
  concluding an artifact is absent** -- "I scanned the volume and it was empty" is not a finding when
  two volumes share the name.
- **A checkpoint's `provenance_sha256` IS the producing run's `run_identity_sha256`**
  (`train_tracelet_gm.py:1117`), so a checkpoint can be matched to its run by HASH rather than by
  recipe similarity. Scanning every candidate `manifest.training.json` matched `a1907ed1...` to
  exactly one run, `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`. Use this instead of inferring
  lineage from matching hyperparameters.
- **A MATCHING STORED VALIDATION HISTORY PROVES NOTHING ABOUT REPRODUCIBILITY.** An identical
  step-1..1000 curve across two run identities looks like code-change invariance and is not: the
  history is a list carried INSIDE the recovery checkpoint and restored on resume, so it is shared
  ancestry, not independent reproduction. The tell is `resume_checkpoint` in the recipe arguments.
  (Claim made and withdrawn by the agent mid-task; worth keeping because the artifact is genuinely
  persuasive at a glance.)
- **CONTINUED TRAINING DOES NOT FIX THE SMALL-RING DEFECT, AND THE OUTCOME METRIC HIDES IT.**
  Matched n=70/arm, identical trajectory seeds, pinned kernel, ring catalog byte-identical between
  checkpoints (`50337de077f374db`) so ring-size SUPPORT is fixed and any shift is policy:
      quality                        0.1714 -> 0.3000   (+0.129)   <- looks like success
      mean heavy atoms               28.70  -> 26.87
      rings / molecule               3.743  -> 3.300
      molecules with a 3/4-ring      0.4429 -> 0.5143   (WORSE)
      per-RING strained fraction     0.1679 -> 0.1948   (WORSE)    <- the mechanism metric
      ring sizes   6-rings 152 -> 111,  3-rings 26 -> 38
  Quality rose because molecules got SMALLER, not because ring chemistry improved. **Endpoint
  prevalence and the per-ring closure policy diverge, and only the per-ring metric catches it** --
  exactly the failure mode that motivated adding it. Nothing reaches 95% at n=70, so these are
  DIRECTIONS not verdicts; but there is no hint of the intended improvement in either measure.
- **SIZE-PARTIALLED REGRESSION SETTLES THE 2026-09-21 CONFOUND: the SA half is genuine strain, the
  QED half is size.** Replicated at both checkpoints:
      SA  ~ strain +0.791+-0.217 (s1000), +0.496+-0.218 (s2500)   heavy +0.024/+0.058
      QED ~ strain +0.033+-0.039, -0.037+-0.037 (NO effect)       heavy -0.024+-0.003 (|z|~8)
  So "strained molecules have worse QED" was entirely molecule size; the SA effect survives control.
  Validated by a NEGATIVE CONTROL: on synthetic data where strain has zero effect and size carries
  everything, the fitted strain coefficient collapses to <1e-6 while the raw contrast still reads
  +0.7. An instrument that cannot return "it was size" cannot be trusted when it returns "it was
  strain".
- **THE RING DEFECT IS UPSTREAM OF THE LEARNED RATES, so no reward term can fix it.** Exact support
  audit, 265 ring events over 100 rollouts: unconditional catalog small-ring mass **0.030**; uniform
  over the LEGAL SUPPORT where ring events actually fire **0.443**; the model's prior given that
  support **0.284**; produced **0.280**. The model is ALREADY pushing away from small rings relative
  to its own support, which is ~15x enriched in them versus the catalog. An SA or small-ring penalty
  would push an already-anti-small-ring policy against a support that leaves it no alternative --
  which is why the correction belongs in the training DISTRIBUTION, not the objective.
- **`event_schedule="exact_early_ring"` IS A BUILT, TESTED, INERT REPAIR -- the region-law lesson
  again.** `rewrite/commuting_schedule.py` moves whole ring transactions to the earliest state at
  which they are executable (detected BY EXECUTION, `atom_insert`/`atom_delete` as phase barriers,
  swaps accepted only when both orders are array-exact), so ring decisions get supervised at states
  with free slots where large-ring templates are still legal. Added 2026-07-20 -- ONE DAY after the
  step-1000 checkpoint -- 6/6 tests passing, and the string appears nowhere outside
  `tree_transport.py`. Lineage B trained on `"sequential"`. Plumbing gap is five additive hops:
  recipe argument -> gate CLI flag -> `build_tree_transport_path_records(..., event_schedule=)` ->
  its two `compile_carbon_tree_to_target` call sites -> Modal `build_tracelet_recipe_argv`.
  **Mutation-test at the CALL sites: two hops share one sink, so a single consultation test stays
  green** (the exact near-miss the region-law wiring hit).
- **Shard globbing pooled two different trajectory-seed families into one headline number.**
  `score_available`/`score_seed` glob `seed{seed}_*.json`, which merged a legacy `total=1000` shard
  with new `total=200` shards for the same seed while every count still looked reasonable. This sat
  in the PUBLISHED-benchmark path. Guarded in both the app and the analysis script; the guard fired
  on real data with a precise diagnosis. Key a shard by its full identity, never by a seed prefix.

## 2026-09-21 (PMO located: the parent allocator spends the budget on garbage, and there is no chemical prior)

- **MEASURED, zero oracle calls: the PMO search starts from excellent chemistry and walks off-manifold on
  EVERY task.** The init bank (`docs/PMO_INIT_BANK.json`, 100 SMILES, objective-blind, fixed before any
  task) is median **QED 0.763 / SA 3.575 / 21 heavy atoms, 85% at or above QED 0.6**. After 250 calls the
  scored molecules sit at median QED 0.41-0.45 with only **20-26%** above 0.6. So the weak-start hypothesis
  is FALSIFIED and the degradation is universal, not oracle-specific.
- **On a fingerprint-predictor oracle that drift becomes REWARD HACKING.** gsk3b: r(score, QED) **-0.433**,
  r(score, SA) **+0.684**; best molecule 0.290 at **QED 0.03 / SA 7.81**, carrying bare phosphorus,
  hypervalent `[IH]` and a peroxide-hydrazine chain `OOCNNCO`; top-10 median QED 0.113, SA 7.155. The
  similarity/descriptor oracles are NOT fooled (correlations run the healthy way, top molecules reasonable),
  so hacking is the CONSEQUENCE of going off-manifold, not the cause. **That 0.290 is not a reportable
  number.** `r(score, heavy_atoms)` is +0.67 to +0.81 on all three tasks -- a universal size gradient any
  scale-learning component could trivially latch onto instead of learning chemistry.
- **THE PMO PATH CONTAINS NO LEARNED MODEL.** A full real proposal round (16 candidates, 61 attempts)
  instantiates `WholeGraphRateModel` / `FactorizedRateModel` / `FactorizedTraceletRateModel` **0 times** and
  calls `torch.load` **0 times**; grepping the whole runtime for `rate_model` / `load_factorized` / `*.pt`
  returns nothing. Proposals are `shallow_rng` / `structured_rng` / `arbitration_rng` draws over the legal
  fiber plus heuristic tilts; the only checkpoint loaded is the dead jump library's plan latents. `torch` is
  in the import closure -- **imports are not calls.** The legal fiber is vastly larger than the drug-like
  manifold, so sampling it near-uniformly MUST drift: validity-closure is a VALENCE guarantee and says
  nothing about plausibility. GenMol/IVG sample from learned distributions over real molecules and never
  meet this. Part of what reads as an algorithmic gap is a missing generative prior.
- **THE UPSTREAM CAUSE IS THE PARENT ALLOCATOR, AND IT IS SCORE-INDEPENDENT.** Archive median **16** heavy
  atoms; parents actually DRAWN median **9**, 51.2% under 10. `parent_allocation='niche_score'` ->
  `exploration_niches(endpoints, utilities, max_niches=4, per_niche=2)` at the production call site with bare
  defaults gives at most **4x2 = 8 molecules ~80% of parent mass regardless of archive size**; everything
  else gets only the `0.2/N` floor. Replayed on the REAL final archives, **3 of 4 niche centres are 1-3
  heavy-atom fragments** (`[SH4]`, `CCl`, `NN(N)F`, `P`, `NF`, `O=C=[SH4]`): the drug-like niche holds
  189-203 of 234 endpoints and contributes **2** parents, the three fragment niches contribute **6**, scoring
  0.000-0.12 against incumbents at 0.22-0.50.
  **THE MUTATION TEST THAT NAILS IT:** swinging four one-atom fragments from score 0.01 to 0.99 changes their
  parent mass by **EXACTLY 0.0000**, while a control (making aspirin best) moves its mass 0.0083 -> 0.1083,
  **13x**. A structurally-alone molecule receives full niche mass however badly it scores, because
  `|group| == 1`; score only breaks ties WITHIN a niche. A fragment that enters once becomes a permanent
  niche centre drawing ~10% of every round.
- **DELETION IS REFUTED AS THE CAUSE, WITH THE SIGN REVERSED.** Realized edits GROW molecules:
  `atom_insert:atom_delete = 2.42:1`, mean delta **+1.16** heavy atoms, 45.5% grow vs 17.5% shrink;
  reproduced reward-independently on a synthetic-reward run whose reward PEAKS at 26 heavy atoms and
  penalises small molecules -- **the collapse happened anyway** (44 archive entries under 5 heavy atoms).
  Sub-10 archive members are 87-96% children of sub-10 parents: a ratchet sustained by ALLOCATION, seeded by
  only 3-8 large-deletion events per task. The archive median falling 20 -> 15.5 is DOWNSTREAM of allocation.
- **The benchmark objective never reaches a decision.** `top_k` / `auc` / frontier appear nowhere in
  `pmo_population_controller.py`, `fiber_control.py` or `pmo_credit.py` except two prose comments;
  `archive_top_k` is called only inside the round SUMMARY. The controller maximises predicted ENDPOINT SCORE.
  Marginal contribution to the top-10 mean does not exist as a quantity anywhere in the runtime.
- **The exploration constants are calibrated for docking and swamp PMO's signal by 10-30x.** Measured mean
  positive improvement per child is **u = 0.0077** (6 channel x task cells, 0.0052-0.0114). The parent tilt
  `1 + upside + 0.25/sqrt(trials+1)` needs **n > 1042 trials on ONE parent** for upside to beat the constant
  (max possible 250), so it is a pure novelty bonus. `PopulationCredit`: `value(untried) = 0.2500` strictly
  outranks `value(1 trial, +0.05 delivered) = 0.2268`; a 1-trial cell needs **>= 0.0732** merely to tie, 10x
  the measured mean. Result: **180 active cells / 91 basins from 202 scored children ~ 1.1 trials per cell**,
  every top-earning cell at `trials: 1`. Proliferation beats credit.
- **Plateau escape LATCHES PERMANENTLY into the dead lane.** `rounds_without_improvement >= 3 ->
  escape_rounds_remaining = 2`, decremented 1 per allocation, so on a plateau it is RE-SET faster than it
  drains: measured **escape True for 8 consecutive rounds, never releasing**, holding jump at 3/16 and
  cutting shallow to 1/16. The jump lane is measured dead both with the real oracle (3 executions / 1,091
  proposals) and reward-independently (147 attempts -> 4 eligible, 2.7%, vs shallow 72.7% / structured 65.3%).
- **Parent MODES are labels, not behaviour.** `propose_batch` builds `schedules = {channel: [self._parent()...]}`
  and `_parent()` takes no channel argument, so `global_explore` / `jump_from_elite` / `refine_elite` all draw
  from ONE distribution; every occurrence of `mode` in the controller is a WRITE into provenance. `jump_from_elite`
  does not restrict to elites. Do not describe these as different policies.
- **Capacity, measured, and STRONGER than the standing trap:** PMO states are **48 slots** (all 16 init states,
  `n_real` 15-27, 0 tight) with the 40-heavy ceiling enforced at CONSTRUCTION, not on endpoints; states come
  from `decode_state`, never a SMILES re-parse, so the tight-graph hazard cannot arise here. At 40 slots the
  path **hard-refuses** ("expected an exact supported 48-slot source", 0/13 families execute) rather than
  silently dropping `atom_insert`. For PMO, 48 is a requirement.
- **Smaller measured facts worth keeping:** stereochemistry costs exactly ZERO (`useChirality=False` at both
  occurrences in the oracle source; Tanimoto(reference, stereo-stripped) = 1.000000 on the one stereo-bearing
  reference). `bond_reorder` is the one Active8 executor rule PMO can never emit (absent from
  `GENERIC_MODULES`). Heavy elements S/P/Cl/Br/I/B are reachable ONLY by `atom_restate_semantic`, never by
  insertion -- every growth module is hardcoded CNO(F). 8 of 16 init states are permanently anionic and
  `CalcMolFormula` (what `Isomer_scoring` parses) then carries a trailing `-` and one fewer H; AUC cost
  UNDETERMINED. The archive is append-only with no prune/evict path, and PMO endpoint eligibility is
  RDKit-parseability only -- correct for no-prescreen, but there is no floor of any kind.

## 2026-09-21 (de novo: `exact_early_ring` is INERT on the mechanism, and the slot-scarcity story is a rollout-time explanation)

- **DO NOT RETRAIN ON `exact_early_ring`, and do not bother relaxing its barrier.** Zero-training probe,
  n=24 traces / 40 ring events, production compiler and scheduler, pinned kernel:
      metric                        A sequential   B shipped   C barrier relaxed
      ring position (frac of trace)     0.939        0.652          0.652
      accepted / attempted swaps          --       236 / 276      236 / 276
      MEAN FREE SLOTS                   21.925       21.925        21.925
  Ring events DO move materially (0.939 -> 0.652). **Free slots do not change at all -- per-event
  identical A==B 40/40 and A==C 40/40**, same distribution `{16:2,19:3,20:12,21:2,22:2,23:8,24:3,25:6,
  27:1,29:1}`. Arm C reproduced arm B on 24/24 traces with **identical attempted swaps (276 = 276)**,
  which proves **arm B never reached a barrier at all** -- so removing the barrier is provably a no-op,
  eliminated by direct mechanism rather than by the conjunction.
- **CORRECTS the standing diagnosis: "ring growth is supervised LATE on crowded states near the 40-atom
  cap" is a ROLLOUT-time story and does not hold at TRAINING time.** In training traces the ring decision
  already sits at **~22 free slots of 40** (min 16). Slot scarcity is NOT the binding variable there. In
  the flexible-graft compiler every `atom_insert` precedes the topology phase, so free slots at the ring
  decision already equal `40 - final heavy count` before any ring event fires, and `ring_system_grow`
  cyclizes existing atoms consuming **no** slots -- reordering within the topology phase cannot move the
  number. What actually refuses the swaps is the exactness test on **`bond_reroute` 38/40 and
  `bond_reorder` 2/40**: the topology events that build the very bonds the ring system attaches to. So
  the support degeneracy comes from bond topology, not crowding.
- **CONSEQUENCE: the fix is the compiler's EMISSION ORDER, not a post-hoc reordering.** The ring
  transaction must be emitted at a different point in the dependency order -- interleaved with the
  `bond_reroute` topology phase rather than appended after it. Bubble-sorting a poor trace cannot repair
  a trace whose ring decision was never placed well.
- **The reference 0.443 support mass is a mean over TWELVE states, not the 265 ring events**, and it is
  extremely heterogeneous: four of the twelve have supports of only 2, 2, 5 and 12 templates, three of
  which are 100% small (mass 1.0). It is driven by a handful of near-degenerate states, not a uniform 44%
  tilt. Quote it with that caveat.
- **METHOD: a probe reducer must refuse to emit a verdict from zero data.** The reducer initially emitted
  `FIX_COMPILER_ORDERING` -- the "don't retrain" branch -- from **zero compiled traces**, because a
  separate bug (passing the catalog INTO `compile_carbon_tree_to_target`, which refuses unsupported
  traces) had killed 12 of 12 real molecules. The trainer compiles with `ring_catalog=None` and filters
  afterwards with `structured_ring_trace_supported`. Both fixed; the reducer now returns
  `INCONCLUSIVE_NO_DATA`. A verdict branch reachable with an empty input set is a landmine, and it was
  pointing at the right answer for the wrong reason.
- **Instrument check before arms, not after.** Rather than hoping arm A landed near 0.443, the probe
  recomputed the committed reference audit's own 12 states through the production support path and got
  **12/12 agreement to 16 digits**. The ring catalog is byte-identical between step-1000 and step-2500
  (`template_repr_sha256 25bb9f52...`), so the instrument transfers between arms.

## 2026-09-21 (parent-first construction: addition is damaging at EVERY size, and ring closure is a 4x mitigation)

- **CORRECTS my own "damage scales ~7x with program length" reading.** That curve is real but it lives
  INSIDE the region-EXCISION branch (3-5 primitives +0.081 mean dQED, 6-9 -0.028, 10-14 -0.135,
  15+ -0.159). Length-controlled with synthetic intents, **pure ADDITION is net-negative at every size
  including a single atom**: 1 insert **-0.023**, 2 **-0.061**, 3 **-0.100**, 4 **-0.160**. So the
  +0.081 belongs to REMOVING a fragment, not to short programs as such. Two branches, two curves --
  do not generalise one to the other, and the sharper statement is "adding without removing is
  damaging at any size".
- **RING CLOSURE IS A ~4x DAMAGE REDUCER, AND QED AND SA DISAGREE ABOUT IT.** 3 atoms + 1 closure is
  **-0.022 against -0.100** for 3 atoms alone; 5 + 1 closure **-0.067 against -0.160** for 4 alone --
  a closure turns a floppy chain into a ring. But mean dSA moves the OPPOSITE way: **+0.88 / +1.07
  against +0.25 / +0.34**. **Quoting either metric alone inverts the conclusion.** A placement prior
  should be built to exploit closure at a known SA cost rather than rediscover it.
- **The missing prior is element PLACEMENT, not element VOCABULARY.** Drawing the chain uniformly from
  an intent's element set gives polyperoxides -- `C1COOCOOCCC2OOOO1` from `('C','O')` -- which are
  valid, executable and RDKit-parseable, and not molecules. Narrowing or widening the element SET
  cannot fix this; the damage is in where elements sit relative to each other.
- **Branch-split chemistry against the run's own random selection (mean dQED -0.0837, 66.1% losing):**
      region_excision  yield 24.3%  mean dQED **-0.065**  mean dSA **-0.25**  endpoint SA med 4.13
      zero_excision    yield 55.4%  mean dQED **-0.125**  mean dSA **+1.97**  endpoint SA med **6.23**
  The excision branch is slightly BETTER than random selection; the construction branch is about twice
  as damaging and pushes SA to effectively unmakeable. **Yield is 94-98% for builds of <=4 atoms, so
  REACH IS NOT THE LIMIT -- CHEMISTRY IS.**
- **A library can pin a branch to the worst end of a curve.** All 8 catalog plans with no excision
  demand ask for 12-13 insertions, so the zero-excision branch was measured at the worst length by the
  LIBRARY, not by the move class. The first reading ("short parent-first programs improve
  drug-likeness") was too broad and a length control refuted it. Control the variable the library
  happens to fix before attributing an effect to the mechanism.
- **A PHANTOM MUTATION SURVIVOR: the test of the test was missing.** A mutation string written with
  single quotes against double-quoted source made `str.replace` a silent no-op, so the unmutated file
  passed and read as a weak guard. Same class as a tautological expectation. **A mutation runner must
  diff the file and refuse to report a verdict when the mutation did not apply.**
- **A field that echoes the request cannot witness that the request was met.** `rings_closed` was
  copied from `intent.close_bonds` -- the DEMAND -- rather than counted from the realized actions, and
  survived the battery. Count outcomes from executed actions, never from the specification that asked
  for them. Companion finding: **a support floor is only tested where it binds** -- a floor test that
  never hit a negative margin let a law mutated from re-ranking into filtering pass.
- **An assumption encoded as a test can be refuted by the test.** A single inserted atom CAN close a
  three-ring onto a neighbour of its anchor; the branch was more capable than assumed, and the test
  written to pin the assumed limit failed instead. Worth doing deliberately.
- `enumerate_cycle_close_edges` runs the semantic admission mask and is expensive: 92 zero-excision
  requests cost roughly the wall time of 999 excision ones. Worth it -- a hand-rolled pair search would
  move the ring chemistry out of the kernel and into the module -- but budget for it.

## 2026-09-21 (the chemical prior fixes SA, not QED -- and it was measured at a call site reaching 8.8% of proposals)

- **PREDECLARED PREDICTION FAILED ON QED, AND THE PRIOR IS A LARGE CLEAN WIN ON SA.** Both real,
  neither cancels the other. Paired on **913 identical decisions** (same parent, same family, same
  candidate list, same RNG; only the selection rule differs), 80 real production parents, 2,448
  attempts/arm, pinned kernel WITH torch so no mixed-environment caveat applies:
      mean dQED   A -0.0383   B -0.0402   paired delta **-0.0018 +- 0.0035  (sigma -0.5)**  FAILED
      mean dSA    A +0.5192   B +0.3313   paired delta **-0.1879 +- 0.0297  (sigma -6.3)**  WIN
      %SA<=4      26.3% -> 33.9% (T1.0) -> 36.4% (floor .02/T0.15)
  The falsifier was sealed at `ae53cd07` BEFORE measuring and is byte-unchanged; it was not relaxed.
- **TWO CONTROLS MAKE THE QED NULL A REAL ANSWER RATHER THAN A SHRUG.** (1) **93.3% disagreement
  rate** -- the prior changes the decision on essentially every draw, so this is not "guidance too
  weak to bite", and sweeping temperature to 0.15 does not rescue QED. (2) **Paired dHeavy =
  0.000000 with ZERO variance** -- the SA win is structural, not a size artifact, because all five
  families at that site preserve heavy-atom count. Support identity held 306/306 (B's candidate list
  is always a permutation of A's), so the prior re-ranks and never filters.
- **THE SCOPING FINDING, AND IT REDIRECTS THE WORK: the call site reaches only 8.8% of production
  proposals (503 of 5,742 archived entries), and all five of its families PRESERVE heavy-atom
  count.** So `atom_insert` (n=498) and `atom_delete` (n=324) -- the two largest net-negative
  families in the per-edit census -- **never route through it**. **A QED null measured there is not
  a result about the proposal stream.** I sent the agent to the wrong site; the damaging families
  and the model's strongest capability are both somewhere else.
- **THE MODEL DOES SCORE PLACEMENT, which is exactly what the parent-first construction branch
  needs.** `atom_insert` marks carry `neighbors` as well as `atom_type`. Enrichment against a
  uniform draw over the same legal insert marks, oxygen-bearing production parents:
      C_onto_C 7.98 | C_onto_N 6.33 | O_onto_C 2.85 | N_onto_C 1.42 | O_onto_N 0.52 | N_onto_N 0.45 | O_onto_O 0.15
  Heteroatom-onto-heteroatom suppressed **~15-18x** relative to C-C -- precisely the polyperoxide /
  `N#CCNOCO` class the construction branch emits. **CAVEATS THE AGENT STATED AND I AM KEEPING: the
  O_onto_O row rests on 5 states (indicative, not measured), and the MAGNITUDES ARE NOT STABLE
  across parent subsets** (an all-parent sample put O_onto_N at 0.118 where the oxygen-restricted
  one puts it at 0.52). **The ORDERING reproduces; never quote a single magnitude.**
- **CONSEQUENCE: wire the prior into the CONSTRUCTION lane, not `current_state_program`.** That is
  where the placement knowledge and the measured damage are both located. `path_log_likelihood`
  ships for ranking a constructed path -- rank different-length paths by `per_step`, same-length by
  `total`, always report `unjoined_steps`.
- **A per-family attribution from 9-15 states was BACKWARDS against 199-300.** An earlier signal
  probe reported signal in atom_restate/cycle_close and none in bond_reroute; the better-powered
  paired data has **bond_reroute improving most** (dQED -0.0311 -> -0.0188). The paired numbers
  supersede it. Also: SA improves in ALL FIVE families while QED improves in none materially, and
  `cycle_close` is dQED-POSITIVE in both arms with the worst dSA (+0.9) -- independently
  reproducing the "QED and SA disagree about ring closure" finding from the construction branch.
- **DIVERSITY IS THE TRADE AND IT IS MILD AT THE DEFAULT.** Distinct endpoints 1464 (A) -> 1359
  (92.8%) at floor .05/T1.0 -> 1136 (77.6%) at floor .02/T0.15. Parents covered identical (67 of
  80) in every arm. The sharpest arm buys +10.1 points of %SA<=4 for -22% distinct endpoints;
  floor 0.05 / T1.0 is the defensible operating point.
- **Do NOT weight the FAMILY choice with this model.** It puts 0.38 on atom_restate but only 0.063
  on cycle_close -- and cycle_close is the one family whose edits GAIN QED. Keep family uniform.
- **THE CHECKPOINT IS PROVISIONAL AND WAS SINGLE-COPY.** `ringcore_a7546e2_best.pt` (16,000 steps,
  scope `3721d69851110fdd`, ring catalog `639ff6078c32d43c` matching the frozen production value)
  is a `PROVISIONAL_EDITING_CHECKPOINT` selected by hazard-inclusive GM loss, which
  `configs/ringcore_v1_checkpoint_selection.json` **forbids for frozen results** -- so it is fine
  for a mechanism probe and NOT for a published number. **Its Modal run directory is gone from both
  profiles**; backed up to `~/compose_ckpt_backup/` and verified byte-identical (`24117dfe...`).
  Every remote sibling is <=3000 steps.
- **Precedent worth knowing: `control/learned_proposal.py` already exists** --
  `return_weighted_proposal_nce_v1`, a 4096-dim linear NCE over Morgan fingerprints trained on
  campaign RETURN, consumed by rejection sampling, whose result was a **best-score null** and whose
  README says "do not extend this recipe unchanged". It does NOT bear on a chemistry prior: it is
  task-trained, so it carries objective information and is a different object. If anything its null
  supports the diagnosis that the missing thing is chemistry, not return.

## 2026-09-21 (teacher-route prior: recognises regions, generates WORSE programs -- do not put it under PMO)

- **DECISIVE NEGATIVE: a teacher-route structural prior LOWERS complete-program yield against a
  generic control, on BOTH populations.** 48 draws per parent per arm, 11,232 draws, module count
  drawn once per (parent, draw) and SHARED by every declared arm so no arm can win by declaring
  shorter programs:
      arm                    T4 held-out (15 parents)   generic ZINC (24 parents)
      declared_uniform CTRL        0.850                      0.962
      declared_prior TEACHER       0.749                      0.929
  Paired: T4 **-0.1014** [95% -0.164, -0.042], sign +3/-8, p=0.227; generic **-0.0330**
  [-0.057, -0.006], +5/-17, **p=0.017**. Diversity moves the same way (generic -2.50 distinct
  endpoints/parent, p=0.0043). **Exact teacher-endpoint recovery is 0 of 11,232 draws in EVERY arm.**
- **ATTRIBUTION IS CONSISTENT ACROSS BOTH POPULATIONS AND NAMES ONE COMPONENT:** the REGION LAW costs
  the yield (T4 -0.064 p=0.0034; generic -0.026 p=0.0001) while the FAMILY projection is NEUTRAL
  (T4 -0.019 p=1.00; generic +0.010 p=0.63). Mechanism: teacher-like regions are LARGER, and larger
  excisions are refused more often by the executor.
- **THE RECOGNITION-vs-GENERATION LESSON REAPPEARED ONE LEVEL DOWN, AND THIS TIME INSIDE THE REGION
  LAW.** Leave-source-out against an ANALYTIC uniform control (exact hypergeometric, no sampling
  error on either side) the learned law ranks the teacher's region better -- recall@5 **0.867 vs
  0.665**, MRR 0.471 vs 0.410 -- and generates significantly WORSE complete programs. And the
  ranking edge itself is **not established**: the exact Poisson-binomial test (each source has its
  own support size and target count, so the control is NOT one binomial rate) gives p=0.054 on the
  single best cell of four and 0.39/0.48/0.52 on the rest, at n=15 sources. **Better at recognising
  where a teacher cut; worse at producing a program.**
- **ONE COMPONENT SURVIVES, AND IT IS A T4 RESULT ABOUT REGION SCALE, NOT A PMO PRIOR.** On the
  production path, 15 parents x 48 draws, released atoms per draw:
      production_v1 (shipped)              yield 1.000   mean 1.44   max  8.33
      production_route_law (LEARNED)       yield 0.994   mean 3.86   max 21.00
      production_free_gate_v1 (engineered) yield 0.999   mean 3.27   max 15.00
  Paired vs v1 the learned law is **+2.424 atoms, unanimous +15/-0**, and it beats the engineered
  task-gated law (+0.596, +11/-4) **without consuming the similarity reference and delta that
  `free_gate_margin_v1` requires** -- which is why it is expressible for PMO where neither exists.
  Every clean eligible witness in the five exhausted T4 delta=0.6 cells is a 7-15 heavy-atom
  excision, **a class v1 cannot express at max 8.33.** Decide this on T4's own rescue evidence.
- **CONTAMINATION FINDING, act on separately: `t4_compositional_structural_subgoal_generator` is NOT
  admissible as a no-prescreen PMO prior.** Its `training_trace_counts` are
  `{"pmo_dependency_region": 85, "t4_complete": 50}` in fold 0 and 85 of 191/193 in folds 1-2 --
  **44-63% of its training routes are PMO routes**, from a corpus whose own result file states "All
  PMO supervision is answer-known, panel-informed or winner-informed development evidence".
- **Data hygiene that worked: make the fitter RAISE rather than rely on discipline.**
  `route_prior_fit.py` refuses `pmo_route_distillation/`, `pmo_teacher_route_gap_v1.json`,
  `pmo_winner_program_curriculum/` and `pmo_public_winner_recovery/`, and the certificate records
  `pmo_oracle_values_read: 0`, `pmo_task_identities_read: 0`, `docking_scores_read: 0`.
- **A FLOOR GUARD THAT ASSERTS `min(weight) > 0` IS SATISFIED BY THE UN-FLOORED LAW.** That mutation
  initially SURVIVED; repaired to assert the floor actually BINDS. Second time tonight this exact
  shape appeared -- a floor is only tested where it binds.
- **Teacher routes do not decompose into production modules:** only **65 of 238** dependency
  components match a single family's emission signature, so a fitted family head is the wrong shape
  and the family term had to be projected through a MEASURED emission matrix instead.
- **Limit that belongs beside every number here:** the teacher routes are COMPILED WITNESSES -- a
  search reconstructed a reported endpoint -- so the corpus rule mix is partly a property of that
  compiler. Yield is unaffected, but **no claim that the prior learned chemistry rather than
  route-compiler habit is supported**, and 15 source molecules carry every leave-source-out number.

## 2026-09-21 (resuming a scored run: four ways a resume is not a resume)

- **A flag accepted by `__init__` but NOT by `restore()` makes the fresh path work and the
  RESUME path raise, and nothing notices until you resume.** `run_program_campaign` passes
  `optimizer_kwargs` to BOTH the constructor and the `restore` classmethod, so
  `PmoPopulationController.restore()` died with `TypeError: unexpected keyword argument
  'enable_online_memory'` on the first 1k extension -- after the ledger had already been
  widened, before a single call was charged. The 250-call runs never exercised it because
  each lived in one container. Worse than the crash: had `restore` silently accepted and
  dropped it, a resumed arm B would have rebuilt as arm A. The guard test asserts both
  signatures carry the same arm parameters, so a future flag cannot repeat it.
- **An extension that archives-then-unlinks is NOT IDEMPOTENT, so its own failed attempt
  reads as a forbidden retry.** The extension path archives `result.json` to
  `result_at_<N>_calls.json` and removes it BEFORE the run starts. The attempt that died on
  the TypeError therefore left a folder with no result at all, and the guard -- keyed on
  `result.json` -- refused to resume the very state it had created. Fix: the ARCHIVE is
  strictly stronger evidence than `result.json`, because it proves the prior run completed
  AND that this extension has run before. Gate resuming from it on the prior attempt having
  TERMINATED (`started.json` AND `failure.json`); `started` without `failure` means a
  container may still be live and must be refused, not resolved. Cross-check the archive's
  name against the charged-call count inside it -- two records of one number.
- **In-process state that was never serialised cannot be "restored" -- but it can often be
  RECONSTRUCTED, and the distinction must be stated.** Arm B's online memory accumulated
  across 250 calls inside one container; persistence landed afterwards, so its snapshot
  carried no `online_memory` key (verified by reading the real snapshot, not inferred). A
  plain resume would have restarted it COLD, quietly turning a 1000-call memory experiment
  into a 250-call one followed by a reset, with nothing in the artifact to show it. It was
  recoverable because EVERY quantity the memory holds is a sum or a bounded max over counted
  observations, all durable in the archive -- only the audit ordinal on donor rows depends on
  arrival order. Replaying them through the SAME production observation path reconstructed it
  (234/234 entries, 0 attribution failures, ordinal 234 -> 250 by the first resumed round).
  **Byte-equality with the live object is unavailable and must not be claimed.**
  GOTCHA: `observations` is keyed by RECEIPT id and `entries` by its own key -- key overlap
  is ZERO -- so the join is on ENDPOINT. A join on the key reconstructs nothing, which is why
  the reconstruction REFUSES rather than resuming empty when it replays 0 of N.
- **A revision boundary is MEASURABLE, not a matter of argument.** Restore one real
  production snapshot under each revision and compare every leaf of the re-emitted state:
  **2 differing leaves of 218,869** for arm A, both new bookkeeping keys carrying `None`.
  Entries, observations, population state, credit, pool continuity and all three RNG streams
  identical. That is what licenses saying "proposal semantics unchanged"; a diff of the source
  would not have, because it cannot show what the code actually reads.
- **A REDEPLOY REBAKES THE CONTRACTS AN IN-FLIGHT RUN IS BOUND TO.**
  `add_local_file(..., copy=True)` fixes the contract at IMAGE BUILD time, and the worker
  refuses any payload its baked contract does not match. So a second fix that moves the
  contract hashes cannot be deployed while a scored run is live without risking that run on
  its next preemption retry. Park the second fix; record which payload each running arm is
  actually bound to rather than re-pinning the repo to match a running job.
- **`modal volume ls <vol> <subpath>` SILENTLY RETURNS THE PARENT LISTING** for a valid
  subdirectory, and `<subpath>/` returns "No such file or directory" for a directory that
  exists. Both readings are wrong and neither errors usefully. Use the Python API
  (`modal.Volume.from_name(...).listdir(path)`), which is exact. Companion to the 2026-09-20
  `modal volume get` collapse gotcha. Also: `Volume.reload()` raises
  `reload() can only be called from within a running function` -- it is not available locally.

## 2026-09-21 (a blank T4 cell had a molecule; the driver was waiting for a human)

- **"No molecule" in the combined table can mean "the rescue arm holding the molecule is not
  merged yet", not "nothing was found".** `5ht1b_s8_d0.6` read as blank while its live
  protonation-rescue arm already held **best -10.8 at 88 charged calls** against GenMol's
  -10.50 -- a WIN sitting unmerged. Read each rescue arm's OWN status before calling a cell
  blank; the panel table and the rescue volumes are different artifacts.
- **`state: "terminal"` in `driver_state.json` is not a failure and not a stall -- it is the
  fail-closed driver WAITING FOR A HUMAN**, and it burns nothing while it waits. The d06 arm
  sat terminal at 88 of its own authorized 248 calls, i.e. 160 calls of an ALREADY-GRANTED
  authorization unspent, and the stall watchdog reported the sibling d04 cell instead. Check
  `driver_state.json` per arm; a watchdog keyed on lock mtime does not see this state.
- **Confirm "the prior call is terminal" by GETTING the call, never by inferring it from a
  zero task count.** `modal.FunctionCall.from_id(cid).get(timeout=0)` raised
  `continuation remained reserved; fail closed without running a phase` -- that RAISE is the
  evidence `--confirm-prior-call-terminal` asserts. A listing showing `tasks=0` is not
  evidence; the same listing misled a phantom-run diagnosis the day before.
- The advance is `modal run --detach <app> --mode advance --run-id <id>
  --confirm-prior-call-terminal`, and `--detach` is mandatory because `main()` uses `.spawn()`.

## 2026-09-21 (PMO 1k A/B: online structural memory wins at every budget)

- **MEASURED, matched 1000-call A/B on `celecoxib_rediscovery`, identical initialization, identical
  code, payloads differing in exactly three leaves (`/arm/name`, `/arm/enable_online_memory`,
  `/extension/extends_payload_sha256`):**
      calls    A best   B best   A top10  B top10
        250    0.2385   0.3441    0.2177   0.3102
        500    0.2857   0.3838    0.2713   0.3556
        750    0.3061   0.3838    0.2930   0.3671
       1000    0.3364   0.3838    0.3287   0.3714
      AUC@1000  0.2489   0.3138   (+26.1%)
  B wins at EVERY checkpoint on EVERY metric, and **B reached its final best at 500 calls** -- A
  never caught it over the remaining 500. The effect did NOT decay with budget, which is the
  question a 250-call result cannot answer. ONE task, ONE seed: this justifies scaling the
  algorithm, it does not establish a benchmark claim.
- **Mechanism is ATTRIBUTED, not self-reported.** Memory provenance is read from the tag the memory
  channel writes at synthesis time (`pmo_online_memory` in the candidate's provenance metadata), not
  from a controller flag: 82 memory-derived calls of 250 rising to 198 of 500, and 3/12 then 4/14
  frontier improvements from memory-derived descendants. Arm A's zeros are STRUCTURAL (no memory
  object exists), which is what makes it a control rather than a missing measurement.
- **A frontier improvement is only defined on the CHARGED SEQUENCE.** The archive has no order, so an
  archive-wide attribution figure belongs to no checkpoint. Compute it over the charged-call prefix.

## 2026-09-21 (CORRECTION: `exact_early_ring` was never inert -- the probe measured the one quantity that cannot move)

- **CORRECTS the 2026-09-21 entry "`exact_early_ring` IS INERT on the mechanism".** It is not.
  MEASURED on 270 molecules / 646 ring decision points, matched three arms, pinned kernel:
      sequential  (what Lineage B trained on)   P_support(3/4-ring) 0.3049 +- 0.0065
      exact_early_ring (shipped 2026-07-20)                         0.1926 +- 0.0049
      ring_dependency_block (new)                                   0.1956 +- 0.0050
  The shipped scheduler delivers **-0.1169 +- 0.0071** against sequential and captures essentially
  the whole available gain. **The earlier n=24 probe concluded "inert" because it measured FREE
  SLOTS, which provably cannot move under a reordering** -- free slots are identical across all
  three arms (11.883). Measuring the one quantity incapable of showing the effect, then believing
  the null, is the error. Pick the metric that CAN move before reading a null.
- **The slot-scarcity story is dead and it pointed the WRONG WAY.** `r(mass, log
  legal_template_count) = -0.87 to -0.92` against `r(mass, free_slots) = +0.13`, **wrong sign**.
  Small molecules have MORE small-ring mass, so "create the minimum required atoms, then commit the
  ring" -- which is what the brief instructed -- moves the damaging direction. Mass tracks support
  SIZE, not crowding.
- **The residual is a per-ring-system-ORDINAL effect that NO schedule can remove.** Mean by ordinal:
  0 -> 0.1371 (passes the <0.15 gate), 1 -> 0.1890, 2 -> 0.2754, 3 -> 0.3255, 4 -> 0.4497. Each
  commitment constricts the support the next is decided against, and a trace carries ~2.5 ring
  systems. Along a real trace the last `atom_restate` cuts legal templates **1034 -> 295** (mass
  0.1199 -> 0.2237) and the first ring grow leaves **28** templates at 0.3571. So the repair is a
  SUPPORT question or a scoped claim, not a scheduling one.
- **CONSEQUENCE worth acting on separately: Lineage B trained on `sequential` while a scheduler
  worth 0.30 -> 0.19 has sat unwired since 2026-07-20** -- one day after the step-1000 checkpoint.
  Five additive plumbing hops, two sharing a sink.
- **A per-item fallback cannot protect a corpus.** Committing a ring early invalidates a LATER graft
  on 3 of 150 real molecules, and the per-system fallback cannot see it because the commitment it
  would undo already succeeded. Without a TRACE-level fallback the schedule silently shrinks the
  corpus ~2%, invisibly in any per-arm mean.
- **Modal `.map()` with ordered output can deliver ZERO rows** from a run that processed 536
  molecules across 62 containers, because one stalled shard queues all the rest -- pass
  `order_outputs=False`. And **killing the local driver does not stop the app**: it held 99
  containers and starved the next launch until `modal app stop`.

## 2026-09-21 (four agents, one laptop: the resource failure modes are the same shape as the data ones)

- **Uncapped local fan-out across concurrent agents took the machine to 65 MB free of 32 GB** with
  19.8M pages purged, and **19 of the 44 python processes had orphaned to `ppid=1`** -- parent dead,
  still holding ~0.25 GB and ~40% CPU each, forever. `pkill -f <script.py>` never matches them
  because a spawn-pool child's cmdline is `python -c from multiprocessing.spawn import spawn_main`.
  Sweep by `ppid==1` and by that string, and cap concurrent local processes per agent (3 here).
  Serial is also FASTER: the repo's own measurement has oversubscription taking a build from 5,507
  entries/h to 250.
- **A grep for orphans matches its own command line.** An agent reported `spawn_orphans=3` that were
  its own `grep`; bracket the pattern (`[m]ultiprocessing.spawn`) or the count is fiction.
- **Disk hit 3.7 GiB free of 926 GiB and an agent's `git worktree add` died with `No space left on
  device`.** A full checkout is ~800M per worktree; `git worktree add --no-checkout` +
  `sparse-checkout set src tests` is **20M**. Use sparse worktrees for agents that only need to run
  tests. Also: `du` cannot read `~/Desktop`/`~/Documents`/`~/Downloads` without Full Disk Access, so
  a home-directory survey can silently miss the majority of the volume -- 362 GB visible against 899
  GB used here.

## 2026-09-21 (fragments: a budget knob and a placement mechanism are ORTHOGONAL, and the prediction test proves it)

- **A COORDINATOR PREDICTION WAS FALSIFIED IN THE USEFUL DIRECTION.** I predicted the attachment
  mechanism's attributable delta would SHRINK once the baseline also ran at the tuned budget, on the
  reasoning that the 24-attempt deltas were budget-confounded. MEASURED at both arms 96:
  motif **41.80 -> 50.70**, decoration **80.67 -> 89.93**. The 24-attempt numbers were UNDERSTATING
  the mechanism. The agent reported this as a negative against the stated prediction rather than
  presenting the larger number as a win, which is the only reason it is credible.
- **WHY: the budget fixes CHEMISTRY, only the mechanism fixes PLACEMENT.** On the baseline arm the
  extra budget moves committed-per-attempt 93.47 -> 99.57 and exhaustions 7.5 -> 0.6 while task
  success moves **3.50 -> 3.57**. It converts exhausted trajectories into committed, chemically
  valid, fragment-CONTAINING molecules and steers none of them to the declared sites. So on the
  baseline a larger budget buys chemical validity that the task then rejects.
- **THE EXPLANATION WAS TESTED, NOT ASSERTED -- and this is the strongest evidence in the fragment
  workstream.** If chance placement is what lifts the baseline at all, the lift must COLLAPSE as the
  number of declared interfaces rises, because satisfying k sites by luck gets rapidly harder.
  Measured over all 20 drug-task pairs: 1 interface -> mean baseline gain **+6.67** (10 drugs);
  2/3/4/5/6 interfaces -> **+0.00 / +0.34 / +0.00 / +0.00 / +0.00**. **19 of 20 multi-interface
  pairs gained exactly nothing.** That is why motif's baseline moved +6.67 (one site) and
  decoration's moved +0.07 (two to six). Deriving a prediction from a mechanism and measuring it
  converts an attribution into a tested claim.
- **The negative control survives the configuration change**: superstructure declares no interface
  and its delta is EXACTLY 0.00 at 24 AND at 96 -- so it is not an artifact of the old setting.
- **Choose a budget knob at the point the budget stops BINDING, not at the largest value tried.**
  96 was chosen because exhaustions fall to 8.6 / 1.0 / 0.8 per hundred attempts, so the remaining
  loss is chemistry and a larger value has almost nothing left to recover. Verified it does not
  narrow the distribution -- motif uniqueness 78.22 -> 89.86 and quality 14.47 -> 21.03, i.e. harder
  trajectories FINISH rather than easy ones being favoured. Without that check the value would have
  been a number that happened to look good.
- **A knob that helps where the deficit was and nowhere else is a mechanism; one that lifts
  everything is a fit.** SPIRAPRIL decoration 49.67 -> 75.67 with exhaustions halved; LESINURAD,
  already 100% with zero exhaustions, unchanged at 100%.
- **Enforce a scientific scope rule STRUCTURALLY.** The knob is global by construction, a test
  asserts the parser offers no `--mark-attempts-for-task` / `--per-drug-*` option, and the value is
  hashed into the sampler identity so the aggregator cannot silently combine two settings. A
  convention someone has to remember fails eventually.
- **NEVER quote a tuned arm against an untuned baseline.** The agent caught this itself: a 96-attempt
  attachment arm against a 24-attempt baseline would have credited the budget's gain to the
  mechanism -- the same conflation corrected twice already in this workstream.

## 2026-09-21 (two proposal channels, one shared defect: the law ranks a molecule the harness never scores)

- **`expand` on the T4 path does NOT gate the molecule a proposal law ranks.** It abstracts the
  synthesized program via `extract_structural_goal`, expands `_variants`, re-binds each subgoal with
  `attachment_bindings(...).assignments[0]`, and gates whatever `instantiate_goal` builds. MEASURED
  over 150 draws: the program's own endpoint appeared in the gated set **zero times**,
  `recovered_fraction = 0.0000`. So a completion law measured at p = 0.00859 on MODULE endpoints
  cannot transfer end to end -- it is conditioning the right object for the WRONG consumer.
- **The PMO path is the opposite and it was verified by EXECUTION, not by reading:
  `recovered_fraction = 1.0000`.** 144 candidates, three independent hops -- endpoint recomputed
  from each candidate's own executed trace through the production decoder, program size profile
  against that trace, and the charged archive entry after a real
  `propose_batch -> lock_query_subset -> observe_batch` -- 0/0/0 mismatches. Structurally,
  `_generate_channel_pool` executes the program and takes `endpoint = trace["endpoint"]`, and
  `program_campaign` charges `ledger.query(candidate["endpoint"])`. No goal abstraction anywhere.
- **LESSON: before conditioning ANY proposal law, verify by execution that the molecule the law
  ranks is the molecule the harness scores.** Pin it as an invariant with a mutation that makes a
  candidate record carry a molecule its program did not produce. Two independent investigations hit
  this today from opposite directions, which is what makes it a shared assumption rather than a
  local bug.
- **A related convergence worth stating: the PMO ALLOCATOR cannot fund basins the proposal law never
  proposes** -- candidate pool holds 1.44 basins/round, 19 of 27 rounds offered zero
  discovery-eligible cells, 15.7% of reserved discovery slots fillable. Both findings point the same
  way: the leverage is in WHAT GETS GENERATED, not in how it is ranked or allocated.

## 2026-09-21 (route corpora: the only general pair compiler destroys the molecule first)

- **`compile_source_to_target` is `delete_to_null_then_construct_v1` and its own docstring says it
  "is intentionally not an edit-minimal alignment".** MEASURED: `retained_fraction = 0.000 on 4 of
  4` generic pairs, all routed through the formal NULL state -- **including a pair differing by ONE
  METHYL, which costs 25 steps on an 11-atom molecule** (12 deletions to nothing, then 13
  constructions). The route shape is INDEPENDENT of how close the pair is.
- **CONSEQUENCE: every semantic route feature is degenerate on its output** -- retained region,
  released region, replacement topology and interface all read "retain nothing, release everything".
  Fitting route prototypes on it would learn the compiler's deletion order, which is the
  `real_endpoint_multistep_path` artifact already on the record. A scaffold-preserving bridge
  compiler is the prerequisite for answer-known proxy tasks at scale, and it does not exist yet.
- **PROVENANCE, checked rather than trusted: `diagnostics/ivg_winner_paths/` is derived from a
  COMPETITOR's published winners.** `origin: github.com/invirtuolabs/InVirtuoGen_results`;
  `by_target` is 5ht1b/braf/fa7/jak2/parp1 and `by_cell` is `docking_*_thr4/thr6`, i.e. **T4
  docking cells, not PMO tasks**; its own `evidence` field says "target-informed locally replayed
  upper-bound witnesses, not blind recovery"; and the LICENSE notice is **non-commercial**. So it
  leaks no PMO answers, but it is competitor-derived, target-informed, small (160 routes / 82
  targets), NC-licensed, and **flatly inadmissible for T4, where those ARE the benchmark's answers**.
  A prior fitted on this exact corpus already failed on yield. Check a corpus's audit file before
  treating its name as a description.

## 2026-09-21 (INFORMATION BOUNDARY CORRECTED: a benchmark-declared target is a task INPUT, not leakage)

- **VERIFIED IN SOURCE, not assumed.** `tdc/chem_utils/oracle/oracle.py:748`:
  `celecoxib_rediscovery = rediscovery_meta(target_smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', fp='ECFP4')`
  The rediscovery/similarity targets are LITERALS in the public task definition. The same holds
  for albuterol/mestranol similarity and the other explicit structural tasks.
- **CONSEQUENCE, and it overrides the earlier reading of the standing rule.** `mission.md` says "Do
  not preload same-task PMO winners, hidden oracle information, target SMILES, or task->program
  lookups". That was written against WINNER ROUTES and PRESCREENED VOCABULARIES. Applying it to a
  target the benchmark itself declares was over-application: it is a task INPUT in exactly the sense
  that T4's delta and start molecule are inputs, and the mission already says those "are benchmark
  INPUTS, never reasons to retune". IVG's no-prescreen flag (`--use_prescreen`) is about not
  pre-scoring the ~250k ZINC set to build task-specific vocabularies, not about ignoring declared
  task structure. **If the benchmark declares the target, use the target.** Owner decision,
  2026-09-21; the mission line should be amended to say HIDDEN oracle information and UNDECLARED
  targets.
- **What must remain general is the CONTROLLER** that turns an arbitrary declared goal into
  executable programs -- never a per-task rule, never `if task == celecoxib`, never hand-written
  molecular content. Still prohibited: same-task winner routes, prescreened vocabularies, oracle
  internals, hidden component scores, uncounted same-task history, and computing a property on an
  uncounted candidate to select it.
- **ARCHITECTURE this licenses (one algorithm, 23 tasks):**
      goal/basin selection -> structural transport program -> exact execution -> online memory
  with only the GOAL SOURCE varying: `T_declared` where the task supplies a structure
  (rediscovery, similarity, SMARTS, formula) and `T_archive` -- diverse high-reward scored
  molecules as pseudo-targets -- for black-box tasks (gsk3b/jnk3/drd2).
- **This avoids the failed teacher REGION prior by construction.** Do not learn which region a
  historical teacher edited; COMPUTE the region from the G-to-T graph difference, yielding
  `(retain core, R_delete, H_install, alpha, D)` directly. The measured negative attributed the
  entire loss to the region law (p=0.0001) with the family projection NEUTRAL (p=0.63), so teacher
  routes are retained for REALIZATION knowledge -- "given a desired transformation, how do I realize
  it coherently" -- which makes this a compiler rather than an imitation model.
- **Memory is the LOCAL optimizer, not the global strategy.** Measured across three matched 250-call
  tasks: celecoxib win, gsk3b win by harder exploitation, perindopril LOSS. A mechanism that is
  task-dependent at the global level and reliable at the local one belongs downstream of basin
  selection, not in place of it.
- **The graph-difference planner IS the scaffold-preserving bridge compiler** that
  `compile_source_to_target` is not (`delete_to_null_then_construct_v1`, retained_fraction 0.000 on
  4 of 4 pairs including one differing by a single methyl). One build serves both the controller and
  the answer-known proxy-task laboratory, and it removes any need for the NC-licensed,
  competitor-derived `ivg_winner_paths` corpus.

## 2026-09-21 (a proposal law bites only where something is DISCARDED)

- **Verifying that the ranked object IS the scored object is NECESSARY BUT NOT SUFFICIENT.** On the
  T4 path `expand` keeps EVERY eligible endpoint -- it builds each variant, calls `fiber.check`, and
  on a pass writes `found[gate["smiles"]]`. No budget, no top-k, no selection among eligible
  endpoints. So a law ranking those endpoints cannot change what `expand` returns even though the
  ranked and gated molecules are provably identical. **Ranking only matters where something is
  discarded.** Ask BOTH questions of any proposal law: is the ranked object the scored object, and
  does anything downstream actually discard candidates.
- **A 24.8% structural discard that changes nothing.** `attachment_bindings` enumerates up to 128
  exact bindings per subgoal and the loop uses only `assignments[0]`; measured over 101 real
  subgoals, 25 carry more than one binding. Widening to k=4/combos=8 gave **byte-identical results
  on 3 of 3 cells**. A discard is only a defect if removing it changes an outcome -- census first,
  then measure, then decide; do not ship the widening on the census alone.
- **GENERATION mechanisms are not exposed to this trap** (they change what exists); SELECTION layers
  are. Site a bandit or re-ranker where the path is genuinely capacity-limited, which on PMO is the
  oracle budget and the per-round query count, not proposal admission.
- **DISCRIMINATING TEST for "outside the support" vs "rare under the prior": does the recovery
  fraction RISE with the draw budget?** The fragment path first read 0.376 at 128 draws, which looks
  exactly like T4's exposure; it rises monotonically to 0.50 / 0.70 / 0.85 at 64 / 256 / 1024. T4's
  0.0000 is a hard zero no budget lifts. Reporting 0.85 as a LOWER BOUND rather than claiming the
  remainder absent is the correct form. Without the power curve a viable design would have been
  withdrawn on a false positive.

## 2026-09-21 (T4 fa7_0 at delta=0.6: REACHABLE but not reached; the barrier is the goal-abstraction layer)

Written for merge into `.claude/context/learnings.md`. Kept in the `t4_fa7_0_*`
namespace rather than appended directly, because other agents were editing the
shared file concurrently.

All numbers MEASURED under the pinned kernel (python 3.11.13 / rdkit 2024.03.5 /
numpy 1.26.4 / torch 2.4.0), `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
ZERO oracle calls, zero docking, zero Modal across every result below.

### The headline: fa7_0 is reachable, and the yield does not clear a rescue bar

- **The cell is NOT infeasible.** All five stored witnesses pass the unmodified
  production `Fiber.check`, and a production configuration independently
  GENERATED an eligible endpoint from the real seed. So "delta=0.6 admits nothing
  for fa7_0" is false and should not be written.
- **It is also not solved.** Across 3 seeds x 480 draws with the region law on,
  the completion law produced **one** eligible endpoint, at one seed. That is
  not reproducible seed-to-seed and does not justify docking calls.
- Report it as a scoped negative: **reachable, not reached at this budget**, with
  the barrier located rather than guessed.

### The one generated molecule, and why the qualitative result beats the count

  OFF arm, 1,440 draws, ONLY endpoint:
    C=c1ccc(CN(CCC(C)C)C(=O)c2cccc3ccccc23)cc1=C
    sim 0.6897 / QED 0.6422 / SA 2.5254, heavy delta -5
    an exocyclic quinoid that has DELETED the amidine -- the FA7 S1-pocket
    pharmacophore. ALERTS 2 -> 0, which is exactly where its QED came from.
  ON arm (completion law), ONLY endpoint:
    COC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1
    sim 0.6610 / QED 0.6264 / SA 2.331, heavy delta -8
    retains BOTH amidine and carboxamide; passes under benchmark_only,
    compose_valid AND legacy_screened; round-trips through the executor.

**Benchmark eligibility and chemical plausibility are two numbers and they
disagree here.** The unconditioned path's only success is a molecule nobody would
defend; the conditioned path's only success is a credible one. State the kind of
molecule beside the count, or the count misleads in both directions.

### Where the barrier IS: the goal-abstraction layer

`expand` does not gate the molecule a proposal law conditions. It abstracts the
synthesized program with `extract_structural_goal`, expands `_variants`, re-binds
each subgoal with `attachment_bindings(...).assignments[0]`, and gates whatever
`instantiate_goal` builds.

  **`recovered_fraction = 0.0000`** over 150 draws -- the program's own endpoint
  appeared among the gated endpoints ZERO times.

So a law that ranks the module endpoint improves a molecule the campaign never
scores. That is the barrier, and it is architectural rather than chemical.

### TWO INERT-REPAIR TRAPS, and the rule that covers both

1. Conditioning the MODULE endpoint is inert because the campaign scores
   something else (`recovered_fraction = 0`).
2. Conditioning the GATED endpoints is inert for the opposite reason: `expand`
   keeps EVERY eligible endpoint (`found[gate["smiles"]] = {...}` on each pass,
   no budget, no top-k). There is no selection, so a ranking cannot change what
   it returns.

**RULE: a proposal law bites only if BOTH hold -- the ranked object IS the scored
object, AND something downstream DISCARDS candidates.** Verifying the identity is
necessary and NOT sufficient. Check for a discard before building a ranker; it is
one read of the consuming loop and it is cheaper than the law.

### Placement is a REAL axis pointing the WRONG way (do not spend on it)

`attachment_bindings` enumerates up to 128 bindings per subgoal and the loop used
`assignments[0]`. Measured over 57 real fa7_0 goals / 101 subgoals:
**24.8% carry more than one binding** ({1: 76, 2: 21, 4: 3, 6: 1}).

Widening to four bindings, 240 draws, wrapped gate, 0 gate disagreements:

    narrow (k=1)   3,568 distinct endpoints offered    0 eligible
    broad  (k=4)   3,961 distinct endpoints offered    0 eligible
    ADDED 393 (+11.0%)   LOST 0   ADDED-AND-ELIGIBLE 0
    first refusal of the added: similarity 264 (67%), unparseable/over-ceiling 94
    (24%), qed 35 (9%)

So the discarded placements are **DISTINCT_BUT_INELIGIBLE**, not duplicates: the
axis is real, 11% more molecules are reachable through it, and the gate refuses
all of them -- two thirds on SIMILARITY. That is mechanistically unsurprising for
a similarity-constrained task: moving the attachment site moves away from the
reference, so `assignments[0]` is already near-optimal for the binding constraint
and every alternative is worse. **Placement is not the lever for T4 at delta=0.6.**
It was also byte-identical on eligible counts across 3 of 3 cells at 480 draws.
Left default-off; do not turn it on (8x the instantiate-and-gate work for nothing).

Side observation worth keeping: 94 of the 393 are unparseable or over-ceiling,
including disconnected fragments (`C=O.CC(C)...`) and radicals (`[CH2]`, `[C]=O`),
which independently corroborates that the intervention layer still emits radicals.

### What DID work, and stands on its own

The replace-completion law (`src/compose_v4/control/replace_completion_law.py`,
contract `proposal.shallow.completion_law`) conditions the completion half of
`segment_replace`, which drew blind: `rng.integers(1, capacity+1)` over 1..8 plus
a uniform C/N/O chain. Measured, every eligible completion inserts ONE or TWO
atoms, so ~75% of the draw mass landed where nothing is eligible.

  480 draws per arm, arms differ ONLY in the contract field:
    fa7_1     7 -> 13 eligible   (+86%)
    braf_2    4 ->  6            (+50%)
    5ht1b_0  20 -> 35            (+75%)
  No control regressed. 6/6 mutations killed, both shared-sink hops separately.
  Absent field = byte-identical OFF; a uniform law is NOT a drop-in (it reproduces
  the OFF draw only ~1/K of the time), which is why none is registered.

**This is a general improvement to the T4 proposal law and is worth having
independently of whether fa7_0 ever closes.**

### Method notes that cost something to learn

- **A mutation that silently fails to apply must ABORT, not score as killed.**
  Two mutations missed on an indentation mismatch; the runner refusing to emit a
  verdict is what caught it.
- **A floor test whose margin RETURNS a bad value never reaches the branch it
  claims to guard** -- it must RAISE. One mutation survived on exactly this and
  the repaired test kills it. A guard is only tested where it binds.
- **A broad per-draw `except` will swallow your own probe's bug.** Slicing a
  2-tuple as `[:3]` reported `executed=0 refused=600`, which reads as "the module
  is inexpressible" and is instead a typo. Assert the return SHAPE.
- Wrap the live gate to see a funnel; `expand` returns only survivors, so its
  interior is invisible from the return value. Require the decomposition to
  predict the real verdict on every call (`gate_disagreements == 0`) or the
  attribution is void.

## 2026-09-21 (CORRECTION to the support test: a zero surviving ONE budget increase is not a hard zero)

- **SHARPENS the "does the recovery fraction rise with draw budget" test recorded earlier today.**
  I stated it as a single increase. That is too weak. MEASURED on the fragment panel: **LOVASTATIN
  motif read 0.0000 at 256 draws AND again at 2048** -- an eightfold increase buying nothing, which
  imitates a hard zero convincingly -- **then lifted to 0.80 at 8192.** BARICITINIB did the same on
  a smaller scale (0.0000 at 256 -> 0.4615 at 2048). Every apparent zero on the panel eventually
  lifted. T4's `recovered_fraction = 0.0000` survives ANY budget, which is what makes it genuinely
  outside the support. **Require REPEATED increases across a wide range before calling a zero hard**,
  and report the realized fraction as a LOWER BOUND, never as a claim that the remainder is absent.
- Panel result for the record: superstructure **0 redirects on 10 drugs** -- no exposure by
  construction, since there is no declared interface to steer toward; motif 0.5728 at 256 draws;
  decoration 0.3763 at 128 (the worst case). In-support on all three, so a program may be scored
  on this path.

## 2026-09-21 (fragment linker v1 FALSIFIED: the transaction never occurs under the prior)

- **v1 fired its own predeclared falsifier.** The path program used `realized_linker_length` as its
  state variable with a per-event monotone path predicate replacing coverage staging. Predeclared
  falsifier: "realized length stays at the seed for >90% of committed endpoints". MEASURED: it
  commits **ZERO** endpoints -- 61 refusals across four drugs while the attachment-only arm commits
  26. Worse than the threshold, so it fires unambiguously. Predeclaring the criterion is what made
  this a clean result instead of an argument about a threshold.
- **ROOT CAUSE, measured three ways, and it is not a strict predicate.** (a) 600 model draws from a
  seeded start: 30 passed the region lock, **all 30 left length unchanged, 0 lengthened it**.
  (b) With the program OFF entirely, **1,440 events across 120 rollouts never once reached a length
  above 1**. (c) Mechanism: both bonds of the seeded bridge are **bridges in the graph sense**, so
  deleting either disconnects the molecule and the executor refuses the state. Lengthening therefore
  requires **close-then-open** -- ring through the far core first, then remove the original bond --
  a coordinated transaction of at least three events.
- **So a per-event predicate cannot gate it:** the first step creates a pendant, which does not
  change path length, so the predicate refuses it and the trajectory dies at event zero. The design
  assumed a predicate could steer an existing capability; **the capability is absent, so there was
  nothing to steer.**
- **RULE for any composite/macro primitive: the in-support check applies to EACH CONSTITUENT.** A
  composite whose constituents are each individually rankable by the prior is an ACCELERATION -- it
  bundles moves the prior could already make in the wrong order. A composite requiring even one
  constituent the prior cannot rank is a CAPABILITY EXTENSION wearing a macro's clothes and must be
  declared as such. Run the rising-power-curve probe on each constituent separately and report the
  curves.

## 2026-09-21 (the session's unifying pattern: absent by construction vs badly searched)

- **All four workstreams converged on the same distinction, and in every case only a measurement
  separated the two readings:**
  - **T4 / fa7_0** -- the barrier is ARCHITECTURAL, located at the goal-abstraction layer
    (`recovered_fraction = 0.0000`), not a proposal-coverage failure. Two obvious repairs measured
    INERT, and placement is a real axis pointing the wrong way under a similarity ball.
  - **De novo / small rings** -- a consequence of a DECLARED v1 host scope (`_eligible_grow_host_graph`
    admits only acyclic neutral carbon and asserts a forest), not a reward, policy or scheduling
    failure. On 3 of 12 audited rollout states EVERY legal template is a small ring.
  - **PMO / transport** -- the required moves are ~21-40 primitives against a measured realization
    ceiling of median 16 / max 23, so a single program cannot cover most transports. Staging is
    forced, not chosen.
  - **Fragments / linker** -- the close-then-open transaction never occurs under the prior at all
    (0 of 1,440 unconstrained events lengthened the path).
- **In every case the tempting reading was "the search is bad" and the correct one was "the
  capability is absent by construction".** They are indistinguishable from an aggregate score and
  are separated only by instrumenting the support the decision is made against. Where the capability
  IS present and merely rare, budget lifts it (fragments' 0.0000 -> 0.80 at 8192); where it is
  absent, no budget does (T4's hard zero, de novo's states with no non-small option).

## 2026-09-21 (goal-conditioned transport, Step 1: read evidence understates what the path installs)

- **A COORDINATOR WARNING FALSIFIED, and the correction is the rule.** I warned that sulfur is not
  installable because every growth module reads `("C","N","O")` / `("C","N","O","F")` in source, so
  S/P/Cl/Br/I are reachable only via `atom_restate_semantic` -- and told the agent to enumerate
  unreachable targets before testing celecoxib. MEASURED on the real proposal path from CNOF-ONLY
  sources: endpoints containing **B, F, I, P and S** across 97 endpoints (S in 4, P 8, I 6, B 21).
  **The vocabulary READ from source understates what the path INSTALLS**, and where read evidence
  and executed evidence disagree the executed one is authoritative. Celecoxib's sulfonamide is not
  blocked. Only `amlodipine_mpo` (Cl) went unreached, correctly labelled a BOUND not a proof --
  presence at n=97 demonstrates installability, absence demonstrates nothing.
- **TRANSPORT SCALE, measured over 108 correspondences on 15 declared-structure tasks: min 19 /
  median 36 / max 56 primitives**, against a realization ceiling of median 16 / max 23 with
  teacher-scale plans at 29-40 never binding. **Staging is forced, not chosen.** Note the estimate
  ROSE from the earlier raw-MCS reading of 21-40 under two corrections; a correction that only ever
  shortens the work it implies is the one to distrust.
- **Three defects that validation found rather than confirmed:**
  (a) RDKit's `completeRingsOnly` does NOT yield a ring-complete core -- it still admits a LONE RING
  ATOM as an attachment point, so celecoxib's core came back as benzene plus one pyrazole atom whose
  other four are deleted, claiming an atom is already correct while its ring must be rebuilt. Prune
  to a FIXED POINT, because dropping one atom can break a ring that was fully covered.
  (b) An atom-only core representation reports `scale = 0` for benzene -> cyclohexane. Retaining an
  atom skeleton does not retain BOND ORDERS; add `core_bond_changes` and guard it with "scale 0
  implies the canonical SMILES are equal".
  (c) **A tautological assertion hid a real ordering defect.** `len(fragments) >= 1` is always true;
  made real (exactly one fragment must survive at EVERY step) it FAILED -- an in-set-degree peel
  reached two fragments, because in-set degree ignores how an atom connects to the retained core.
  The fix was to change the ORDERING, not the assertion.
- **CROSS-STREAM: two independent workstreams hit the same structural requirement.** 7 of 108
  correspondences fail connectivity, ALL on `median1` (bridged bicyclic camphor): when the retained
  core's pieces are joined only through deleted atoms, **no ordering can preserve connectivity** --
  the transport is an excision PLUS a reattachment bond. That is the identical move class as T4's
  `5ht1b_2` witness (8-atom interior two-bridge excision plus one reattachment). Declare it
  (`requires_reattachment`) so the planner picks another alignment or stages the bond. NB
  `bond_insert` is outside the frozen Active8 codec surface, so the reattachment primitive is
  `cycle_close`.
- **A diagnosis field that reads as a failure will be treated as one.** Adding `requires_reattachment`
  to a `validate()` result broke five tests doing `all(checks.values())`, because `False` there is a
  HEALTHY outcome. Separate VERDICT keys from DIAGNOSIS keys explicitly and share the split between
  the driver and the tests.
- **PMO DISCARDS, unlike T4** (the second gate): merged pool 32-33 -> selected 16, **16-17 discarded
  per round, 6 of 6 rounds capacity-limited**, while `lock_query_subset` discards **0**. So a
  selection layer sited at `_credit_allocate` is load-bearing and one at the lock would be inert.
  Gotcha: the full pool is published as `eligible_pool`; `proposal_pool` is the LOCKED batch's key.
- **GOAL AUDIT, read from pinned PyTDC 1.1.15 with `ast` (TDC never imported, no pickle
  downloaded): 19 of 23 PMO tasks DECLARE structure; only 4 are black-box** (drd2, gsk3b, jnk3,
  qed). Declared target structure 5, reference-in-composite 6, SMARTS 3, molecular formula 3, target
  pair 2. A first pass reporting 5 declared / 10 black-box was wrong in the direction that matters,
  from three patterns now each pinned by a test: a keyword naming a MODULE CONSTANT
  (`median1` -> `camphor_smiles`), a reference assigned to a LOCAL inside a function body (every
  composite MPO and `valsartan_smarts`), and a task defined as a CLASS (`jnk3`). Keep `unresolved`
  distinct from `black_box`: one is a finding about the benchmark, the other a gap in the reader.
  SCOPE: for the six composite MPOs the declared reference is ONE TERM of a multi-objective score,
  so transporting to it is a heuristic and those must be validated on TASK SCORE, never on
  similarity-to-reference.

## 2026-09-21 (THE PMO PLATEAU EXPLAINED: the transport path exists and the score signal walks you off it)

- **VERDICT `STAGED_PATH_EXISTS_BUT_SIGNAL_IS_NOT_MONOTONE`.** 44 transports from the init bank to
  declared targets, split at the measured 23-primitive ceiling, intermediates built with RDKit from
  the correspondence so a failure is a failure of the PLAN, not the executor:
      respects_ceiling            100%
      all_intermediates_valid     100%
      all_intermediates_connected 95.5%
      reaches_target              86.4%
      PATH EXISTS (scoped)        44/44   median 2 stages, max 3
      monotone                    27/44
      **DIPS BELOW SOURCE         17/44 (39%)**
- **THE CEILING IS NOT THE BLOCKER -- staging clears it.** The predeclared falsifier ("if staging
  cannot preserve monotonic approach, the ceiling is the publishable blocker") did NOT fire as
  written, and the sharper true statement is better: **39% of transports pass through an
  intermediate scoring WORSE than where the run started, so a score-greedy controller abandons
  exactly the intermediate the path requires.** The signal does not guide you along the path that
  exists. That points at SELECTION, not realization.
- **THIS IS THE MECHANISM BEHIND THE MEASURED CELECOXIB PLATEAU.** On the matched 1000-call A/B, arm
  B's BEST froze at call 500 (0.3838) and never improved through 1000 while top-10 kept filling
  (0.3556 -> 0.3714). That was recorded as "exploitation strong, discovery weak" with no mechanism.
  The dip IS the mechanism: B was not out of budget or out of basin, it was structurally unable to
  accept a worse intermediate.
- **It also retroactively justifies the basin exploration floor**, which had been measured FREE (best
  and U10 identical to four decimals on an adversarial landscape where exploiting is correct). Budget
  protected from score-based selection turns out to be exactly what a transport channel needs -- a
  much stronger argument than "it costs nothing".
- **Three plan defects, each found BEFORE the verdict was read, each having inflated the failure
  rate:** (a) a globally RING-FIRST install order disconnects the molecule, because installing a
  target ring before the linker joining it to the core leaves a fragment -- attributed BY EXECUTION,
  since reverting the anchored pick alone drops connected 95.5% -> 63.6% and complete paths 44/44 ->
  30/44; (b) copying the TARGET's aromatic bonds into a partly built molecule fails, because a ring
  installed without its substituents is not aromatic on its own -- build from KEKULE orders and let
  sanitization re-perceive, as COMPOSE's own states do; (c) the monotonicity trajectory OMITTED THE
  SOURCE, so an initial dip was invisible and celecoxib read as monotone while its first stage sat
  below its own start. Also: a monotonicity verdict over one point is vacuous and must report
  UNEVALUATED rather than True.
- **Scoping that hides the unscoped number is spin; scoping that shows both is analysis.** Unscoped
  the verdict reads 17/44, scoped 44/44, and the difference is two declared category errors --
  `declared_smarts` supplies a reference SCAFFOLD rather than a molecule to reproduce, and a
  `requires_reattachment` correspondence DECLARES that no deletion order keeps it connected. Report
  both numbers and name the exclusions.
- **"A guard is only tested where it binds" bit twice in one battery, and the two cases are
  different.** A mutation of the anchored pick SURVIVED while the aggregate showed it was worth 32
  points of connectivity -- a fixture GAP, fixed with measured anchor-sensitive pairs. A mutation
  installing one ring atom at a time also survived -- that one is a FINDING, not a gap: under
  anchored growth the next adjacent atom is a ring neighbour anyway, so the explicit ring grouping is
  redundant with it, and ring integrity at a STAGE BOUNDARY is a separate guard in the chunker.
  Record the distinction rather than deleting the surviving mutation.

## 2026-09-21 (the dip is an ORDERING artifact -- deep but one stage wide -- and a coordinator-set trap)

- **VERDICT `INTERLEAVING_REMOVES_MOST_DIPS`.** Over 35 declared-target pairs, prune-first dips on
  10; interleaving rescues **9 of 10**, preserving the endpoint on **6 of 10**.
      dip DEPTH relative to source:  median 65.9%, min 13.0%, max 100.0%; 0/10 shallower than 10%
      dip WIDTH:                     median 1 stage, MAX 1
- **"Deep but narrow" is decisive in BOTH directions, which is why depth and width had to be
  measured separately.** A median 66% collapse kills the "tunable selection tolerance" option
  outright -- the objective does not wobble, it collapses, so no plausible threshold absorbs it.
  A width of exactly one stage bounds the fix: **a transport channel needs protected budget for ONE
  ROUND**, not an open-ended commitment. An architectural requirement of a very bounded kind.
- **MECHANISM: the dip is an artifact of stage ORDER, not of transport.** Prune-then-install strips
  structure the target does not want BEFORE adding structure it does, so the midpoint is smaller
  than both endpoints. Installing first keeps the molecule out of the trough with identical
  endpoints: `0.2235 -> 0.1014 -> 0.6515` becomes `0.2235 -> 0.2826 -> 0.6515`.
- **CAVEAT that makes this a policy rather than a default: interleaving preserves the endpoint on
  only 6 of 10.** Stage ordering must therefore be chosen PER TRANSPORT on trajectory shape subject
  to validity and endpoint preservation, never fixed to one order.
- **A COORDINATOR-SET TRAP, and the agent avoided it.** I told the agent that if no alignment were
  monotone, "a score-greedy controller provably cannot follow a transport -- a publishable statement
  about the objective." Its first alignment run returned exactly that: `NO_ALIGNMENT_IS_MONOTONE`,
  0/10 rescued. **It was nearly VACUOUS.** 26 of 35 pairs admit exactly ONE alignment and **7 of the
  10 dipping pairs had no alternative at all**, so "not rescued" was definitionally true for them;
  among the 3 that did have an alternative, **3 of 3 were rescued**. `correspondences` derives its
  top-K from the matches of a SINGLE MCS, so alignment diversity exists only for symmetric
  molecules -- it was never a general lever.
  **TWO LESSONS. (1) Ask whether the thing being tested EXISTS before reporting that it failed** --
  same class as "a metric that cannot vary is not a measurement". **(2) Pre-blessing one branch as
  "publishable" creates pressure to reach it.** State what each outcome would mean, never which one
  would be a good result.

## 2026-09-21 (CORRECTION: "publicly available" is not "supplied by the task" -- traced at source)

- **CORRECTS the entry above, "a benchmark-declared target is a task INPUT".** That entry is TOO
  BROAD and its PMO half is withdrawn. The celecoxib target SMILES IS a literal in TDC's public
  oracle source -- that fact was verified and stands. **The inference drawn from it was wrong:
  publicly available is NOT the same as supplied-to-the-optimizer-by-the-evaluated-task.**
- **TRACED AT SOURCE in the comparator.** IVG's `in_virtuo_reinforce/genetic_ppo.py` does contain a
  dictionary of reference SMILES assigned to `config.target` -- but every use of
  `self.config.target` in the optimization loop passes it to `visualize_top_smiles`, which puts the
  reference at the front of a molecular drawing grid and saves a plot. The data flow is
  **reference SMILES -> configuration -> visualization -> saved plot**. It never reaches the
  proposal model, the reward, the prompter or the training procedure. **It is a DISPLAY reference,
  not a navigational input.** Finding a SMILES in a repository does not settle the question;
  tracing where the string GOES does.
- **The PMO paper is explicit:** "neither the analytic form of oracles nor the derivatives of the
  properties are accessible" -- feedback is scalar oracle evaluations under a query budget.
- **CORRECTED RULE: use a target when the EVALUATED TASK actually supplies it.**
  - **T4 supplies the lead molecule** -- GenMol's lead runner reads the supplied start, builds its
    fingerprint and initializes from its fragments. A genuine task input, like delta.
  - **PMO black-box search supplies scalar feedback only.** A source-to-target compiler does not
    change this: hand it the answer and you have evaluated REFERENCE-CONDITIONED CONSTRUCTION, not
    black-box discovery. That is still good science -- it just needs its own label.
- **Where the comparators' task-specific proposal information actually comes from:**
  - GenMol's released PMO setup: `scripts/exps/pmo/get_vocab.py` reads ZINC250k, evaluates the PMO
    objectives, decomposes molecules into fragments, scores fragments by the mean of the molecules
    containing them, and writes the **top 10,000 fragments PER OBJECTIVE**. That is an
    oracle-informed **prescreened vocabulary** -- precisely what IVG's `--use_prescreen` toggles.
  - IVG no-prescreen: generated initialization plus counted evaluations that drive prompting,
    mutation, adaptive length selection, replay and generator updates. Its `GeneticPrompter` does
    use molecular STRUCTURE aggressively -- but from its own scored search population, not from the
    reference dictionary.
  - NAMING TRAP: `model.sample(oracle=n_oracle)` does NOT hand the generator the property oracle;
    `n_oracle` is a list of selected sequence lengths. Property evaluation happens at
    `self.oracle(smiles)`.
- **PERMITTED vs NOT, for the no-prescreen comparison.** Freely usable: structures the optimizer
  GENERATES, task-independent molecular data, graph relationships among candidates, and scores
  acquired within the run's budget. Not silently supplied: the hidden reference graph, a target
  fingerprint extracted from the evaluator, same-task winning molecules selected using prior oracle
  results, or a checkpoint trained to reconstruct benchmark answers. **A zero-read certificate at
  runtime does not erase information already baked into weights or a library.**
- **THREE SEPARATELY LABELLED MODES, because they answer different questions:**
      goal-given transport DIAGNOSTIC     reference visible     can the compiler CONSTRUCT it?
      teacher-region warm-start DIAGNOSTIC init state informed,  can the local optimizer EXPLOIT it?
                                           no suffix or goal
      no-prescreen PMO SEARCH             nothing hidden shown  can the system DISCOVER it?
- **The 11-task teacher dossier is a CHALLENGE SET, not a training distribution.** 137 programs /
  4,548 primitive actions, but **8 of the 11 task sections share the SAME starting molecule**
  (`COc1ccccc1CNS(=O)(=O)c1cc(C(=O)N2CCCCCC2)cs1`) and lineage diversity is very limited. Fitting a
  prior on 11 PMO winning structures and then reporting those same tasks as task-independent would
  convey answer information regardless of any runtime certificate. **Generality of the algorithm and
  fairness of its information are separate questions.**
- **SCALE of the remaining gap, as bookkeeping not forecast:** IVG's no-prescreen celecoxib AUC at
  10,000 calls is 0.798; our arm B reached AUC 0.3138 at 1,000. Matching 0.798 over 10k after those
  first 1,000 requires an average top-ten level of ~0.852 across the remaining 9,000 calls. A single
  perfect target molecule is not the objective; a strong top-ten CURVE is.

## 2026-09-21 (before naming a vocabulary boundary, check the candidate against what the prior PROPOSES)

- **A NEGATIVE THAT LOOKS LIKE A PROPERTY OF THE MODEL AND IS A PROPERTY OF THE MEASUREMENT.** A
  probe reported `insert carbon` as **NO_HIT up to 32,768 draws on two drugs** -- close to the
  evidence needed to declare a constituent unrankable and a composite primitive a capability
  EXTENSION rather than an acceleration. **It was a units bug: `atom_type` is a VOCABULARY INDEX,
  not an atomic number.** `ELEMENTS = ['null','B','C','N','O','F','P',...]`, so `CARBON = 6` is
  **phosphorus**. The probe was inserting a phosphorus linker and asking whether the prior proposes
  it. It does not, correctly.
- **The tell was in the artifact the whole time** -- a bare `P` in the constructed SMILES. A negative
  result whose molecules nobody reads is a negative result nobody has checked.
- **The fix revealed the answer.** With carbon looked up BY NAME and given three hydrogens (a
  single-bonded carbon is a methyl; an unsupported H-count makes the insert unexecutable), insert
  flipped to IN_SUPPORT -- and **the model's most frequent insert at that site was exactly
  `(carbon, 3 hydrogens)`**, the atom that should have been constructed. The prior was not silent;
  it was answering a different question.
- **GENERAL RULE, adopted for every support probe: before naming a boundary, check the candidate
  against what the prior ACTUALLY PROPOSES at that site.** Asking "does the prior propose X" when X
  is malformed returns a confident no; asking "what does the prior propose here" catches it in one
  call. Companion to "a metric that cannot vary is not a measurement" and "ask whether the thing
  being tested exists before reporting that it failed".
- **Applying the raised standard correctly:** a constituent reading NO_HIT to 16,384 draws was
  recorded **UNRESOLVED**, not as a boundary and not as a non-issue, because the same move class is
  in-support elsewhere on the panel and because LOVASTATIN needed 8,192 to recover from a zero that
  had survived 2,048. Naming something unresolved is more useful than guessing either way.
- **A PRECONDITION derived from failure analysis is better than two unexplained failures.** The
  close-then-open transaction executes on 2 of 4 linker drugs; both failures are the SAME step
  (ring-close) on the two drugs the original structural audit already recorded as having **zero free
  valence at their declared sites**. A saturated anchor cannot accept another bond. So the composite
  fires only where the far-core anchor has free valence -- a structural property of the state,
  activating without any instance identity, that converts two anomalies into one declared scope.
- **Macro vs extension is decided PER CONSTITUENT.** All three constituents in-support on one drug
  means the composite bundles moves the prior could already make but does not make in sequence --
  an ACCELERATION. One unrankable constituent would make it a capability EXTENSION, which must be
  declared as such rather than wrapped in a macro.

## 2026-09-21 (ordering selector: admissibility is a FILTER, never a score term)

- **MEASURED over 35 transports: 0 have no admissible ordering; prune-first would dip on 10; the
  policy finds a dip-free ordering on 5 of those 10. RESIDUAL 5/35 = 14.3%, protected rounds
  median 1 / max 1.** Verdict `FLOOR_IS_CHEAP_INSURANCE`: the transport channel needs budget
  insulated from score-based selection on 14% of transports for exactly one round each. Bounded and
  specific -- insurance, not the mechanism.
- **THE NUMBER THAT MATTERS WENT DOWN: 5/10, not the 9/10 interleaving alone achieves.** Interleaving
  removes the dip on 9 of 10 but **preserves the endpoint on only 6 of 10** -- four of those nine
  land somewhere other than the target. **Admissibility (every intermediate valid and connected, the
  final molecule IS the target) must be a HARD FILTER, never a term in the shape score.** Scoring
  shape without it selects orderings that never arrive: a beautiful trajectory to the wrong
  molecule. This gap is the entire reason to build a selector rather than switch the default to
  interleaved.
- **Express a protected-budget requirement as the chosen ordering's TROUGH WIDTH, not as a budget
  fraction.** The floor then provably protects a SINGLE CROSSING and cannot later be read as an
  open-ended allowance.
- **THREE SURVIVING MUTATIONS, THREE DIFFERENT MEANINGS -- telling them apart is the real work, and
  a battery summary makes them look identical:**
  (a) **fixture gap** -- the guard is real but no fixture exercises it (the anchored-pick mutation
      survived while the aggregate showed it worth 32 points of connectivity). Fix the fixtures.
  (b) **redundancy** -- a separate "does it dip" term could not be killed because a dip-free
      ordering has depth 0.0 AND width 0 by construction. **Remove it as decoration rather than
      keeping it with a test written around it.** A term that cannot be killed because it is implied
      by construction is not a guard, it is a comment.
  (c) **untestable on real data** -- depth-vs-width ordering is undecidable from the measured set
      because no pair offers two DIPPING ADMISSIBLE candidates of differing depth. Pin it on
      constructed candidates through the key function directly.
  The recurring shape underneath all three: a conditional assertion over a fixture that cannot fail
  it. Three distinct fixtures were needed -- mixed, residual, synthetic -- because the dipping
  fixture's candidates are all inadmissible.
- **A classification of what a benchmark DECLARES is not a claim about what it SUPPLIES to the
  optimizer.** The 19-of-23 goal audit was correct and reusable; converting it into "therefore the
  target is a task input" asserted a data flow nobody had traced. Tracing
  `config.target -> visualize_top_smiles -> saved plot` settled it.
- **A goal-conditioned scoring-identity gate must land WITH the realizer, not after it.** It needs
  the proposals to exist, but if it reads 0.0000 the way T4's `expand` did, every downstream
  improvement in goal selection is unobservable and the realizer would be measuring nothing.

## 2026-09-21 (SHARPENED: estimate the per-draw RATE first -- it tells you which rung is decisive)

- **IMPROVES the rule "before naming a boundary, check the candidate against what the prior proposes
  at that site."** The sharper form, contributed by the fragment agent and adopted: **estimate the
  PER-DRAW RATE first, and it tells you which budget would settle the question** -- instead of
  doubling a ladder blindly until something happens.
- **MEASURED instance.** A constituent read NO_HIT at 16,384 draws, which looked like a possible
  vocabulary boundary. Rather than spending 131,072 draws asking whether the prior proposes THAT
  construction, one call asked what the prior proposes AT THAT STATE: **`bond_insert` is 8 of 6,000
  draws (0.13%)**, and the specific atom pair is a fraction of that. **Expected hits in 16,384 draws
  is therefore of order ONE -- a zero there discriminates nothing and carries no information.** The
  rate estimate identified 65,536 as the decisive rung, and the move HIT at 65,536.
- **CONSEQUENCE: UNRESOLVED was not hedging, it was the correct reading of a measurement with no
  power.** A negative from an underpowered probe is not weak evidence of absence; it is no evidence
  at all, and the rate estimate is what distinguishes the two. Compute the expected count before
  reporting a zero.
- **Verdict that followed: all three constituents IN_SUPPORT, so the composite bundles moves the
  prior can already rank but does not make in sequence -- an ACCELERATION, not a capability
  extension.** The macro/extension question is decided per constituent, at the budget the rate
  estimate says is decisive.
- **Aim a falsifier at your own mechanism, not at the substrate.** The added "precondition honesty"
  criterion: if the free-valence gate fires on drugs that DO have free valence, **the gate is wrong
  rather than the chemistry**. That is the hard direction to point a falsifier and the one that
  catches self-serving preconditions.
- **`pgrep | head -1` made a 119%-CPU process read as a 0.0% deadlock.** Same family as a container
  census whose value could not vary: an instrument that cannot report the true state will
  confidently report a false one.

## 2026-09-21 (a PRECONDITION can be an artifact of the ROUTE, not a property of the substrate)

- **THE LESSON, and it is the most transferable thing the fragment work produced.** A free-valence
  precondition looked exactly like a well-formed declared scope: it had a structural predicate, it
  explained two drug failures, it activated without any instance identity, and it converted
  anomalies into a stated limit. **It was a symptom of performing the steps in the wrong order.**
  Before scoping a mechanism to the instances where it happens to work, check whether a DIFFERENT
  ROUTE to the same transformation removes the restriction entirely. A well-formed scope and a
  symptom of a bad route are indistinguishable from the inside.
- **MEASURED, the three columns that settled it** (declared-site hydrogens in the retained core vs
  after seeding, 10 linker drugs):
  - **Declared sites are NOT saturated in the drug** -- every one carries at least one hydrogen in
    the core. So the attachment model is NOT mismatched panel-wide and **the motif and decoration
    results are unaffected.** (The frightening possibility, refuted.)
  - **The seeding is CORRECT**: `after_seed` is exactly `core - 1` on every drug and every site --
    the linker bond consuming precisely the valence the declared site offers. Intended chemistry,
    not a defect.
  - **The route was the problem**: close-then-open ring-closes to the far anchor BEFORE removing the
    old bond, so the anchor must transiently carry TWO external bonds. A site offering one free
    valence -- the normal case -- cannot. It ran only where a site happened to have spare valence
    beyond what the task requires.
- **THE FIX: `bond_reroute` exchanges the bridge ATOMICALLY** ("No disconnected state is ever
  visible"), needs no transient valence, and is TWO events instead of three. MEASURED:
  close-then-open reaches a length-2 linker on **4 of 10** drugs; **insert-then-reroute on 10 of
  10**, every endpoint valid with cores separated. **So the precondition is DELETED, not scoped**,
  which also retires the falsifier written to guard it.
- **A FALSIFICATION CAN BE PRODUCTIVE, and rescoping would have destroyed that.** Holding the
  panel-wide reading of the predeclared vacuity falsifier (90.3% vs a >90% threshold) is what forced
  the valence question, which refuted two possibilities and found the better route. Accepting the
  applicable-subset reading would have left a matched arm running on a mechanism restricted to 2
  drugs by an artifact.
- **"The test was underpowered" is a criticism of the TEST, not an escape from it.** Both were true
  at once: the falsifier fired, AND it was badly designed -- a threshold set casually with no power
  analysis, on a denominator a later addition changed. Re-running larger to see whether it lands the
  other side is measuring until the answer changes. **Fix the NEXT falsifier, do not re-litigate
  this one.** Predeclare threshold, n from a power calculation, denominator, and what a pass
  licenses -- committed to a file BEFORE the first sample.
- **A HAND-CONSTRUCTED sequence bounds what is REACHABLE; it does not predict YIELD.** The same
  mechanism executed on 4/10 drugs by hand and committed on 2/10 under the sampler. Always report
  both and expect the sampler number to be lower.
- **A predicate with no way to make progress must RELEASE.** Gating on path progress in states where
  the transaction cannot fire strangled the trajectory -- committed 38 -> 3 -- and would have read
  as the mechanism destroying yield. Same failure as the previous version, one layer up.
- **A guard test at the START state can pass for the wrong reason.** A
  `sites_offered_for_a_single_core_prompt` mutation SURVIVED because at the start state a
  single-core prompt has no atoms outside its core, so ANY implementation returns None. Re-tested on
  a grown state where the guard actually binds, it is killed. The floor-guard lesson, at fixture
  level.

## 2026-09-21 (TWO production chemistry kernels: PMO is rdkit 2023.9.6, T4/editing is 2024.3.5)

- **CORRECTS a standing instruction I repeated to several agents all session.** There are TWO
  production chemistry kernels in this repo and **the PMO one is NOT 2024.3.5**.
  `modal_apps/pmo_population_v1_app.py` builds its image with `uv pip install --system --no-deps
  'PyTDC==1.1.15'` followed by **`rdkit==2023.9.6`**, numpy 1.26.4, pandas 2.1.4, scikit-learn
  1.2.2, scipy 1.15.0, setuptools 75.6.0, plus an explicit `_rdkit_six_shim()` because PyTDC 1.1.15
  imports `rdkit.six`, which modern RDKit does not ship. **The image holds ONE rdkit, so the PMO
  runtime -- oracle AND COMPOSE executor -- runs on 2023.9.6**, while the T4/editing production pin
  is 2024.3.5. Reproducing PMO numerics on `~/compose_region_pinned_env` (2024.3.5) is therefore NOT
  reproducing the PMO container.
- **Why the obvious mirror fails:** PyTDC 1.1.15 pins `rdkit>=2023.9.5,<2024.3.1`, so
  `uv pip install PyTDC==1.1.15` against rdkit 2024.3.5 is flatly unsatisfiable; production uses
  `--no-deps` plus the shim. `setuptools>=81` additionally breaks it because `tdc/metadata.py`
  imports `pkg_resources`. **Working local mirror: python 3.11, rdkit 2023.9.6, PyTDC 1.1.15
  `--no-deps`, setuptools 69.5.1, plus the shim** -- all 11 atlas-task oracles then construct AND
  score offline.
- **CONSEQUENCE: any local PMO number computed under 2024.3.5 needs a parity statement.** This repo
  has already measured that two rdkit versions can write DIFFERENT canonical SMILES for the same
  molecule, and that canonical SMILES is used as a CACHE KEY -- so a spelling difference is not
  cosmetic in any procedure that dedupes on the string. Parity on drug-like molecules also says
  nothing about parity on the exotic intermediates a search constructs.

## 2026-09-21 (TEST A: construction is NOT the gap -- the refusal is charge, not search)

- **MEASURED, zero oracle calls, 204 attempts (12 destinations x {recorded source, 8 init-bank
  molecules, 8 live PMO parents}), using `winner_paths.find_path` which recomputes its own MCS
  correspondence per pair -- so this is CONSTRUCTION for a new pair, not replay of a recorded
  atom-address sequence:**
      recorded source    12/12 witness_found; fresh compile reproduces the recorded step count
                         exactly on 11 of 12, and finds 35 steps where the record has 38
      transfer           102/192 overall
      **transfer, NEUTRAL 102/108 = 94.4%, with 102/102 EXACT destination recovery**
                         re-replayed independently through the production executor
      search failures    6 of 192 = 3.1%, node budget exhausted at residual 1-6
- **THE DOMINANT REFUSAL IS NOT SEARCH: 84 of 192 (43.8%) are rejected BEFORE any search with
  `unreachable_charge_change`**, because 7 of 16 distinct transfer sources carry a net formal charge
  and every atlas destination is neutral. That is the charge-PRESERVING scope locked on
  2026-07-26/27 surfacing as a PMO ceiling, **exactly as it surfaced as a T4 ceiling on 5ht1b_2**.
  **"COMPOSE cannot build these molecules" is FALSIFIED. Construction is not the gap.**
- **THE TEACHER ROUTES ARE NOT HILL CLIMBS.** MEASURED with 33 diagnostic oracle calls on a
  production-mirroring kernel: **6 of 11 recorded routes pass through a midpoint scoring BELOW their
  own source** -- `perindopril_mpo` 0.360 -> **0.009** -> 0.809 (a 39x drop), `qed` 0.796 -> 0.251
  -> 0.948, plus albuterol, celecoxib, gsk3b, thiothixene. A score-greedy optimizer cannot follow
  these, so a controller that CAN construct a supplied destination may still never reach it blind.
  **This independently corroborates the planned-transport dip finding on REAL recorded routes.**
- **`runtime_supported` in the frozen route distillation is `len(route.actions) <= 32`** -- a route
  LENGTH threshold, not an executor-support predicate. Its census reads as though jnk3, perindopril,
  thiothixene and troglitazone have no supported routes; MEASURED, **194 of 194 routes replay
  EXACTLY** under the current rewrite system, comparing every intermediate by canonical key,
  including the 41-step and 42-step spines. **Read a support-sounding field's DEFINITION before
  quoting it as a capability finding.**
- **THE DOSSIER IS THREE SOURCE MOLECULES, not eleven** (supersedes the earlier "8 of 11 share a
  start"). 194 routes, 11 tasks, **3 distinct sources**, and **9 of 11 task sections (88.7% of
  routes) compile from one molecule**. Every section has exactly ONE lineage: a compiled spine plus
  siblings reusing its action prefix verbatim. **Counting "137 programs" as breadth is counting
  serialisations.**
- **Recorded dossier scores are a DIFFERENT EVALUATOR twice over and must never be carried across:**
  `gsk3b` and `jnk3` there carry `oracle_kind: frozen_tdc_forest` -- a local `.npz` forest, not a
  PyTDC `Oracle` -- and pytdc 0.3.6 against production 1.1.15. `drd2` was deliberately excluded for
  exactly this reason, which is the right precedent.
- **`bond_reroute` appears ZERO times in all 194 routes** (7 of 8 Active8 families exercised). Do not
  describe the dossier as full-vocabulary supervision.
- **A guard whose test ANOTHER guard also satisfies is not load-bearing.** `bypass_input_hash_check`
  survived a battery because the tampered fixture left `payload_sha256` stale, so the PAYLOAD-hash
  guard raised the same exception the FILE-hash guard would have. Make the tamper internally
  consistent (re-hash the payload) so only the target check can catch it.
- **Budget a tree-copying mutation battery as a SERIAL job**: a 780 MB worktree copy per mutation,
  11 pytest runs taking >20 minutes under contention against ~35 s each unloaded.

## 2026-09-21 (kernel parity PASSES for PMO transport -- and the laptop kernel changes MOLECULES)

- **PARITY PASS, and it closes the two-kernel worry: every transport artifact is byte-identical
  (sha256) between rdkit 2023.9.6 (PMO production) and 2024.3.5 (T4/editing).** Correspondences
  108/0 with scale 19-36-56; stage splits 44 transports / 27 monotone / 17 dips / median 2 stages;
  ordering residual 5/35 = 14.3% with median 1 protected round. **Nothing needed re-basing.** The
  three deterministic scripts were re-derived IN FULL at ~1.3 s each rather than sampled -- cheaper
  AND stronger than the sample that was asked for.
- **Endpoint parity is not enough and the full lattice was walked: 37,784 intermediates, of which
  14,312 (37.9%) do not sanitize.** Comparing BOTH the canonical SMILES and the sanitize-or-not
  verdict: **0 SMILES disagreements, 0 verdict disagreements, 0 of 15 canaries moving.** The live
  proposal path too -- the scoring-identity gate re-run under 2023.9.6 reproduces the committed
  2024.3.5 artifact byte-for-byte, 144/144, `recovered_fraction` 1.0000 on all six rows.
- **THREE kernels, not two, and the laptop one is genuinely different.** Against `.venv`'s rdkit
  **2026.03.6**: **1,472 of 37,784 off-path intermediates (3.90%) are a DIFFERENT MOLECULE** --
  0 spelling differences, 1,472 differing by **InChI** (`C=CC(=C)NCCOc1ccccc1` vs
  `CC=C(C)NCCOc1ccccc1`). The planner is NOT implicated: `retain_core`/`core_map`/`delete_order`/
  `install_order`/`core_bond_changes` hash identically on all three kernels. Cause is kekulization
  of the ENDPOINTS moving, so a partly-built ring inherits a different double-bond placement.
  **Our artifacts survive 2026 only because those lattice points are ones the stage splitter never
  lands on -- a measured coincidence of where ring-system group boundaries fall, NOT a property to
  rely on.** Anything walking a different prefix schedule must be re-run pinned.
- **A parity PASS needs probe controls or it is vacuous**: same-kernel comparison must be REFUSED,
  cross-mode comparison REFUSED, and a deliberately mutated dump must be REQUIRED to report one
  spelling flip, one verdict flip and one canary flip. A positive control whose mutation fails to
  apply must ABORT with a traceback rather than print a verdict.

## 2026-09-21 (the THIRD inert mechanism, and a too-broad claim of mine corrected)

- **`donor_program` -- molecular pendant exchange between two complete molecules -- is ABSENT from
  the scored entry point's 118-module import closure**, as is `donor_memory`. Third instance of a
  built, tested mechanism that no production caller reaches, after the region law behind an opt-in
  keyword and `allocation_priority` with zero call sites. Now a COMMAND
  (`scripts/pmo_production_closure_audit.py`) rather than a rule to remember.
  **TWO SUBTLETIES THE AUDIT HAD TO HANDLE:** absence must be established two ways because a
  DEFERRED import would not appear in `sys.modules`; and it must separate DOTTED from BARE-NAME
  references, because `adaptive_program_optimizer` binds a LOCAL VARIABLE called `donor_program`
  seven times -- a source scan alone reads as seven hits on a module that is not there.
- **CORRECTS a coordinator claim that was too broad.** I generalised `compile_source_to_target`'s
  `retained_fraction = 0.000` into "a scaffold-preserving bridge does not exist today." That is
  right for ARBITRARY pairs and WRONG about donor recombination: **`compile_transplant` does not
  take an arbitrary pair** -- it builds its target from an explicit retained/added split and RAISES
  if replay changed a retained slot, so it cannot route through null. Preservation is guaranteed by
  construction there. Do not quote the demolish-to-null number outside its scope.
- **THE BINDING CONSTRAINT IS THE CUT DISTRIBUTION, and the STEPS column is what decides it.**
  Matched arms, 60 pairs replayed from a completed 250-call charged celecoxib ledger, same pairs,
  same order, same compiler, differing ONLY in the cut draw:
      arm                     compiled  novel  retained med  retained>=0.5  **steps med**
      uniform (SHIPPED)       142/240    136      0.324           46          **36**
      retentive               155/240    139      0.722          121          **17**
  PMO's measured realization ceiling is **23 primitives**. So the shipped
  `cut_distribution: uniform_oriented_single_bridge` puts the **MEDIAN donor transplant OUTSIDE what
  the controller can realize**, not merely wasting draws. Chemistry is not the cost: QED
  0.426 -> 0.434, SA 3.58 -> 3.71. Honest cost: `self_proposal` rises 1 -> 24 of 240.
  **Same shape as the T4 region-law defect: an unconditioned region draw, not the executor, is the
  binding constraint.** INFERRED and worth checking first -- `pendant_cuts` returns oriented
  single-bond bridges, the same object class `BridgeRegionLaw` weights, so the repair is likely a
  REUSE of a law already built and measured on T4 rather than a new mechanism.
  LABEL: one task, one ledger, 60 pairs, offline compilation only -- **not a scored result.**
- **Exclude a contaminated bank by a TEST, not a comment.** `pmo_banks_all.json` (celecoxib
  0.374-0.458, above the A/B's best) carries `uncounted_calls: 249455`. Also worth knowing: the 1k
  A/B molecules are NOT on local disk (counters only) -- they live on the Modal volume at
  `pmo_population_controller_v1/<run_id>/<task>/oracle/query_*/result.json`.

## 2026-09-21 (CORRECTION: `ppid=1` does NOT mean abandoned -- a blind orphan sweep nearly killed Test C)

- **CORRECTS the earlier entry treating every `ppid=1` python process as a leaked orphan to sweep.**
  That is true for a pool whose PARENT DIED, and false for a job an agent launched DETACHED on
  purpose. Both look identical in `ps`.
- **NEAR MISS, and only a silently failing `kill` prevented it.** A blind sweep of all five `ppid=1`
  python processes would have destroyed, 35 minutes in: two `pmo_atlas_blind_search.py` groups at
  95% CPU (Test C, the decisive discovery measurement), `pmo_atlas_entry_probe.py` at 100% (also
  Test C), and `fragment_path_composite_scoring` at 111% (the v3 constituent probe). All four were
  doing exactly the work that had been commissioned.
- **THE DISCRIMINATOR IS CPU AND ELAPSED TIME, NOT PARENTAGE.** A leaked pool child burns CPU with
  no artifact advancing; a detached job burns CPU and its output file keeps growing. Before killing
  anything, read the **command line** and check whether its artifact is being written. Sweep only
  what is both parentless AND either idle or matching a pattern nothing is expected to be producing.
- **A kill that "fails silently" must be checked, not assumed.** The sweep printed no confirmation
  and the orphan count did not move; had it succeeded the work would have been gone with no record
  of what was lost.
- **MEMORY PRESSURE IS NOT EVIDENCE ABOUT WHICH PROCESSES TO KILL.** At 59 MB free the four live
  probes held **2.14 GB total** -- killing them could not have fixed a 32 GB machine, and the
  pressure was in the compressor (~797k pages stored) and elsewhere. Measure the candidates' actual
  RSS against the shortfall before treating them as the cause.

## 2026-09-21 (the constituent question has TWO answers: action-level and transition-level)

- **CORRECTS the rule I set earlier ("a composite is an acceleration if every constituent is
  individually rankable") -- that question has a per-PROMPT answer AND two different levels, and I
  had collapsed them.** The linker v3 composite executes
  `BondReroute(a=path_atom, b=far_anchor, u=new_atom, v=far_anchor)`, whose moved endpoint `u` is
  NOT an endpoint of the cut, while the model's graft is a RESTRICTED reroute in which the moved
  atom IS a cut endpoint. So:
      as an ACTION      unavailable on **10 of 10** released prompts (universal, explains WHY)
      as a TRANSITION   compared on the SUCCESSOR MOLECULE, recovered on **2 of 10**,
                        missing on 8 at a 16,384-draw cap
  **Report both. The action statement explains the mechanism; the TRANSITION statement decides
  whether the composite is an acceleration or an extension, and only the second governs.**
- **VERDICT: v3 is a capability EXTENSION on 8 of 10 prompts and an acceleration on 2**, and must be
  declared rather than wrapped. That is the same class as de novo's ring-installation host scope: a
  declared capability boundary, measured rather than assumed.
- **THE TRADE MUST BE SHOWN, both arms:** the IN-SUPPORT route (close-then-open) reaches a length-2
  linker on **4 of 10** drugs; the REACHING route (insert-then-reroute) reaches it on **10 of 10** by
  hand but is outside the model's support on 8. Neither number alone is the result.
- **Refusing to generalise the one explanation that fits is what separates a finding from a story.**
  ERLOTINIB recovers because the model offers to move the far-core anchor 100 of 520 times -- the
  only model-shaped route to that molecule. LIOTHYRONINE recovers with the far anchor moved **zero**
  times, so a second route exists the census does not identify (most likely two different actions
  landing on one canonical molecule). Recorded as UNEXPLAINED rather than folded into the first
  mechanism.
- **Two independent instruments at different budgets agreeing is what makes a negative safe.** A
  4,000-draw support census and an escalate-to-16,384 scoring probe agreed on every prompt where
  both reported; where they could not both be powered (MARIBAVIR drew 11 reroutes in 4,000), the
  census negative was explicitly marked the WEAK one and the escalating probe was the instrument.
- **A KNOB CAN BE PLUMBED THROUGH EVERY LAYER A READER CHECKS AND STILL NOT BE RECORDED.** The suite
  runner accepted `--linker-bridge-atoms` and `--path-program`, passed both correctly, and wrote
  shards containing **no realized length, no seeded length, no transaction count**. A 60-shard
  matched arm was launched and stopped 27 minutes in on that. **Wiring a knob and recording what it
  did are two different jobs, and only the second makes the run answerable.**
- **`starmap` yields IN ORDER, so a slow first shard makes a healthy 60-container fan-out look
  dead** -- nothing prints until shard #1 returns, and containers capture subprocess output so
  `modal app logs` is empty too. Check the TASK COUNT, not the log, and do not restructure a fan-out
  mid-run on the strength of silence.

## 2026-09-21 (PMO located at the manifold; and two overstatements of mine, corrected by the owner)

- **THE PMO DIAGNOSIS, now stated without hedging:** COMPOSE has sufficient LOCAL optimization and
  sufficient TRANSPORT capability, but its blind GLOBAL proposal distribution does not reliably
  remain in, or enter, productive molecular basins. Three measurements carry it: celecoxib blind
  U10 = 0.3714 at 1,000 calls against U10 = 0.7362 in **64** calls for the same frozen local
  controller placed in the productive region; 0 of 11 atlas tasks reach the anchor rung blind; and
  the blind population systematically leaves the drug-like manifold.
- **THE MANIFOLD MEASUREMENT (`diagnostics/pmo_atlas_v1/manifold_drift_v1.json`, recomputed from the
  committed Test C blind ledgers, ZERO new oracle calls, rdkit 2023.9.6).** All 11 tasks start from
  the same init bank at median QED 0.716. By the final call-quartile:
      qed +0.033 | mestranol -0.096 | celecoxib -0.130 | albuterol -0.151 | thiothixene -0.151
      isomers -0.155 | median1 -0.166 | troglitazone -0.263 | jnk3 -0.311 | perindopril -0.348
      gsk3b -0.359
  **qed is the ONLY task that does not drift, and it is the only task whose OBJECTIVE IS
  drug-likeness** -- its reward pins the manifold that every other task's does not. That is the
  mechanism, and it explains the Test C pattern without the narrow-vs-broad taxonomy the complete
  data already refuted. `isomers` is the predicted EXCEPTION and confirms it: the only task whose
  heavy-atom median FALLS (21.5 -> 14.5), drifting TOWARD its small C7H8N2O2 target, and the only
  non-qed task above 90% of anchor. **Drift is not bad per se; drift AWAY from the target's region
  is.** INFERRED, NOT ESTABLISHED: r(%anchor, QED drift) = +0.475 at n=11 is p~0.14. The qed
  contrast is the solid part; do not quote the correlation as a finding.
- **THE ALLOCATOR REPAIR IS NECESSARY AND NOT SUFFICIENT, and this was nearly a wasted lever.** Every
  one of those blind runs already used `niche_evidence` -- the repair that replaced the 1-3 heavy-atom
  fragment niche centres -- confirmed in each campaign manifest, and drifted anyway. Check what a run
  ACTUALLY used before proposing a fix it already has.
- **WHY IVG does not meet this, and it is not a cleverer reward.** Its generator is a pretrained
  distribution over real molecules, so every proposal lands on the manifold; the legal fiber COMPOSE
  samples is vastly larger, and validity-closure is a VALENCE guarantee that says nothing about
  plausibility. IVG gets manifold adherence free on all 23 tasks; we get it on qed only. Its
  no-prescreen setup is 100 pretrained-model samples scored in-run + a live scored population +
  a genetic prompter + a 300-example replay buffer + PPO -- an ONLINE bank, not the prescored ZINC
  bank, which is `--use_prescreen` only.
- **`docs/PMO_INIT_BANK.json` holds 100 molecules and the controller draws `count: 16`.** Matching
  IVG's initialization breadth is a CONFIG VALUE, not new data.
- **CORRECTION, mine: "100-init is inadmissible at a 250-call budget" is WRONG.** It is
  PROTOCOL-VALID at any budget provided all 100 evaluations are counted toward the total; it is
  merely a poor ALLOCATION there (40% of 250, 20% of 500, 1% of 10,000 -- the last being
  IVG-matched). **`protocol-valid` and `budget-efficient` are different predicates and both must be
  measured.** Calling a bad allocation "inadmissible" would have deleted a legitimate arm.
- **CORRECTION, mine: "its 122 unspent authorized calls won't help" was too categorical.** What is
  MEASURED is that the CURRENT PROPOSAL MECHANISM exhausted its yield. More calls through that same
  proposal law probably do not help; that is not the same claim. The distinction is exactly the
  lesson being applied to PMO -- **more oracle budget != better search when the proposal
  distribution never produces the needed chemistry** -- and collapsing it hides the reason.
- **A BASIN SHOULD BE DEFINED BY FUTURE VALUE, NOT BY RESEMBLANCE TO A KNOWN ANSWER (owner).**
  `V_local(G) = E[top-10 after 64 local calls | G]`; a good basin is high `V_local`, and the teacher
  region is a POSITIVE CONTROL rather than the target. This prevents overfitting development to one
  known answer, and makes "C found a structurally different region with equal future value" a
  SUCCESS instead of a miss. **COST TRAP: exact `V_local` is 64 oracle calls per candidate**, so it
  is unavailable to a zero-charged-call gate -- it needs an offline proxy whose agreement with true
  `V_local` is validated on points already scored (the teacher region and the blind endpoints, free),
  and a proxy whose fidelity is unmeasured is not a basin definition.

## 2026-09-21 (T4 5ht1b_2: both protonation rescue arms closed a cell that was null at BOTH deltas)

- **RESULT, reconciled by exact identity and verified independently.** 5ht1b_2 reads `null` at both
  thresholds in the locks-based reconciliation -- the one blank 5HT1B cell.
      arm    contract     delta (EXECUTABLE)  calls     status                best
      d04    c51c6144     0.4                 249/249   complete_budget       **-12.5**
      d06    d95fb5a1     0.6                 126/248   candidate_exhaustion  **-10.8**
  Both at code revision `e86a2181`. GenMol for 5HT1B seed 3 is -11.6 (d0.4) and -10.5 (d0.6), so
  both beat the comparator. **They must be reported as a NAMED protonation-rescue phase** -- their
  own `claim_boundary` forbids splicing them into the unchanged-v1 panel table.
- **Executable `delta` was diffed against the prose `claim_boundary` on BOTH arms and matches**, which
  is the check that caught the jak2 d06 contract carrying 0.4. Do it every time.
- **Both winners re-scored independently under the pinned T4 kernel, and both are eligible:**
      d04  sim **0.4000**  QED 0.6736  SA 3.9776  heavy 36
      d06  sim 0.6769      QED 0.6263  SA 3.7766  heavy 31   net charge 0
  The d04 winner sits EXACTLY on the similarity bound, so it survives only because the shipped
  comparator is non-strict -- the asymmetry recorded on 2026-09-20, now load-bearing on a real win.
- **Both came from the predicted mechanisms, which is what makes them more than luck.** d04 took SA
  from the source's 4.687 to 3.978, a **0.709 repair**, and 5ht1b_2 was diagnosed as demanding a
  substantial SA repair. d06 is NEUTRAL against a charged seed, reached by EXCISING the [NH+]
  fragment -- the route the witness census predicted, not a protonation edit.
- **The stall watchdog reported this cell as `STALLED_AT_ROOT ... locks=1 newest=round_000_lock.json
  ckpt=False` while it was at round 31 with a completed result.** The watchdog shells out to
  `modal volume ls`, which silently returns the PARENT listing -- the 2026-09-20 gotcha, now
  producing a false STALL rather than a false OK. Use `modal.Volume.listdir` (exact) before acting on
  a watchdog verdict.

## 2026-09-21 (basin-entry gate: NOTHING enters, and the gate itself is partly blind)

- **MEASURED, celecoxib, 3,211 proposals, ZERO charged oracle calls, entry threshold 0.30:**
      arm                              best sim  entries  QED drift  1-step QED drop
      nc1 uniform random legal edits     0.2133     0      +0.0295      -0.2121
      arm 0  deployed B, real run 911    0.2083     0      -0.0993      -0.0231
      arm 1  B regenerated               0.2125     0      -0.0172      -0.0852
      arm 1 @ init_bank_16 (control)     0.1831     0      -0.0187      -0.0553
      arm 2 @ init_bank_100              0.1912     0      +0.0294      -0.1124
  Best anywhere is **71% of the entry threshold**; closing it needs +0.087 absolute / +41%
  relative. NOT under-powered: every arm reports `still_climbing = False`, the maximum stopped
  moving between 500 and 1,000 draws at 10x the initial budget, and B's final step extrapolates
  to ~15,700 further proposals to reach entry (an optimistic bound, since sample maxima grow
  sub-linearly).
- **FALSIFIED -- "100-init is a cheap entry win."** I relayed it as cheap and it is not: zero
  entry, ZERO novel-and-nearer molecules, and a WORSE one-step manifold drop (-0.1124 vs -0.0553)
  precisely because it starts from a better pool (parent median QED 0.7627 vs 0.7155) and falls
  further. Its best molecule toward the region is already inside the 16 (both 0.1486). It buys
  parent diversity and lift, nothing else. Recorded as PROTOCOL-VALID at any budget with all 100
  evaluations counted, allocation cost reported separately (40/20/10/1% at 250/500/1k/10k).
- **FALSIFIED -- "B never moves toward the productive region." It MOVES, then SATURATES.**
  0.1486 (16 init) -> 0.1791 (call 50) -> 0.2029 (call 100) -> 0.2059 (call 200); the last charged
  call that improved it was **156**, and nothing in the final 94 calls. 911 further proposals from
  those parents add **+0.0024**. Say saturation, not failure to move -- they imply different fixes.
- **B'S STRUCTURAL SEARCH IS NOT BETTER THAN RANDOM.** On IDENTICAL parents at a matched 300
  draws: uniform-random 0.2133 vs B 0.2093; at 1,000 draws B reaches 0.2125, still below the
  control's 300-draw value. A negative control that BEATS the production arm is the strongest
  single statement in the PMO diagnosis.
- **DONOR TRANSPORT IS 42% OF B'S BUDGET AND 0 OF THE NEAREST 25 APPROACHES** (384 of 911
  proposals); `broad_exploration` is 31% and 20 of 25; `local_search` 245 and 5 of 25. Provenance
  read from the synthesis-time tag, not re-derived.
- **NOVELTY ALONE IS WORTHLESS AS A SIGNAL.** Every arm is ~99.5% novel against B's 475 distinct
  molecules, because a legal edit almost always invents a new molecule. Only "novel AND nearer
  than B ever got" discriminates, and it reads 3 / 0 / 0 / 1 across the arms. Report the qualified
  column or a 99.9% novelty rate will read as discovery.
- **THE V_LOCAL PROXY VALIDATES ONLY PARTIALLY, AND THE LIMIT IS STRUCTURAL RATHER THAN
  STATISTICAL.** Test B had already paid 64 charged calls at each of 33 points (11 tasks x 3 route
  positions), which IS `V_local`, so reading it was free. Within task, Kendall tau:
      structural similarity to anchor  +0.818   (excl. anchors +0.455)  wrong on 3 tasks
      the molecule's own oracle score  +0.818   -- NOT offline for a new proposal
      heavy atoms                      +0.692
      QED                              +0.212   (excl. anchors -0.091)  wrong on 6 tasks
  The similarity agreement rests on the anchor points where it is 1.0 BY CONSTRUCTION. **Every
  proxy that validates is teacher-referenced, so it scores a high-value molecule in an unrelated
  basin as ZERO -- exactly the case the future-value definition exists to credit.** Verdict
  `PARTIAL_PROXY_ONLY_AND_IT_CANNOT_SEE_A_NOVEL_BASIN`: the structural ladder is sound as a
  POSITIVE-CONTROL detector ("did the arm re-find the known region") and cannot answer "did the arm
  find a different one".
- **The owner's future-value refinement is MEASURED-correct, not merely better-argued.** Three
  tasks rank WRONG under similarity: `median1` (similarity 0.39 -> 0.49 -> 1.00 while V_local stays
  0.280/0.267/0.325, so the teacher anchor has LOW future value) and **`perindopril_mpo`, where
  early (sim 0.156) has V_local 0.366 and BEATS near-anchor (sim 0.330, V_local 0.214)** -- fully
  inverted -- plus thiothixene.
- **CONSEQUENCE: we cannot yet distinguish "COMPOSE cannot enter productive basins" from "our gate
  only sees the teacher's basin."** The costed resolution is 192 calls (3 x 64) of OFF-ROUTE
  `V_local` at the blind run's own highest-scoring molecules: low off-route value means the null
  holds under both definitions; high means B already occupies a basin the gate cannot see and the
  GATE is what needs replacing. Arm candidates at similarity ~0.21 are explicitly NOT worth their
  192 calls.
- **A permutation control is what licenses "no task-specific approach":** the same proposals scored
  against all 11 regions put celecoxib's own region **7th of 11**, with zero entries in 41,844
  (proposal, region) pairings so the predicate never fires spuriously. CAVEAT the agent stated and
  I am keeping: regions differ in accessibility to generic molecules, so the ranking conflates
  proposal signal with region accessibility; the narrow reading is only that celecoxib's region is
  not preferentially approached.
- **METHOD, MINE, AND IT IS THE SECOND TIME TODAY: I asserted a precondition I had not measured.**
  I authorised superseding a sealed predicate "ONLY because no arm number has been read yet".
  **Twelve had been**, across all five gate tasks. The agent reconstructed them from the committed
  shards and their commit hashes rather than taking my word, and recorded `precondition_holds:
  false`. The supersession still stands, but on the WEAKER argument -- all twelve returned the
  identical result (zero entrants), so no arm could have been favoured -- and the artifact keeps
  the false precondition visible rather than tidying it away. **Never state the condition that
  licenses changing a sealed artifact without checking it; the check is cheap and the seal is the
  whole point.**

## 2026-09-21 (construction-lane prior: the drift fix works, and the arm that cannot show it)

- **MEASURED on celecoxib, zero charged oracle calls, PMO kernel rdkit 2023.9.6:**
      arm         design                                              OFF drift   ON drift
      closed loop score-free lineage, 15 generations x 16 proposals,    -0.2393    -0.1017
                  parents drawn UNIFORMLY (no objective read anywhere)
      open loop   the same 702 decisions at the real run's OWN parents  -0.1717    -0.1628
  Median QED 0.661 -> 0.422 (OFF) against 0.661 -> 0.560 (ON): a **57% drift reduction**. Final
  %QED>=0.6 0.273 -> 0.410, %SA<=4 0.199 -> 0.312, and the ON arm's median SA is FLAT across all
  fifteen generations while OFF climbs 4.40 -> 4.87. Yield control: both arms produced exactly 16
  children in every generation, 240 each, so it is chemistry and not attrition.
- **THE OPEN-LOOP ARM IS STRUCTURALLY INCAPABLE OF SHOWING THE EFFECT, and reporting its flat
  Q1->Q4 as a null would have been reading the wrong arm.** It replays both selection rules at
  parents an ALREADY-DRIFTED run chose, so it inherits that drift by construction and can only show
  whether the chemistry at each decision is better. Drift is a property of a TRAJECTORY; measure it
  on a closed loop the mechanism actually steers.
- **Paired, n=702, 80.5% disagreement: dQED +0.0367 +- 0.0058 (sigma +6.3), dSA -0.3671 +- 0.0485
  (sigma -7.6)**, permutation p = 5e-5 for both; %SA<=4 42.7% -> 58.3%.
- **CORRECTS the standing line "the chemical prior fixes SA, not QED" -- that was a statement about
  a CALL SITE, not about the prior.** At the construction draw QED improves at sigma +6.3. The
  earlier entry's own caveat ("a QED null measured there is not a result about the proposal
  stream") was right and I propagated the headline instead of the caveat.
- **These families CHANGE molecular size, so the QED claim needs size controls the old site did not
  (its paired dHeavy was exactly 0.000000).** Two, both given: regression intercept at zero size
  change +0.0379 +- 0.0058 (sigma 6.5), and the 401 decisions where both arms produced the same
  heavy-atom count +0.0330 +- 0.0056 (sigma 5.9). The fitted size slope is NEGATIVE
  (-0.0053 QED/atom) and the prior adds +0.24 atoms, so its size effect works AGAINST the gain
  rather than explaining it.
- **Placement census, 1,152 realized insertions per arm: heteroatom-onto-heteroatom 34.6% -> 5.5%
  (6.3x)**, C-onto-C 19.2% -> 47.0%. NB the uniform lane's element OUTCOME is not uniform even
  though its DRAW is -- an oxygen tip cannot extend, so completed chains are hydrazine-biased, and
  its most common insertion is N-onto-N at 24.6%.
- **Rank placement and element JOINTLY at the first growth step.** The measured signal is a
  PLACEMENT signal, so no rule that fixes the anchor first can see it.
- **A tautological guard survived a 10-guard battery by recomputing the seed rule locally**, so its
  expectation moved with the code; rewritten to drive the real `phase_measure` and then killed.
  Same shape as every other tautology on this record, now at fixture level in a mutation battery.
- **A BASELINE THAT COLLECTED NOTHING IS NOT A BASELINE.** A baseline attribution run aborted on
  `ModuleNotFoundError: No module named 'tools'` and reported ZERO failures, which made the branch
  diff read as 14 regressions. Check the collection summary line before differencing failure sets.

## 2026-09-22 (`Pages free` is NOT free memory on macOS, and a mutation battery looks exactly like an orphan)

- **I nearly raised a memory alarm and swept live work on a metric that cannot answer the question.**
  `vm_stat`'s `Pages free` read 0.34 GB and looked like the 65 MB crisis from the day before. But
  `memory_pressure` reported **System-wide memory free percentage: 85%**: `Pages free` EXCLUDES the
  860,691 INACTIVE pages (~13 GB) that macOS reclaims on demand. **Use `memory_pressure`, not
  `Pages free`.** Same error class as the container census whose value could not vary -- an
  instrument that cannot report the true state reports a false one confidently.
- **The actual consumers were ordinary desktop apps** -- Chrome ~3 GB across its processes, the
  Claude process 0.87 GB, Outlook 0.56, Messages 0.35 -- against 1.7 GB for ALL 19 python processes
  combined, largest single 0.40 GB. Check the top-RSS list before attributing pressure to your own
  work.
- **A MUTATION BATTERY AT 0% CPU WITH `ppid=1` IS INDISTINGUISHABLE FROM AN ORPHAN AND IS NOT ONE.**
  Two such processes were `pmo_discovery_mutation_battery.py` and `pmo_macro_option_mutation_battery.py`
  -- commissioned work, idle because a battery spends its time blocked on SUBPROCESS pytest runs
  while the parent waits. Sharpens the 2026-09-21 correction: the discriminator is CPU **plus a
  growing artifact**, and a parent that DELEGATES its work to subprocesses shows 0% CPU while being
  perfectly healthy. Read the command line every time.
- **Killing them would have freed 40 MB against a supposed 0.34 GB shortfall.** Re-proves the
  standing rule: measure the candidates' actual RSS against the claimed shortfall before treating
  them as the cause. If the arithmetic does not close, the diagnosis is wrong.

## 2026-09-22 (fa7_0 final: a general completion-law win, and four claims of MINE it falsified)

- **SHIPPED AND STANDS ALONE, independent of whether fa7_0 ever closes:**
  `src/compose_v4/control/replace_completion_law.py` + `completion_law_contract.py`, field
  `proposal.shallow.completion_law`. The completion half of `segment_replace` drew BLIND --
  `rng.integers(1, capacity+1)` over 1..8 plus a uniform C/N/O chain -- while **every** eligible
  completion inserts ONE or TWO atoms, so ~75% of the draw mass landed where nothing is eligible.
  480 draws/arm, arms differing ONLY in the contract field:
      fa7_1     7 -> 13  (+86%)
      braf_2    4 ->  6  (+50%)
      5ht1b_0  20 -> 35  (+75%)
  No control regressed. 6/6 mutations killed with both shared-sink hops killed separately; absent
  field = byte-identical OFF; the margin is taken from the PRODUCTION region-law factory so both
  halves share byte-identical chemistry. **T4 is frozen so this is not in the panel -- it is banked
  for whatever runs next.**
- **FOUR CLAIMS IN MY OWN BRIEF, FALSIFIED BY THE AGENT.** Worth recording because I stated all four
  as fact: (a) "5 witnesses, all replace/-9" -- actually replace -9, replace -9, grow -8, prune -8,
  replace -8; (b) "constrained Pareto max QED 0.633" -- a witness sits at QED **0.7728** / sim
  0.6102, so that figure bounded a RESTRICTED SUPPORT, not the cell; (c) "the witnesses relieve the
  alerts" -- only ONE does, and witness2 passes with **ALERTS=2 UNCHANGED**, its QED coming from the
  excision cutting MW 423.6->311.4 and ALOGP 5.97->3.52, i.e. SIZE not alert relief; (d) a redirect
  I sent (rank the gated endpoints) would have been INERT, caught by reading the consuming loop
  before building. A coordinator's brief is not evidence; an agent that checks it is doing its job.
- **`for-each-ref refs/remotes` CANNOT TELL YOU WHETHER A BRANCH IS ON THE REMOTE.** It enumerates
  LOCAL TRACKING refs, which exist only after a fetch created them, so a pushed branch nobody has
  fetched reads as ZERO and looks like unreplicated single-copy work. **`git ls-remote --heads
  origin <branch>` queries the remote.** Companion errors I made the same night: comparing
  `rev-parse --short` (8 chars) against `cut -c1-7` (7) and reading eight matching branches as
  DIFFER, and reading `vm_stat`'s `Pages free` as free memory when `memory_pressure` reported 84%
  free. **Three instrument errors in one session, all the same shape: a measurement that cannot
  answer the question, believed because it returned a number.**
- **A worktree sweep is safe only with a per-branch remote check.** Removing 26 checkouts freed
  8.7 -> 28 GB; the guard required each branch present on origin at a MATCHING FULL SHA, which is
  why `pmo-realization-repair-20260920` was correctly kept back. Worktree removal never deletes a
  branch -- objects live in the parent repo -- so the check is about REPLICATION, not about loss.
  Recover any checkout with `git worktree add <durable-path> <branch>`.
- **STILL EXPOSED, owner decision pending: ~35 branches dated 2026-09-08..19 exist on NO remote.**
  None are from the current work and none were touched by the sweep, but they are single-copy on one
  disk, which the standing rule calls the failure mode already paid for.

## 2026-09-22 (PMO design decision: macro-options over a global-delta vocabulary, and the reward split)

- **THE ABSTRACTION CHANGED, and the earlier framing was too local.** Asking "where does probability
  disappear along the route" decomposes a transformation into primitives. COMPOSE executes compound
  edits as ONE program, so the controller never has to reproduce a teacher route step by step. The
  right question is **what GLOBAL structural transformation separates a blind molecule from a
  productive basin, and does the controller have a generic macro family covering that TYPE?**
  Represent the move as `omega = (retain region, edit regions, operation family, scale, completion
  constraints)`, execute it atomically, and score only the endpoint.
- **THE REWARD SPLIT, and this is the fundamental part.** Two regimes were being conflated:
      local refinement    r_t = f(G_{t+1}) - f(G_t)      -- measured to work well once in a basin
      global transport    R(omega) = f(G_{t+k}) - f(G_t) -- ENDPOINT return, no internal pruning
  A useful route can run 0.36 -> 0.10 -> 0.02 -> 0.81, and a greedy controller kills it at step one.
  Intermediate states must stay chemically VALID; they need not monotonically improve the oracle.
  **Using a local reward to judge a global move is the classic error, and the whole advantage of an
  exact executable program is that the intermediates need never be exposed to selection.**
- **DONOR LOGIC REVERSED.** Donor transport consumed 42% of B's proposals and contributed 0 of its 25
  nearest approaches. The likely defect is not the idea but the ORDER of the questions. Today it asks
  "which donor + cut + transplant should I make?" It should ask "I need a global transformation of
  type X at scale Y -- can a donor supply a compatible completion?" **Donor becomes a CONTENT
  PROVIDER for a macro move, not the global strategy itself**, competing with the learned chemical
  prior, the recombination bank and a de-novo fragment generator to fill the same declared slot.
- **THE FA7 FINDING IS THE TEMPLATE OF WHAT TO LOOK FOR.** `segment_replace` drew completion size
  uniformly over 1..8 while EVERY eligible completion inserted 1 or 2 atoms -- ~75% of the draw mass
  where nothing could be eligible -- and fixing that one conditional lifted eligible endpoints
  +86%/+50%/+75% on three cells with no control regressing. Nobody could see it until the NEEDED
  transformation was measured against the PROPOSED distribution. The PMO analogue to test (MEASURE,
  do not assume) is whether productive transitions need coherent 6-12 atom replacements while B
  overwhelmingly proposes tiny edits or mismatched donor programs.
- **Rank bottlenecks by LOST PROBABILITY MASS**, so the output reads "X% of the failure is region
  selection, Y% scale, Z% completion content" rather than "discovery seems hard".
- **FAIRNESS BOUNDARY, explicit.** Development diagnostics MAY use answer-known material -- exact
  anchors, compiled routes, teacher structures, V_local. The product must be a GENERIC EDIT GRAMMAR
  ("large replacement", "coherent excision", "two-region edit"), never a task-specific runtime cheat.
  A prior fitted on teacher routes is already a measured decisive negative here (lower
  complete-program yield on both populations, p=0.0001, attributed to the region law); the atlas
  teaches which generic CAPABILITY is missing, it does not become the capability.
- **SCOPE CAVEAT ON THE RUNNING A/B, declared by its own agent before measuring:** the construction
  prior reaches only the shallow lane (`DynamicProgramOptimizer._mutate` fresh synthesis and the
  warm-memory lane). `structured_program_channel`, `joint_dependency_region_jump` and `recombination`
  are NOT prior-aware. **So a null there is a null about the shallow construction lane, not about
  chemical priors in PMO** -- which is exactly why the global-delta census is the more fundamental
  experiment and runs in parallel rather than after.
## 2026-09-21 (the July de-novo artifacts were never lost -- they are on a DIFFERENT PROFILE'S volume)

- **CORRECTS the 2026-09-21 entry "A STRONGER DE-NOVO GENERATOR CANNOT BE PROMOTED, ONLY TRAINED".**
  That entry recorded a full recursive scan of `compose-v4-artifacts` finding ZERO de-novo training
  artifacts, and correctly noted the volume was created **2026-08-09, AFTER the July runs**. It then
  drew the wrong conclusion. There are FOUR Modal profiles here (`nitya`, `rahul` -> workspace
  kosha-labs, `rarospec2`, `rahul-94866`) and **each has its own `compose-v4-artifacts`**. The
  `rahul` profile's copy was created **2026-07-17** -- the Lineage B window -- and holds 191 run
  directories including every `tree_fcd_transfer` / `stage3` de-novo run. **A volume name is not a
  volume identity; check `modal volume list` on EVERY profile before concluding an artifact is gone.**
  The same reasoning error would have justified a multi-day retrain that reproduced work already
  sitting on disk.
- **`checkpoint.best_so_far.pt` and `checkpoint.recovery.pt` are DIFFERENT ARTIFACTS with different
  contracts, and reading resume-capability off the wrong one inverts the conclusion.**
  `best_so_far` is `checkpoint_kind = 'interim_best_evaluation_model'`: weights plus
  `selected_validation`, and **no optimizer, scheduler or RNG state** -- so it genuinely cannot be
  resumed. `checkpoint.recovery.pt` (written by `recovery_every: 500`) is
  `checkpoint_kind = 'exact_training_recovery'` and carries `current_state_dict`,
  **`optimizer_state_dict`, `torch_rng_state`, `cuda_rng_states`**, `best_state_dict`,
  `best_metrics`, `history` and `completed_steps`. Sizes differ 3x (38 MB vs 113 MB) because AdamW
  keeps two moments per parameter -- **the size ratio is the cheap tell.** "The run cannot be
  resumed" was measured on `best_so_far` and was false of the run.
- **A checkpoint's `provenance_sha256` IS the producing run's `run_identity_sha256`, so the exact run
  is FINDABLE by scanning volume manifests.** `modal_apps/train_tracelet_gm.py:1117` sets
  `recipe["arguments"]["provenance_sha256"] = run_identity_sha256`, and each run writes
  `manifest.training.json` carrying that value. Downloading every candidate run's
  `manifest.training.json` and matching the hash identified Lineage B as
  **`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`** by EXACT HASH rather than by inference from a
  recipe match. Do this before reconstructing anything: the identification is minutes of work and it
  is proof, not a guess. Note `run_identity` also folds in `run_label` and the data manifest, so the
  value cannot be recomputed from source alone -- the manifest is the only route.
- **The from-scratch reproduction gate was ALREADY RUN, in July, and it PASSED -- by two runs with
  DIFFERENT code identities.** `compose-v4-stage3-flexible-graft-3k-fullcache-c39520c-v2`
  (`run_identity af57588e...`, a later source revision) shares Lineage B's recipe and seed 20260717,
  and its stored validation `history` is **byte-identical to Lineage B's at every evaluation through
  step 1000**: 56.655693 / 19.152582 / 18.760945 / 15.715200 / **14.545557**, the last matching the
  step-1000 checkpoint's `early_stopping_reference_loss` exactly. It then continued to step 2500.
  So the de-novo training path is empirically invariant across those revisions, and the trajectory
  extension the retrain was meant to produce **already exists on disk**. A stored `history` list is a
  reproduction reference; look for one before spending GPU to regenerate it.
- **Only `best_so_far` and `recovery` survive per run, so intermediate steps are NOT recoverable.**
  `best_so_far` is overwritten whenever evaluation improves and `recovery` every 500 steps, so the
  surviving Lineage B trajectory is exactly TWO measurable states: step 1000 and step 2500. The
  sibling runs (`-cert-cdac478-v1/v2`, `-fullcache-c39520c-v1`) kept manifests only. Plan a
  checkpoint sweep around the states that EXIST rather than around a desired grid.
- **The current code still samples the July checkpoint identically, measured two ways.** The archived
  July step-1000 ring histogram (`results/diagnostics/step1000_ring_topology_comparison.json`,
  n=100) is `{3:54, 4:23, 5:107, 6:202, 7:4, 9:1}` -> **19.69% of RINGS strained**; an independent
  n=50 sample drawn under current code three months later gives 32/160 = **20.0%** (two-proportion
  z = 0.08, p = 0.93). Ring-size composition is reproduced. This is evidence about SAMPLING, not
  about training.
- **Report the per-RING strained fraction beside the per-MOLECULE prevalence.** Prevalence (share of
  molecules carrying any 3/4-ring) moves when molecules get bigger or smaller even if the closure
  policy is unchanged; the per-ring fraction `P(size | a ring exists)` is invariant to both molecule
  size and ring count, so it is the size-independent read of the learned closure policy. Measured
  `ring_removing_events == 0` over 50 trajectories (the de-novo model has zero
  `ring_system_delete` validation examples and fires none at sampling), which is what licenses the
  ENDPOINT ring census to stand in for ring CREATION -- check that, do not assume it.
- **The n=50 strained-vs-clean gap is PARTLY confounded with molecule size, and the two axes behave
  differently.** OLS with size partialled out (`scripts/denovo_checkpoint_sweep.py`,
  `compose_v4.eval.denovo_ring_decomposition`): **SA** raw strained difference +0.946 falls to a
  strain coefficient of **+0.614 (SE 0.280)** with heavy atoms at +0.083 (SE 0.021) -- a genuine
  strain effect survives, about 65% of the raw gap. **QED** raw difference -0.120 falls to
  **-0.054 (SE 0.039)**, i.e. NOT distinguishable from zero, while heavy atoms carry -0.0165
  (SE 0.0029). So the QED half of the association is essentially all size. Quoting the raw
  conditional split alone would have attributed both to ring strain.

## 2026-09-21 (the de-novo small-ring defect is a SUPPORT defect, and its fix was built and never wired)

- **The model is NOT biased toward small rings; the legal SUPPORT at ring-formation time is.**
  Measured, from the committed `diagnostics/ring_calibration/step2500_exact_support_audit_12.json`
  (265 ring events over 100 rollouts at step 2500): the typed ring catalog's own unconditional
  small-ring mass is **0.0302** (285 small of 3,092 templates = 9.2% by count), but uniform over the
  legal support AT THE STATES WHERE RING EVENTS ACTUALLY FIRE is **0.4431**. The model's prior given
  that support is **0.2837** and it produced **0.2804** (observed small-event fraction 0.25). So the
  learned policy sits well BELOW uniform-over-support -- it is already pushing away from small rings
  -- while the support it is conditioned on is ~15x enriched in small rings relative to the catalog.
  **CONSEQUENCE: an SA penalty or a small-ring rate penalty is the wrong instrument.** It would push
  a policy that is already anti-small-ring against a support that leaves it no alternative. The
  defect is upstream, in WHICH STATES the ring decision is taken at.
- **Why the support is degenerate: ring growth is supervised LATE, on crowded states.** The de-novo
  transport compiler emits one sequential program in which cardinality events (`atom_insert`/
  `atom_delete`) and ring events interleave in compiler order. By the time `ring_system_grow` fires,
  the molecule is near the 40-atom cap with few free slots, and large-ring templates are no longer
  executable -- so conditioning on "a ring grows" renormalises onto whatever remains, which is
  mostly small. The 2026-07-19 audit independently localised **every newly created small ring to
  `ring_system_grow`, not to Graft**, which is consistent.
- **THE FIX ALREADY EXISTS, IS TESTED, AND IS UNREACHABLE.** `src/compose_v4/rewrite/
  commuting_schedule.py` (added `50880537`, **2026-07-20 -- one day AFTER the step-1000 checkpoint**)
  detects commuting adjacent events BY EXECUTION and moves whole ring transactions to the earliest
  state at which they are actually executable, with `atom_insert`/`atom_delete` as phase barriers and
  a swap accepted only when both orders are legal and array-exact after the pair -- so slot gauge and
  the corpus endpoint are unchanged. `compile_carbon_tree_to_target` exposes it as
  `event_schedule="exact_early_ring"` (default `"sequential"`), and `tests/test_commuting_schedule.py`
  passes 6/6. **But `exact_early_ring` appears NOWHERE outside `tree_transport.py`**: no trainer flag,
  no recipe key, no Modal passthrough, and `build_tree_transport_path_records` has no
  `event_schedule` parameter at all, so neither of its two `compile_carbon_tree_to_target` call sites
  (`experiments/tracelet_conditional.py:192`, `:454`) can forward one. Lineage B therefore trained on
  `"sequential"`. This is the 2026-09-20 region-law lesson again, in a second place: **a validated
  repair behind an opt-in keyword is INERT until a caller passes it, and "the module has tests" hides
  that completely.** Grep for the keyword at CALL sites, not definition sites.
- **The plumbing gap is exactly five hops**, all additive and default-preserving: recipe argument ->
  gate CLI flag -> `build_tree_transport_path_records(..., event_schedule=...)` -> its two
  `compile_carbon_tree_to_target` call sites -> Modal `build_tracelet_recipe_argv` passthrough. Any
  wiring must be mutation-tested at the CALL sites (drop the keyword at each hop and require a named
  test to go red), because two hops share one sink and a single consultation test would stay green.
- **This is a training-DISTRIBUTION correction, not an objective.** It changes which states the ring
  decision is supervised at; it adds no SA term, no QED term and no reward, and the compiled endpoint
  is provably unchanged. That is what makes it admissible where a benchmark-chasing penalty is not.

## 2026-09-21 (CORRECTION: the identical validation curve was INHERITED BY RESUME, not reproduced)

- **CORRECTS the entry above ("The from-scratch reproduction gate was ALREADY RUN ... and it PASSED
  -- by two runs with DIFFERENT code identities"). That claim was WRONG and I am withdrawing it.**
  The continuation run `compose-v4-stage3-flexible-graft-3k-fullcache-c39520c-v2` did not train from
  scratch. Its own `manifest.training.json` -> `run_identity.recipe.arguments` carries
  `resume_checkpoint = /artifacts/compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/
  checkpoint.recovery.pt` together with `allow_resume_provenance_mismatch: true`. So it RESUMED from
  Lineage B's exact recovery state -- weights, optimizer moments and RNG -- and the step 1..1000
  `history` entries are byte-identical because `history` is a LIST CARRIED INSIDE THE RECOVERY
  CHECKPOINT and restored on resume. They are the same numbers, not two independent measurements of
  the same number. **A matching stored history proves shared ancestry, not reproducibility.** Before
  reading any agreement between two runs as an equivalence result, diff their recipe arguments for
  `resume_checkpoint` / `initialize_*`; ancestry is recorded there and nowhere in the metrics.
- **What survives, and it is the more useful fact:** step 2500 IS the genuine continuation of the
  Lineage B trajectory, resumed from its exact optimizer and RNG state rather than restarted. So
  "Lineage B trained 1,500 steps longer" already exists as an artifact and answers the continued-
  training question directly -- better than a from-scratch rerun would, since a rerun could only
  approximate the trajectory it is resuming.
- **The standing caveat that replaces the withdrawn claim:** steps 0->1000 and steps 1000->2500 ran
  under DIFFERENT source revisions (`source_sha256` 98ca9b28... vs 99354484...), which is why the
  launcher needed `allow_resume_provenance_mismatch`. The corpus is identical across both
  (train sha256 70526d92..., reference 1f8e92f7..., same byte counts), and the recipes differ only in
  run-scoped paths plus the resume flags. So a step-1000 vs step-2500 comparison is a comparison
  across a code-revision boundary as well as a training-step boundary, and must be reported that way.
  Whether the de-novo path is invariant across those revisions remains **UNMEASURED**; the cheap test
  that would settle it is the gate's `--load-checkpoint` mode, whose report emits the recomputed
  `initial_validation` beside the stored `selected_validation` for the same weights.

## 2026-09-21 (continued training does NOT fix the small-ring defect; quality rose by SHRINKING molecules)

- **MEASURED, matched n=70 per arm on identical trajectory seeds, pinned production kernel**
  (`diagnostics/denovo_checkpoint_sweep_v1/`), step 1000 vs step 2500 of the SAME Lineage B
  trajectory (step 2500 resumed from step 1000's recovery state; ring catalog byte-identical,
  fingerprint `50337de077f374db`; corpus identical):
      validity/attempts     1.0000 -> 1.0000     uniqueness   1.0000 -> 1.0000
      diversity             0.8906 -> 0.8856     quality      0.1714 -> 0.3000  (z=+1.79)
      molecules w/ 3-4 ring 0.4429 -> 0.5143  (z=+0.85)
      per-RING strained     0.1679 -> 0.1948  (z=+0.77)     <- the closure-policy shape
      rings/molecule          3.74 -> 3.30      heavy atoms  28.70 -> 26.87
      ring-forming events/traj 3.31 -> 2.86
      ring sizes  {3:26,4:18,5:63,6:152,7:3} -> {3:38,4:7,5:73,6:111,7:2}
  **The defect did not improve on either axis; both edged UP.** No endpoint difference reaches 95%
  significance at n=70, so read directions rather than verdicts -- but there is no hint of the
  improvement continued training was supposed to deliver.
- **The quality gain is attributable to SHRINKING, not to better ring chemistry, and only the
  per-RING metric shows it.** Quality rose 0.171 -> 0.300 while molecules lost 1.83 heavy atoms and
  0.44 rings each and the model fired 0.46 fewer ring-forming events per trajectory. The per-molecule
  prevalence cannot separate those; the per-ring fraction `P(size | a ring exists)` is invariant to
  molecule size and ring count, and it went the WRONG way. Chemically the histogram is worse too:
  6-membered rings 152 -> 111 while 3-membered rings 26 -> 38. **Always report the per-ring shape
  beside any aggregate quality gain, or "quality improved" will be read as "chemistry improved".**
- **QED strain effect is ZERO once size is held fixed, REPLICATED at both checkpoints**
  (+0.033+-0.039 at step 1000, -0.037+-0.037 at step 2500), while heavy-atom count carries it
  (-0.024 per atom, |z|~8 in both). The SA strain penalty is real at both (+0.791+-0.217 and
  +0.496+-0.218). So of the original n=50 conditional split, the SA half was a genuine strain effect
  and the QED half was molecule size -- and that decomposition now replicates across two checkpoints.
- **This converges with the support audit and settles the mechanism.** The model already sits below
  uniform-over-support on small rings (0.284 vs 0.443), so it is not the policy that is broken; more
  gradient steps cannot repair a support that is ~15x enriched in small rings relative to the
  catalog. Two independent lines -- a longitudinal checkpoint comparison and a static support census
  -- agree that the defect is upstream of the learned rates. **Do not spend further training on it.**

## 2026-09-21 (de novo ring schedule: the support defect is real, localized, and the repair plateaus)

- **The n=800 three-arm probe's OWN predeclared rule returns FIX_COMPILER_ORDERING**, and reducing it
  settles two things the earlier n=24 probe could not. Arm A (sequential, what Lineage B trained on)
  **0.2919 +- 0.0038**, arm B (shipped `exact_early_ring`) **0.1894 +- 0.0027**, arm C (same, phase
  barriers removed) **identical to arm B to sixteen digits** -- which proves the barrier is never
  reached and relaxing it is a provable no-op.
- **CORRECTS the standing slot-scarcity framing at TRAINING time, with the right metric.** Free slots
  are IDENTICAL across all three arms (11.883) while the support mass moves 0.29 -> 0.19, so slots
  cannot be the mechanism. What the mass actually tracks is support SIZE:
  `r(mass, log legal_template_count) = -0.87` (A) and **-0.92** (B), against
  `r(mass, free_slots) = +0.13` with the WRONG SIGN -- smaller molecules have MORE small-ring mass,
  not less. Anyone proposing "commit the ring when only its minimum atoms exist" is proposing to move
  along the +0.13 axis in the damaging direction.
- **The collapse is ONE STEP, and it is decoration.** Full support curve along a real 32-step trace
  (pinned kernel): indices 0-28 sit at legal 300-1600 and mass 0.10-0.16; the LAST `atom_restate`
  before the ring cuts legal **1034 -> 295** and lifts mass **0.1199 -> 0.2237**; the first ring grow
  then leaves only **28** legal templates at mass 0.3571 for the second ring system. So decoration
  constricts the support, and each ring commitment constricts it further for the next.
- **Post-hoc adjacent-transposition scheduling is EXHAUSTED where it stops.** Blocker census over
  2,017 ring events: after bubbling as far left as it can, **82.2% are blocked by a `bond_reroute`**
  and **95.3% have no exact commutation available at all**. The shipped scheduler cannot cross the
  graft phase, so it can only ever park a ring event immediately after the last graft.
- **The repair is an EMISSION-ORDER change, `event_schedule="ring_dependency_block"`.** Each ring
  system is committed at the earliest graft prefix the executor accepts it at, which also places it
  ahead of all decoration. Feasibility is decided BY EXECUTION; the tree-edge test beside it is a
  cheap pre-filter, MEASURED to leave every compiled trace byte-identical when removed.
- **A per-item fallback cannot protect a corpus; you need a TRACE-level one.** Committing a ring
  early can invalidate a LATER graft that touches one of its atoms -- 3 of 150 real training
  molecules compiled under `sequential` and raised under the block. The per-system fallback cannot
  see this because the commitment it would have to undo already succeeded. Without abandoning the
  whole reordering as a unit the schedule silently shrinks the corpus by ~2%, and that loss is
  INVISIBLE in any per-arm mean computed over whatever survived.
- **Reach is bounded and must be reported with the win:** 30% of real molecules defer at least one
  ring system to the legacy end-of-route position, and ~2% fall back entirely. Both are flagged per
  trace (`ring_dependency_block_deferred`, `ring_dependency_block_fell_back_to_sequential`).
- **Compile cost 1.087x**, measured in executor applications per compiled trace (50.2 -> 54.6, n=150)
  -- a load-independent counter, because this machine is shared with three other agents.
- **A knob measured INERT was removed rather than shipped.** A graft-gap spreader between consecutive
  ring commits produced byte-identical output at gap 0, 3, 6 and 20 across 116 molecules, because
  ring dependencies complete too late in the graft phase for the gap to ever bind. Shipping it would
  have implied a tuning lever that does nothing.
- **`sequential` is byte-identical after the refactor**, fingerprinted over 113 real molecules
  against the branch base: 0 differing traces, same 13 pre-existing compile failures.
- **ACCEPTANCE: the schedule repair FAILS the <0.15 gate on the mean and passes on the median, and
  it does NOT beat the scheduler that already ships.** Matched three-arm run, pinned kernel,
  molecules drawn uniformly at random from the recipe's own train partition, 250 molecules / 590
  ring decision points, zero oracle calls:
      arm                     mean              median   legal templates
      sequential              0.3089 +- 0.0069  0.2716   349
      exact_early_ring        0.1921 +- 0.0051  0.1381   743
      ring_dependency_block   0.1957 +- 0.0051  0.1394   720
  Paired per ring event: repair minus sequential **-0.1132 +- 0.0071** (462 improved / 96 worsened),
  repair minus shipped scheduler **+0.0037 +- 0.0017 with 515 of 590 events UNCHANGED** -- which is
  marginally and DETECTABLY WORSE, about two standard errors; state the sign rather than rounding it
  to "equivalent". Endpoint exactness holds **250 of 250 in every arm**, by array identity AND
  canonical key.
- **So the deliverable is PLUMBING, not an algorithm.** The support gain worth having is already
  available from `exact_early_ring`, which shipped 2026-07-20 and was never wired into the training
  recipe. The dependency block reproduces it at 1.087x compile cost instead of 14,897 verified
  adjacent swaps per 800 traces, and places rings INSIDE the graft phase, but it adds no support
  quality on top. Do not present it as a further gain.
- **The residual is a ring-system ORDINAL effect and no schedule can fix it.** Block arm by ordinal
  within a molecule: **0.1403 / 0.1977 / 0.3026 / 0.2982 / 0.5320** on n = 156 / 125 / 67 / 23 / 2.
  The FIRST ring system passes the gate; each commitment constricts the legal support the next is
  decided against, which is unavoidable in a sequential trace carrying more than one ring system
  (mean 2.5 systems per molecule). Closing it needs a support-level change or a scoped claim.
- **The pooled size histogram and the per-event mean disagree, and the gate is the per-event mean.**
  Pooled 3+4-ring share moves only 17.44% -> 13.80% while the per-event mean moves 0.310 -> 0.201,
  because pooling weights each event by its support SIZE and the per-event mean does not -- and the
  events that matter are exactly the ones whose support has collapsed to a handful of templates.
  Report both and say which one the gate is stated over.
- **Modal `.map()` ordered output can deliver ZERO rows from a run that did most of its work.** An
  n=2000 fan-out processed 536 molecules across 62 containers and handed the driver nothing, because
  shard 0 had stalled on one expensive molecule and ordered output queues every finished shard
  behind it. Pass `order_outputs=False` whenever results are reduced as a set. Separately: killing
  the LOCAL driver does not stop the app -- it held 99 containers afterwards and starved the next
  launch until `modal app stop`.

## 2026-09-21 (the ring-support collapse is the HOST, and it is a documented v1 scope)

- **ANSWERED: committing one ring system drops the legal template count 1034 -> 28 because
  `_eligible_grow_host_graph` (`ring_system_fiber.py:2160`) admits only ACYCLIC, CARBON, NEUTRAL,
  SINGLE-BONDED atoms and asserts the result is a FOREST.** Its docstring says so in one line:
  *"Carbon, neutral, acyclic single-bond support for v1 ring installation."* So committing a ring
  system permanently removes its atoms from the scaffold every later ring decision is made against,
  and installing a heteroatom removes one more while raising a bond order SPLITS the host without
  removing any atom at all. A molecule's ring systems compete for one shrinking carbon forest.
- **This is a SCOPE restriction, not a catalog gap, and the catalog proves it by itself: 0 of 3,092
  templates require a cyclic host.** Every template's `source_bonds` pattern is a forest, so ring
  systems are installed ATOMICALLY onto acyclic carbon -- fused systems are single templates, never
  a ring grown onto an existing ring. Nothing in the catalog can reuse a ring the molecule already
  built. MEASURED, needs no molecules, so no sample of states can explain it away.
- **The small-ring share of the catalog is a steep function of remaining host, and that single table
  is the whole mechanism:**
      host atoms  3      4      5      6      7      8      9     10     12     14     18    30+
      templates   2      5     13     34     77    154    324    564    955  1,883  2,965  3,092
      3/4-ring  100%   100%  46.2%  47.1%  57.1%  42.9%  25.0%  17.0%  15.3%  10.5%   9.4%   9.2%
  A large ring needs a large contiguous carbon tree; a three-ring needs three atoms. A support
  measured on a small host is small-ring-enriched BY CONSTRUCTION. The knee is host ~7-9.
- **This retro-explains the correlation that looked backwards.** The acceptance run measured
  `r(mass, free_slots) = +0.13` -- smaller molecules showing MORE small-ring mass, which read as
  nonsense. Smaller molecule -> smaller host -> the survivors are the small rings. The sign was
  right; the mediator was the host, not the slots.
- **MEASURED on real traces (90 ring decisions, 34 molecules, block schedule):** host atoms
  26.9 -> 20.1 -> 14.2 -> 13.3 by ring-system ordinal, carried almost entirely by committed cycles
  (0.0 -> 17.2) and NOT heteroatoms (0.4 -> 1.2) under a schedule that commits rings first --
  which is what it should be. `r(small mass, largest_host_tree) = -0.673`,
  `r(log legal, largest_host_tree) = +0.772`.
- **The catalog table is a LOWER BOUND; observed mass runs 1.4x-2x above it** (ordinal 2: predicted
  ~15.3%, observed 29.8%). Two further mechanisms the atom count alone misses: fitting needs the
  right branching SHAPE, not just enough atoms; and **host COMPONENTS rise 1.21 -> 1.83 -> 2.35**
  with ordinal, because a committed ring can SPLIT the remaining forest and a template needs one
  contiguous piece. Report `largest_host_tree`, never `host_atoms` alone.
- **CONSEQUENCE, and it closes the de-novo small-ring line: no schedule, reward, or catalog addition
  can fix this.** Scheduling moves WHEN a ring is committed and cannot stop it consuming its atoms;
  the model already sits below uniform on small rings and cannot pick a template outside the
  support; and 2,965 of 3,092 templates already fit an 18-atom host, so adding templates changes
  nothing while the host is small. Exactly two doors remain: extend `_eligible_grow_host_graph` to
  cyclic/heteroatom scaffolds (a real capability change that invalidates the forest assertion its
  DP relies on), or scope the claim to atomic ring installation on acyclic carbon.
- **SEPARATE, still open, and cheap: `exact_early_ring` reaches no training path.** Lineage B
  trained on `sequential` while a scheduler worth 0.3049 -> 0.1926 has sat unwired since
  2026-07-20. **CORRECTS the recorded "five additive hops": it is five hops but NINE call sites** --
  `train_tracelet_cnof_gate.py` calls `build_tree_transport_path_records` at FOUR sites (1226, 2885,
  2897, 2909) and that builder has TWO `compile_carbon_tree_to_target` sinks
  (`tracelet_conditional.py` 192, 454). The two sinks share one parameter, so a consultation test
  exercising only one stays green -- the exact near-miss the region-law wiring hit. Mutation-test at
  the CALL sites, not the definition.
- **THE ROLLOUT 44.3% IS THE SAME MECHANISM, AND IT IS BIMODAL, NOT ENRICHED.** The committed
  reference audit stores both the statistic and the rollout state it was measured on, so this is
  directly attributable (`scripts/denovo_rollout_host_attribution.py`, which reproduces the audit's
  own stored mean before reporting anything). Over its 12 states:
  `r(small mass, largest_host_tree) = **-0.890**`, `r(log legal, largest_host_tree) = **+0.962**`,
  mean largest host tree 11.4. States with an intact host score **0.116-0.192**; states whose host
  has been eaten score **0.417-1.000**. Quoting the 0.4431 mean alone describes NEITHER population.
- **On 3 of 12 audited rollout states EVERY legal template is a small ring** -- 2 to 5 legal
  templates on a host cut to a 3-5 atom fragment by 20-23 atoms already locked into committed rings.
  **The model has no non-small option to choose.** No amount of training, reward shaping or better
  sampling can fix a state whose support contains no alternative. This closes the loop on why ~50%
  of generated molecules carry a 3/4-ring: at their later ring decisions there was nothing else.
