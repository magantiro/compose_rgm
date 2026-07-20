# The COMPOSE generator, from zero (for the lipid lane)

**Audience:** someone who has never seen this generator. No prior knowledge
assumed. By the end you should have a correct mental model of what the model is,
how every piece fits, which file does what, how a run works end to end, and
where each lipid-specific change goes. Read this, then read the files it points
you at; you'll know what you're looking at.

Companion: `HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md` (what to reuse + the
paper goals). This doc is the *internals*.

---

## 1. The one-paragraph mental model

A molecule is a graph (atoms + bonds). This model **generates a molecule by
starting from a simple seed graph and applying a sequence of small, chemically
legal edits ("rewrites") until it stops** — insert an atom, delete an atom,
change a bond order, attach a whole ring, etc. It is a **continuous-time Markov
chain (CTMC)** over molecular graphs: at any current molecule it has a set of
legal edits, each with a learned **rate** (how fast that edit "wants" to fire);
it samples which edit fires and when, applies it, and repeats. A neural network
predicts those rates from the current molecule. **Crucially, every intermediate
is itself a complete, valid, connected molecule** — there is never a half-built
or invalid state. This is "Rewrite Generator Matching": the rewrite rules define
*which* edits are possible; the network learns *when/where/which* edit fires.

That's it. Everything below is detail on: the graphs, the edits, the network
that scores them, how it's trained, how you sample from it, and how you steer it.

---

## 2. Vocabulary (read this once; terms recur everywhere)

- **State / molecular graph:** a molecule as a padded fixed-size array — atom
  types + a bond-order matrix (aromatic-aware). "Padded to 40 slots" = arrays
  sized for up to 40 heavy atoms; lipids need ~48+.
- **Rewrite / operator / mark / action:** one legal edit (e.g. "insert a nitrogen
  double-bonded to atom 3"). "Mark" ≈ the fully-specified edit incl. its payload.
- **Family:** the *category* of edit. There are 10: `atom_insert`, `atom_delete`,
  `atom_restate` (retype an atom), `bond_reorder`, `bond_reroute` (= **Graft**,
  detach+reattach a subtree), `cycle_insert`, `cycle_attach`, `ring_system_grow`
  (install a whole ring system atomically), `ring_system_delete`,
  `ring_system_restate`. (Legacy names: `bond_reroute`=Graft; `cycle_*` are older
  ring primitives — the *production* ring mechanism is `ring_system_*`.)
- **Rate / hazard:** a non-negative number; the CTMC fires faster where rates are
  higher. "Total hazard" = sum of all legal edit rates at a state (the clock).
- **Fiber / legal support:** the set of legal edits at a state (computed by the
  executor). The network only ever assigns probability *within* this set.
- **Executor / kernel:** the code that knows the rewrite rules — which edits are
  legal, and how to apply one to produce the next valid molecule. This is the
  "chemistry engine." Shared, Codex-owned, meant to stay frozen/versioned.
- **Teacher / certified program:** a *known* sequence of legal edits that turns a
  seed into a specific target molecule from the corpus. Training imitates these.
- **Source / prior:** the seed distribution. Here: a randomly sampled
  degree-bounded **carbon tree** (an acyclic C-only skeleton), sizes 4–40.
- **Ancestral sampling:** generate by simulating the CTMC forward from a fresh
  seed with **no target and no lookahead** — just repeatedly "score rates → pick
  an edit → apply → repeat." This is inference.
- **Canonical successor / quotient:** several different edits can land on the
  *same* resulting molecule; we aggregate ("quotient") their rates at that
  molecule so the math counts each distinct outcome once. Also removes "self
  edits" that map a molecule to itself.
- **Generator Matching (GM):** the training framework — learn the CTMC's rates so
  its marginal distribution matches the data. The loss is a Poisson/Bregman
  objective over the teacher edits. You don't need the theory to work here; you
  need to know the loss rewards putting rate on the teacher's edits.

---

## 3. State space, seed, and how a target is reached (the "transport")

- **State:** `src/compose_v4/chem/molecular_graph.py`, `chem/state.py`,
  `chem/aromaticity.py`. A molecule is stored as integer atom types + an
  integer bond matrix; aromatic rings are stored as editable Kekulé
  (alternating single/double) bonds and re-perceived as aromatic on the RDKit
  round trip. So "aromatic" is a *derived* property, not a 5th bond order.
- **Seed:** `chem/source_prior.py` samples the carbon tree (acyclic, C only,
  degree-bounded) with an empirical size law. **The model must *build* all rings,
  heteroatoms, and bonds from this bare skeleton** — nothing is pre-seeded. (This
  is why ring-building is hard for drug-like molecules and why the ring family is
  *restricted*, not perfected, for ring-simple lipids.)
- **Transport:** `rewrite/tree_transport.py` builds, for a (seed, target) pair, a
  **certified program** — the concrete legal edit sequence that turns the seed
  into the target. It uses two tree couplings per target and
  `flexible_size_graft` so the molecule can grow/shrink around the seed. This is
  what produces the training "teachers." The `teacher_ordering=sequential`
  setting linearizes ring installs to the *end* of the program (this is the root
  of the drug-like ring pathology — irrelevant for lipids).

---

## 4. The rewrite operators + executor (the chemistry engine)

Files: `rewrite/operators.py`, `rewrite/kernel.py`, `rewrite/compiler.py`,
`rewrite/tracelet_compiler.py`, `rewrite/tracelet_fiber.py`,
`rewrite/factorized_fiber.py`, `rewrite/typed_ring_catalog.py`,
`rewrite/ring_system_fiber.py`, `rewrite/ring_junctions.py`,
`rewrite/commuting_schedule.py`.

Mental model:
- **`kernel.py` / `de_novo_rewrite_system()`** is the entry point: given a state
  and a family+action, `.apply()` returns the next state; it also enumerates
  legal actions. Every `.apply()` result is sanitized/valence-checked/connected —
  **validity closure**. This is what you call to "take an edit."
- **Whole ring systems are one atomic action.** `ring_system_grow` installs an
  entire connected cyclic component (single/fused/bridged/spiro) — all its atoms,
  bond orders, aromatic edges, charges, hydrogens — in a single transaction, from
  a **catalog** of ring-system templates (`typed_ring_catalog.py`). The catalog
  has ~4096 raw templates (deduplicated to ~2074 production selector keys / ~130
  topology-cycle groups). It never exposes a half-built ring.
- **Fibers** (`tracelet_fiber.py`, `factorized_fiber.py`, `ring_system_fiber.py`)
  compute the *legal action set* at a state and the tensors the network scores.
  This is the "which edits are legal + their features" layer. It's the expensive
  part (chemistry), which is why runs pre-compile it into caches (§7).
- **For lipids:** this is the **kernel** (gate P2-G2). The size cap (40→~48+),
  any new atom types (P?), protonation/charge policy, and the *restricted ring
  catalog* (single 5/6 head rings only) live here. **This is shared RGM core —
  coordinate changes with Codex; don't fork.**

---

## 5. The rate model — the neural network (`model/factorized_tracelet_rate_model.py`)

This is the brain. Input: the current molecular graph (+ a time value, + optional
conditioning). Output: **rates for every legal edit**, produced by a
*factorized* set of heads so the huge action space is tractable:

1. **Graph encoder** — a message-passing GNN turns the molecule into per-atom and
   global vectors.
2. **Total hazard head** — one number: the overall clock rate.
3. **Family head** — a distribution over the 10 families (which *kind* of edit).
4. **Mark heads** — within a family, *which specific* edit: `grow_root_head`,
   `grow_connected` (which atom + bond order + new type), `atom_delete` (which
   atom), `atom_restate`, `bond_reorder`, `bond_reroute` (Graft placement),
   `ring_system_grow` (which template + placement + electronics), etc.
5. **Ring sub-structure heads** — because ring installs are complex:
   - `ring_topology_group_head` — picks a `(topology_class, cycle_sizes)` group
     *before* the exact template (the `topology_cycle_hierarchical` factorization;
     the older mode is a flat softmax over all templates).
   - ring electronic heads — atom types + aromatic roles inside the ring, either
     `factorized_local` (per-atom) or `factorized_contextual` (adds category-pair
     potentials for joint patterns like adjacent [nH]).
6. **Empirical mark prior** (`empirical_mark_prior_mode`): optionally add
   *fixed corpus base-rate log-probs* under the learned heads, so the network
   learns a **residual** above corpus frequencies instead of from scratch
   (`corpus_residual_v1`; `none` disables). This is a ~14 s scan of the teacher
   actions, stored as buffers.

Hard rule everywhere: the **executor masks** decide what's legal; the network
only shapes probability *inside* the legal set. It cannot invent an illegal edit.

**Trainable scopes** (`experiments/factorized_mark_conditional.py` →
`configure_factorized_trainable_parameters`): freeze/unfreeze subsets by name
prefix — `all`, `chemistry_marks_only` (atom/bond/ring-electronic heads),
`ring_topology_only` (just the topology group head), `chemistry_and_topology`
(both). Freezing keeps the family/Graft/timing rates stable while you fix one
thing.

---

## 6. Training objective (`gm/loss.py`, the experiments/* backends)

- The teacher gives, at each state along a certified program, *the edit that was
  taken*. The **Generator-Matching / Poisson-Bregman loss** rewards the model for
  putting rate on that edit and calibrating the total hazard. Files:
  `gm/loss.py`, `experiments/factorized_mark_conditional.py`,
  `experiments/tracelet_conditional.py`.
- **Canonical-successor quotient:** `experiments/canonical_successor_distillation.py`
  aggregates edits that reach the same molecule and virtualizes self-edits, so the
  learned law is over *distinct molecular outcomes*, not syntactic aliases. This
  is the principled fix for "Graft thrashing" (the model otherwise wastes rate on
  no-op/back-and-forth Graft edits). (You mostly consume this via the sampler in
  §7, which already virtualizes self-Grafts at inference.)
- **What "training" optimizes:** lower loss = better *imitation* of teacher edit
  rates. **Warning that cost us this whole project:** lower teacher loss does NOT
  guarantee better *rollouts* (samples). Always judge on a **rollout** (generate
  molecules, measure their properties), never on loss/accuracy alone.

---

## 7. The sampler (`experiments/calibrated_rewrite_sampling.py`) — how you generate

This file you should read fully; it's small and central. It wraps the rate model
for inference and does three things via **exact CTMC thinning** (a standard trick:
to lower a specific rate, sample the edit but *reject* it with some probability;
a rejected edit becomes a "virtual" time-only jump — the molecule doesn't change):

- **Family/mark calibration knobs:** `atom_delete_log_rate_adjustment` (−0.5 to
  stop over-deletion → correct size) and `small_ring_log_rate_adjustment` (−1.5 to
  suppress 3/4-membered rings). Non-positive only; they *lower* selected rates
  without renormalizing everything else.
- **Self-Graft virtualization + immediate-backtrack safety:** if a Graft's
  canonical successor equals the current molecule (a no-op) or immediately returns
  to the previous molecule, it's routed to a virtual jump. **This is the
  graft-thrashing fix** — it's why the calibrated sampler has 0.14% backtracking.
- `AnalyticPancakeQuotientSampler` / the calibrated wrapper compose these; the
  ancestral loop lives in the rollout code (`scripts/evaluate_tracelet_rollouts.py`
  and the Modal sharded evaluator).

**For lipids:** use this sampler as-is (the graft/size fixes transfer directly);
just re-check the −0.5 size knob at ~48 atoms.

---

## 8. Conditioning / optimization (Arm A/B) — how you *steer* it

The unconditional model above generates "typical" molecules. To optimize toward a
property (QED on drug-like; pan-lung delivery on lipids) there are three levers
(`paper1_compose_methods.html` §04), all reusing the same valid generator:

1. **Property conditioning (classifier-free):** feed the target value as an input,
   train it in on ~85% of batches / drop it on ~15% so one model does both
   conditional and unconditional. File:
   `experiments/molecular_property_conditioning.py`.
2. **Frozen-residual sidecar** (`FrozenQEDResidualAdapter`, built in
   `scripts/run_griddd_analytic_zero_sidecar_smoke.py`): keep the base generator
   **frozen**; add a small adapter that outputs a **family-hazard residual** and a
   **within-family residual** which nudge the rates *only when a target is
   present* (missing-target → exact base, "missing-condition identity"). This is
   what the v5 QED run trained. Composition: load the frozen base + `load_state_dict`
   the sidecar weights.
3. **Valid-successor guidance / reward-FT:** at each state, draw a few legal
   candidate edits, execute them, score the resulting molecules with the frozen
   **oracle**, and tilt/resample toward better ones — or fine-tune the CTMC toward
   the oracle under a KL anchor. Files: `experiments/guided_rewrite_sampling.py`,
   `experiments/griddd_conditional.py` (the three-arm **direct / controller /
   combined** executor + **exact oracle-call accounting** — every oracle call is
   counted, with padding, so budgets are matched).

**Arm A = base generator + oracle ranking; Arm B = oracle-guided generation.**
For lipids you swap the QED oracle for the pan-lung oracle; the machinery is
identical. **Reality check from this session:** trained conditioning is currently
*weak* — the QED sidecar did not improve molecule-level QED in the real-lead smoke
(it degraded it). So the conditional lever needs strengthening (reward-FT /
stronger controller), not just a benchmark launch. Expect the same caution on
lipids.

---

## 9. The Modal pipeline + recipe format (how a run actually executes)

Everything runs on Modal (cloud). A **recipe** is a JSON in `recipes/` with a
`name`, `purpose`, and an `arguments` dict (every hyperparameter: sizes, families,
ring modes, learning rate, scope, data paths…). The launcher
`modal_apps/train_tracelet_gm.py` reads a recipe and runs one **stage**:

1. **Compile paths** (`--compile-only`): build certified teacher programs from the
   corpus → `compiled_paths.pt.shards` (this is the expensive CPU step).
2. **Compile support** (`--support-compile-only`): project the paths into the
   per-training-step support tensors → content-addressed `_shared/training_support`.
3. **Train** (`--train-only`): GNN training on GPU; warm-start via
   `--initialize-compatible-from-source-checkpoint`; writes `checkpoint*.pt` +
   `metrics.json` (watch `selected_step` and the per-family accuracies).
4. **Evaluate** (`modal_apps/evaluate_rollout_shards.py`): sample N molecules in
   parallel shards, compute the metric panel (reuse
   `scripts/analyze_unconditional_sufficiency.py`, swapping the lipid reference).

Caches are keyed by content, so a config that keeps the path/electronic settings
reuses existing paths/support (no recompile). **Three launch gotchas that cost us
time:** always `modal run --detach` (else spawned tasks are killed); `--recipe-name`
needs the `.json`; new `--trainable-parameter-scope` values must be added to the
argparse `choices` in `scripts/train_tracelet_cnof_gate.py`, not only the runtime.

---

## 10. Where each lipid change goes (the punch list)

| Change | File(s) / knob | Owner |
|---|---|---|
| Size cap 40→~48+, atom types, protonation/charge | `chem/*`, `chem/source_prior.py`, `rewrite/compiler.py`, `rewrite/tree_transport.py` (P2-G2 kernel) | **shared core → Codex** |
| Restrict ring family to single 5/6 head rings | recipe: down-weight ring-grow, `ring_template_factorization` off, restricted `typed_ring_catalog` | lipid lane (config) |
| Lipid corpus | recipe `arguments` data path; the empirical prior re-scans lipid teachers | lipid lane (their corpus) |
| Base config (proven) | `empirical_mark_prior_mode=corpus_residual_v1`, `ring_electronic_mode=factorized_local`, calibrated sampler on, atom-delete −0.5 | lipid lane (config) |
| Optimization (Arm A/B) | `molecular_property_conditioning` / `guided_rewrite_sampling` / `griddd_conditional`, **swap QED→pan-lung oracle**; linker preservation as a hard legal-fiber condition; recovery curriculum before round-2 | lipid lane |
| Eval panel | `scripts/analyze_unconditional_sufficiency.py` + lipid marginals (head/linker/tail, protonation, reaction-family transfer) | lipid lane |

---

## 11. Suggested reading order (maps your 6-item plan to files)

1. **Sampler first** (small, central): `experiments/calibrated_rewrite_sampling.py`
   — you'll immediately understand rates, thinning, the graft fix.
2. **Kernel/executor:** `rewrite/kernel.py` → `rewrite/tracelet_fiber.py` /
   `factorized_fiber.py` → `rewrite/typed_ring_catalog.py` /
   `ring_system_fiber.py`. This is where the lipid kernel bump + restricted rings go.
3. **Rate model:** `model/factorized_tracelet_rate_model.py` — the heads in §5;
   skim, don't memorize.
4. **Objective/quotient:** `gm/loss.py`,
   `experiments/canonical_successor_distillation.py`.
5. **Conditioning trio:** `experiments/molecular_property_conditioning.py`,
   `guided_rewrite_sampling.py`, `griddd_conditional.py` (+ the sidecar in
   `scripts/run_griddd_analytic_zero_sidecar_smoke.py`).
6. **Pipeline/recipe:** `modal_apps/train_tracelet_gm.py`,
   `modal_apps/evaluate_rollout_shards.py`, any `recipes/*.json`
   (e.g. `tree_fcd_transfer_unconditional_chemistry_topology.json`).

Read those and you can responsibly modify the kernel, the ring restriction, and
the oracle swap. Everything else is detail on top of the model in §1.
