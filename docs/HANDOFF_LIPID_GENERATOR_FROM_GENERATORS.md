# Generator → Lipid handoff (comprehensive, lossless)

**From:** Claude generator lane (`claude/generator-cond-uncond`)
**To:** Claude lipid/oracle lane (`claude/lipid-corpus-oracle`)
**Date:** 2026-07-20 · **Rev 2** (base decision now evidenced)

> **Rev-2 changelog (read §0.5):** the recommended architectural base moved from
> the pre-quotient **pancake** checkpoint to **Lineage B** (quotient-correct
> flexible-Graft), now backed by a matched controller measurement (B **33.3%** vs
> pancake **16.7%** constrained-optimization success on an identical panel), plus a
> precise element-set analysis (§7) and the in-flight-improvement reassessment
> (§13). All Rev-1 content below is preserved and still valid.

Everything the lipid workstream needs to port the COMPOSE generator to lipids —
the code/config/weights, **and** how the ported generator maps onto the Paper 2
goals, gates, and claim boundaries in the governing HTML plans
(`docs/research_plans/paper2_compose_lipid.html`,
`compose_two_paper_execution_plan.html`, `paper1_compose_methods.html`). Read the
HTML plans as canonical; this doc is the generator-side index into them.

> **New to the generator? Read `docs/GENERATOR_INTERNALS_FROM_ZERO.md` first.**
> It assumes zero prior knowledge and explains, from first principles: what the
> model is (a learned CTMC that edits molecules via legal chemical rewrites), the
> state space, operators, the rate-model heads, the training objective, the
> sampler, the conditioning trio, the Modal pipeline + recipe format, a glossary,
> and a suggested reading order that maps each file to what it does and where the
> lipid changes go. This handoff assumes that mental model.

---

## 0. How the code gets there — git, not copy

All three checkouts are **git worktrees of one repo** (shared `.git` object
store): Codex core, generators (`claude/generator-cond-uncond`), lipid
(`claude/lipid-corpus-oracle`). So the lipid worktree already sees the generator
branch and merges it:

```bash
# in compose_rgm_claude_lipid, on claude/lipid-corpus-oracle
git merge claude/generator-cond-uncond      # or rebase through main
```

Then modify freely on your own branch. Branch is on origin for review/backup.

---

## 0.5. Base decision (EVIDENCED): Lineage B, not pancake

**What changed and why.** An audit of the unconditional run history found the
generator lane had three evidence-bearing lineages, not one, and that the earlier
pancake recommendation was the **weakest** of the three:

- **Lineage A = "pancake"** (pre-quotient, SHA `47716924…ae2bf`): thrashes — the
  sampler spends the majority of events on Graft/reroute and canonical
  self-transitions, dumps rings late (≈99% of ring events in the last decile,
  time position 0.978), and shows long delete runs. Drug-like C/N/O/F weights.
- **Lineage B = quotient-correct flexible-Graft + whole-ring-system** (Modal run
  `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`): **clean edit dynamics** —
  zero canonical self-events, no delete-to-one collapse. This is the
  distinguishing property for lipids.
- **Lineage C = factorized-tree** (`tree_fcd_transfer_stage1_factorized_v1`): good
  topology calibration but an initial delete-collapse run and off-thesis primitive
  ring substrate — worse for an editing task.

**Measured evidence (not a guess).** Both A and B were run through the *same*
hard-constrained valid-fiber controller (a ceiling estimator for constrained QED
optimization) on an identical 12-lead panel, identical settings, only the
checkpoint differing:

| metric (QED ≥0.90 @ Tanimoto ≥0.40) | pancake (A) | **Lineage B** |
|---|---|---|
| success rate | 2/12 = **16.7%** | 4/12 = **33.3%** |
| beat-lead-and-feasible | 9/12 | **12/12** |
| mean best-feasible QED | 0.824 | **0.881** |
| proposal productivity (distinct scored, mean / min) | 355.6 / **2** | 389.1 / **282** |
| leads collapsed to no-improvement (sim=1.0) | **3 of 12** | **0** |

B nearly doubled the success rate and **erased the no-op collapse** (pancake had
three leads where the search never found a single improving molecule in 640
oracle calls — one with only 2 distinct feasible states; B never stalls). B is
**not** uniformly better per-lead (pancake wins 2 leads); the win is
distributional + productivity, exactly what "zero self-events" predicts. Artifacts:
`diagnostics/griddd_valid_fiber_controller_panel12_lineageB_ceiling.json` vs
`…_ceiling.json` (pancake).

**Provenance (pin these exactly; never substitute a same-step different-hash file):**
- Volume/path: `compose-v4-artifacts` : `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`
- SHA-256: `c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c` (verified, 38,387,536 bytes)
- Local pull used here: `/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt`
- B's exact config: `tree_transport=flexible_size_graft`, `source_prior=carbon_tree`
  (`DegreeBoundedCarbonTreePrior` sizes 4..40, max_degree 4, `tree_size_prior=empirical`),
  `teacher_ordering=sequential`, `training_backend=factorized_marks`,
  `ring_proposals=typed_catalog`, `ring_electronic_mode=factorized_local`,
  `bond_representation=aromatic`, `hidden_dim=256`, `message_passing_steps=6`,
  `operational_horizon=16.0`, `training_steps=3000`, `seed=20260717`.
  Rate knobs are the **pre-fix defaults**: `rate_factorization=hierarchical`,
  `ring_template_factorization=flat`, `ring_family_mass_mode=boolean`,
  `empirical_mark_prior_mode=none` (so B carries the drug-like ring defect the
  §13 fix targets — irrelevant to you, see §13).

**CRITICAL — "base" means architecture + dynamics + optional warm-start, NOT
drop-in weights.** B's weights are **drug-like C/N/O/F**; do **not** ship them as
the lipid generator. Adopt B as: (a) the architecture/config/sampler template
(flexible-Graft quotient substrate + clean editing), and (b) an *optional*
compatible-init encoder / general-mark warm-start source **iff** the lipid kernel
shapes overlap. The lipid **model is still trained on the lipid corpus** (§6,
"general first, linker fine-tune second"). This supersedes the Rev-1 §6/§13
pointer at the pancake/combined GuacaMol checkpoints as the warm-start source.

**Conditional re-qualification status (§10-style gates, for when you do Arm A/B).**
Of the five gates that qualify a new base for guided optimization: the **frozen
fixed-lead panel, identical oracle accounting, and matched-vs-pancake comparison
are already satisfied** by the run above; **canonical-successor execution** and
**zero-sidecar equivalence** are one bounded script each and are in progress on
the generator lane. Note the pancake base qualified only *with* a calibration
band-aid (`atom_delete −0.5`, `small_ring −1.5`); B is being checked for
qualification **without** any calibration (its native design removes the
self-transitions/collapse that band-aid compensated for). Treat B as **converging
to a re-qualified base**, not yet frozen — but already the correct target.

---

## 1. What the generator is FOR in Paper 2 (the goal)

Paper 2 thesis (paper2 HTML): COMPOSE-Lipid adapts Rewrite Generator Matching to
ionizable-lipid chemistry, **generates complete molecules through
chemistry-guarded stochastic rewrites rather than ranking a fixed library**, and
prospectively tests whether generated candidates improve the **in-vivo
lung-delivery hit rate under a matched experimental budget**.

**Central novelty claim (do not weaken, do not overclaim):** *to our knowledge,
the first learned de-novo ionizable-lipid generative model whose model-sampled
candidates — rather than candidates selected from a pre-enumerated virtual
library — are prospectively synthesized, formulated, and validated in vivo.*
Comparators are positioned precisely: LUMI-lab and MOLEA are AI+in-vivo but
predictive/active-learning/optimization, **not learned generators**; LiGen and
synthesis-aware lipid models are learned generators **without prospective
in-vivo validation**; GEM 2026 is a self-cited related work with **no reused
data/preprocessing/code/model/checkpoints**. The "first" is locked only after a
dated, model-by-model prior-art audit at submission and keeps "to our knowledge".

The generator (what this handoff ports) is the **engine** behind Figures 1–3 and
6 and Arms A/B. It does not, by itself, establish any biological claim.

### The four-claim boundary (keep these separate — paper2 Fig 3)

A generated molecule must never let one claim leak into another:
1. **2D graph validity** (guaranteed by construction — what we own),
2. **ionizable-lipid phenotype** (head/linker/tail architecture, ionizable amine),
3. **synthesis feasibility** (reaction class, building blocks, route — a rewrite
   path is *not* a synthetic route),
4. **biological plausibility / delivery** (the prospective assay adjudicates,
   not the oracle score).

### Two generation strategies (paper2 §02, exec-plan L3)

- **Arm A (required workhorse):** base RGM samples target-free ancestral
  trajectories → ranked by the frozen multi-study lung oracle (+ AD/uncertainty,
  synthesis filters, diversity-aware Pareto).
- **Arm B (gated upside):** same base checkpoint, lung-guided validity-preserving
  RGM — tilt only legal successor rates, or KL/trajectory-anchored reward-FT.
  Guidance changes probabilities, never enables an illegal rewrite. Dropped if it
  exploits the oracle / collapses diversity / leaves the applicability domain.
- Plus matched non-AI controls (random/DoE, expert/rational). **Primary contrast
  is AI vs non-AI hit-rate enrichment; A-vs-B is prespecified secondary.**

---

## 2. Which Paper-2 gates/contracts the ported generator must satisfy

The generator is directly responsible for these (thresholds from the HTML plans):

| Gate / contract | What the generator must show |
|---|---|
| **P2-G2 · lipid kernel** | ≥99% exact source→target program support on the declared eligible corpus; **zero committed validity/connectivity failures**; practical throughput at lipid sizes. The C/N/O/F base is **not** assumed lipid-ready. |
| **P2-G3 · frozen base generator** | ≥10,000 independent ancestral samples; 100% committed-state validity/connectivity; ≥99% uniqueness; ≥90% exact novelty **plus scaffold/linker novelty**; no delete-to-one/topology pathology; credible architecture/property coverage; adequate yield after ionizable-lipid + synthesis filters. |
| **P2-G5 · Arm B readiness** | reward improvement across seeds and oracle members, **unchanged hard support**, matched compute, no diversity collapse / extreme chemistry / uncertainty inflation / AD escape. |
| **L1 · generator contract** | 10k samples/seed × ≥3 seeds; compare LiGen, synthesis-DAG/MCTS, matched enumeration where scope permits; report validity/connectivity, uniqueness, exact/scaffold/**linker** novelty, NN-memorization curves, **head/linker/tail architecture, tail count/length/branching, degradable motifs, charge/protonation/size marginals**, reaction-family + **linker-held-out** transfer, synthesis-filter yield, blinded route review, throughput. |
| **L3 · selection** | same base checkpoint/oracle/AD/budget/diversity-selector/candidate-count across arms; Arm B additionally freezes reward/KL/updates/seeds/uncertainty penalty; resolve cross-arm duplicates before testing. |
| **L5 · round-2** | linker-frozen source-conditional RGM optimization of a reproducible round-1 hit (see §8 recovery curriculum). |

The generator does **not** own P2-G1 (data), P2-G4 (oracle), P2-G6 (candidate
lock), or P2-G7 (wet-lab). Those are the lipid/oracle lane + PI.

---

## 3. Pathology ledger — what's solved vs open (this session, GuacaMol)

| Pathology | State | Relevance to lipids |
|---|---|---|
| Graft/self-transition thrashing | **Solved** — self-Graft virtualization + immediate-backtrack safety (0.14% backtracking) | Transfers directly (sampler/config technique, not weights) |
| Validity / connectivity / self-events / collapse / stalls | **Solved** (100% valid, 0 pathological events) | Same code path — directly serves P2-G3's zero-failure bar |
| Chemistry marks (triples 54%→19% etc.) | **Technique validated** — `corpus_residual_v1` base + learned residual | Re-fit the base on the *lipid* corpus (~14 s scan); lipids barely have triples |
| Size / over-deletion | **Solved via one small knob** — atom-delete log-rate −0.5 (`exp(-0.5)≈0.61`) → mean atoms 25.1→27.1 vs ref 26.35 | Re-check the mean at ~48 atoms; worst case re-tune this one number |
| Rings (fused / cyclization / small-ring) | Improved, **not** solved (fused 9→21% vs ref 58%; under-cyclization open) | **Moot** — lipids ring-simple (~0.6 rings/mol, single 5/6 head rings, no fused, no 3/4). Restrict/down-weight the ring family. |

**Net: nothing non-ring blocks a lipid smoke.** The ring saga stays in Paper-1
land; it is not on the lipid critical path (exec-plan lipid-transition gate).

---

## 4. Code to use (all on the merged branch)

Base generator + sampler:
- Shared RGM core (Codex, frozen): `rewrite/*`, `model/factorized_tracelet_rate_model.py`, ancestral sampler.
- `experiments/calibrated_rewrite_sampling.py` — **graft-thrashing fix** (self-Graft virtualization + backtrack safety). Use this sampler path.
- `experiments/factorized_mark_conditional.py` — `corpus_residual_v1` empirical mark prior + `configure_factorized_trainable_parameters` scopes (`chemistry_marks_only`, `ring_topology_only`, `chemistry_and_topology`, `all`).
- `scripts/analyze_unconditional_sufficiency.py` — reuse for the P2-G3/L1 panel; swap the reference set for the lipid held-out corpus and add the lipid-specific marginals (head/linker/tail, tail count/length, degradable motifs, protonation).

Conditional / optimization (Arm A/B, Fig 4/6):
- `experiments/molecular_property_conditioning.py` — property conditioning (classifier-free) → Arm A/direct targeting.
- `experiments/guided_rewrite_sampling.py` — valid-successor oracle tilting → Arm A ranking + Arm B guidance.
- `experiments/griddd_conditional.py` — the three-arm (direct/controller/combined) execution + **exact oracle-call accounting** (reuse the accounting; swap QED→pan-lung oracle).
- `experiments/canonical_successor_distillation.py` — the principled graft/rate repair (if a cleaner trained base is wanted later).

Pipeline (debugged this session):
- `modal_apps/train_tracelet_gm.py`, `modal_apps/evaluate_rollout_shards.py`.

---

## 5. Config (the validated "best formulation")

- Graft fix: **on** (calibrated sampler virtualization + backtrack wrapper).
- `empirical_mark_prior_mode=corpus_residual_v1` (re-scanned on lipid teachers).
- `ring_electronic_mode=factorized_local`.
- **Restrict the ring family** to observed lipid head-rings (single 5/6); **skip `ring_template_factorization=topology_cycle_hierarchical`** — drug-like fusion machinery lipids don't need; down-weight/disable ring-grow per the lipid distribution.
- Keep atom-delete −0.5 as a start; verify mean size at lipid scale.

---

## 6. Weights — do NOT reuse GuacaMol weights as the lipid model

GuacaMol checkpoints (pancake `47716924…`, combined
`compose-v4-uncond-chem-topology-train-20260720-v1`) are **C/N/O/F drug-like
specific** (triple-heavy, ring-heavy). **Train the lipid generator on the lipid
corpus.** Optional accelerator: compatible-init warm-start of encoder + general
mark heads from the combined checkpoint, *only if* the lipid kernel overlaps in
shape. The corpus determines the model; per the memory, **train general first,
linker fine-tune second** — the base corpus stays broad/reaction-diverse; the
Michael/linker specialization is a downstream fine-tune, not the base.

---

## 7. Lipid chemistry the kernel must support (P2-G2)

Audit and declare before training (paper2 P2-G2, lipid-design memory):
- **Atom types / charges — the real gating question, scoped precisely:** the
  atom-type *vocabulary* already includes `B, C, N, O, F, P, S, Cl, Br, I` (plus
  `null`/`SCAR`), so the **state representation is NOT the blocker**. What is
  CNOF-restricted is the trained **rewrite fiber** (`enumerate_tracelet_cnof_fiber`),
  the ring/attachment **catalog**, the **corpus**, and the training **gate**
  (`train_tracelet_cnof_gate.py`). Consequence:
    - **Ionizable amino-lipids are mostly C/H/O/N** (esters, amides, ethers,
      amines) → **in scope of the existing CNOF fiber**. Good news: no extension
      needed for the common case.
    - **P (phosphate heads / phospholipids) and S (thioesters, some ionizable
      lipids) are OUT of the trained fiber.** Extending is a scoped
      "add fiber actions + templates for P/S + a P/S-containing corpus + retrain"
      job — **not** a representation overhaul (the vocab already has them). It is
      shared-RGM-core, so **coordinate with Codex**; it pairs with the ~48-atom
      kernel bump. **Action: histogram your corpus's elements FIRST**; if a
      material fraction needs P/S, scope the fiber extension before training.
  Also declare: **neutral and protonated forms** of the ionizable amine;
  formal-charge policy; stereochemistry policy.
- **Size:** heavy-atom range to ~**48+** (AGILE mean 47.77); long/repeated tails;
  practical throughput at this size is a real P2-G2 requirement.
- **Architecture (Fig 2 marginals):** ionizable **head**, **linker(s)** (incl.
  **degradable** ester/carbonate motifs), multiple **tails** (count / length /
  branching / degree of unsaturation). These are the reportable L1 marginals.
- **pKa:** ionizable amine target intrinsic pKa ~9 / apparent ~6.4 — a property
  target for Arm A/B, computed with a **validated pKa predictor**, not a
  hand-rolled heuristic (intrinsic ≠ apparent).
- **Reaction-family diversity:** the corpus must span **≥6 reaction families**
  (user requirement) — never framed as picking one. The kernel/support must admit
  all of them; L1 reports reaction-family + linker-held-out transfer.
- **Rings:** restricted single 5/6 head rings (see §5).

---

## 8. Optimization levers — Paper-1 engine → Arm A/B mapping

The RGM engine (paper1 §04 "Conditional") gives three steer levers on one valid
generator; they map straight onto the lipid arms:
1. **Property conditioning (classifier-free):** feed the target (e.g. pKa /
   predicted delivery) as input, drop it on ~15% of batches. → direct Arm-A-style
   targeting; the primary matched comparison.
2. **Modular inference-time guidance:** tilt only legal successor rates with the
   frozen **pan-lung oracle** or a learned value. → Arm A ranking / Arm B guidance.
3. **Reward fine-tuning:** KL/trajectory-anchored update toward the frozen oracle;
   hard chemistry support unchanged. → Arm B. Watch oracle exploitation, diversity
   collapse, AD escape (P2-G5).

**Hard application conditions (the RGM advantage, paper2 §05):** linker/scaffold
preservation is enforced by **removing violating rewrites from the legal fiber**,
not a soft penalty. For **round-2 linker-frozen optimization (Fig 6 / L5)**: start
from a measured lead, **freeze the novel linker in the legal fiber**, and learn/
guide short valid recovery trajectories around it. This requires the **recovery /
revision curriculum** (paper1 §04 "Transfer"): apply k≈1–4 valid perturbations to
a data molecule and learn the inverse rewrite distribution (shrink, Graft
reversal, ring-system delete, scaffold-preserving correction). The forward
carbon-tree→data teacher alone does **not** establish learned local editing — the
recovery objective is a hard prerequisite before Fig-6 candidates are locked.

---

## 9. Dependencies & sequencing

- **Corpus (P2-G1 / L0):** lipid/oracle lane owns it; it is the real prerequisite
  — separate virtual structural pretraining, measured lipids, the deduplicated
  pan-lung response corpus, prospective internal data; provenance/licensing,
  cross-source dedup, reaction/study IDs, **leave-study/scaffold/linker/
  reaction-family/temporal splits**, endpoint counts. LNPDB is the cross-study
  anchor, **not** the whole oracle corpus; non-lung measurements → off-target/
  auxiliary heads, **never** target reward.
- **Kernel extension (P2-G2):** shared RGM core → **coordinate with Codex**; do
  not fork. Adapt via config + lipid-specific modules/subclasses.
- **Oracle (P2-G4):** lipid/oracle lane; frozen study-aware pan-lung oracle is
  the reward source for Arm A/B.
- **Sequencing (exec-plan dependency contract):** lipid *training* is gated
  behind the Paper-1 **credible unconditional gate + one matched QED
  optimization proof** (P1-G7). But **corpus/kernel prep and an exploratory
  restricted-ring lipid smoke are authorized now** — no need to wait for Paper-1
  ring perfection (explicitly stated in the exec-plan lipid-transition gate).

---

## 10. Claim discipline (do not overclaim — shared artifact S1/S12)

- Keep the **four-claim boundary** (§1). Graph validity ≠ synthesizable ≠
  ionizable-lipid ≠ delivered.
- Central novelty stays **"to our knowledge"** and is locked only after the dated
  prior-art matrix (P2-G0). LUMI-lab/MOLEA are AI+in-vivo but not learned
  generators; LiGen/synthesis-aware are generators without prospective in-vivo.
- **GEM 2026 independence (S12):** cite for lineage; reuse **no** data,
  preprocessing, code, model components, or checkpoints. COMPOSE-Lipid is an
  independent RGM build on its own corpus.
- The **assay, not the oracle score, adjudicates success.** Report all attempted
  syntheses/candidates in denominators (all-attempt), not just successes.

---

## 11. Launch gotchas & reproducibility (learned this session)

- **`modal run --detach`** always for spawned stages — non-detached kills the
  spawned tasks (zero artifacts).
- **`--recipe-name` needs the `.json` extension.**
- New `--trainable-parameter-scope` values must be added to the argparse
  `choices` in `scripts/train_tracelet_cnof_gate.py`, not only the runtime
  selector in `factorized_mark_conditional.py`.
- Rollout evals shard on Modal; if the merge stalls on straggler shards, download
  the shard `.pt` files and merge locally (`rollout.final_state` →
  `molecular_graph_to_smiles`). Reuse `analyze_unconditional_sufficiency.py`.
- Register every result in the shared result registry (S8): checkpoint/config/
  data/cache hashes, all-attempt metrics, CIs, compute.

---

## 12. One-line summary to paste into the lipid chat

> **Generator → lipid handoff (Rev 2, base=B).** Merge, don't copy:
> `git merge claude/generator-cond-uncond`. Read `docs/GENERATOR_INTERNALS_FROM_ZERO.md`
> then this doc (§0.5 base decision, §7 elements, §13 in-flight fixes). **Base =
> Lineage B, not pancake** — quotient-correct flexible-Graft, clean dynamics;
> evidenced 33% vs 17% on a matched constrained-QED controller. Pin
> `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`
> (SHA c9d9275103…) by run-ID+SHA, but treat "base" as architecture/config/sampler
> + optional encoder warm-start — **train the lipid model on the lipid corpus**
> (general first, linker fine-tune second; B's weights are drug-like C/N/O/F).
> Start now (base-agnostic + mergeable): pan-lung oracle swap (reuse
> `griddd_conditional` accounting), corpus, optimization loop; enforce linker
> preservation as a **hard legal-fiber condition**; build the **recovery
> curriculum** before round-2 (Fig 6). Don't wait for our ring fix (off your
> critical path); `git merge` again later to inherit it. **Real gate = element
> set:** vocab has P/S but the trained fiber (`enumerate_tracelet_cnof_fiber`) is
> CNOF-only — amino-lipids fit; phospholipid P / thioester S need a
> fiber+corpus+retrain extension (shared RGM core → **coordinate with Codex**,
> pairs with the ~48-atom kernel bump). **Histogram corpus elements first.**
> Restrict/down-weight the ring family (lipids ring-simple). Targets
> P2-G2/G3/G5, L1/L3/L5. Keep the four-claim boundary + "to our knowledge".
> `modal run --detach`, `.json` recipes.

---

## 13. Reassessment: what the generator lane is improving, and when to re-merge

**Start now; do not wait for any of the below.** Your long-pole — pan-lung oracle
swap, lipid corpus + element audit (§7), optimization-loop scaffolding (§8) — is
**base-agnostic and mergeable**. Because the handoff is **git-merge, not copy**,
you `git merge claude/generator-cond-uncond` again later to inherit every
improvement below for free. Nothing here is on your critical path.

**Three improvements in flight on `claude/generator-cond-uncond`:**

1. **Ring-hazard fix (§9.3) — OFF your critical path.** Repairs de-novo drug-like
   small-ring/bridged over-production via `rate_factorization="superposed"` +
   `ring_template_factorization="topology_cycle_hierarchical"` (the topology-group
   intensities compete unnormalized, so losing common-ring support *lowers* ring
   hazard instead of renormalizing onto rare small rings). The **mechanism is
   proven** by a zero-GPU falsifier unit test
   (`tests/test_topology_group_intensity_ring_hazard.py`): superposed drops ring
   mass on support loss, hierarchical (the defect) does not. It then warm-starts
   Lineage B into that combo for a bounded convergence (audit-scoped: ~250–500
   updates, 200 rollouts, 5–10 min A100). **Why it's irrelevant to you:** lipid
   optimization is an *editing* task on ring-simple molecules, so clean edit
   dynamics matter, not de-novo ring-building marginals. You should still
   **restrict/down-weight the ring family** per §5 regardless.

2. **Lineage-B re-qualification as a guided-optimization base (§0.5).** Three of
   the five qualification gates are already satisfied by the matched controller
   run; the remaining two (canonical-successor execution, zero-sidecar
   equivalence) are one bounded script each, in progress — plus the check that B
   qualifies **without** the pancake calibration band-aid. When this lands, B is a
   frozen, re-qualified base you can warm-start from with confidence.

3. **Calibrated sampler (already landed, use it now):** `calibrated_rewrite_sampling`
   — self-Graft virtualization + immediate-backtrack safety + the `atom-delete
   −0.5` size knob. This is a sampler/config technique (not weights), so it
   transfers directly (§4). It is what gives the clean dynamics measured in §0.5.

**Re-merge signal.** Re-run `git merge claude/generator-cond-uncond` when the
generator lane registers, in the shared result registry (S8, §11): (a) the
re-qualified Lineage-B checkpoint (run-ID + SHA), and (b) the ring-fixed
unconditional checkpoint with its sufficiency panel. Until then, build on B's
architecture/config/sampler (§0.5) and **pin B by run-ID + SHA** (never a
same-step different-hash file).

**Do not hard-wire:** pancake checkpoint paths; `hierarchical`/`flat`/`boolean`
rate-mode assumptions; or CNOF-only support if your element audit (§7) shows P/S.
