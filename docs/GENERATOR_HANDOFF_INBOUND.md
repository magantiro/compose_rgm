# INBOUND handoff: COMPOSE generator → lipid generator port

**Received:** 2026-07-20 (pasted by the user into the corpus/oracle lane session).
**From:** the generator session (`compose_rgm_claude_generators`, branch `claude/generator-cond-uncond`).
**To:** this lane (`claude/lipid-corpus-oracle`) — I now own the generator port.
**Status:** SPEC ONLY — do not act until the training corpus is committed and the user gives the go. This is the counterpart to the outbound `docs/GENERATOR_HANDOFF_FROM_CORPUS.md`.

**Authoritative source:** full spec at `docs/HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md` on branch `claude/generator-cond-uncond` (pushed to origin). Also read the three HTML plans: `docs/research_plans/paper2_compose_lipid.html`, `compose_two_paper_execution_plan.html`, `paper1_compose_methods.html`. Summary below (verbatim substance).

---

## 1. Get the code (no copy — shared git worktrees)
In this `compose_rgm_claude_lipid` worktree on `claude/lipid-corpus-oracle`:
`git merge claude/generator-cond-uncond` (or rebase through main). You then have the full generator + conditional/optimization code in your tree; modify on your own branch.

## 2. Model spec / base formulation (validated on GuacaMol this session, no bandages)
- **Sampler:** `experiments/calibrated_rewrite_sampling.py` — graft/self-transition thrashing is solved here (self-Graft virtualization + immediate-backtrack safety, 0.14% backtracking). Use this path.
- **Chemistry:** `empirical_mark_prior_mode=corpus_residual_v1` (corpus-base + learned residual) — re-scan it on the lipid teacher actions (~14 s). `ring_electronic_mode=factorized_local`.
- **Size:** keep atom-delete log-rate **-0.5** (fully fixes size on GuacaMol: 25.1→27.1 vs ref 26.35); just re-check the mean at ~48 atoms and re-tune that one number if needed.
- **Rings:** restrict/down-weight the ring family to observed lipid head-rings (single 5/6-membered). **Skip** `ring_template_factorization=topology_cycle_hierarchical` — that's drug-like fusion machinery; lipids are ring-simple (~0.6 rings/mol, no fused, no 3/4). The unconditional ring pathology is a Paper-1 problem and is **NOT** on the critical path.
- **Scopes** available in `configure_factorized_trainable_parameters`: `all`, `chemistry_marks_only`, `ring_topology_only`, `chemistry_and_topology`.

## 3. Weights — do NOT reuse GuacaMol weights as the lipid model
They're C/N/O/F drug-like specific. **Train fresh on the lipid corpus.** Optional: compatible-init warm-start of encoder + general mark heads from `compose-v4-uncond-chem-topology-train-20260720-v1` **only if** the lipid kernel overlaps in shape. Per program policy: general/reaction-diverse corpus first, linker fine-tune second.

## 4. How to train (pipeline = 4 Modal stages; mind the 3 gotchas)
**Gotchas:** always `modal run --detach`; `--recipe-name` needs the `.json` extension; any new `--trainable-parameter-scope` must be added to the argparse choices in `scripts/train_tracelet_cnof_gate.py`.
Stages (make a lipid recipe with your corpus path + kernel config + restricted rings):
- **(a) paths:** `modal run --detach modal_apps/train_tracelet_gm.py --compile-only --recipe-name <lipid>.json --run-label <paths>` (compile lipid teacher paths from your corpus).
- **(b) support:** `... --support-compile-only --recipe-name <lipid>.json --run-label <support> --source-run-label <paths> --support-steps N --support-containers 2 --support-workers 12` → writes to `_shared/training_support`.
- **(c) train:** `... --train-only --recipe-name <lipid>.json --run-label <train> --source-run-label <paths> [--initialize-compatible-from-source-checkpoint --initialization-source-run-label <warmstart> --checkpoint-name checkpoint.recovery.pt] --training-steps N --schedule-steps N`.
- **(d) eval:** `modal run modal_apps/evaluate_rollout_shards.py --run-label <eval> --source-run-label <train> --checkpoint-name checkpoint.pt --samples 10000 --shards 100`. If the merge stalls on straggler shards, download the shard `.pt`s and merge locally (`rollout.final_state → molecular_graph_to_smiles`). Panel: `scripts/analyze_unconditional_sufficiency.py` (swap in the lipid reference set + add head/linker/tail, tail count/length, degradable-motif, protonation marginals).

## 5. Optimization (Arm A / Arm B, Figs 4/6)
Reuse `molecular_property_conditioning.py` (classifier-free targeting = Arm A/direct), `guided_rewrite_sampling.py` (valid-successor tilting = Arm A ranking / Arm B guidance), `griddd_conditional.py` (three-arm exec + exact oracle-call accounting). **Only change:** swap the QED oracle for the pan-lung oracle. Enforce **linker/scaffold preservation as a HARD legal-fiber condition** (remove violating rewrites), not a soft reward. Round-2 linker-frozen optimization (Fig 6/L5) requires the **recovery curriculum** (k≈1–4 valid perturbations → learn the inverse rewrite dist: shrink, Graft-reversal, ring-delete, scaffold-preserving) — the forward teacher alone does not establish learned local editing; build it before locking Fig-6 candidates.

## 6. What the generator must satisfy (paper gates)
- **P2-G2:** kernel ≥99% exact support, zero committed validity/connectivity failures, lipid-size throughput.
- **P2-G3:** 10k samples, 100% committed validity, ≥99% uniqueness, ≥90% exact + scaffold + linker novelty, no delete-to-one/topology pathology.
- **P2-G5:** Arm B reward gain across seeds/oracle members, unchanged hard support, no diversity collapse / AD escape.
- **Contracts:** L1 (report architecture + reaction-family + linker-held-out marginals), L3 (matched arms), L5 (round-2).

## 7. Lipid chemistry the kernel needs (P2-G2)
C/N/O(/F)(+P if in scope); neutral + protonated ionizable amine; ~48+ heavy atoms; head/linker/tail incl. degradable linkers; multi-tail (count/length/branching/unsaturation); pKa target ~9 intrinsic / ~6.4 apparent computed with a **validated** pKa predictor (not hand-rolled); corpus spanning ≥6 reaction families; restricted 5/6 head rings.

## 8. Dependencies + coordination
The **corpus (this lane) is the real prerequisite**; the ~48-atom kernel bump is **shared RGM core → coordinate with Codex, don't fork** (adapt via config + lipid modules). Lipid training is policy-gated behind the Paper-1 QED optimization proof, but corpus/kernel prep + an exploratory restricted-ring smoke are authorized now.

## 9. Don't overclaim
Keep the four-claim boundary (graph-valid ≠ ionizable-lipid ≠ synthesizable ≠ delivered); novelty stays "to our knowledge" pending the prior-art matrix; GEM 2026 = citation only, zero reused artifacts; the assay adjudicates success, not the oracle score; all-attempt denominators.

---

## My reconciliation notes (corpus lane — how this maps to what's already built)
- **§8 corpus prerequisite** ↔ the training-ready corpus I'm materializing now: `r1_reaction_grounded_corpus_v1.csv` + `training_corpus_manifest_v1.json` (R0 anchor + R1 reachable support, realism weights, kernel profile). This is exactly the "real prerequisite."
- **§7 kernel needs** ↔ my `kernel_readiness_profile` already reports C/N/O/S/P, ~48+ heavy atoms, charge/stereo — feeds the lipid recipe's kernel config. Note: my corpus has **P** (iPhos) and **S**; the spec says "C/N/O(/F)(+P if in scope)" — confirm S/P scope with Codex for the shared kernel bump.
- **§2 restricted rings** ↔ corpus is already ring-simple (audit `rings` JS matched R0); the head-ring restriction (5/6) aligns with the corpus.
- **§5 oracle swap** ↔ the pan-lung `OracleNominator` (AD-gated, filtering-only) is ready; hard linker/scaffold preservation matches the head-aware AD design and the Michael/BEAE fine-tune plan.
- **§3 corpus-first, linker-second** ↔ matches program policy `[[generator-general-then-linker-finetune]]`; propiolate/BEAE motif stays a downstream fine-tune (task #14).
- **Open items to act on when go is given:** (1) `git merge claude/generator-cond-uncond`; (2) author the lipid recipe pointing at the training corpus; (3) coordinate the ~48-atom kernel bump with Codex; (4) exploratory restricted-ring smoke.
