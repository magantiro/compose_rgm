# Generator → Lipid handoff: what to reuse for the lipid generator

**From:** Claude generator lane (`claude/generator-cond-uncond`)
**To:** Claude lipid/oracle lane (`claude/lipid-corpus-oracle`)
**Date:** 2026-07-20

This tells the lipid workstream exactly which generator code, config, and
weights to use (and not use) to stand up a lipid generator + lipid-property
optimization, based on what was validated on GuacaMol this session.

## 0. How the code gets there — git, not copy

All three checkouts are **git worktrees of the same repo** (shared `.git`
object store): Codex (`agent/…`), generators (`claude/generator-cond-uncond`),
lipid (`claude/lipid-corpus-oracle`). So the lipid worktree already sees the
generator branch locally and just merges it:

```bash
# in the compose_rgm_claude_lipid worktree, on claude/lipid-corpus-oracle
git merge claude/generator-cond-uncond      # or rebase through main
```

Then modify freely on your own branch — it never touches the generator lane and
vice versa. The branch is also on origin for review.

## 1. Pathology ledger (what's actually solved vs open)

| Pathology | State on GuacaMol | Relevance to lipids |
|---|---|---|
| Graft/self-transition thrashing | **Solved** — self-Graft virtualization + immediate-backtrack safety wrapper (0.14% backtracking) | Transfers directly (sampler/config technique, not weights) |
| Validity / connectivity / self-events / collapse / stalls | **Solved** (100% valid, 0 pathological events) | Same code path |
| Chemistry marks (e.g. triples 54%→19%) | **Technique validated** — `corpus_residual_v1` empirical base + learned residual | Re-fit the base on the *lipid* corpus (a ~14 s scan); lipids barely have triples |
| Size / over-deletion | **Solved via one small knob** — atom-delete log-rate −0.5 (`exp(-0.5)≈0.61`) brings mean atoms 25.1→27.1 vs ref 26.35 | Re-check the mean for ~48-atom lipids; worst case re-tune this one number |
| Rings (fused/cyclization/small-ring) | Improved, **not** solved (fused 9→21% vs ref 58%; under-cyclization open) | **Moot** — lipids are ring-simple (~0.6 rings/mol, single 5/6-membered head rings, no fused, no 3/4). Restrict/down-weight the ring family. |

Net: **nothing non-ring is a blocker.** The ring saga stays in Paper-1 land.

## 2. Code to use (already on the merged branch)

Base generator + sampler:
- Shared RGM core (Codex, frozen): `rewrite/*`, `model/factorized_tracelet_rate_model.py`, ancestral sampler.
- `experiments/calibrated_rewrite_sampling.py` — **the graft-thrashing fix**
  (self-Graft virtualization + backtrack safety). Use this sampler path.
- `experiments/factorized_mark_conditional.py` — `corpus_residual_v1` empirical
  mark prior + `configure_factorized_trainable_parameters` scopes.
- `scripts/analyze_unconditional_sufficiency.py` — reuse for the lipid eval
  panel (swap the reference set for the lipid held-out corpus).

Optimization (the part you asked about):
- `experiments/griddd_conditional.py`, `experiments/guided_rewrite_sampling.py`,
  `experiments/molecular_property_conditioning.py` — valid-successor guidance,
  property conditioning, reward-FT. **Swap the QED oracle for the pan-lung
  oracle**; the guidance/reward machinery is otherwise unchanged.

Pipeline (debugged this session — mind these three gotchas):
- `modal_apps/train_tracelet_gm.py`, `modal_apps/evaluate_rollout_shards.py`.
- **Always launch with `modal run --detach`** (non-detached kills spawned tasks).
- **`--recipe-name` needs the `.json` extension.**
- New `--trainable-parameter-scope` values must be added to the argparse
  `choices` in `scripts/train_tracelet_cnof_gate.py` (not only the runtime).

## 3. Config (the "best formulation")

- Graft fix: **on** (calibrated sampler virtualization + backtrack wrapper).
- `empirical_mark_prior_mode=corpus_residual_v1` (re-scanned on lipid teachers).
- `ring_electronic_mode=factorized_local`.
- **Restrict the ring family** to observed lipid head-rings (single 5/6);
  **skip `ring_template_factorization=topology_cycle_hierarchical`** (drug-like
  fusion machinery lipids don't need). Down-weight/disable ring-grow per the
  lipid distribution.
- Keep atom-delete −0.5 as a starting point; verify mean size on lipids.

## 4. Weights — do NOT reuse GuacaMol weights as the lipid model

The GuacaMol checkpoints (pancake `47716924…`, combined
`compose-v4-uncond-chem-topology-train-20260720-v1`) are **C/N/O/F drug-like
specific** (triple-heavy, ring-heavy). **Train the lipid generator on the lipid
corpus.** Optional accelerator: a **compatible-init warm-start** of the encoder
+ general mark heads from the combined checkpoint, *only if* the lipid kernel
overlaps in shape — but the corpus determines the model.

## 5. Hard dependencies (coordinate — not the generator's to decide)

- **Corpus** — the lipid/oracle lane owns it; it is the real prerequisite.
- **Kernel extension** (max_atoms ~48+, confirm atom types incl. any P) —
  this is the **shared RGM core → loop in Codex** (gate P2-G2). Do not fork it;
  adapt via config + lipid-specific modules/subclasses.
- Per the two-paper policy, lipid *training* is gated behind the Paper-1 QED
  optimization proof, but an **exploratory restricted-ring lipid smoke is
  authorized** now alongside corpus/kernel prep.

## 6. One-line summary to paste into the lipid chat

> Merge `claude/generator-cond-uncond`. Base = `calibrated_rewrite_sampling`
> (graft/self-events solved) + `corpus_residual_v1` chemistry re-fit on the
> lipid corpus + atom-delete −0.5 (size); **restrict the ring family** (lipids
> are ring-simple). Optimization = reuse `guided_rewrite_sampling` /
> `molecular_property_conditioning` with the pan-lung oracle in place of QED.
> Train fresh on the lipid corpus (don't reuse GuacaMol weights). Don't fork the
> RGM core; coordinate the kernel size bump with Codex. Launch with
> `modal run --detach`, `.json` recipe names.
