# Initial locked-family conditioning: matched development qualification

Status: pre-outcome protocol for the seed-10, all-prompt zero-oracle assay. This
is not the official three-seed, 100-attempt fragment benchmark and does not
promote a controller. The seed-9 three-prompt pilot was inspected before this
protocol was frozen; it is development evidence only.

## Scientific object and hypothesis

The generated object is a complete, connected, supported molecular graph,
produced by a sequence of exact rewrite marks. In superstructure generation,
the supplied core is locked. The claim under test is not that the base editing
law learned a fragment task, but that sampling its first-event family conditional
on construction when all current atoms are locked reduces wasted refused draws
and increases output at a fixed 24-draw/event budget. This is a task-generic
structural condition and reads no drug name, hidden target, quality value, or
oracle. The executor and region lock remain final admission checks. The current
family mask is not a complete conditional law over legal marks or operands.

## Frozen matched assay

- Inputs: the frozen ten GenMol/IVG superstructure prompts from
  `data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv`
  (SHA-256 `a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9`); the
  already trained broad-organic fragment checkpoint
  `/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt`
  (SHA-256 `24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`); and the
  pinned local IVG evaluator at commit `b50bb3ae` verified by
  `tools/fetch_official_fragment_evaluator.py` (metrics blob SHA-256
  `3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099`).
- All ten prompts, seed 10, 20 attempts per prompt, 32 maximum events,
  operational horizon 16, and 24 mark draws per event. Each prompt starts from
  the same deterministic RNG seed in every arm. The runner uses one sequential
  RNG stream per prompt, so later attempt-level draws are **not paired** after
  the two proposal laws consume different numbers of random variates. No
  oracle calls or new training.
- Three arms: historical attachment controller; the same controller with
  `condition_initial_locked_family`; and conditioned plus the opt-in effective-
  chemistry hard lock. No prompt-specific setting or result-dependent retry.
- Persist all attempted trajectories, canonical committed and emitted SMILES,
  rule/refusal counters, event counts, exact seeds, evaluator and input identity,
  and per-prompt official metrics. Reduce prompt rows only after verifying all
  600 attempt records and input hashes.

The primary endpoint is no-output attempts (the official evaluator counts their
empty placeholders as invalid). Report chemical validity per committed endpoint,
independent fragment containment, official validity/quality/uniqueness/diversity,
first-event conditioned offers and accepts, lock refusals, work, and wall time
separately. Quality and diversity are safeguards, not a reward used in sampling.

## Predeclared falsifiers

Reject the conditional first-family repair if any committed endpoint is
chemically invalid, if any conditioned output violates the required fragment,
if any arm loses input/checkpoint/evaluator identity, if the conditioned family
is not actually offered and accepted on the all-locked initial states, or if
the conditioned arm has no fewer no-output attempts than historical control.
Treat a gain confined to the seed-9 development prompts as insufficient. Report
per-prompt regressions, including quality, uniqueness and diversity, without
hiding them in an average. For the strict-lock arm, require independent
fragment preservation for every committed endpoint; do not infer a theorem
about all representable molecules from a finite sample.

Even if this assay passes, promotion requires an independently frozen full
official sample and a separate decision. This protocol cannot relax the
historical benchmark denominator or erase the original failed attempts.
