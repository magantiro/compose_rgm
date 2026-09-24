# Completed-interface redirection: development gate

Status: pre-outcome development contract, not an official fragment benchmark
result or a change to the released sampler. The historical Boolean
`redirect_attachment=True` remains the control. The candidate has its own
serialized value, `redirect_completed_interfaces`, and is derived only from
the prompt's declared interfaces and current coverage.

## Defect and falsifiable mechanism

During attachment-first staging on a multi-interface prompt, an insertion
anchored to a declared site whose required external-neighbour count is already
met cannot reduce the unsatisfied-interface set. The historical redirector
exempts **every** declared anchor, so this insertion reaches the staging gate
and is refused while another declared site remains open. The candidate
rebinds only this redundant insertion to an unsatisfied declared site, keeping
its element, bond-order realization, and executor/region-lock checks. It does
not alter the source model, random seed derivation, chemical support, budget,
or post-coverage behavior. It cannot read drug names, task names, targets, or
quality scores.

The mechanism is irrelevant unless it actually fires. The runner must count
offers, successful rebindings, and accepted resulting transitions separately.
If accepted rebindings are absent or rare, an apparent metric change is not
attributed to the mechanism.

## Matched development assay

- Inputs: pinned fragment checkpoint
  `/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt` (SHA-256
  `24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`)
  and the frozen fragment manifest
  `data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv`
  (SHA-256
  `a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9`).
  Verify the cached IVG evaluator revision and blob hashes before each arm.
- Primary set: the four **development** scaffold-decoration prompts
  BARICITINIB, ERLOTINIB, LIOTHYRONINE, MARIBAVIR. These were the development
  prompts in the prior interface-release pilot, not the six held prompts in
  `docs/FRAGMENT_SINGLE_INTERFACE_QUALIFICATION_2026-09-23.md`.
- Negative controls: BARICITINIB motif extension (one interface) and
  LOVASTATIN superstructure (zero interfaces). Their attempt records must be
  identical across arms. Any difference is a wiring failure.
- Two arms: historical strict attachment control and strict control plus
  completed-site redirection. Same checkpoint, sampler, 20 attempts per
  prompt, seed 0, 32 events, 24 mark draws/event, operational horizon 16.
- Persist each prompt/arm as an atomic unit with every attempt, emitted and
  committed SMILES, failure counters, exact seed, config, source and input
  hashes, evaluator identity, and software versions. Do not pool prompts before
  computing uniqueness or diversity.

Report on the 80 primary attempted trajectories: produced/committed endpoints,
chemical validity on committed endpoints, graph-lock and independent fragment
preservation, declared-site coverage, full task success, official-formula
quality, uniqueness, within-prompt diversity, distance, refused marks,
mechanism offers, rebindings, accepted rebindings, and wall time. Report every
negative/null attempt on the attempted denominator.

## Frozen development decision rule

Reject the candidate if either negative control differs attempt-by-attempt,
any committed endpoint is chemically invalid or violates the exact region
lock, or an official input/evaluator hash differs. Also reject it as an
ineffective mechanism if fewer than five completed-site rebindings become
accepted transitions across 80 decoration attempts.

For development qualification, require at least a 10% reduction in staging
rejections across the primary set, no more than two fewer full task successes
out of 80, no more than a five-percentage-point loss in official-formula
quality, and no more than a 0.03 loss in mean within-prompt diversity. These
are conjunctive; a validity or constraint failure cannot be compensated by
quality. If the candidate fails, preserve the result and do not tune it on
the held prompts. Passing this assay authorizes a separately frozen held
qualification, not a paper claim or automatic promotion.
