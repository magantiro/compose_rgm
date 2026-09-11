# Persistent two-option planning: parallel PMO component experiment

User authorization: active controller goal and request for a complementary PMO
track in parallel. No external-winner inputs, new prescreening, docking, reference
training, or executor changes. This specifically tests scheduling of the existing
population-derived transplant channel. It does not replace the full controller's
generic, atom/bond/ring, local or global channels with donor-only chemistry.

The prior `pmo_continuation_choice` experiment used witnessed maxima to select a
first move, then discarded the selected continuation and sampled fresh reference
draws. Its negative result does not test retention of found complete sequences.
Here both policies retain every sampled exact trajectory; no surrogate is used.

## Fixed comparison before proposals

Use two exact paid starting states: the latest 0.6747477698 one-edit diagnostic
champion and the distinct 0.6720215050 broad-controller champion. Freeze both and
their complete historical labels before proposals. Donors are the same fixed
116-row bank from the latest broad-controller comparison, not IVG molecules.
Reuse its inverse-score-rank prior with 20% uniform exploration and uniform
oriented single-bond-bridge cuts. Compiler: existing 64-step/128-expansion bounds.

For each root, sample four complete first-option proposals. Score their canonical
novel union, then freeze the second-phase parent allocation:

- Immediate arm: select the best scored state among the root and complete first
  proposals, then make eight second-option draws from it.
- Lookahead arm: make two second-option draws from each of all four first-option
  slots, including worse but valid states. Failed first proposals leave failed
  continuation slots; no silent retry. No parent-score pruning.

Each arm has four shared planning opportunities and eight continuation slots per
root. Shared exact parent and draw-RNG tasks are reused; the repeated occurrence
retains its logical slot. Retain best found root/first/second endpoint together
with its actual complete primitive path. Do not throw away a witnessed path and
resample it. Planning evaluations count as oracle calls for both policies; their
shared physical union is charged once.

Two roots, at most 40 new canonical calls across both arms. Random seed 20261006,
streams derived from seed, root index, phase, first-slot, second-draw. Immediate
continuations use selected-first-slot identity, so the corresponding two
lookahead draws share exact tasks; remaining immediate draws use distinct stream
indices. Incumbent selection uses first-slot -1.

Primary: a new champion above 0.6747477698 and a higher best for lookahead than
immediate earns a fresh-seed full-controller integration test. Otherwise stop
this unchanged proposal/scheduling combination. Report two-root results, all
failures, mean/best/top-ten endpoints, improving and temporary-loss recovery
paths, intended release versus realized change, primitive depth, query counts
and proposal time. A recovered path below the champion is supporting evidence,
not competitive performance. No exact Doob or typical-return estimate is claimed.

## Compute and durability

One local CPU, no new neural-law enumeration, no GPU or Modal deployment. Existing
local transplant batches took roughly 20-50 seconds for 64-128 attempts. At most
40 compiled tasks here, 180-second total administrative bound; finished tasks
remain reusable. Persist locked source/recipe/inputs and snapshot, each compiled
program, phase candidate locks before calls, oracle reservation/results, selected
paths and final interpretation. Local oracle identity must match the previous
qualified oracle. Use the frozen source snapshot for provenance, focused tests,
no unrelated suite or commits. Account prior 249455 prescreen and 1978 development
physical calls, including the just-completed independent 32-call local-selector
test as lookup-only history. Its results do not alter the two starting states,
donor bank, proposal probabilities or selection recipe.
