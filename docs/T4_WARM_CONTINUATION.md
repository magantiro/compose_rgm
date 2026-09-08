# Guided T4 warm continuation

Prospective authorization, 2026-09-08: the user approved the proposed faster
multi-round development test. Reuse the repaired guided arm's complete first
round, then run two additional round-synchronous batches of at most 20 dockings
each. This is 40 new calls, at most 60 cumulative. No reference continuation,
training campaign, weight tuning, or additional rounds are authorized.

## Identity and question

COMPOSE learns an executable marked molecular rewrite law. Its generated
objects are complete supported molecular graphs, not retrieved ring templates.
This experiment asks whether its frozen region/option/primitive controller can
accumulate useful changes through objective-feedback-driven parent selection.
The platform claim remains the executable stochastic process described in
AGENTS.md. This single inspected PARP1 seed0 cell is development evidence only.

Compare each new round with the saved first-round archive. A matched reference
continuation is the necessary future causal ablation, not part of this run.
Neither improvement nor proximity to an inspected IVG winner establishes
guidance benefit, generalization, or benchmark superiority. No IVG structures,
fragments, scores, or reference-distance rewards enter the optimizer.

## Frozen semantics and accounting

Inherit the exact task and six input identities from
`configs/t4_matched_pilot.json`. Keep Q(M), the applicability-balanced untrained
Q(o), generic, R_theta, committor, kappa=1, exploration settings, horizons,
eight lineages, three region draws, one particle, and oracle allocation fixed.
The existing bootstrap-ridge acquisition consumes all available docking pairs;
it is an inherited heuristic, not a calibrated uncertainty model. Similarity
remains against the original seed at d=0.4, never the current parent.

Support remains connected broad-organic, charge-preserving, non-stereochemical
graphs with at most 40 active atoms. The fused program retains its aromatic C6
restriction. No whole-ring template, multi-neighbor birth, executor change, or
new support restriction is introduced. Region enumeration and sampling run
unchanged on parent SMILES metadata; selected region indices are then mapped
onto the saved exact persistent slots. Executable states are never rebuilt
from those SMILES. Per-parent coherent displacement is not relabeled as
cumulative displacement; cumulative graph counts and molecular ancestry are
reported separately.

Reuse all 20 saved guided docking records, including failures if present, and
recover their exact endpoints from the candidate lock's sampled transitions.
Bind both source file hashes. Restore its post-selection RNG state and fit the
existing surrogate where the normal optimizer would fit after round one.
This is an explicit warm-start adapter, not an unsupported assertion of a
bit-identical restart from a legacy checkpoint.

Each new round seals its candidate pool, selected candidates, exact states,
ancestry, configuration, and RNG before any docking. Update the archive only
after the whole batch. Exclude previously evaluated canonical molecules.
Retain the inherited unseeded docking protocol and its noise limitation.
No backfill, outcome-based macro tuning, or automatic oracle retry is allowed.

## Compute and restart boundary

One CPU, 8 GiB, no GPU, one worker, zero retries, one-hour whole-job stop.
Retain the 20,000 public-executor application ceiling separately per new round.
A ceiling or timeout is a reported negative result, not authority to increase
the budget. Completed parent units, locked rounds, and docking receipts are
restart units. An interrupted oracle batch or failed preparation requires an
explicit accounting decision before retry. No second-round regeneration.

The previous guided round used 449.687 proposal seconds and 16,335 executor
calls. Two rounds with larger parents may cost more than twice that work.
The hard stop bounds the new run; do not promise an exact completion time.
Inventory the remote source and destination before launch. Reuse the complete
source round, frozen assets, runtime gates, and existing saved-product checks.
Record the launch revision, input hashes, software, hardware, wall time,
executor work, exclusions, and all failures in task-specific artifacts.

## Acceptance and launch

Focused tests must verify exact endpoint recovery and region-slot mapping,
archive completeness, RNG restoration, original-seed constraints, round locks,
oracle accounting, no implicit retries, and no incomplete program candidates.
Run repository-wide verification once on the frozen launch candidate; classify
known baseline failures without relaxing gates. Newly implicated scientific
failures stop launch. Do not claim a green repository when it is not green.

From a clean committed worktree, use `python3 tools/preflight.py --strict`,
then `modal deploy modal_apps/genmol_t4_opt_app.py`, then
`python3 tools/t4_launch.py --warm-continuation`. Never use `modal run --detach`.
Emit durable 30-second heartbeats and parent/round/docking checkpoints.

Report options and completion reasons, proposal versus selected versus docked
chemistry, intended versus realized per-parent scale, cumulative cycle rank,
ring-system and heavy-atom deltas, ancestry, candidate diversity, feasibility,
all docking results, best-so-far scores, and proposal time. Distinguish new ring
formation from splitting an old system by opening it. Freeze the result before
deciding the next scientific change.
