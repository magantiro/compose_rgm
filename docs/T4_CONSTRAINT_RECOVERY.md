# Completed-option constraint recovery, 2026-09-09

Authorization: the user approved proceeding after the macro-beam constraint
diagnosis and the COMPOSE/IVG process comparison. This is one bounded,
winner-blind, zero-new-docking development comparison, not model training or
the older unlaunched 40-call experiment. Final T4 eligibility is unchanged.

Question: does graded intermediate guidance recover eligible new molecules
where terminal-only guidance becomes flat? COMPOSE's frozen executable process,
local/global region prior, applicability-balanced option prior, generic channel,
complete-option execution and kappa=1 remain unchanged. The output is exact
primitive paths and locked complete-option molecules, not a new generator.
Support remains the existing charge-preserving achiral 1..40-active-atom graph
process. No IVG structures, paths or winner-derived targets enter the search.

Three arms start from the same best eligible observed parent in the complete
51-call development archive, with the same saved task-value snapshot:

1. post-hoc: no task scoring during generation;
2. terminal: existing hard-gated terminal desirability;
3. recovery: exp(-v / 0.10) times the saved model's ungated docking desirability.

Here v is the existing maximum relative QED/SA/original-seed-similarity
violation. The scale 0.10 is inherited from the T4 application's TAU_V, not
selected using this experiment. All three arms retain the exact starting parent
in one active beam slot. The remaining two slots use the existing quality draw
(uniform for post-hoc, kappa=1 tilt with 0.1 exploration otherwise) and max-min
Morgan diversity. Complete outputs equal to the incumbent are excluded only
from those two duplicate beam slots, never from the attempt ledger. The parent
is not a new candidate, primitive self-transition or extra oracle observation.

The graded score is an immediate heuristic, not a calibrated recovery
probability, learned future-value model or exact Doob transform. It does not
change final eligibility or make the medchem screen a pathwise gate. Ungated
docking predictions outside the observed archive are unvalidated; their use is
bounded by the existing empirical-pool tilt and exploration. Improvement in
this surrogate alone will not establish successful optimization.

Four search levels, width three, three option bundles per parent, at most
30 attempts per arm (90 total). A retained incumbent can restart at later
levels, so molecular chains have at most four options, not necessarily four.
The 44-primitive bookkeeping horizon never cuts a registered option mid-program.
Same deterministic RNG derivation as macro-beam, excluding arm from the seed.
Every complete option is replayed. Persist every attempt and selection;
resume unchanged completed units. Reuse exact-state laws only after existing
dependency and physical-input compatibility checks. Old candidate results
cannot stand in for the new search because incumbent retention changes it.

Three CPU workers, one core and 8 GiB each, no GPU, retries zero,
30-second heartbeats and an 1800-second administrative timeout. The previous
warm-root runs took 795 seconds proposing plus about 172 seconds initialization
each. Expected wall time 10..20 minutes plus deployment; timeout means incomplete,
not molecular impossibility. Maximum reservation is 1.5 CPU-hours and 12
GiB-hours excluding initialization/build overhead; dollar rates unverified.
No oracle or executor-call stopping limit alters a completed program.

Report distinct new eligible molecules and actual ineligible-to-eligible
complete-option transitions, with recovery-to-an-already-observed molecule
separated from novelty. Also report eligible ring-changing products, option
chains, intended vs realized scale, canonical diversity, surrogate outcomes,
all failures, actual law/executor work and time. If recovery only recreates the
incumbent, that is not novel optimization. A null outcome is a complete diagnostic.
No automatic docking or follow-up fan-out.

Focused checks: graded score and unchanged terminal gate, incumbent retention
without invoking post-hoc scoring, exact replay/resume, probability floor and
three-worker launcher. Clean committed source, strict preflight, deploy the T4
app, then tools/t4_launch.py --constraint-recovery. Full-suite verification is
for milestone qualification, not this bounded development comparison.
