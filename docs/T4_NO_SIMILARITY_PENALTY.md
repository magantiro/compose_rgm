# Intermediate similarity ablation, 2026-09-10

Question: does removing the direct original-seed similarity penalty from
completed-option retention improve recovery of new eligible molecules? The user
approved this one bounded development diagnostic. It uses no new oracle calls,
training, winner input, executor change, or automatic follow-up.

COMPOSE remains a frozen executable stochastic process over complete supported
1..40-active-atom, charge-preserving achiral molecular graphs. This tests one
controller ranking heuristic, not a new model or a claim of general performance.
Outputs are exact replayed primitive paths and locked complete-option molecules.

Use the existing four-level, width-three, three-branch search from the observed
best eligible parent in the 51-call archive. Preserve its exact incumbent slot,
local/global region prior, applicability-balanced option prior, generic channel,
whole-option execution, kappa=1 retention tilt, exploration, saved task value,
and seed derivation. Intermediate execution already has no hard T4 feasibility
cutoff. The only new ranking field excludes similarity from the violation
penalty. QED/SA penalties retain their thresholds and scale 0.10. Final QED>=0.6,
SA<=4, original-seed Morgan similarity>=0.4 and the medchem screen are unchanged.
The separate Morgan diversity slot remains unchanged too.

This is an immediate surrogate heuristic, not a calibrated probability of future
recovery. Predictions outside the observed archive remain unvalidated. Removing
this penalty could make recovery worse; report that outcome without tuning.

Reuse the three completed constraint-recovery controls. Verify unchanged search
and retention function bodies, unchanged scientific dependency files, matching
physical inputs and task snapshot. The new score branch is contract-opt-in.
Reuse exact-state reference laws only through existing compatibility checks.
No previously generated candidate is inserted into the new search.

One CPU worker, 8 GiB, no GPU, zero retries, 30-second heartbeat, 1800-second
administrative timeout. At most 30 completed-option attempts, 44 primitive steps
per lineage; the bookkeeping horizon does not interrupt a registered program.
Previous graded control took 741 seconds proposing and 947 seconds overall.
Expect 10..20 minutes plus deployment, with 0.5 CPU-hours and 4 GiB-hours maximum
reservation excluding initialization/build. Dollar rates are unverified.
Resume units and law-cache identities are inherited unchanged. A timeout is
incomplete evidence, not an impossible molecular path.

Primary outcome: distinct new eligible molecules and actual ineligible-to-eligible
complete-option transitions. Report unchanged-parent recoveries separately from
novelty, ring and structural deltas, intended region scale, candidate diversity,
selection probabilities, all failures, internal calls and time. A zero-oracle
surrogate improvement cannot establish docking improvement. This inspected
development cell is not a held-out evaluation.

Focused checks cover similarity-independent ranking at fixed QED/SA/prediction,
retained QED/SA penalties, unchanged endpoint gate, exact search/replay/resume,
legacy function compatibility, and the one-worker launch receipt. Run strict
preflight on clean committed source, deploy the T4 app, then launch with
`python3 tools/t4_launch.py --no-similarity-penalty`. Full-suite qualification is
not required for this bounded diagnostic and will not be claimed.

IVG comparison: the primary paper describes soft penalties for QED, SA and
similarity violations in sampled molecules. It does not establish a hard
similarity cutoff on each internal denoising state. Its internal states differ
from COMPOSE's complete molecules. This ablation tests our own control choice,
not a claim that IVG ignores similarity during optimization.
Source: https://arxiv.org/html/2509.26405v1#S3.S4
