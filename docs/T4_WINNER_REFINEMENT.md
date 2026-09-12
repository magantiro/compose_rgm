# Winner-initialized T4 refinement

## Authorization and question

On 2026-09-12 the user explicitly approved the proposed small winner-initialized
refinement experiment. This task tests whether the existing broad COMPOSE support
and option controller can improve a published PARP1 seed0, delta=0.4 winner once
it is supplied as the initial state. It does not test autonomous seed-to-winner
discovery, train a model, or establish superiority on the original benchmark.

The fixed starting winner is the exact endpoint of the already replayed first
whole-ring plan. Its previous production docking receipt is retained as historical
development evidence, not substituted for the fresh control replicates.

## Locked recipe

- One complete positive-mass primitive-successor census from the winner, using
  the production reference evaluator and executor. No new support enumerator.
- Sixteen independent reference-policy streams, each with at most two completed
  existing options and fourteen primitive edits. Preserve the production WHERE
  distribution, balanced applicable WHAT prior, generic channel, and lazy HOW law.
  Ordinary edits and ring/carbonyl options remain available. No endpoint predictor,
  winner-distance score, learned actor, or learned future-value head is used.
- Select at most fourteen distinct eligible endpoints, aiming for seven from
  completed multi-edit options and filling remaining places from the complete
  primitive census. Balance option/family groups and use fingerprint diversity
  within groups. This is oracle allocation, not a change to the molecular law.
- Retain the original benchmark seed for QED >=0.6, SA <=4 and Morgan radius-two
  2048-bit similarity >=0.4. Apply the existing endpoint medicinal-chemistry gate.
  Intermediates require executability only. Record failures and all attempts.
- Three fresh winner dockings, fourteen candidate dockings, then two additional
  dockings of the lowest-scoring successful candidate. Maximum nineteen calls;
  failure counts as an attempt. No retries of ambiguous started calls.
- Keep the existing Open Babel preparation and QuickVina2 settings. QuickVina's
  seed is now explicit, 1701/1702/1703 for the three replicate positions, equally
  applied to winner and selected candidate. Open Babel gen3D is still stochastic;
  save prepared ligand and pose hashes. Do not claim deterministic ligand preparation
  or a matched external-benchmark reproduction.
- At most eight proposal workers or eight docking workers plus one driver, CPU
  only. Each proposal worker has a 1200-second ceiling, each docking call at most
  480 seconds, driver at most 3600 seconds. Reserve at most $10; no follow-on run.
  A timeout is incomplete evidence, not permission to shrink support or retry.

## Decision and interpretation

Compare the selected candidate's three scores with the winner's three scores,
including their ranges and individual replicate differences. A negative mean
difference is a development signal; three repeats do not establish broad
statistical superiority and selection uses the first replicate. Report repeat-only
differences separately. A repeat-only improvement warrants an independently
locked follow-up, not an automatic new optimization round. Failure to improve
does not prove a molecular optimum. If complete-option proposals are mostly
ineligible or fail, report that separately from docking quality.

Persist candidate locks, exact executor traces, original-seed properties,
option/region ancestry, structural displacement, topology deltas, candidate
diversity, proposal time, all docking receipts and total call accounting under
`t4_winner_refinement/<run_id>` on `compose-v4-artifacts`. The result is a
winner-informed development assay, not a future-aware controller qualification.
