# Unchanged local-guidance replication

Authorized 2026-09-11 after the first comparison passed its declared best-score
criterion. Preserve `configs/pmo_local_guidance.json` and its prepared inputs.
The new contract is `configs/pmo_local_guidance_replication.json`.

Use seed 20261009, the next fixed seed, with the same original sixteen exact
parents, 116 donors, 1044-label endpoint model, four rounds, sixteen attempts
per arm per round, mixture, all primitive support and archive rules. No warm
start from the newly discovered winner, refitting, mixture tuning or public
endpoint input. This is a warm development replication, not no-prescreen PMO.

The first run's 85 physical oracle labels are reusable requested-only scores.
They must not change the original `observed` proposal exclusion set, initial
archive, donor bank or fitted model. Only a generated and locked request exposes
a cached score to the current arm. Record cached requests and new physical
calls separately. Historical costs before replication: 249455 prescreen and
2141 development physical calls. Prior model-fitting labels are in this history.

Primary positive criterion remains guided best greater than both the original
initial best and simultaneous baseline best. Report both seeds, top-ten means,
diversity, failures, proposal time and physical calls regardless of direction.
A second positive retains the channel for subsequent generalization work, not
an external superiority claim. A null or reversal stops the unchanged mixture.
An operational failure is inconclusive; do not silently replace its seed.

At most 128 new calls total, 29 workers plus one driver, one CPU and 8 GiB per
worker, no GPU, 180-second worker and 900-second driver ceilings, no retries,
$10 reserved cap. First-run driver took 497 seconds with 470.5 proposal seconds;
expect roughly 4-12 minutes after deployment, depending on the realized paths.
At most 128 worker tasks. Reuse existing compatible law caches and immutable
inputs. Restart units are saved worker and round receipts. Thirty-second
heartbeats and durable spawn survive the client. Clean source and strict
preflight precede deployment. Focused cache/exclusion, proposal and ledger tests
are the development checks; this does not declare the overall milestone done.
