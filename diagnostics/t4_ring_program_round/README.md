# Parameterized ring programs: one audited T4 round

Computed development result, 2026-09-08, session `compose_iclr`. Seven of seven
selected new construction programs completed their requested ring, but only one
met the unchanged T4 endpoint constraints. Best docking improved from -9.6 at
46 calls to -9.7 at 51 calls through a **local** edit, not new ring construction.
The newly constructed, eligible pendant heterocycle scored -8.2 versus its
parent's -8.7. This is positive construction evidence and negative evidence for
an immediate docking benefit from the sampled new rings.

## Authoritative artifacts and recipe

- Scientific revision: `7c21e1525b578c25cad9dfb9e4dfd30006df625a`, deployed from a
  clean detached worktree after preflight. No training or automatic continuation.
- Run: `ddb2f823de568fdb904b4e08341ee3effc2e9828a9043de7836bfc202144cb0d`.
- Modal call: `fc-01M21B4DEZ0ZAY87MHDEA1Y1HZ`, function `t4_ring_program_round`.
- Contract: [t4_ring_program_round.json](../../configs/t4_ring_program_round.json).
  File SHA-256: `2a797d6b946d37594d88b3b2b62cd30924ea4d160d306f18b5056117f1c72704`.
- Complete machine-readable offline audit: [review.json](review.json), SHA-256
  `cfc4be3c199893c335a00144200d2aa7edd22073a10cd899d044b5289566f62a`.
- Downloaded immutable run: [attempt_1/run](attempt_1/run). The review hashes
  every input JSON, including the executor ledger. The raw executor ledger is
  retained locally and on `compose-v4-artifacts` under
  `t4_ring_program_round/<run_id>/round_4/executor_attempts_0.json`, not in Git.
- Source archive SHA-256:
  `5ade7fb788e54e37a4aa2534132547be12fdb24d57ebd8f2f0dcd39bf0fec7e1`.
  This is the complete 46-call archive, not only the best molecule.

PARP1 seed0, similarity delta 0.4, one new-recipe round from round 3. Eight
parents, three unchanged region draws each, one particle per bundle, frontier
cap eight, at most three representatives per bundle. Equal parent allowances
of 2,500 executor applications, total cap 20,000; at most 20 new dockings.
Existing archive RNG state reused. This is not a bit-identical resume of the
old option recipe. Q(M), R_theta, structural committor, kappa=1, generic support,
purpose-balanced untrained Q(o), and the executor/gates remained unchanged.
No IVG winner fragments, frequencies, or docking-based option-weight tuning.

Remote software: Python 3.11.12, NumPy 1.26.4, RDKit 2024.03.5, PyTorch
2.4.0+cu121. One CPU, 8 GiB container, no training/GPU allocation. Offline
review used Python 3.12.9, NumPy 2.5.3 and RDKit 2026.03.6 and records that
distinct environment. Exact endpoint topology and closure verification passed
there; it is not an assertion that all RDKit versions are equivalent.

## Construction and allocation

All 24 region/option draws were retained. Selected options were seven new
parameterized programs, generic (5), append (2), grow (2), restate (2), shrink
(2), and one each of local, aromatize, rebuild and historical build_ring_system.
No standalone cyclize option was drawn. Seven new programs gained exactly one
cycle each: five pendant programs gained one ring system and two fused programs
preserved ring-system count. All seven had zero detected spiro and bridgehead
atoms. There were no terminal-halogen or sulfur additions in these programs.

| New request | Draws/completed | Edits | Intended release | Realized coherent change | T4 eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pendant 5, C4N1 aromatic | 2/2 | 6 | 0.103, 0.174 | 0.207, 0.261 | 1/2 |
| Pendant 5, C4O1 aromatic | 1/1 | 6 | 0.308 | 0.231 | 0/1 |
| Pendant 6, C5O1 nonaromatic | 1/1 | 7 | 0.071 | 0.250 | 0/1 |
| Pendant 6, C6 aromatic | 1/1 | 7 | 0.567 | 0.233 | 0/1 |
| Fused 5, C4N1 nonaromatic | 1/1 | 4 | 0.464 | 0.179 | 0/1 |
| Fused 6, C6 aromatic | 1/1 | 5 | 0.733 | 0.200 | 0/1 |

Composition denotes the whole new cycle, including shared anchors for fused
rings. All new programs used zero optional refinement steps in this round.
The construction witness uses exact persistent-slot paths, not just SSSR counts.
Nonaromatic fused construction does not imply every shared bond is saturated.

The historical 11-step build_ring_system draw stopped after three search edits
with `no_option_support` and emitted no endpoint. Three parents exhausted their
2,500-call allowances; their available valid intermediates were retained under
the existing generic policy. These outcomes are not counted as successful
program completions. Grow and rebuild emitted no new cycles in this round.

## Oracle bottleneck and scores

The final representative pool contained **27 canonical-distinct molecules from
23 bundles**, not a large uncompressed frontier. Only **5/27** were T4-feasible,
so all five received docking, each from a distinct bundle. None was duplicated;
there were zero oracle failures. The 20-call allowance was a cap, not a mandate
to dock infeasible molecules. No candidate was retroactively filtered using its
docking result.

All 22 excluded endpoints failed seed similarity >=0.4. Overlapping failures:
five also had QED <0.6 and three had SA >4. Among the seven new ring endpoints,
six failed similarity, four failed QED and two failed SA. These are unchanged
benchmark thresholds, not new ring-quality vetoes.

| Docked option | Parent score | New score | New cycle / ring system |
| --- | ---: | ---: | --- |
| local | -9.6 | -9.7 | 0 / 0 |
| grow | -9.3 | -9.6 | 0 / 0 |
| restate | -9.2 | -9.5 | 0 / 0 |
| grow | -9.3 | -9.5 | 0 / 0 |
| pendant 5, C4N1 aromatic | -8.7 | -8.2 | +1 / +1 |

Pool intended release min/median/max: 0.033/0.269/0.889; realized coherent
change: 0.038/0.103/0.292. Docked release: 0.036/0.174/0.889; docked realized
change: 0.069/0.074/0.261. The 0.889-release grow bundle realized only 0.074
coherent change. Large search permission is still not a large realized rewrite.

Mean pairwise Morgan/Tanimoto distance was 0.69751 in the 27-candidate pool and
0.56299 among five docked candidates. All 27 had zero detected spiro/bridgehead
atoms; this is a structural descriptor result, not proof of medicinal quality.

## Work, verification, and interpretation

Proposal time: **679.525 s**; preparation wall time: 685.687 s; docking:
15.458 s; whole remote call: **851.028 s (14.18 min)**. Public executor ledger:
10,570 calls comprising 8,491 executed, 2,077 invalid rewrites and two
budget-interrupted attempts. Invalid proposals were not committed states.
Peak memory and billed USD cost were not measured. This is not a matched speed
comparison or evidence that adding the new menu made the old recipe faster.

Focused dependency checks passed **141/141** in 9.30 s on the clean scientific
commit; touched standalone code passed Ruff, the Modal app compiled, preflight
and diff whitespace checks passed. The independent local synthetic-fixture
audit completed 21/21 requests in 1.075 s using 307 executor calls; see
[ring_program_audit](../ring_program_audit). That uniform-reference fixture is
not learned T4 evidence. The repository-wide suite was not rerun for this
bounded development audit; this is not full-milestone qualification.

The existing offline auditor passed hash bindings, exact archive continuity,
program completion, endpoint topology, executor accounting, candidate-lock
chronology, per-step KL bounds, canonical deduplication and deterministic archive
reduction. Reproduce from revision `7c21e15` with the downloaded ledger present:

```sh
PYTHONPATH=src:. python tools/t4_warm_audit.py \
  diagnostics/t4_ring_program_round/attempt_1/run \
  --output diagnostics/t4_ring_program_round/review.json
```

**Decision:** the integration now supplies constructive ring chemistry in this
observed production sample; inability to close the requested rings is no longer
the explanation for these seven proposals. It has not shown better docking
from new construction. The next useful diagnostic is the already-saved
parent-to-endpoint constraint loss, especially attachment-site and seed-similarity
preservation, before more docking or another option catalog. Do not relax the
benchmark gates or change Q(o) using these docking outcomes.

This is one inspected development cell with seven selected construction draws,
not a general completion-rate estimate, causal ablation, algorithm ranking or
held-out IVG comparison. Docking was unseeded; the 0.1 score improvement can
include oracle noise. No additional round was launched.
