# Paired T4 frontier comparison

Status: complete positive development result. This is one inspected,
unreplicated PARP1 seed0, delta=0.4 round, not evidence of benchmark superiority
or generalization.

This is not yet competitive with the verified external results for this cell.
The three released InVirtuoGen runs report -13.5, -13.6, and -13.6, and GenMol
reports -10.6. The broader current controller is also below the workshop
COMPOSE result of -10.7 at 500 calls, although this warm-start arm has used only
59 total calls. These budgets and procedures are not a matched superiority
comparison.

The same 51-call warm archive (best feasible score -9.7), frozen reference model,
executor, local/global region law, option support, primitive horizon, endpoint
constraints, and round-frozen docking predictor were used in both arms. The
post-hoc arm used task value only after generation. The in-loop arm additionally
used two-transition resumable planning during generation. Both candidate locks
were audited before the first oracle call.

## Result

| Measure | Post-hoc | In-loop |
| --- | ---: | ---: |
| Completed candidate pool | 55 | 56 |
| Eligible and docked | 4 | 8 |
| Unique canonical docked | 4/4 | 8/8 |
| Distinct docked bundles | 4 | 8 |
| Best new docking score | -9.9 | **-10.0** |
| Improvement over shared -9.7 champion | 0.2 | **0.3** |
| Topology-changing docked molecules | 1 | 1 |
| Slowest lineage proposal time | 793.1 s | 1,065.1 s |
| Aggregate proposal compute | 3,523.0 s | 5,674.6 s |
| Executor calls | 21,722 | 58,135 |

The in-loop arm used eight successful new docking attempts and the post-hoc arm
used four, for 12 total against the authorized maximum of 40. There were no
failed docking calls. The completed reduction and docking recovery took 627.3 s;
proposal work had already been completed and hash-locked in the reusable source
partitions.

The best endpoint in each arm came from the same semantic option class,
`construct:pendant:6:5,1,0:nonaromatic:0`. Each added six heavy atoms, one cycle,
and one ring system through ordinary valid primitive transitions. The post-hoc
ring scored -9.9 at similarity 0.458; the in-loop ring scored -10.0 at similarity
0.444. This is direct evidence that the option layer can produce constructive,
feasible ring chemistry on this development cell. It does not show that rings
are generally beneficial.

Intended region release ranged from 0.074 to 0.593 post-hoc and 0.074 to 0.724
in-loop. Median intended release was 0.393 in both arms, while median realized
coherent change was 0.073. The successful ring endpoints came from intended
release 0.074 but realized coherent change 0.259. Intended search scope and
realized molecular change therefore remain distinct quantities.

Candidate diversity was 4/4 and 8/8 unique canonical molecules within arms,
each from a different bundle. Across arms, 11 of 12 dockings were unique. The
one repeated canonical molecule scored -9.2 and -9.1, an observed 0.1 kcal/mol
repeat difference in the inherited unseeded docking pipeline.

## Decision

The immediate failure is no longer inability to construct a useful ring. The
round-frozen endpoint predictor ranked the observed best ring last in each arm
(4/4 post-hoc and 8/8 in-loop). In-loop planning doubled endpoint eligibility,
but WHERE and WHAT received no task-value contrast; only eight HOW decisions were
task-guided. The next controller experiment should therefore target reusable
delayed value and oracle allocation across option-complete outcomes, while
retaining explicit bundle/option exploration. It should not narrow support,
reward ring count, tune macro weights to these docking outcomes, or call the
current endpoint predictor a future-value model.

No automatic next round is authorized. A replicated or multiround T4 comparison
requires a new prospective contract and oracle authorization. PMO oracle work
also remains unauthorized.

The IVG comparator values and public endpoint provenance are recorded in
`../ivg_winner_paths/audit.json`; no IVG endpoint entered this run.

## Provenance

- Source run: `c663e7ac0532b0664a40b02de1d53868a3c0bf69726f2957c68fc26777cfa843`
- Source code: `2f535a96d734a3fe9f7d6616ad55aea43d0505ac`
- Contract body SHA-256: `87d2e8c1841d7647f008914660e27e58efdb950c96cb410baaf064e74c129b7a`
- Full source result SHA-256: `1574f7066d59e8a0703532f5c2314b89ba8c330159d888e003a8a0af65d0fec4`
- Machine-readable verified review: `result.json`
- Remote namespace: `/t4_frontier_compare/c663e7ac0532b0664a40b02de1d53868a3c0bf69726f2957c68fc26777cfa843`

The machine-readable review binds every downloaded JSON artifact by SHA-256 and
rechecks the sealed candidate locks, pre-oracle barrier, lock-to-docking identity,
attempt accounting, endpoint eligibility, shared warm champion, and absence of
an automatic continuation.
