# Matched donor/reference comparison, exposed prescreened development

Initial seed 20260926, run
`478421560799673857cd5aa0d2fc36df5b2ee4582828936ca6eb428699bd02fd`,
source `7a8cce2bcaab`, complete. Authoritative analysis: `report.json`.
Raw result and trajectories: `compose-v4-artifacts/pmo_donor_comparison/<run>`.
Both arms start at 0.5595028849441883 from identical 16 original prescreened
parents. Neither the probe winner nor the public winner is injected.

| Measure | Broad baseline | Donor/reference mixture |
| --- | ---: | ---: |
| Best | 0.559503 | 0.603023 |
| Top-ten mean | 0.550680 | 0.557554 |
| Completed / attempted options | 87 / 91 | 84 / 93 |
| Improving offspring | 16 / 87 | 16 / 84 |
| Mean offspring score change | -0.06886 | -0.13399 |
| Unique generated candidates | 86 | 82 |

156 new physical oracle calls across arms, 184 logical workers, 2443.108
recorded worker-seconds, 1881.283 neural-law seconds, 0.175 oracle execution
seconds. The comparison uses 249455 reported historical prescreen calls plus
the separately recorded 194 probe/parity calls. It is not no-prescreen PMO or
a 10k-query AUC result. Selection decisions replay exactly, with numerical
weights checked at 1e-12 cross-platform tolerance. This is not an exact-original-
reference Doob claim, nor a learned future-value experiment.

The best new molecule arose at boundary two by replacing a methyl side chain
with a propyl group through four executor primitives. It scores 0.6030226891555273:
`CCCC(NC(=O)c1nc(-c2ccc(C)cc2)n2c1CCC1CCCCC1C2)C(=O)OCC`.
This productive edit changed no rings. The hybrid also explored larger changes,
but more aggressive changes were not uniformly beneficial.

## Interruption and recovery

A worker was preempted at boundary four. Its unchanged task then failed the
old Python equality check because donor-cut tuples reload from JSON as lists.
Completed boundaries 1–3 and all 72 queries were preserved. The exact original
driver task was resumed on the same deployed source, using JSON-reloaded parents;
29/30 boundary-four workers were already complete. No completed query was repeated.
The first call was `fc-01M28R0185X0BTA6JPN0P1PD3B`; recovery was
`fc-01M28RH05WJNXT7TPPT6S3YNK7`. The report binds both receipts.

The result's 184.422 seconds is only the final driver session, not total run time.
The analysis reports submission-to-finish wall time including manual recovery
downtime. Worker totals cannot recover compute lost to preemption. The permanent
repair compares persisted content identities, retaining rejection of changed
values. A focused regression reproduces both cases.

## Next decision, before replication outcomes

Repeat unchanged scientific recipe with seed 20260927. Reuse compatible frozen
exact-state neural laws, not prior scores or new starts. Two deterministic wrapper
files adapt the completed run to the existing cache interface; original molecular
law files remain unchanged. Matching executor/model dependency hashes and input
hashes still gate reuse. No full panel or longer budget is authorized by this
single positive result. The replication contract records positive/null/failure
decisions before its outputs.
The replication uses at most 14 workers plus one driver, reserving the other
15 containers for a parallel bounded T4 lane. Scheduling changes no draws,
candidate locks, score calls or resampling decisions.

## Replication outcome

Seed 20260927 completed in 627.439 seconds with 149 new physical queries.
Hybrid best 0.5869734364739604 versus baseline 0.5595028849441883; top-ten means
0.549018 versus 0.544569. The original prescreened initialization is unchanged.
The best-score gain therefore repeated, though it is smaller than the first run.
This does not establish performance on another PMO task or against external AUC.

Run `88906fe9b25c5b283d4d68571dd8273aabbb16ba10929f182dc6f19f6df6ea5c`,
source `731a62699d02`, authoritative analysis `replication_report.json`.
186 workers, 2365.461 recorded worker-seconds, 1970.001 neural-law seconds,
0.165 oracle execution seconds. All particle decisions replay exactly. No
recovery was needed. Hybrid completed 75/91 attempted options, baseline 93/95;
greater compilation failure remains a cost of this mixture.

Best new SMILES:
`CCOC(=O)C(CN)C1CC2CCCCC2N1C(=O)Nc1nc(-c2cccnc2)cs1`.
