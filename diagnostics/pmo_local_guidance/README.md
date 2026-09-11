# Broad search plus local endpoint selection: first development comparison

The predeclared best-score criterion passes in one warm development seed. The
guided mixture reaches **0.6835298931**, versus **0.6803013430** for the retained
broad baseline, from an identical initial best of **0.6747477698**. This is not
yet replicated or a matched external PMO comparison. The guided top-ten mean and
archive diversity are lower, so the result does not show uniform improvement.

| Measured quantity | Baseline | Guided mixture |
|---|---:|---:|
| Best actual Perindopril MPO score | 0.6803013430 | 0.6835298931 |
| Top-ten mean | 0.6747775400 | 0.6743329457 |
| Attempted option slots | 64 | 64 |
| Completed options | 53 | 50 |
| Unique generated molecules | 51 | 50 |
| Mean pairwise Morgan distance, queried archive including starts | 0.6453534253 | 0.6026577905 |

Both arms use the same sixteen exact initial states, 116 fixed donors and
archive-parent allocation. The baseline allocates 50% donor / 50% reference
draws; guided allocates 50% donor / 25% reference / 25% local endpoint selection.
The local model remains frozen at its prior 1044-label fit. Generic and multi-edit
reference channels remain available; no executor, frozen reference, template
vocabulary or benchmark-objective change was made. Seed: 20261008. The exact
contract is `configs/pmo_local_guidance.json`, hash
`0682c9efc6dea9051ef8bee34a6d0d3dca4f95b84eba8cf8285e6a4381bb07aa`.

## Winning path and ranking limitation

The guided winner descends from a round-1 local choice with score 0.6747477698.
That choice closes a pendant three-membered ring. A round-4 generic option then
executes three atom restatements to obtain 0.6835298931. The four primitive edits
were independently replayed through the unchanged semantic executor, and the
exact states connect across the two options. Cycle rank increases from four to
five and ring-system count from two to three; heavy-atom count remains 39.
These are two controller decisions in this run, not a claim about the shortest
route from the original prescreen source or about learned foresight.

On the same round-1 parent, the baseline proposed a legal atom deletion scoring
0.6775074859. It was present in the local pool but ranked 110/1618 with prediction
0.6221295839. The local choice ranked seventh, with prediction 0.6461271366 and
actual score 0.6747477698. Thus the predictor missed a better immediate move,
while its selected precursor later yielded the final winner. This post-hoc
observation supports keeping alternative continuations; it does not establish
that the endpoint model predicted future value.

The recorded winner is:

```text
CCCC(NC(=O)c1ccc(N)c2c(C3CC3O)c[nH]c12)C(=O)N1C2CCCCC2CC1C(C)C(=O)OCC
```

## Compute, provenance and verification

The four rounds used **85 new physical oracle calls**, 116 distinct workers,
497.011 seconds driver wall time and 470.540 seconds proposal wall time. Summed
worker time was 1948.350 seconds; summed frozen-law time was 902.640 seconds.
Recorded oracle-function time was 0.065896 seconds, excluding orchestration.
The guided arm's thirteen local draws predicted 19832 pooled candidates, with
219.480 summed seconds in selection. Exhaustive mark counts are not pooled with
stochastic proposal-attempt counts. Historical costs remain 249455 prescreen
calls and **2141 development physical calls including this run**. The 1044
training labels are reused from that history, not an additional unreported bank.
Billing has not been independently reconciled; the authorized reserved cap was
$10, at most 29 workers plus one driver, no GPU.

Execution used clean revision `797d7085e737`, Python 3.11.12, Torch 2.4.0+cu121,
NumPy 1.26.4 and RDKit 2024.03.5, CPU only. Strict preflight passed. The existing
focused integration checks were reused. Six focused reporting tests passed;
formatting, lint and diff checks passed. The source snapshot, exact contract,
oracle ledger, archive-parent decisions, per-slot channel identities and winner
replays were audited with no new oracle or neural-model calls. No full-suite or
overall-milestone completion is claimed.

`report.json` contains the population audit, input hashes and computation counts;
`evidence.json` contains the exact winner replay, saved-ranking diagnostic,
decision and source dependencies. Parent-normalized structural-change ratios
can exceed one after growth from a very small parent; they are not bounded
percentages of an endpoint molecule. No ordinary ring-quality or medicinal
efficacy claim follows from the PMO score.

Remote namespace on `compose-v4-artifacts`:
`pmo_local_guidance/2f58bc8ace9d1591520d0c2a670a0b04cb1bd1cad4879f1ca4420361c8570e72`.
Durable call `fc-01M297KAGZKERT3EE8P8SJW9MM` completed. Its exact source snapshot
has also been uploaded into that namespace. Local raw receipts are in the
matching run directory under `/private/tmp/compose-pmo-local-guidance-runs`.

**Decision:** the primary criterion earns one unchanged fresh-seed replication
from the same original starts, donor bank and fitted model. Do not warm-start
that replication from the new winner, refit the model, or tune the mixture.
The current authorization covered this single run; no replication or second-task
experiment has been launched. The overall competitive-controller goal is unmet.
