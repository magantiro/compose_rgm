# Score-information audit

Scope, 2026-09-10: diagnose the completed 100-query development pair using only
its saved labels, ancestry and production allocation code. No model fitting,
oracle calls, new candidates, policy changes, or deployment. Output is a
provenance-bound accounting of information used and discarded by allocation.
This diagnoses the tested approximation, not COMPOSE's representable support.

Checks: replay production credit updates against the recorded final credit,
compare parent-relative score changes with archive reward, count zero-credit
outcomes, and inspect contextual variation within option labels. Root-parent
comparisons are explicitly excluded because individual root labels are not in
these final receipts. No labels are inferred or filled in. Report both arms and
all failed/repeated attempts; a selected example is not a causal estimate.

The next policy remains a proposal. This audit does not authorize a new run or
replace the locked archive-improvement objective with parent-relative reward.

## Findings

Production credit replay exactly matches both final receipts. All 60 available
adaptive prefix records also reproduce their WHERE/WHAT probability rows to
absolute tolerance 1e-12. This is the rule actually executed, not a stale-code
explanation. The full outcome audit takes under one second locally and makes zero
oracle calls.

Information flow in the tested implementation:

- Parent allocation uses observed score rank, not molecular features or predicted
  future improvement (`archive_allocation.parent_distribution`).
- Adaptive WHERE uses pooled credit for three scale bins. Adaptive WHAT uses
  pooled credit for option labels. Ring size/composition can affect WHAT via its
  label, but the credit estimate is not conditioned on the molecular graph, its
  current score, exact region, or attachment site (`ArchiveCredit.values`).
- HOW samples the frozen reference within the chosen option. There is no PMO
  score/value input to that call (`execute_option`, `MolecularHierarchy.sample_reference`).
- Completed endpoints are queried and retained, including worse endpoints. There
  is no within-option surrogate or task-aware lookahead in this pilot.

| Retrospective accounting | Balanced | Adaptive |
|---|---:|---:|
| Attempted options | 98 | 102 |
| Zero archive reward | 61 | 62 |
| Non-root parent comparisons available | 87 | 91 |
| Improvements over those parents | 21 | 19 |
| Parent improvements with zero archive reward | 10 | 6 |
| Parent losses with positive archive reward | 14 | 20 |

These are different estimands, not inconsistent scores. A worse child can fill an
empty top-ten slot or displace another archived molecule and earn archive reward.
A locally improved child can remain outside the top ten and earn none. The
archive-AUC objective is legitimate; the information loss comes from using this
scalar reward, pooled by option label, as the only task-specific WHAT statistic.
Do not replace the objective or hard-prune declining intermediates on this basis.

Concrete adaptive record `attempts/0052`: after four previous attempts of
`construct:pendant:6:5,1,0:aromatic:0`, its probability on this parent was raised
from 0.0099777778 to 0.2399261013. Its parent scored 0.0834612883 and its endpoint
scored 0.0041881415. At the end of the run, this option remains second by pooled
estimated credit, despite seven endpoint scores of 0.3630, 0.1292, 0.00556,
0.0000138, 0.00419, 0.0000267 and 0.1254. Descendants of its first use contribute
substantial credit, then that label-level estimate transfers to other contexts.
This demonstrates context-blind credit transfer. It does not prove all those
alternatives were known bad in advance or identify an unobserved optimal edit.

Proposed next intervention: predict outcomes conditional on the actual molecule
and proposed edit/option, learning from raw measured scores as well as archive
returns. Check chronological predictive performance on saved observations before
using a new predictor for guidance. Such an offline check cannot establish
counterfactual policy improvement. A later prospective same-generator comparison
must test whether guidance improves the locked objective, preserve exploration
and complete-option execution, and charge every new oracle observation. No new
ring templates, free winner labels, or support/benchmark changes are implied.

Artifacts: `diagnostics/pmo_feedback_audit/result.json`, produced by
`tools/pmo_feedback_audit.py`. It records input/source hashes, code revision,
software, tolerance, exclusions and every audited attempt. Reproduce with:

```sh
python tools/pmo_feedback_audit.py --adaptive-snapshot /absolute/adaptive_topology_snapshot --output /absolute/result.json
```

The optional snapshot is the already downloaded prefix. Omitting it still audits
both complete results, but does not replay individual selection rows. Two focused
production-allocation tests passed in 1.40 s; Ruff and `git diff --check` passed.
No production controller changed and no full-suite/milestone claim is made.
