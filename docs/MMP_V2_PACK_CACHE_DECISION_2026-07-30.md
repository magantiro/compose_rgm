# MMP exact-state pack completion and V2 reuse decision

## Outcome

The content-addressed exact-state MMP pack completed successfully on 2026-07-30. It is authorized as
an immutable development artifact and potential pair-local compile cache. It is not the final
editing-V2 corpus and does not authorize Gate 0, T1, or training.

The authoritative local completion artifact is:

`diagnostics/coherence/mmp_v2_packer_complete_f942fc2315cf9210f4c7_2026-07-30.json`

Its SHA-256 is:

`1a09c4944fb73a72ceddea47c2a8579c27bbd1ca46e5ac47e1e3e5987425e9b4`

## Verified completion facts

- launch commit: `7d3069c15acb283dc117187b90230b7b1003afb2`
- authorization gate: `e1164073dea09cf149682b57aeaa04b3a3b67a7ae8f88b230811385b2048ad5c`
- input identity: `222e11a2abc0ffc56501c8ab387ab0839650f355ef366ae7d3ad4274c1e75b8b`
- output identity: `f942fc2315cf9210f4c7e120ea7db374833e8fe742e85692d329ba11a0b603f2`
- frozen source-pool rows: 344,153
- packed traces: 341,817
- packed progress states: 2,243,770
- physical shards: 17
- duplicate row identifiers: 0
- source overlap under the legacy split: 0
- scaffold overlap under the legacy split: 0
- reducer-audited entries: 167,093

The pack rejected 2,336 rows only because their endpoints straddled the legacy three-role split. The
completion artifact records 2,265 delete-plus-insert paths and 71 Graft paths among those rejects. It
does not contain atom-count or graph-cycle-rank deltas for them.

## Scientific boundary

This derivative uses the legacy `train`, `validation`, and `test` roles, the `ringcore-v1` split salt,
and caps applied before the final editing-V2 structured split. The final editing-V2 design instead
requires five evidence lanes, four roles, component assignment before path admission, role-local
density controls, and a newly sealed final holdout.

Direct reuse or relabeling would preserve pre-split selection coupling and is therefore prohibited.
The 2,336 legacy straddlers must return to the final candidate and component census rather than remain
a permanent exclusion overlay.

## Authorized reuse

An exact packed trace may be reused as a pair-local compile-cache hit only after the final V2 admission
process independently selects the endpoint pair and verifies exact agreement of:

- endpoint molecular identities;
- compiler identity;
- support and operator contracts;
- executor and canonicalizer identities;
- codec and runtime identities;
- exact replayed source, intermediate, and target states.

Final V2 record envelopes, partition assignments, packed shards, Active8 decisions, successor-fiber
caches, sampler streams, and all address-bound gates must be rebuilt.
