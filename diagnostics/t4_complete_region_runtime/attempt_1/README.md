# T4 complete-region runtime, attempt 1

## Outcome

The bounded zero-oracle correction passed both authorized gates. The deployed
runtime now accepts one to four complete address-free dependency-region target
patches per protected program, rather than one or two primitive-like construction
events.

- Runtime support: 77/77 Full-146 complete routes, comprising 147 region
  decisions. The decision-count distribution is 26 routes with one region, 35
  with two, 13 with three, and 3 with four.
- Teacher-forced runtime realization: 77/77 routes committed exactly one endpoint;
  all 77 canonical endpoints equal the teacher endpoint.
- Precision among committed endpoints: 77/77 (1.0).
- Teacher primitive actions used by the runtime: 0.
- Oracle calls, docking calls, Modal launches, network calls, and GPU seconds: 0.

These are answer-known representation and realization gates. They do not show
that a learned policy can autonomously propose the region patches.

## Runtime semantics

Each decision contains a whole `StructuralSubgoal` patch: source-region roles,
target topology, atom and bond attributes, boundary attachments, and dependencies.
All decisions except the last carry `CONTINUE`; the final decision carries `STOP`.
The runtime publishes and locks only the completed endpoint.

Independent regions are combined and realized globally at `STOP`. If a later
region consumes an atom created by an earlier region, it uses a relative
`(prior_region_index, output_role_index)` reference. Those dependent regions are
realized sequentially inside one protected transaction. Intermediate states are
never evaluated or locked; abort commits no endpoint and exposes no partial
primitive program.

## Authoritative artifacts

- Contract: `configs/t4_complete_region_runtime_v1.json`
  - physical SHA-256: `f3e01a74cb96050d50dcf171bb6107e95551f92fea6a0d3c5927186db6c4b972`
  - payload SHA-256: `18f630f6c4f6b2d9555f63afd3384a38ff32a8d604e94deddbe06fd5977af4b1`
- Support gate: `support.json`
  - physical SHA-256: `33c29e5369ec49c95af5cb309a02f9cb9388e5989ed51c69166ec82a6f0e43f1`
  - payload SHA-256: `e5c1310cd17719d104e60224f339710deea185c3d9492da022dd1a1fbadd9f03`
- Realization aggregate: `result.json`
  - physical SHA-256: `8ce954f8a17210f0334fa94162ab620210b22f6d3a352c2be084eb512c7abb90`
  - payload SHA-256: `141650bca69afa8e79faa84fe963ca7f50027f014d3d76e55572ce541e280a20`
- Per-route receipts: `routes/*.json`; `result.json` binds every physical and
  payload hash.

All 77 route payloads were recomputed under implementation revision
`9df8819af82f9baf4ca4f3007ba2ec8215a0e7d4`; the canonical payload mismatch
count was zero. Operational elapsed time is intentionally excluded from persisted
scientific payloads so reruns are byte-stable.

## Verification

Focused checks executed:

```text
pytest -q tests/test_complete_region_program.py tests/test_t4_complete_region_runtime_audit.py
6 passed

ruff check tools/t4_complete_region_runtime_audit.py tests/test_t4_complete_region_runtime_audit.py
All checks passed

black --check tools/t4_complete_region_runtime_audit.py tests/test_t4_complete_region_runtime_audit.py
2 files would be left unchanged

git diff --check
passed
```

The regression fixtures include a later region that references an atom created by
an earlier region, invalid early `STOP`, and protected abort atomicity.

## Deferred work

Learned factor ranks and autonomous corrected-runtime generation require new
split-first modeling contracts. They were not implemented, fitted, or evaluated
under this bounded milestone.
