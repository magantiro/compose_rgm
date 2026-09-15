# T4 selector utility prelock

## Scope

This zero-oracle milestone freezes a very small prospective development panel
from the immutable complete-macro selector order. It does not change the
generator, selector, candidate pools, delta-0.4 eligibility rule or the existing
four-request raw utility lock. The existing baseline-learned requests are the
raw-order counterparts.

The selector architecture was selected using offline held-teacher component and
exact-patch diagnostics. A later scored result would therefore be development
evidence, not an independent final benchmark.

## Frozen selection

The self-hashed contract is
`configs/t4_selector_utility_prelock_v1.json`, with payload SHA-256
`0f0958b9aa09a09f6de18e77018cc6be5264f7643f7aa290fe59ebce40f54143`
and physical SHA-256
`6ea268cdd67d67d06a1a058408619cca223148fb4189b304abc5766a91e424e3`.

For each of the five previously declared cells, the reducer joins the immutable
selector rank to the prior baseline-learned eligibility ledger by exact
candidate identity. It selects the first eligible selector-ranked endpoint only
after excluding every candidate and physical docking-request identity already
in the raw lock. It does not recompute eligibility or inspect a task or docking
score.

The reduction locked two new requests:

| Cell | Selector rank | Raw generator rank | Status |
| --- | ---: | ---: | --- |
| `jak2_1` | 1 | 36 | locked |
| `parp1_0` | 2 | 22 | locked |
| `5ht1b_0` | | | no eligible baseline-learned candidate |
| `braf_1` | | | no eligible baseline-learned candidate |
| `fa7_0` | | | no eligible baseline-learned candidate |

The existing raw panel remains four unique requests. This milestone adds no raw
counterpart and has a maximum future call ceiling of two.

## Artifacts

Authoritative outputs are under
`diagnostics/t4_selector_utility_prelock/attempt_1`:

- request lock: physical SHA-256
  `9802da3b356ae5f41460a29377640c9a0e3fb866f2e82c46ea581624074ba046`,
  payload SHA-256
  `6d49e2a397b662a4425a4a5ee0d17a6223d64c4ac3e0effa594dc8d8c1c52b29`;
- selected candidate lock: physical SHA-256
  `715302518749e3120041f48a515e1c94e0df13631a1f423f9aa06554d48b9a00`,
  payload SHA-256
  `b76b41c2d1400ccc115798cc21b02a5768c24de4eea381628cc8ae13c42fc549`;
- result: physical SHA-256
  `b2ff9020630caa2b702cff16deddf16c9ac2ca5b8cdc079ad4a5a1be56fb8b25`,
  payload SHA-256
  `922d688b781e55e71a0054f98fc1862e4e318ee9a7fc04257072dea9cce2cf21`.

A complete independent rerun produced byte-identical candidate, request,
abstention, result and interpretation files. Eleven focused selector/prelock
tests passed. The reducer made zero oracle or docking calls, launched Modal zero
times and accessed no live run.

## Launch boundary

The request lock is not launch authority. Scoring requires a later explicit
authorization naming the exact request-lock physical and payload hashes and the
two-call ceiling. Retries, replacements and backfill remain forbidden.
