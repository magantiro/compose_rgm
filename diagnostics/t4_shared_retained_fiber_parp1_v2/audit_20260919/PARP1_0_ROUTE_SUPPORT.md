# PARP1-0 production-route support probe

This is a zero-oracle, answer-known diagnostic. It does not modify a live campaign.

| Route | Marginal rank | Single | Expansion | Frontier | Pool | Exact realization |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `3099a7787e15` | 24 | 3 | 9 | 9 | 9 | committed, 0.284s |
| `8008b5b0d18d` | 29 | 5 | 15 | 15 | 15 | committed, 0.281s |
| `8b0a15e5c56e` | 18 | 4 | 12 | 12 | 12 | committed, 0.178s |

## Decisive mechanism

The teacher programs are present, source-bound, highly ranked by scale-balanced production ordering, inside the first 96 realization slots, and exact-realizable. They are lost only because the route expert realizes the entire prefix serially and returns atomically; a later pathological proposal blocks completion until the whole worker times out, discarding already completed teacher candidates.

The first diagnostic timebox in production order is rank 34. The preserved live root worker also failed at its 1,800-second function timeout.

## Smallest generic fix

Isolate each route-program realization behind a bounded per-candidate deadline and persist/collect successful candidate receipts incrementally in the existing ranked order. A timed-out proposal should abstain without erasing earlier committed programs.

This fix is proposed only. It was not implemented or launched by this audit.
