# Authoritative current-line T4 inventory, 2026-09-19

The current shared-controller development line now has a scientifically comparable settled result or settled checkpoint for all 30 T4 cells. Twenty-nine cells are terminal: 26 used the full 49-call budget and three terminated by candidate exhaustion. One cell, PARP1 seed 2 at delta 0.4, remains incomplete at its last settled 17-call checkpoint.

Among the 29 terminal cells, COMPOSE is at or better than the corrected published InVirtuoGen (IVG) mean on 23 cells: 22 strict wins and one tie. It is worse on six. The incomplete PARP1 cell is already better at its settled checkpoint, so 24 of 30 cells are at or better than IVG if that nonterminal checkpoint is shown separately.

This is not yet one frozen full-benchmark result. It is a development mosaic spanning nine run IDs and multiple controller revisions. The numbers are scientifically comparable evidence from the same shared-controller line, but they must not be advertised as a single prospectively frozen controller evaluated across all 30 cells.

## Exact terminal misses

| Cell | Delta | COMPOSE | IVG | Gap | Status | Mechanism visible in evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| JAK2 seed 0 | 0.6 | -9.6 | -9.7 | +0.1 | 49/49 complete | shallow/exploration champion; small full-budget reward miss |
| PARP1 seed 0 | 0.4 | -13.1 | -14.1 | +1.0 | 49/49 complete | shallow/model champion; full-budget reward miss |
| BRAF seed 0 | 0.6 | -8.9 | -9.7 | +0.8 | candidate exhaustion at root | no queryable eligible descendant entered scoring |
| 5HT1B seed 0 | 0.4 | -12.6 | -13.3 | +0.7 | 49/49 complete | route-scale-floor champion; full-budget reward miss |
| 5HT1B seed 1 | 0.4 | -11.8 | -12.0 | +0.2 | 49/49 complete | shallow/model champion; small full-budget reward miss |
| 5HT1B seed 2 | 0.6 | -9.8 | -10.6 | +0.8 | candidate exhaustion at root | no queryable eligible descendant entered scoring |

Lower docking score is better. Gap is COMPOSE minus IVG, so a negative gap is a COMPOSE win.

## Minimal next panel

For matrix completeness, the only missing action is robust continuation of PARP1 seed 2 at delta 0.4 from the exact settled 17-call checkpoint to either 49 calls or a recorded terminal state. Its settled best is -11.1 versus IVG -9.0. A later frozen -11.8 observation was not settled and is excluded.

For the stated all-cell performance objective, the narrow development panel is the six terminal misses above. The two exhaustion cells are proposal/eligibility-support failures. The other four consumed the full budget and require reward/proposal improvement rather than continuation bookkeeping.

For a final paper-level benchmark claim, the selected controller must then be frozen and evaluated unchanged across all 30 cells. The present multi-revision development mosaic cannot substitute for that prospective frozen campaign.

## Evidence hierarchy

1. Settled immutable terminal per-cell result payloads.
2. Settled immutable campaign summaries, with hashed per-cell results used for champion provenance.
3. The last settled immutable checkpoint for the one incomplete cell.
4. Frozen but unsettled observations are excluded.
5. Older Dynamic, Full-146, route-aware, and NoDistill results remain separate development or ablation evidence.

The machine-readable ledger is `result.json`. It records the exact run ID, contract hash, code revision, input hashes, current status, comparator, gap, and champion proposal provenance for every cell.
