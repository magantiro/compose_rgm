# PARP1 Full versus NoDistill read-only audit v1

## Outcome

This audit localizes the observed medium/large candidate loss to selection and
continuation, not docking. Across 264 durable selections, 263 immutable query
receipts are complete and scored. The only absent receipt is Full `parp1_2_d04`
round 3 query `q06`, a small-band proposal. All five medium and the one large
round-3 selections have complete receipts.

The comparison is retrospective and zero-oracle. It reads already durable
artifacts and makes no causal claim that trajectory distillation alone explains
the outcome. It made no remote mutation, launch, resume, cancellation, or oracle
call.

## Band accounting

Selected counts are shown as small/medium/large for each durable round:

| Cell | Full | NoDistill |
| --- | --- | --- |
| `parp1_0_d04` | `3/4/1, 4/1/3, 3/5/0, 6/1/1, 4/0/4, 7/0/1` | `4/3/1, 3/2/3, 2/6/0, 6/2/0, 5/1/2, 6/1/1` |
| `parp1_1_d04` | `3/4/1, 1/4/3, 7/1/0, 7/0/1, 8/0/0, 8/0/0` | `3/4/1, 4/3/1, 3/4/1, 8/0/0, 5/3/0, 6/0/2` |
| `parp1_2_d04` | `1/6/1, 4/3/1`; unsettled round 3: `2/5/1` | `4/3/1, 3/3/2, 2/1/5, 6/1/1, 3/1/4, 5/1/2` |

These labels do not all have the same provenance:

- NoDistill stores native `proposal_scale_band` on every selected row.
- Full route rows store native `realized_primitive_band` when they retain a
  protected primitive count.
- Full shallow and anchored rows have no stored band. Their labels here are
  retrospective, using `extent = max(1, created + deleted, regions)`, with
  small at most 3, medium at most 11, and large otherwise. These labels were
  not used by the Full controller.

The clearest collapse is Full `parp1_1_d04`: medium/large selection changes
from `4/1, 4/3` in rounds 1 and 2 to `1/0, 0/1, 0/0, 0/0`. NoDistill still
selects medium candidates in round 5 and large candidates in round 6. Full
`parp1_0_d04` loses medium selections in rounds 5 and 6 but retains large
selections.

## Champions and ancestry

Lower scores are better. Gaps are left score minus right score.

| Cell | Full settled | NoDistill | IVG | Full - NoDistill | Full - IVG | NoDistill - IVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `parp1_0_d04` | -13.1 | -11.3 | -14.1 | -1.8 | 1.0 | 2.8 |
| `parp1_1_d04` | -13.7 | -11.9 | -13.4 | -1.8 | -0.3 | 1.5 |
| `parp1_2_d04` | -11.1 | -11.6 | -9.0 | 0.5 | -2.1 | -2.6 |
| mean | -12.6333 | -11.6000 | -12.1667 | -1.0333 | -0.4667 | 0.5667 |

Full `parp1_2_d04` remains settled at -11.1. Its frozen but unsettled round-3
best receipt is -11.8. Substituting that receipt only as a labelled diagnostic
gives a Full mean of -12.8667 and a mean Full-minus-IVG gap of -0.7000. It must
not be reported as the checkpoint champion.

The full parent chains in `result.json` establish that final proposal provenance
alone is insufficient:

- Full `parp1_0_d04` ends with a shallow proposal at -13.1, but its ancestry
  contains a round-2 route proposal at -12.7.
- Full `parp1_1_d04` ends with a shallow proposal at -13.7, after route-derived
  ancestors in rounds 1 and 2 at -13.0 and -13.3.
- Full `parp1_2_d04` settles on an anchored proposal at -11.1 after a round-1
  route ancestor. Its unsettled -11.8 anchored receipt also has a round-2 route
  ancestor.
- None of the three NoDistill champion chains contains a
  `route_complete_region` ancestor.

Thus the Full p1_0 and p1_1 winners are shallow endpoints but route-seeded
descendants. Describing them as direct shallow-only successes would erase their
measured ancestry.

## Exact limitation

`selection_input.json` persists exact eligible counts only by expert. It does
not persist row-level eligible candidates or eligible band/family labels. Full
particle-reduced route candidates are also not published as one final row-level
post-merge pool. Exact post-merge eligible-by-band counts are therefore
unrecoverable from the sealed artifacts. The audit reports exact expert totals,
exact selected/docked bands, and a separately labelled direct-receipt census;
it does not fabricate the missing post-merge band census.

The IVG source is `diagnostics/t4_win_audit.json`, file SHA-256
`c07b15299a64da21c21723f17b7813dca860b24d31d316323f6e5f5a04aa5b20`.
That source flags the p1_2 reference record, and the audit preserves that flag.

## Reproduction and verification

The machine-readable authority is `result.json`, payload SHA-256
`383e80f17442373c6f69639caaf1c0e48856d91b362f1f3331767446fb351e7d`.
It binds the two run IDs, six volumes, launch revisions and hashes, checkpoint
file and payload hashes, every durable selection/plan/lock payload hash, the
seven present unsettled receipt hashes, and the one exact missing receipt path.

Regenerate and byte-check it with:

```bash
python3 diagnostics/t4_compose_nodistill_parp1_v1/full_vs_nodistill_audit_v1/build_audit.py
python3 diagnostics/t4_compose_nodistill_parp1_v1/full_vs_nodistill_audit_v1/build_audit.py --check
```
