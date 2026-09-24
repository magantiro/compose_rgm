# Matched linker diagnostic, not an official benchmark row

The exact matched-arm result is
`matched_analysis_with_provenance.json` (SHA-256
`1260fc923f1c28fa5fa298e13265e3941233d4d2201585d6a6544dcb5b6469a0`).
It records the physical SHA-256 of all 60 input shards, the analysis-script
hash, source revision, sampler identity, checkpoint identity and per-arm
denominators. Both arms use ten released linker prompts, three seeds, 100
attempts per prompt and a one-atom *seeded* bridge. They differ only in the
`path_program` controller switch. This is retrospective diagnostic evidence;
the official linker row remains withheld.

| Outcome | Attachment only | Coherent path program |
| --- | ---: | ---: |
| Attempted trajectories | 3,000 | 3,000 |
| Committed, connected, chemically valid endpoints | 2,217 | 422 |
| Endpoints longer than the seeded bridge | 15/2,217 (0.68%) | 421/422 (99.76%) |
| Task success / attempted | 73.90% | 14.07% |
| Official-evaluator quality, percent of attempts | 17.17% | 4.07% |
| Mean emitted diversity | 0.52 | 0.44 |

The attachment-only arm largely inherits the benchmark adapter's bridge:
2,202 of its 2,217 committed endpoints still have exactly one internal linker
atom. Calling its 73.90% task success *linker design* would therefore be
misleading. The coherent path program produces genuine length changes, but
the hard path-progress gate leaves most attempts without a committed endpoint
and loses 13.10 quality percentage points. Both arms retain 100% chemical
validity **conditional on commitment**; 14.07% is not chemical validity.

The path arm records 516 successful composite transactions, 426 transaction
refusals after obtaining a candidate payload, and 11,117 per-event path
rejections. These counters do **not** yet identify how often no prior-sampled
insertion payload was found. The next causal audit must separate unavailable
site, absent payload, insertion failure, reroute failure, lock failure,
controller failure and lack of path lengthening. No new linker operator or
benchmark claim is justified by this aggregate alone.

The composite is independently classified as a proposal-support extension on
eight of the ten prompts and an acceleration on two in
`diagnostics/fragment_path_v3_classification_v1.json`. Thus the comparison
cannot be presented as simply changing allocation among existing model
actions. The one-atom bridge is a constructed starting condition, not a
generated linker, and must remain explicit in any future table.
