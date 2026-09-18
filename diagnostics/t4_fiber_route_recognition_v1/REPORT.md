# FiberControl complete-route recognition diagnostic

## Result

The current `ProgramValue` model did not reliably recognize injected, historically
high-value complete programs. With each probe score and every later outcome excluded
from fitting, the model ranked the 5HT1B probe **30/30**, the BRAF probe **5/5**, and
the FA7 probe **2/6**. Lower rank is better.

| Cell | Historical probe score | Earlier scored outcomes | Ordinary shallow distractors | Pessimistic rank |
| --- | ---: | ---: | ---: | ---: |
| `5ht1b_0` | -12.4 | 5 | 27 | 30/30 |
| `braf_1` | -11.1 | 7 | 4 | 5/5 |
| `fa7_0` | -9.2 | 7 | 5 | 2/6 |

The frozen contract's literal promotion rule, top-5 rank on at least two of three
probes, evaluates to true. That rule is not a useful headline conclusion here because
the realized BRAF and FA7 menus contain only five and six candidates. In ordinal
terms, the model placed two probes last and recognized only the FA7 probe well. This
degenerate gate is preserved rather than changed after seeing the data.

## What was tested

The probe in each cell was selected by a fixed retrospective rule: choose the
lowest-scoring exact complete route with at least four earlier same-cell outcomes,
which is the minimum history at which `ProgramValue` fits. The candidate menu contains
the injected historical complete program plus unique current-code shallow proposals
from the same root. The complete protected program is one FiberControl decision. Its
primitive intermediate states are not treated as queryable decisions.

Two views are recorded:

1. Frozen recognition immediately before the probe outcome, using only lower query
   indices from that cell.
2. Chronological replay after each earlier outcome. The probe's score is never used in
   fitting or ranking, and no later score enters the fit.

The chronological result reinforces the frozen result. Once fitting begins, the 5HT1B
probe is rank 30/31 and then 30/30. The BRAF probe remains last as its menu contracts
from 8 to 5 candidates. The FA7 probe moves between ranks 3/9, 5/8, 2/7, and 2/6.

## Interpretation

This is a negative recognition result for the current value representation, not a
negative result for the integrated controller. Injecting a known-good complete program
does not make `ProgramValue` reliably prefer it over ordinary proposals. Improving
proposal support alone therefore may not be sufficient: the online controller should
retain explicit exploration or proposal-family quotas until value learning has enough
direct evidence to discriminate the new route expert.

This diagnostic does not measure autonomous route proposal probability. It also does
not justify training on the probe answers. Its legitimate use is to show that the
current value model is not already an oracle for known route programs.

## Evidence and provenance

The authoritative machine-readable result is [`result.json`](result.json). Its payload
SHA-256 is
`55c604a801e73c26ce43f2341b80a5be60f46bc17f7133a4c8bbb384c443ebe6`, verified after
publication. The analyzer ran from revision `27628d0f316f37d29b2c0295053fc80775a76ced`
with RDKit `2026.03.6`. The result records the frozen contract hash, analyzer and
FiberControl source hashes, exact program and receipt identities, and zero oracle,
docking, and Modal calls.

Historical complete-route artifacts do not contain FiberControl intervention-family
labels. Those features were set to zero rather than inferred from endpoint or winner
identity. The result therefore applies to the feature support actually recoverable
from these archives. Exact multi-boundary trajectories with scored intermediate
endpoints were not locally available, so the diagnostic correctly abstains from
claiming sequential route-following accuracy.
