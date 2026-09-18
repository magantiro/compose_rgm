# Anchored-replacement proposal-support comparison

## Outcome

The frozen zero-oracle gate passed. At the same 512-program budget per lane on
JAK2 seed 1 at delta 0.6, the new generic anchored-replacement lane produced 13
unique eligible amide-plus-diamine endpoints. The unchanged shallow and progressive
structured lanes produced zero each.

| proposal lane | unique eligible | eligible amide | eligible diamine ring | eligible basin |
| --- | ---: | ---: | ---: | ---: |
| shallow | 144 | 2 | 0 | 0 |
| progressive structured | 137 | 6 | 0 | 0 |
| anchored replacement | 69 | 31 | 25 | **13** |

All 192 independently seeded workers completed without failure. The sum of worker
wall times was 724.66 seconds for shallow, 1,082.98 seconds for structured and
244.22 seconds for anchored replacement. Modal executed the independent attempts in
parallel, so these sums are not elapsed launch time.

## What changed

The proposal expert composes two already supported Dynamic-v1 modules into one
protected program: delete a one-boundary pendant fragment, retain its anchor, then
construct a saturated five- or six-member C/N ring through that same anchor. Generic
carbonyl-anchor, hetero-fragment and inverse-size weights affect proposal probability.
No target name, route identity, endpoint, docking score or runtime lookup enters the
proposal law. The exact executor, endpoint gate and declared primitive support are
unchanged.

## Claim boundary

This is proposal-support evidence, not docking-utility evidence. It establishes that
the coupled proposal law enters an eligible structural basin that the two prior lanes
missed at the matched budget. It does not establish that any novel endpoint binds well
or beats InVirtuoGen. The next required step is exact historical identity matching,
followed by a prospectively locked small scored panel for the remaining novel endpoints.

## Provenance

- Contract payload SHA-256:
  `c4743d0b3621e4c23b387f823488b7172fefa59d99b0daac5906fe3bc297cd4b`
- Result payload SHA-256:
  `970fe1eba85307bd5000c0028625ac22d9cd416b81a5352580c831e906f503bf`
- Result file SHA-256:
  `6a9aef8d22611859a0fa2f1de78f64ef3bc743563dac1999b26acf12885c5985`
- Complete attempt ledger SHA-256:
  `97590ebf095978487ddd47aff0399524cf1fa13c81633cef6df93b231232db34`
- Candidate lock SHA-256 inside the result:
  `ad712979cd20765c36b09b2a78ae2f41752f688281fc4b7b7b5fc3a7f42d03b2`
- Code revision:
  `1a74071bb99ff61f316d2a126886b1906d8c49e1`
- New oracle calls: 0
