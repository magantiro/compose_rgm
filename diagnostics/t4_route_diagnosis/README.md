# Saved-route diagnosis

The frozen guide prefers the incumbent to every completed stage of the known
route, including the IVG endpoint. Removing intermediate similarity penalties
alone does not fix this ranking. This is a retrospective, answer-known diagnostic,
not autonomous winner recovery or a docking result.

Authoritative output: `result.json`, SHA-256
`6dd20679073978791ec6bd97a9583a203cf92810b7695124a8976cd2f85480fa`.
Its input hashes, source dependency closure, versions, exact-state identities,
configuration and provenance are recorded in the artifact. Analysis code is
committed at `dc6acdc`; production scoring imports came from clean `d86c055`.

| Completed stage | Cumulative primitive edits | Predicted docking | Eligible endpoint |
|---|---:|---:|---|
| Remodel linker | 4 | -7.61625 | yes |
| Add pendant benzene | 11 | -7.88722 | yes |
| Add fused six-membered ring | 16 | -7.97256 | yes |
| Add peripheral carbonyl | 17 | -8.03331 | yes |
| Remodel core carbonyl, exact known endpoint | 21 | -8.15650 | yes |

The incumbent's predicted score is -9.14809. Lower is preferred, so the guide
ranks even the known endpoint 0.99159 above, and therefore worse than, the
incumbent. These are predictions, not observed docking values. This diagnostic
does not itself verify the published endpoint docking score in our pipeline.

All five completed program stages satisfy the unchanged endpoint screen.
Primitive intermediates 18, 19 and 20 do not; those states are internal to the
last program. The result distinguishes a primitive-level feasibility barrier
from the separate guide-ranking problem at completed option boundaries.

The known endpoint SMILES is:

```
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21
```

The incumbent SMILES is:

```
Cc1nc(=O)nc(-c2ccc3n2-c2c(F)cc(CN(C)C)cc2CNC3=O)n1C
```

Reading their molecular graphs exposes different peripheral ring chemistry,
linker arrangement and core carbonyl structure. Fingerprint similarity alone
does not describe those differences. The five-stage witness starts at the
original benchmark seed, not this warm incumbent. It is therefore neither a
shortest-path proof nor a mapped route from the actual current search root.

The analysis scored 22 saved states in 1.352 seconds including input checks,
excluding Python imports. It made zero new docking calls, learned-law calls,
executor replays or model updates. Frozen training features and the saved
conditional-option predictions reproduced before publication. Three focused
analysis tests passed; lint and format checks passed. No repository-wide
milestone completion is claimed.

Next: dock the complete fixed 16-candidate eligible repair census without
surrogate selection. This tests missed value in proposals we actually have;
it does not establish autonomous traversal of the known route. Keep winner
comparisons diagnostic and out of proposal generation or model fitting.
