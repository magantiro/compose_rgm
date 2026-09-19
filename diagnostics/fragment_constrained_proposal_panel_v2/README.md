# Fragment-constrained proposal panel v2

This frozen zero-oracle regression panel attempted one deterministic COMPOSE
proposal for each of the 50 released prompts. It preserves v1's negative audit
and changes only the generic superstructure site rule: an atom must have a
replaceable implicit hydrogen. The prompts, variant, support limits, attempt
count, validation, and metrics are unchanged.

The prospective contract is
`configs/fragment_constrained_proposal_panel_v2.json`, with payload SHA-256
`cad6f08a8534acd21f8b2a184b8cc3db606322a81f8ff1c40fe3411d270aaf0c` and
physical SHA-256
`8b2eaf148c730632068344fa2b724d1660c658f8e1c695a9dbdccc04041a5c38`.

## Computed result

| Task label | Coverage | Constraint precision | Exact-valid execution yield | Unique endpoints | Abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linker design | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| Scaffold morphing | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| Motif extension | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| Scaffold decoration | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| Superstructure generation | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| Overall | 50/50 | 50/50 | 50/50 | 40/50 | 0 |

Unexpected failures and constraint failures were both zero. Oracle, scoring,
and Modal calls were zero. Every endpoint independently rechecked exact receipt
consistency, all committed states for chemical validity and connected-or-null
structure, and the released fragment constraint.

The 40 overall unique endpoints do not indicate within-task duplication. The
released scaffold-morphing prompts reuse the linker-design inputs, and the
generic deterministic proposal law therefore emits the same ten endpoints for
those two labels. This one-sample support audit is not benchmark uniqueness.

The authoritative machine-readable artifact is `result.json`, SHA-256
`d3258d8253a869db94f96a7033998896e246d83fecfd49d8f55d4a10ac678905`.
This result contains no objective, quality, comparator, or 100-sample benchmark
evidence.
