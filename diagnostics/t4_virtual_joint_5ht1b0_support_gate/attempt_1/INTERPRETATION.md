# 5HT1B-0 delta 0.4 virtual-joint support gate

## Outcome

Deferred final-target validation restores known-answer execution support for all
three strong routes, but the unchanged default autonomous proposal allocation
does not expose any of them. The known-answer support subgate passes at 3/3 exact
reconstructions and executions with zero primitive teacher actions. The complete
autonomous gate fails at 0/3 exact endpoint recovery and 0/3 radius-2
transformation-equivalent recovery.

This separates representation from proposal allocation. The joint runtime can
execute all three transformations when supplied their known constituents and
bindings. All nine constituents are also present in the autonomous root binding
census, but their marginal ranks are 58 through 89, outside the fixed expansion
width of 48.

## Known-answer execution and constituent support

The marginal template probability is `0.0040632603406326046` and the one-region
log score is `-6.6932130904422955` for each of these nine low-frequency
constituents.

| Route | Reported label | Teacher primitives | Joint runtime primitives | Constituent ranks | Event count / scale | Selected |
|---|---:|---:|---:|---|---|---|
| `2b0aff16...` | -13.1 | 16 | 16 | 88, 73, 89 | 34 large, 4 medium, 13 large | 0/3 |
| `88e87055...` | -14.0 | 22 | 11 | 58, 72, 82 | 31 large, 11 medium, 20 large | 0/3 |
| `bbd7db0d...` | -12.8 | 17 | 15 | 66, 80, 70 | 27 large, 11 medium, 4 medium | 0/3 |

All nine exact teacher bindings were enumerated. The `bbd7db0d...` region-1
binding census was truncated after finding eight bindings, but the exact teacher
binding was among them. No teacher constituent is absent because of binding
failure or the global binding budget.

The sequential same-root baseline remains 0/3 across all 18 region orders. The
pre-joint autonomous shard also recovered 0/3 strong routes among 61 committed
endpoints. Therefore the change repairs the earlier sequential representation
failure, but it does not improve strong-route autonomous recovery at the frozen
default allocation.

## Autonomous fixed-budget result

The source produced 101 unique bound constituents. The global marginal selected
48: 9 of 24 local, 23 of 53 medium and 16 of 24 large constituents. Planning
considered 48, 64, 64 and 64 prefixes at depths one through four. It made 240
final-target checks, with 214 final-target abstentions. Only depth-one targets
were valid: 26 checks yielded 17 unique retained targets. Exact realization
committed 14 of 14 successful targets, while three targets abstained with
`frontier_exhausted_abstention`. No jointly valid depth-two through depth-four
target survived the selected top-48 constituent set.

No global work limit was exhausted:

- binding visits: 5,882 / 16,384;
- planning expansions: 2,139 / 4,096;
- retained targets: 17 / 128;
- realization attempts: 17 / 128;
- realizer expansions: 41 / 65,536;
- per-realization expansion-cap abstentions: 0 at the 4,000 cap;
- primitive support: at most 32.

The negative autonomous result is consequently an allocation/support result, not
a work-budget truncation result.

## Legacy scale-balanced allocation diagnostic

A separate derived diagnostic applied the already-existing
`select_scale_balanced_proposals` round-robin semantics to the sealed 101-row
bound census at the same width of 48. It selected exactly 16 local, 16 medium and
16 large constituents, but still selected 0/9 teacher constituents. This
diagnostic made no binding, planning or realization call and did not modify the
proposer. Scale balancing alone therefore does not resolve the low-frequency
constituent exclusion.

## Template-balanced allocation diagnostic

A second distinct diagnostic used the same width of 48 and the same sealed 101
rows. It ordered the 27 bound templates by frozen marginal probability and
template ID, then selected one constituent per template before taking a second
from each template, while preserving the sealed constituent order within each
template. This generic template-balanced allocation selected 7/9 teacher
constituents. Its selected scale census was 8 local, 22 medium and 18 large.

The seven recovered constituents were each rank 1 of 1 within their template.
The two remaining exclusions were `2b0aff16...` region 2, rank 4 of 4 within its
template, and `bbd7db0d...` region 1, rank 5 of 8. Thus template coverage removes
the between-template frequency failure, but width 48 still does not cover deep
binding alternatives within two teacher templates. This was a sealed-census
allocation diagnostic only, not an autonomous planning or endpoint-recovery
run.

## Reproduction and claim boundary

The measured gate command was:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
tools/t4_virtual_joint_5ht1b0_support_gate.py \
--output diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_1/result.json
```

The machine-readable result was reproduced byte-for-byte in an independent
second execution. Its payload SHA-256 is
`bea8ae2f5028b8e6bcff3a3b2a25809bc293f2d9fb8db03607667c155aca0c32`.
The separate legacy-allocation payload SHA-256 is
`b70161434f5c06ba9f368b5b86ef856777619b7769f6899ad2dd4b78aff4de9b`.
The separate template-balanced allocation payload SHA-256 is
`ce6ab8d40b42e2ec3fe4f08ed0c179d90bde20b4e32bdedf86f961f8bca6a057`.

The known endpoints were used only in the separately labeled teacher-forced
execution and post-generation recovery measurements. They were not inputs to the
autonomous proposal call. This audit made zero oracle, docking, Modal or scored
artifact calls and supports no docking-utility or prospective-optimization
claim.
