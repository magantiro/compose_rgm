# Interim PARP shared-controller audit

This is a zero-oracle audit of a running campaign snapshot. It is not a final campaign result.

| Cell | Calls | Best | IVG mean | Gap | Calls left |
| --- | ---: | ---: | ---: | ---: | ---: |
| parp1_0 | 33 | -9.2 | -12.3 | +3.1 | 16 |
| parp1_1 | 25 | -12.6 | -11.7 | -0.9 | 24 |
| parp1_2 | 25 | -9.9 | -10.7 | +0.8 | 24 |

## Measured mechanism

- Every known PARP delta=0.6 teacher-region template is present in the live shared checkpoint.
- PARP1-1 entered a strong basin through a complete-region route proposal, then FiberControl selected shallow descendants that improved it further.
- The weaker cells therefore require analysis of binding, composition, ordering, and scored basin quality rather than another compiler change.

## Evidence boundary

Teacher matches are diagnostic only. The live controller received one shared template distribution, not a cell-to-winner lookup. Docking scores come only from preserved campaign receipts.
