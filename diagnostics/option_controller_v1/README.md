# Option-controller V1 retrospective gate

The reusable option-boundary controller is implemented, but the existing
policy-matched PMO continuations do not support fitting its improvement value.
Among 600 horizon-threshold target cells, 288 are identified after correct
right-censoring and all 288 are negative. A fitted classifier would therefore
be a trivial all-negative head, not evidence of ranking or calibration. The
gate abstains.

The input is the stored balanced-reference arm of the completed Perindopril MPO
option-particle experiment. Optimized arms are not relabeled as reference
rollouts. The audit used 50 option-boundary rows from four source groups and
eight terminal lineages, and spent zero new oracle calls. The exact input hash,
behavior-policy identity, option census, software versions and decision are in
`report.json`.

Reproduce locally with:

```bash
PYTHONPATH=src python tools/option_controller_retrospective.py \
  --input /private/tmp/compose-pmo-option-particle-runs/30fb08d196021d7f8a8d524c86b17c74532b72feac92cebaa1c003322e96441f/result.json \
  --output diagnostics/option_controller_v1/report.json
```

Next decision: retain the controller implementation and collect only the
smallest policy-congruent continuation set that contains both improving and
non-improving outcomes. Do not scale particles or deploy learned guidance from
this all-negative bank.
