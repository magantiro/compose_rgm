# Frozen integrated-controller BRAF transfer

## Question

Does the exact integrated controller frozen after the three-cell JAK2 development
run transfer to BRAF without a BRAF-specific proposal, feature, scheduler, or
selection change?

The controller remains:

```text
shallow + anchored_replacement + leave-target-out route_complete_region
  -> strict endpoint fiber
  -> shared FiberControl value model
  -> locked docking batch
  -> current-cell online update
```

Only the benchmark adapter changes: the three BRAF sources, BRAF receptor and
box, a split-clean leave-BRAF-out route checkpoint, and a new output namespace.
Every scientific controller field is copied from
`configs/t4_integrated_route_fiber_v1_1.json`.

## Prospective protocol

Run `braf_0`, `braf_1`, and `braf_2` concurrently at delta 0.6. Each cell has a
49-call ceiling consisting of one root and six batches of at most eight. The
same controller seeds, docking seed, proposal widths, parent law, two-call
exploration floor, two-round expert exposure, endpoint support, and no-retry
policy used for JAK2 are retained.

The route prior is fitted after excluding every BRAF route. Its runtime
checkpoint stores only address-free templates and a balanced marginal. It
contains no BRAF source, target name, endpoint, route identifier, executable
teacher route, or absolute atom address.

Candidate exhaustion is a valid transfer outcome. A cell with no eligible
non-root candidate stops rather than weakening the QED, synthetic-accessibility,
similarity, or 40-heavy-atom gates.

## Prelaunch negative evidence

The split-clean route expert generated and exactly realized complete programs on
all three BRAF roots, but produced zero strict-eligible route endpoints on all
three. This is preserved as a failed production-yield gate in
`diagnostics/t4_integrated_route_fiber_braf_v1/route_expert_smoke.json`. The
scored transfer does not relabel that gate as passing and does not add a repair.

An operational root census with the frozen seeds found no eligible shallow or
anchored endpoints on `braf_0` or `braf_1`; `braf_2` had three shallow endpoints.
This census was used only to anticipate abstention and did not change the
controller, budget, or launch set.

## Claim boundary

This is a prospective frozen cross-target development transfer under one docking
seed. It is not a full T4 benchmark, a held-out final evaluation, or a causal
ablation of route distillation or FiberControl. The anchored expert was motivated
by answer-known JAK2 development. Report abstentions and all charged root calls.

## Commands

```bash
PYTHONPATH=src modal run -d modal_apps/t4_integrated_route_fiber_braf_app.py --mode launch
PYTHONPATH=src modal run modal_apps/t4_integrated_route_fiber_braf_app.py --mode status --run-id RUN_ID
```
