# Ablation recipes

Each comparison changes one declared part of the method while keeping its
listed inputs fixed. The molecular output remains a complete, executable graph
within the supported atom, bond, size and charge rules. Task evaluators can
impose narrower endpoint conditions. For a paired comparison, match input
identities, seeds and budgets, and require completed receipts for both arms.

| Comparison | Arms | What changes | Run status |
| --- | --- | --- | --- |
| Superstructure edit law | learned, uniform family and mark | primitive selection probabilities | local runner, verified assets required |
| Fragment fixed-panel selection | deployed, uniform | selection on identical saved offers | replay command, verified assets required |
| PMO proposal construction | structured, uniform chain, created-atom rebinding | proposed executable programs | local runner, real oracle assets required |
| T4 reference selection | off, shadow, active | reference scoring and non-floor selection | panel preparer, real docking assets required |
| QED future control | local, future-aware | continuation policy | fitted value head and source workflow. A matched comparator is not frozen. |

The fragment, PMO and T4 recipes below run after their stated assets are
installed. QED's fitted value head is separate from the fixed molecular reference.

## Fragment edit probabilities

The superstructure comparison changes the edit-family and native-mark selection
law. Both tasks retain the same executable support, structural conditions and
learned stopping rule. The uniform law is not uniform over canonical molecules.

```bash
PYTHONPATH=src python -m compose_v4.experiments.fragments generate \
  --task superstructure_generation --assets local_assets/fragments \
  --attempts 100 --output runs/fragments/superstructure-reference

PYTHONPATH=src python -m compose_v4.experiments.fragments generate \
  --task superstructure_uniform --assets local_assets/fragments \
  --attempts 100 --output runs/fragments/superstructure-uniform
```

Both commands read the seeds and ten prompts from
[`generation.json`](fragments/generation.json). Each output records all attempted
slots, input hashes and the checkpoint identity. See the
[fragment guide](fragments/GENERATION.md) for the pinned environment and assets.

## Fragment selection on fixed panels

First generate a motif, decoration or linker panel with the fragment runner.
Then replay the reference-scored selector and a uniform selector against the
same saved eight-offer panels:

```bash
PYTHONPATH=src python -m compose_v4.experiments.fragments selection-ablation \
  --input runs/fragments/motif-reference/result.json \
  --assets local_assets/fragments --seed 17 \
  --output runs/fragments/motif-selection.json
```

For example, `motif-reference` can be the output of `generate --task
motif_extension --attempts 100`. Repeat with `scaffold_decoration` and
`linker_design` outputs. The command verifies the saved generation result,
provenance and shared checkpoint. It uses the same deterministic quantile for
both selection laws in each attempted slot. Empty panels stay empty. With
`--assets`, the pinned fragment evaluator computes metrics for each prompt and
seed. Without it, the command reports selected molecules and probabilities.

This is a retrospective selector comparison. Proposal construction is fixed
to the saved panels. In linker design, reference selection also retains the
deployed fourfold preference for previously unseen outputs, so this comparison
does not isolate neural scoring from novelty control. It is not a fresh
smaller-pool campaign.

## PMO proposal construction

The PMO runner accepts `structured`, `uniform_chain` and
`created_atom_rebinding` in the `arm` field. The three included
configurations change only that field. They hold the task, seed, budget,
initialization, oracle, checkpoint and guidance settings fixed. Give each run
a distinct output directory:

```bash
PYTHONPATH=src python -m compose_v4.experiments.pmo validate \
  --config experiments/pmo/example.json
PYTHONPATH=src python -m compose_v4.experiments.pmo run \
  --config experiments/pmo/example.json --output runs/pmo/structured
PYTHONPATH=src python -m compose_v4.experiments.pmo run \
  --config experiments/pmo/uniform_chain.json --output runs/pmo/uniform-chain
PYTHONPATH=src python -m compose_v4.experiments.pmo run \
  --config experiments/pmo/created_atom_rebinding.json --output runs/pmo/rebinding
```

`validate` makes no oracle call. Each `run`
does. All three arms can use the same frozen molecular reference. The
uniform-chain arm replaces structured molecular edit choices after successful
structured construction supplies the planned edit count. It does not remove
the learned reference. Created-atom rebinding changes a narrower part of the
program. See the [PMO guide](pmo/README.md) for receipts, incomplete runs and
the separate PyTDC environment. To prepare every objective and seed in the
22-objective protocol without scoring, run
[`tools/prepare_pmo_panel.py`](../tools/prepare_pmo_panel.py) as shown there.

## T4 reference selection

Prepare matched off, shadow and active configurations from the same lead panel,
program template priors and docking inputs. This step makes no docking call:

```bash
PYTHONPATH=src python tools/prepare_t4_panel.py \
  --output runs/t4/guidance-configs \
  --obabel /path/to/obabel --qvina /path/to/qvina02 \
  --receptors /path/to/receptors \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --base-seed 20260918 --replicates 3 --strength 0.25 \
  --guidance-modes off shadow active
```

The manifest lists every generated cell, arm, seed and configuration hash.
Off mode does not load the neural reference. Shadow mode loads and scores it
without changing selection. Active mode reweights only non-floor choices.
Every other search and docking field is copied unchanged across the three
arms. The strength above is illustrative and must be fixed before scoring.
`python -m compose_v4.experiments.t4 run --config <cell-arm.json> --output
<distinct-run-directory>` performs real docking. The [T4 guide](t4/README.md)
gives validation, local run and resume rules. The [offline panel
example](../examples/t4_reference_panel.py) checks selection without docking.

## QED future-value control

QED uses the same fixed molecular reference and a separately fitted
finite-horizon value head. The [QED guide](qed/README.md) gives the complete
source split, reference rollouts, head fitting, source evaluation and
reduction commands. The fitted head is bundled. A matched local-versus-
future-aware performance comparison needs that head and a frozen comparator
policy. Do not treat a reference-only successor sample as the full QED
ablation.
