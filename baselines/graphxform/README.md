# GraphXForm — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.** This file records what
was verified from primary sources and what an adapter would have to do.

## Identity

| field | value |
|---|---|
| paper | Pirnay, Rittig, Wolf, Grohe, Burger, Mitsos, Grimm — *GraphXForm: Graph Transformer for Computer-Aided Molecular Design* |
| venue | Digital Discovery 4(4), 1052–1065 (2025), DOI [10.1039/D4DD00339J](https://doi.org/10.1039/D4DD00339J), CC-BY |
| preprint | [arXiv:2411.01667](https://arxiv.org/abs/2411.01667) (v1 2024-11-03, v2 2025-03-20) |
| code | <https://github.com/grimmlab/graphxform> |
| license | MIT (`LICENSE`). Bundled `objective_predictor/GH_GNN_IDAC/` and `GDI_NN_IDAC/` carry their own separate licenses — check before redistribution |
| checkpoint | `https://syncandshare.lrz.de/dl/fiJs7ZHuCFsVskeoab5aZg/graphxform_pretrained.zip`, ~347 MB, verified reachable (HTTP 200) |
| citation key | `pirnay2025graphxform` (already in `paper_iclr_stochastic_rewriting/references.bib`; the entry lacks the arXiv id) |

## What was verified

- **Action space is strictly constructive.** `AddAtom`, `AddBond`, `DontChange`.
  There is no atom or bond removal; `molecule_design.py::take_action` only
  appends. The paper states removal is future work (§3.3.3).
- **Exact source conditioning exists** via `config.start_from_smiles`, dispatched
  in `core/gumbeldore_dataset.py`; `MoleculeDesign.from_smiles(..., do_finish=False)`
  replays the molecule as an action prefix without terminating, so the model
  continues editing the supplied molecule.
- **Per-objective fine-tuning is required** for the paper's reported results:
  self-generated top-`s` sequences become cross-entropy pseudo-labels
  (`main.py::train_for_one_epoch`). Only the last linear layers are unfrozen, but
  they *are* gradient-updated per objective. A pure-search mode exists
  (`num_epochs = 0`) and is not what the paper reports.
- **Action masking is genuinely per-step** (`update_action_mask` recomputed after
  every action; `logits[mask] = -inf` pre-softmax) but covers only **valence,
  atom type, atom count, no self-bonding, no re-bonding**.
- **Substructure constraints are a terminal filter, not a mask.** The paper's
  ring-size and disallowed-bonding-pattern constraints are implemented in
  `molecule_evaluator.py::infeasible_by_special_constraints`, asserted on
  `mol.synthesis_done`, returning `-inf`. There is **no SMARTS or substructure
  matching anywhere in the repository.**
- **No objective switching.** `objective_type` is a single immutable config
  string read once at evaluator construction.

## Adapter work required

1. Objective shim mapping the COMPOSE frozen oracle (`src/compose_v4/drd2_oracle.py`
   and the frozen goal language) into `MoleculeObjectiveEvaluator.predict_objective`,
   with an oracle-call counter wrapping every call including the ones consumed by
   dataset generation during fine-tuning.
2. Source-panel driver that sets `start_from_smiles` per source. Note that the
   `start_from_smiles` branch creates **one** root instance and does not honour
   `repeat_start_instances`, so multi-seed runs need an outer loop.
3. Atom vocabulary alignment: GraphXForm's allowed-atom set must be restricted to
   the COMPOSE element set, or the deviation recorded.
4. Trajectory export: `molecule.history` plus per-state SMILES, so intermediate
   states can be scored for pathwise metrics.
5. Guard against the documented `data/generated_molecules.pickle` merge-across-runs
   behaviour — it silently continues from a previous run's file.

## Known friction (from the official repo)

- `guacamol` import fails out of the box (`scipy.histogram` removed); README
  documents the one-line patch.
- `torch_scatter` must be installed manually for the local torch/CUDA build.
- `Dockerfile` pins `torch 2.5.1` while `requirements.txt` demands `>=2.6.0`.
- README says `beam_width` defaults to 16; shipped `config.py` says 32; the paper
  used 512.
- Only one issue has ever been filed (a pretraining OOM, closed).

## CPU feasibility

The shipped `config.py` already defaults to CPU (`training_device`,
`objective_gnn_device`, `devices_for_workers`), so CPU-only operation is
*supported*. The paper ran on a single H100 at beam width 512; no CPU runtime is
published, so any CPU estimate here is a projection, not a measurement.
`main.py` derives `num_gpus` from `CUDA_VISIBLE_DEVICES` before `ray.init`, which
must be set explicitly in a CPU container.
