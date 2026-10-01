# Fragment-constrained generation

Generate complete molecules while preserving a supplied fragment and its allowed
attachment interfaces. The shared executor checks every intermediate state.

Start with the [installation and run guide](GENERATION.md). Inputs are defined in
[assets.json](assets.json) and settings in [generation.json](generation.json).

## Tasks

| Task | Construction and selection |
|---|---|
| Motif extension | Construct eligible programs around a retained motif, then select using frozen reference scores |
| Scaffold decoration | Complete permitted attachment sites, then select using frozen reference scores |
| Linker design | Join fixed molecular boundaries, then combine reference scores with a preference for unseen outputs |
| Superstructure | Sample constrained primitive edits from the frozen reference law |

Scaffold morphing uses the linker output convention. The uniform superstructure
ablation changes family and mark selection while retaining the learned stopping
law, executor and structural constraints.

Program-panel selectors score the exact executable traces under the frozen model.
They do not replace the structural proposal constructor. Superstructure uses the
model directly for edit selection. Both paths use the same hash-verified weights.

## Code

- `src/compose_v4/benchmark/`: task constraints, program construction and selectors
- `src/compose_v4/model/reference_checkpoint.py`: verified frozen model loading
- `src/compose_v4/experiments/fragments/`: local driver, assets and evaluation
- `tests/test_fragment_*`: behavior, input validation and regression tests

Each output includes attempted slots, exact configuration, input identities and
source hashes. Missing assets fail explicitly. No private worktree, source
archive or cloud account is needed by the driver. The checkpoint and catalogs
still need a public distribution location before fresh-clone use is available.
