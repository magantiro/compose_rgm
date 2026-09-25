# Motif official run runtime recovery

The approved motif contract payload is
`c22e64562dacb57bf6b497a408ffe7030df50c83d5c2fb47787bdeb88670aabe`.
Its manifest SHA-256 is
`3ac6675ec5a747a14e629b95410e6c86767787b4c4e3393f78485f091c52057c`.

The first launch completed and saved all 100 Baricitinib seed-2 attempt receipts,
then stopped before writing the prompt metric row. The official evaluator import
raised `ModuleNotFoundError: No module named 'pandas'` in the pinned Python
3.11.13, RDKit 2024.03.5 environment. This was an environment dependency
failure, not a molecular-output or benchmark-metric result.

The unchanged runner was resumed from those same saved attempts with the
existing local `.fragment_eval_deps` directory added to `PYTHONPATH`. That
directory supplies pandas 2.2.3 and its dependencies; the pinned NumPy, RDKit,
PyTorch and evaluator remained unchanged. The official evaluator source hash
remained `3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099`.
No attempt was redrawn, no candidate-selection rule or prompt changed, and the
first 100-attempt row then sealed successfully. The run continues under the
original contract; this record is not a change to its metric definition.
