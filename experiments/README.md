# Experiments

Each task has its own input definitions, run commands, and output contract.
Start with a single local example before running a full panel.

| Task | Entry point |
| --- | --- |
| Fragment-constrained generation | [fragments/GENERATION.md](fragments/GENERATION.md) |
| Similarity-constrained QED editing | [qed/README.md](qed/README.md) |
| Black-box molecular optimization | [pmo/README.md](pmo/README.md) |
| Similarity-constrained docking | [t4/README.md](t4/README.md) |

Generated outputs belong outside the source tree in `runs/`. Every task records
its model and input identities, code revision, settings, seeds, and attempted
candidates. External assets must be installed with the declared hashes before
a task can run. Use the task guides for complete-panel commands and the local
examples for quick checks.

The [ablation guide](ABLATIONS.md) identifies each intervention and its run
command.
