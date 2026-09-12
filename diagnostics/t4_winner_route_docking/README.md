# Winner-first docking: complete

The exact diagnostic IVG target scored **-13.6** with the unchanged production
PARP1 docking pipeline. It passed the declared first-call screen, so all six
locked endpoints were docked. There were six calls and zero oracle failures.
The job took 70.68 seconds, including 35.03 seconds inside the docking calls;
deployment took 62.02 seconds separately. It has stopped.

| Saved endpoint, in route order | Observed docking | Prior surrogate prediction |
| --- | ---: | ---: |
| Remodel linker | -7.7 | -7.419 |
| Add pendant benzene | -9.5 | -8.054 |
| Fuse six-membered ring | -11.8 | -8.427 |
| Add peripheral carbonyl | -12.5 | -8.565 |
| Insert core carbonyl, exact diagnostic target | -13.6 | -8.748 |
| Previous best, redocked control | -9.8 | -9.939 |

The target was actually docked first. The table reorders the observations along
the saved execution route for interpretation; it is not the job's time order.
All six pass unchanged QED, SA, seed-similarity and endpoint chemistry checks.
The target has QED 0.7311, SA 2.7573 and similarity 0.4746 at delta=0.4.

The destination's strong score is reproducible in the sense of obtaining a new
strong score through our own pipeline, not exact replication of a previously
quoted -14.1 or a replicate-mean claim. Each molecule has only one new docking.
Our previously observed -11.0 molecule scored -9.8 on this new evaluation,
demonstrating that individual best-of-search scores are noisy. No uncertainty
interval or experimental affinity claim follows from this panel.

## What this changes

The completed saved route has progressively better observed docking in this
panel. No downhill interval is observed at these completed-program boundaries;
that says nothing about every internal primitive intermediate or alternative
routes. The main open problem is autonomous discovery and continued allocation
to such sequences, not whether the target or ring-building stages can score well.

The frozen surrogate orders the route stages progressively but severely
underestimates their improvements. In particular, it would rank the target
behind the incumbent although the new docking observations reverse that order.
This establishes a concrete out-of-distribution ranking failure for this
diagnostic comparison, not that the guide lacks signal everywhere.

Next proposed controller work should address general linker/branch remodeling,
coverage of macro realizations and attachment sites, and task feedback on
structurally different completed programs. The known route, target and these
diagnostic labels must remain outside winner-blind proposal selection and model
training. This run did not generate or autonomously discover the target, fit a
model, change the controller, or authorize an additional optimization episode.

## Provenance and verification

- Run: `fece76dedf244574e46892c7c8eb4bf3d5a9ae114e3abcd2be9e47bb779f03c9`.
- Modal call: `fc-01M25Z31GP3JZ414Z0736KVPE5`.
- Source: `6986b84d4950`, branch `t4-winner-route-docking`, clean worktree
  `/private/tmp/compose-winner-route-docking`. Committed, not pushed.
- Raw results, locks, per-row start/result receipts and ligand/pose files are
  retained in the local run directory and on `compose-v4-artifacts` under
  `t4_winner_route_docking/<run>/`.
- `audit.json` SHA-256:
  `d07681c9e8d2c2187a8f2eed2e2e95fe40bc1d1a3db511ea75ab3894335b94fc`.
  It binds inputs, run revision, analysis implementation, configuration, software
  and every measured row. Sealed payloads, identity and call order passed;
  prepared ligand achiral graphs match the locked SMILES; pose-file scores match
  all six reported docking scores. Receptor and binary hashes match production.
- Seven focused tests passed in 3.31 seconds before deployment, including
  once-only row handling, lock/order/stop behavior and launcher bounds. Strict
  preflight passed. Production `_dock` was AST-identical to the preceding run.
  Analysis and collector Ruff checks passed. No broad suite was run.
- Production docking uses Open Babel 3D preparation and QuickVina2 with one CPU,
  exhaustiveness 1 and ten modes. Seeds remain unset, as in the preceding runs.

Reproduce the offline verification, without new docking:

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:/private/tmp/compose-winner-route-docking/src:. \
  OMP_NUM_THREADS=1 .venv/bin/python diagnostics/t4_winner_route_docking/audit.py
```

Analysis and collected artifacts remain local and uncommitted. Unrelated main
worktree changes were preserved.
