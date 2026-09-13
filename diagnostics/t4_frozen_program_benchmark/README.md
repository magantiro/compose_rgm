# Frozen T4 benchmark operational receipt

The full delta=0.4 T4 benchmark is active on Modal. It runs the sealed shared
program controller from clean commit
`c272b881bd23ebf1f46bbd1989e6e2871ad20687`; later branch commits cannot change
the deployed recipe.

| Item | Value |
| --- | --- |
| App | `compose-t4-frozen-program-benchmark` |
| App ID | `ap-leszzlH1iWePa3nhCUAjPr` |
| Function call | `fc-01M2CE78Y7ZRQAXMNSVD775CH4` |
| Durable run ID | `54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da` |
| Modal volume | `compose-v4-artifacts` |
| Volume prefix | `t4_frozen_program_benchmark/54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da` |

`launch.json` is the compact local handle. `preflight_v2.json` is the successful
zero-oracle local preflight; `failed_preflight_v1.json` preserves the rejected
SMILES-reparsed source-state lock. `launch_boundary_review.json` records the
pre-existing unrelated full-suite collection failure and the focused launch
checks that passed. None of these files is an aggregate benchmark result.

Monitor the existing call without relaunching:

```sh
MODAL_PROFILE=nitya PYTHONPATH=src:. .venv/bin/python \
  tools/t4_frozen_program_benchmark.py status
```

After completion, retrieve the sealed result with the same tool's `retrieve`
mode. Do not automatically retry unresolved docking calls. The complete frozen
scientific and operational contract is `docs/T4_FROZEN_PROGRAM_BENCHMARK.md` and
`configs/t4_frozen_program_benchmark_v2.json`.
