# Three-edit ring expansion

Implemented revision: `4d8ce50424f76b8681597f5c5c754cead5400eac`.
Scope and decisions: `docs/RING_EXPANSION_OPTION.md`.

## Result

The opt-in `expand_ring` program completed four synthetic requests with exactly
three sampled, executor-valid edits each: open a ring bond, insert one neutral
backbone atom, close back to the other endpoint. The sampled insert was carbon
in these fixed-seed examples; the channel also admits supported neutral N/O
births. That broader composition support is not a measured completion rate.

| Request | Outcome | Sampled edits | All public executor calls | Local wall seconds |
|---|---|---:|---:|---:|
| Isolated 5->6 ring | completed | 3 | 9 | 0.0791 |
| Isolated 6->7 ring | completed | 3 | 10 | 0.0404 |
| Isolated 7->8 ring | gate-rejected closure | 2 | 11 | 0.0364 |
| Isolated 8->9 continuation | dependency blocked | 0 | 0 | not run |
| Fused scaffold 7->8 ring | completed | 3 | 10 | 0.0463 |
| Fused scaffold 8->9 continuation | completed | 3 | 11 | 0.0549 |

The fused continuation starts from the preceding exact executed state, not a
SMILES reconstruction. Its endpoints are `c1ccc2c(c1)CCCCCC2` and
`c1ccc2c(c1)CCCCCCC2`. Each expansion adds one atom while keeping graph cycle
rank two and aromatic-ring count one. Both have zero RDKit bridgeheads and
spiro atoms. Exact old-slot induced bonds are retained except for the chosen
edge, replaced by the requested two-bond path through one new slot.

Completion is 4/5 attempted trajectories, or 4/6 declared requests including
the blocked dependency. All four completed trajectories satisfy their exact
structural witness (4/4 acceptance precision). These selected synthetic
fixtures do not estimate population coverage, medicinal quality, learned-law
discovery probability, or docking improvement. The isolated eight-ring closure
was executor-valid but rejected by the unchanged `isolated_ring:8` medicinal
gate; it is not a missing-primitive failure or a reported successful endpoint.

All 51 public executor calls, including unused candidate products, are saved
with source/action/product or failure receipts. Proposal choices condition on
descriptors before product execution; the new channel's support cap is applied
within those matching descriptors. The initially tested alternative ordering
discarded a required closure in the old global shortlist; the new option fixes
that ordering explicitly. Existing fused, generic, and ordinary macro laws and
their regression tests retain their original cap policy.

The report used one CPU worker, float64 uniform primitive-reference weights,
seed zero, a 128-public-call ceiling per request, and a 30-second deadline.
No R_theta, committor training, cloud job, winner input, or docking was used.
These subsecond local timings exclude learned-law enumeration and initialization
and are not a production speedup benchmark. Focused tests ran concurrently on
the host, so timings are observations, not isolated performance measurements.

## Verification and provenance

`result.json` SHA-256:
`feb5f84f5f4ea53a7939594e668b9370c5f898c02dca2591a319efd79172b61b`.
It binds exact input/code hashes, configuration, software, hardware, states,
progress, probabilities, receipts and all failed/blocked requests.

Ran from a clean worktree at the implementation revision:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python tools/ring_expansion_audit.py
```

Use the repository virtualenv's absolute path when running in a clean worktree
without its own virtualenv. No concurrent uncommitted scaffold/model changes
entered the clean run. Its result was copied byte-for-byte into this directory.

132 focused dependency tests passed in 9.00 seconds (`clean_focused.xml`),
covering expansion, the existing fused and generic laws, continuation, exact
state payloads, parent budgets and warm continuation. A shared-workspace run
also passed 132 tests (`focused.xml`). Ruff lint and format checks passed for
all six touched Python files, and `git diff --check` passed. The full repository
suite was not rerun; this is not a completed repository/release milestone.

## Integration boundary

The program is available in `OptionContinuationKernel` and its explicit opt-in
option registry, with progress-aware cache identity and per-step likelihoods.
Q(M), kappa, R_theta, generic support, the existing option-group prior and all
frozen T4 configurations are unchanged. The program is **not enabled in the
batched T4 app or deployed**. Its production learned-law support still requires
a bounded check before end-to-end use. Existing pendant/fused builders remain
available; this program modifies ring size rather than adding a new cycle.

Next: check this short program against the actual frozen marked law, then
enable it explicitly in a separately identified T4 development run if that
check supports proceeding. Do not infer that its local witness guarantees a
better score or use IVG endpoints as templates or rewards.
