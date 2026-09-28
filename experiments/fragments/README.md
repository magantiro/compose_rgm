# Fragment-constrained generation

This is the task entry point for COMPOSE's fragment experiments. It covers motif
extension, superstructure generation, scaffold decoration, and linker design.
Scaffold morphing uses the linker outputs and is not a separate generation run.

**Available now:** offline verification and reproduction of COMPOSE's benchmark
and ablation tables from frozen per-prompt/per-seed metrics. **Not yet available
from this checkout:** a portable, verified end-to-end generation command. The
historical generators and complete traces are preserved, not silently replaced.

## Reproduce the saved results

From the repository root, using Python 3.10 or newer:

```bash
PYTHONPATH=src python3 -m compose_v4.experiments.fragments verify
PYTHONPATH=src python3 -m compose_v4.experiments.fragments tables --output results/fragments/local
```

These commands need only Python's standard library. They do not use network
access, credentials, private worktrees, model weights, RDKit, or PyTorch. To run
from elsewhere, set `PYTHONPATH` to this checkout's `src` directory and pass
`--root /path/to/checkout` **before** `verify` or `tables`.

The output directory must not already exist. A completed report contains:

- `tables.json`: benchmark means and sample SDs, per-seed values, all four
  ablation comparisons, per-prompt effects, and the 1/2/4/8-offer prefix results.
- `tables.md`: readable benchmark, ablation, prefix, and per-prompt tables.
- `provenance.json`: input and implementation hashes, configuration, revision,
  software, verification scope, and hashes of the two output files.

The command checks seven pinned artifact hashes, complete prompt/seed coverage,
absence of duplicate rows, finite metrics, source-summary agreement, and relevant
cross-artifact identities. Output is staged and published as one directory;
existing reports are never replaced. Repeat reductions produce byte-identical
tables in the same interval mode; the provenance receipt records execution time.

### Recompute the bootstrap intervals too

Without this option, intervals are explicitly labeled as **saved**, while means,
SDs and paired prompt effects are recomputed. To repeat the 20,000-draw bootstrap:

```bash
python3.11 -m venv .venv-fragment-reduction
.venv-fragment-reduction/bin/python -m pip install -r experiments/fragments/requirements-bootstrap.txt
PYTHONPATH=src .venv-fragment-reduction/bin/python -m compose_v4.experiments.fragments verify --recompute-intervals
PYTHONPATH=src .venv-fragment-reduction/bin/python -m compose_v4.experiments.fragments tables --recompute-intervals --output results/fragments/bootstrap
```

NumPy 1.26.4 is the validated reduction environment. Some original bootstrap
reductions used NumPy 2.5.3; both historical versions are recorded in the
[manifest](manifest.json). Every recomputed interval must agree with its saved
counterpart to absolute tolerance `1e-10`. This is a numerical reduction check,
not a molecular-kernel parity test.

## What each comparison means

| Experiment | Generation seeds | Compared change | Evidence |
| --- | --- | --- | --- |
| Motif extension | 2, 3, 4 | Reference-score selection versus uniform selection on the same saved endpoints | Benchmark and locked-panel/prefix tables |
| Scaffold decoration | 8, 9, 10 | Reference-score selection versus uniform selection on the same saved endpoints | Benchmark and locked-panel/prefix tables |
| Linker design / scaffold morphing | 6, 7, 8 | Reference score **and fourfold novelty preference** versus uniform selection | Benchmark and locked-panel/prefix tables |
| Superstructure generation | 0, 1, 2 | Learned family/native-mark probabilities versus uniform family/native-mark probabilities; learned hazard retained | Benchmark and primitive-selection ablation |

Each task/arm has ten structural prompts and 100 attempted slots per prompt and
generation seed. Failed slots remain in the denominator. Means first average
prompts within each seed, then average the three seeds. Benchmark SD is the
**sample** SD across those three seed means, not a population SD or uncertainty
across 3,000 supposedly independent molecules. Prompt-bootstrap intervals first
average seeds within each prompt and then resample the ten prompts.

Benchmark quality is uniqueness-adjusted and differs from faithful quality-pass
yield. Smaller prefixes truncate the original attempted-offer order, including
failures and duplicates; they are not new generation runs or measured speedups.
The imported motif interval artifact also contains an older linker comparison.
The reducer deliberately uses **only its motif subtree**, and uses the deployed
novelty-4 linker result for the current linker comparison.

## Layout and provenance

```text
src/compose_v4/experiments/fragments/
    evidence.py       schema, path, and hash checks
    reduction.py      pure row aggregation and optional bootstrap
    reporting.py      readable reports, provenance, safe publication
    __main__.py       thin command-line interface
experiments/fragments/
    manifest.json     explicit experiment-to-evidence map
    README.md         usage and scientific scope
diagnostics/          immutable source results and versioned reductions
tests/test_fragment_evidence.py
```

This uses the existing repository conventions: shared code by responsibility,
experiment navigation by task, and artifacts separate from executable code.
There is no separate reviewer API or second implementation of COMPOSE.

The main checkout already contained the locked motif, decoration, and linker
selection reductions. Four small artifacts were imported **byte-for-byte** from
the preserved `fragment-interface-ablation-20260923` and
`fragment-qed-ablations-20260925` worktrees: the two superstructure arm results,
their prompt-level comparison, and the historical motif interval result. Their
original revisions, source hashes, and absolute provenance paths remain inside
the files. The new runtime never follows those historical absolute paths.

## Known reporting difference

The saved motif rows give validity **99.9667%**, with between-seed sample SD
**0.0577 percentage points**. The submitted Table 1 prints 100%. The reducer
preserves the measured values and flags the difference. It does not modify the
manuscript or rewrite frozen results to match a printed number. Published
external baseline columns are not re-verified by this command.

## What still blocks end-to-end reproduction

| Component | Current state | Required next step |
| --- | --- | --- |
| Saved metric rows | Included and hash-verified | None for table reduction |
| Full attempted proposal panels and trajectories | Preserved in historical worktrees, not distributed here | Inventory, hash-check, package with access/license information |
| Fragment checkpoint | Identified by SHA-256 in the manifest; local-only access | Establish a permitted, stable distribution route |
| Training-derived catalogs and priors | Preserved with the historical tasks | Bring the exact frozen assets into the portable dependency closure |
| Official InVirtuoGen evaluator | External, pinned revision and metrics hash | Preserve its distinct license/access requirements and verify its fetch path |
| Generator launchers | Historical worktree revisions indexed in the manifest | Separate orchestration from reusable code, reconcile dependencies, and test parity |
| Shared model/executor | Frozen fragment versions differ from primary checkout | Audit the differences without overwriting files pinned by other experiments |

Do not use a similarly named development runner as a substitute. The historical
launchers also check original interpreter paths, ancestry, development records,
and file hashes. Copying one script into this checkout is not enough to reproduce
the run. The [integration contract](../../docs/FRAGMENT_REPOSITORY_INTEGRATION.md)
records the safe migration boundary. Full generation is **not claimed verified**.

## Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/pytest -q tests/test_fragment_evidence.py tests/test_fragment_locked_panel_selection_v1.py
.venv/bin/ruff check src/compose_v4/experiments/fragments tests/test_fragment_evidence.py
```

The integration tests build a minimal source export containing only the declared
inputs and reduction package, then run it with Python's site packages disabled.
They cover corrupted/missing inputs, duplicate or incomplete panels, metric and
lineage mismatches, deterministic output, and refusal to overwrite user files.
