# Exact enumeration performance repair

User-authorized on 2026-09-10 while the paired 100-query archive pilot runs.
This branch does not change, redeploy, stop, or restart that pilot.

## Identity and acceptance

Problem: repeated molecular serialization dominates CPU legal-edit enumeration.
Output: the same canonical molecular strings, validity decisions, exact primitive
successors and marked probabilities, computed with less Python work. COMPOSE's
platform identity and molecular support remain as declared in AGENTS.md and
PMO_ARCHIVE_PILOT.md. This is a computational equivalence claim, not a new search
policy, chemistry expansion, reference-model update, or performance claim on PMO.

Baseline: committed pilot source 8ef5eb2. Primary paired check: unchanged exact
persistent-slot states, model initialization, action support, and canonical
outputs before/after the implementation. Validate ordinary organic molecules,
aromatic/fused rings, charged states, null slots, scars and invalid products in
small fixtures. A bounded production-law comparison must retain the same ordered
marks and probabilities. Record timing separately from profiled timing, exact
input hashes, code identity, environment and repeat count. No oracle calls.

Initial measured profile: the adaptive pilot's first four real attempts spent
154.2 s in 16 fresh marked-law enumerations, of which molecular_graph_to_smiles
accounted for 65.6 s self time across 137,499 calls under cProfile. This profile is
instrumented, not an uninstrumented throughput estimate. Raw input is
diagnostics/pmo_archive_pilot/early_adaptive_profile.pstats in the main workspace.

First repair: visit only nonzero upper-triangle bonds during RDKit construction,
in the same row-major order, preserving null-slot handling, invalid-bond behavior,
scar contraction, sanitization and canonicalization. The first paired profile
showed only 1.08x and 1.13x speedups on two exact development sources.

Second repair: a context-local, bounded (512-entry) LRU cache of canonical
serialization outcomes during one production row's preparation. The key contains
all four exact arrays, shapes and dtypes, not object identity or canonical SMILES.
Mutation cannot reuse a stale entry; failures returning None are retained exactly;
exceptions still propagate. Each scope discards its entries on exit, including
exceptional exits. Validators still run and use identical serialized strings.
There is no cache shared globally across jobs, no support truncation, no changed
guards, and no new dependency. Prepared-batch callers are unchanged.

Acceptance measurements in paired_scoped_cache.json: 1.91x and 1.76x median
production-law speedups over three alternating pairs per source. Each pair had
identical ordered marks/probabilities (642 and 700 respectively), exact executed
products, canonical product identities and validity. These are initialized small
model CPU checks on two 40-slot development sources, not full-campaign or learned
model quality results. The frozen repeated artifact after commit is authoritative.

Acceptance: focused parity and regression checks pass, a representative paired
benchmark shows a speed gain, and final diff/artifacts are inspected. This bounded
repair is not a full scientific release. No further optimization campaign is
authorized by this document. Any inspected external winners are diagnostics, not
controller inputs, reusable templates, or blind evaluation data.

## Verification and limits

Focused dependency checks: 19 passed, comprising the seven serialization-cache
cases, semantic cycle-close runtime tests, and the production semantic cycle-open
mark test. Ruff passes on the new test, profiling tool, and production-kernel
module. The chemistry module retains five pre-existing Ruff violations; the base
revision had those same five plus two now resolved import/annotation violations.
Changed regions are formatted and `git diff --check` passes. No repository-wide
suite was run for this bounded development repair, and no full milestone or
deployment qualification is claimed.

Reproduction (pinned RDKit environment, repository root):

```sh
python tools/pmo_enumeration_profile.py --baseline 8ef5eb2 --repeats 3 --output /absolute/result.json
python -m pytest -q tests/test_molecular_serialization_cache.py tests/test_semantic_cycle_close_runtime.py tests/test_semantic_cycle_open_model_integration.py::test_production_kernel_emits_only_semantic_cycle_open_marks
```

The two initial JSONs retain their original dirty-tree identities and file hashes;
they are developmental measurements, not silently relabeled frozen-source runs.
The final receipt will identify the committed implementation. Nothing here changes
the two live pilot arms, their scores, or their query budgets. Compatibility with
old persistent law caches must be established explicitly before any future reuse;
this repair does not bypass existing source-identity guards.
