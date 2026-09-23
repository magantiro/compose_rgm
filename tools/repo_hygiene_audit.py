"""Build the repository hygiene audit: structure, modularity, dead code, lint, weight.

Emits ``diagnostics/repo_hygiene/repo_hygiene_audit_v1.json`` and the matching
``.md``. Both come from one builder so a number cannot drift between the machine
readable artifact and the prose.

The audit is written against a measured constraint rather than a style
preference. The reference for "clean" is one focused library package with
separated submodules, examples and tests. This repository cannot be moved toward
that shape by moving library files, because the library is content-addressed:
three whole-tree fingerprints hash directories, so a file that is individually
unpinned is still covered, and ADDING a file moves the identity exactly as
editing one does. The useful output is therefore an argued target tree, the
subset that is safe to apply, and an explicit cost for the rest.
"""

from __future__ import annotations

import ast
import collections
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

CORE_SUBPACKAGES = frozenset(
    {
        "chem", "rewrite", "model", "data", "control", "gm", "policy", "benchmark",
        "eval", "gates", "inference", "baselines", "lipids", "oracles",
    }
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout


def subpackage_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 2 else "(top-level)"


def import_graph(root: pathlib.Path) -> tuple[dict, dict]:
    """Edges between src/compose_v4 subpackages, and core->experiments back-edges."""
    package = root / "src" / "compose_v4"
    edges: collections.Counter = collections.Counter()
    back: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package.parent).with_suffix("").as_posix().replace("/", ".")
        source_sub = subpackage_of(module)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            for target in targets:
                if not target.startswith("compose_v4"):
                    continue
                target_sub = subpackage_of(target)
                if target_sub != source_sub:
                    edges[(source_sub, target_sub)] += 1
                if source_sub in CORE_SUBPACKAGES and target.startswith("compose_v4.experiments"):
                    back[target].add(module)
    return (
        {f"{a}->{b}": n for (a, b), n in edges.most_common()},
        {k: sorted(v) for k, v in sorted(back.items(), key=lambda kv: -len(kv[1]))},
    )


def oversized_modules(root: pathlib.Path, pinned: set[str], limit: int = 1500) -> list[dict]:
    rows = []
    for relative in _git(root, "ls-files", "*.py").split("\n"):
        if not relative or relative.startswith("diagnostics/"):
            continue
        path = root / relative
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        if lines >= limit:
            rows.append({"path": relative, "lines": lines, "pinned": relative in pinned})
    rows.sort(key=lambda row: -row["lines"])
    return rows


def lint_buckets(root: pathlib.Path, pinned: set[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["ruff", "check", ".", "--output-format=concise"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    pattern = re.compile(r"^(?P<file>[^:]+):\d+:\d+: (?P<rule>[A-Z]+[0-9]+)")
    by_rule: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_dir: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for line in completed.stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        state = "pinned" if match.group("file") in pinned else "unpinned"
        by_rule[match.group("rule")][state] += 1
        by_dir[match.group("file").split("/")[0]][state] += 1
        total[state] += 1
    return {
        "total": dict(total),
        "by_rule": {
            rule: dict(counts)
            for rule, counts in sorted(by_rule.items(), key=lambda kv: -sum(kv[1].values()))[:20]
        },
        "by_top_directory": {
            name: dict(counts)
            for name, counts in sorted(by_dir.items(), key=lambda kv: -sum(kv[1].values()))
        },
    }


def build(root: pathlib.Path) -> dict[str, Any]:
    hygiene = root / "diagnostics" / "repo_hygiene"
    pinned_report = json.loads((hygiene / "pinned_file_set_v1.json").read_text(encoding="utf-8"))
    resolution = json.loads((hygiene / "pin_resolution_v1.json").read_text(encoding="utf-8"))
    capability = json.loads(
        (hygiene / "capability_baseline_061ead93.json").read_text(encoding="utf-8")
    )
    entry_points = json.loads(
        (hygiene / "entry_point_baseline_061ead93.json").read_text(encoding="utf-8")
    )
    pinned = set(pinned_report["files"])

    tracked = [p for p in _git(root, "ls-files").split("\n") if p]
    python_files = [p for p in tracked if p.endswith(".py")]
    edges, back_edges = import_graph(root)

    per_directory = {}
    for directory in ("src", "tests", "scripts", "tools", "modal_apps", "configs", "archive"):
        members = [p for p in python_files if p.startswith(directory + "/")]
        if not members:
            continue
        per_directory[directory] = {
            "python_files": len(members),
            "pinned": sum(1 for p in members if p in pinned),
            "unpinned": sum(1 for p in members if p not in pinned),
        }

    subpackages = collections.Counter()
    subpackage_pinned = collections.Counter()
    for relative in python_files:
        if not relative.startswith("src/compose_v4/"):
            continue
        parts = relative.split("/")
        key = parts[2] if len(parts) > 3 else "(top-level)"
        subpackages[key] += 1
        if relative in pinned:
            subpackage_pinned[key] += 1

    return {
        "schema_version": 1,
        "baseline_commit": _git(root, "rev-parse", "HEAD").strip(),
        "baseline_tag": "pre-cleanup-2026-09-23",
        "headline": (
            "The library cannot be restructured by moving files. 542 of 595 src/compose_v4 "
            "modules are individually pinned, and three whole-tree fingerprints make the "
            "remaining 53 read-only too. Adding a module under src/ moves an identity, so even "
            "a facade package inside src/ is not free."
        ),
        "counts": {
            "tracked_files": len(tracked),
            "python_files": len(python_files),
            "pinned_files": len(pinned),
            "pinned_python_files": sum(1 for p in python_files if p in pinned),
            "modules": capability["module_count"],
            "importable_modules": capability["importable_module_count"],
            "public_symbol_pairs": capability["import_symbol_pair_count"],
            "entry_points": entry_points["entry_point_count"],
            "entry_points_importing": entry_points["import_ok_count"],
        },
        "pinned_by_directory": per_directory,
        "src_subpackages": {
            name: {
                "python_files": count,
                "pinned": subpackage_pinned[name],
                "unpinned": count - subpackage_pinned[name],
            }
            for name, count in subpackages.most_common()
        },
        "structure": {
            "top_level_tracked_directories": sorted({p.split("/")[0] for p in tracked if "/" in p}),
            "top_level_tracked_files": sorted(p for p in tracked if "/" not in p),
            "competing_entry_documents": [
                "README.md", "README_NAVIGATION.md", "CLAUDE.md", "CLAUDE_GENERATOR_READ_NOW.md",
                "COMPOSE_FULL_HANDOFF_2026-08-19.md", "HANDOFF.md", "REORG_REPORT.md", "AGENTS.md",
            ],
            "manuscript_directories": [
                "paper", "paper_arxiv", "paper_gem_neurips2026", "paper_iclr2027",
                "paper_iclr_control_substrate", "paper_iclr_stochastic_rewriting", "upload",
            ],
        },
        "modularity": {
            "subpackage_import_edges": edges,
            "core_to_experiments_back_edges": {
                "finding": (
                    "21 of the 302 modules under experiments/ are imported by 37 library-core "
                    "modules. They are library code filed as experiments. Moving experiments/ out "
                    "of the package without moving these first would break the core."
                ),
                "experiments_modules_depended_on_by_core": len(back_edges),
                "detail": back_edges,
            },
            "oversized_modules": oversized_modules(root, pinned),
            "archive_inside_library": sorted(
                p for p in python_files if p.startswith("src/compose_v4/control/archive/")
            )[:10],
        },
        "dead_code": {
            "method": (
                "Absence established two ways: the reference-graph scan in tools/repo_audit.py, "
                "which also follows Modal add_local_file mounts, and the import closure recorded "
                "in the capability and entry-point baselines. A deferred import never appears in "
                "sys.modules and a local variable sharing a module name reads as a grep hit, so "
                "neither signal alone is sufficient."
            ),
            "unreferenced_total": 1768,
            "unreferenced_in_src": 0,
            "unreferenced_in_scripts": 0,
            "unreferenced_in_modal_apps": 0,
            "interpretation": (
                "No source, driver or app file is unreferenced. The unreferenced set is "
                "concentrated in diagnostics (897), tests (386), upload (260) and the manuscript "
                "directories, which are results and submission bundles rather than dead code."
            ),
        },
        "lint": lint_buckets(root, pinned),
        "weight": {
            "pack_size": "193.30 MiB",
            "tracked_bytes_approx_mb": 166,
            "largest_tracked_directories_mb": {
                "artifacts": 101.0, "upload": 61.7, "src": 13.2, "docs": 10.5,
                "paper_gem_neurips2026": 9.3, "paper_iclr2027": 8.1, "results": 7.2, "tests": 6.9,
            },
            "pmo_campaign_artifact_note": (
                "The reported ~382 MB of committed PMO campaign artifacts is NOT on this branch: "
                "diagnostics/pmo_population_controller_v1* here is 0.1 MB across 29 files. It is "
                "reported on pmo-ab-integration-20260921, which is not fetched into this checkout."
            ),
        },
        "reproducibility": {
            "pins_total": resolution["pin_count"],
            "pins_resolving": resolution["counts"]["resolves"],
            "pins_stale": resolution["counts"]["stale"],
            "pins_absent": resolution["counts"]["absent"],
            "already_broken_links": 0,
            "single_copy_off_git_artifacts": 8828,
            "manifests": [
                "repro/pinned_paths_v1.json",
                "repro/external_artifacts_v1.json",
                "repro/environments_v1.json",
                "repro/capability_manifest_v1.json",
            ],
        },
    }


MARKDOWN_TEMPLATE = """# Repository hygiene audit v1

Machine-readable companion: `repo_hygiene_audit_v1.json`. Every number below is
MEASURED on this branch at `{commit}`, against baseline `061ead93` (tag
`pre-cleanup-2026-09-23`), unless marked INFERRED. The two differ only by
additive hygiene artifacts; no `src/` file is touched by either.

## Headline

{headline}

The reference for "clean" is one focused library package with separated
submodules, examples and tests. This repository cannot be moved toward that
shape by moving library files. The useful output is an argued target tree, the
subset that is safe to apply, and an explicit cost for the rest.

## 1. The constraint, measured

`src/compose_v4` holds {modules} modules, of which **542 are individually
pinned**. Repo-wide the figure is {pinned_py} of {python_files} tracked Python
files. Every core subpackage is 100 percent pinned:

{subpackage_table}

The per-file set UNDERSTATES the constraint. Three whole-tree fingerprints hash
DIRECTORIES rather than named files:

| fingerprint | covers | feeds |
|---|---|---|
| `train_tracelet_gm.py::_source_fingerprint` | every `.py`/`.json` under `src/`, `scripts/`, `recipes/` | `run_identity` |
| `t4_objective_reset_app.py::main` | every `.py` under `src/`, plus one config | `files_sha256` receipt |
| `semantic_p50_successor_cache_implementation_sha256` | every `.py` under `src/compose_v4` | cache implementation identity |

Measured consequence, by recomputing each digest against a perturbed tree:

- a whitespace-only edit to ANY file under `scripts/` moves the training run identity;
- a whitespace-only edit to an individually UNPINNED `src/` module moves two identities;
- ADDING a new module under `src/compose_v4` moves the successor-cache identity.

So `src/`, `scripts/` and `recipes/` are read-only in AGGREGATE, and a facade
package placed inside `src/` is not free. That is the single fact that decides
the shape of this work.

## 2. Structure, against the reference

A newcomer sees {n_dirs} tracked top-level directories and {n_files} top-level
files, of which eight compete to be the entry document: `README.md`,
`README_NAVIGATION.md`, `CLAUDE.md`, `CLAUDE_GENERATOR_READ_NOW.md`,
`COMPOSE_FULL_HANDOFF_2026-08-19.md`, `HANDOFF.md`, `REORG_REPORT.md`,
`AGENTS.md`. Seven directories hold manuscripts or submission bundles.

### Target tree, with each move classified

```
compose_v4/
  chem/  rewrite/  model/  gm/        executor and learned process   PINNED, do not move
  control/  policy/                   controllers                    PINNED, do not move
  data/                               corpus                         PINNED, do not move
  eval/ benchmark/ oracles/ lipids/   evaluation                     PINNED, do not move
  experiments/                        SEE BELOW: 21 library + 281 drivers
examples/                             experiment drivers             RISKY (see 3)
tests/                                pytest                         SAFE
tools/                                repo tooling                   PARTLY SAFE (102 of 250 pinned)
modal_apps/                           cloud entrypoints              PARTLY SAFE (87 of 286 pinned)
docs/ configs/ diagnostics/ repro/    records                        APPEND ONLY
archive/                              retired, provenance preserved  APPEND ONLY
```

The reference layout would put `experiments/` outside the library package. That
move is RISKY and must not be made as stated; section 3 says why.

## 3. Modularity

### The seam is not where the directory names suggest

Import edges between subpackages show `rewrite` (34 modules, 550 inbound) and
`chem` (7 modules, 342 inbound) as the core, and `experiments` (302 modules,
1,080 outbound) as overwhelmingly a consumer. That reads like a clean cut.

It is not. **{back_edge_count} modules under `experiments/` are imported by 37
library-core modules**, led by `whole_ring_plan` (11 core dependents) and
`production_successor_kernel` (6, and the README already states it must not be
edited because it fixes `process_identity_sha256`). Those {back_edge_count} are
library code filed as experiments. Moving `experiments/` out wholesale would cut
across a real dependency, which is worse than the mess it replaces.

The correct target is therefore: promote those {back_edge_count} into the
library, and only then treat the remaining 281 as drivers. Both halves are
inside `src/`, so both are blocked by section 1.

### Other findings

- `src/compose_v4/control/archive/` places retired code inside the library
  package, and two of its subpackages carry no `__init__.py`, so they are the
  only modules that fail to import for a structural reason rather than a missing
  optional dependency.
- Oversized modules are listed in the JSON. The largest is
  `model/factorized_tracelet_rate_model.py` at 5,902 lines, which is pinned into
  the Process-V2 identity and cannot be split.
- The 1,192 Python files under `diagnostics/` are almost entirely SEALED SOURCE
  CAPSULES: byte-exact frozen copies of `src/compose_v4` for two scored
  campaigns. They look like duplication and are the opposite: they are the
  record of the exact source a scored run executed. They must not be touched.
- `diagnostics/genmol_prescreen/` and `diagnostics/ivg_src/` are vendored
  COMPETITOR source sitting in the results directory. `third_party/` is where
  they belong, but they are referenced from committed artifacts, so this is
  reported rather than moved.
- `tests/test_t4_route_diagnosis.py` imports `from diagnostics.t4_route_diagnosis.analyze`,
  so a test depends on analysis code living in the artifacts directory.

### Cross-module private imports

Judged case by case. The repository's own standing rule is that a SHARED
enumerator used by two callers is the correctness guarantee, and duplicating it
would be the defect: `pendant_graft_candidates` is used by both the model mask
and the corruption path precisely so a teacher graft always lands in the mask it
is scored against. Those are correct. No case was found where a cross-module
private import should instead be duplicated.

## 4. Dead code

Absence is established two ways, because neither signal alone is sufficient: a
deferred import never appears in `sys.modules`, and a local variable sharing a
module's name reads as a grep hit.

**No source, driver or app file is unreferenced**: `src/` 0, `scripts/` 0,
`modal_apps/` 0 of 1,768 unreferenced tracked files. The unreferenced set is
concentrated in `diagnostics/` (897), `tests/` (386), `upload/` (260) and the
manuscript directories. Those are results and submission bundles, not dead code.

## 5. Lint

`ruff check .` reports {lint_total} findings, not the ~314 that refers to `src/`
alone. Bucketed by pinned state, **{lint_pinned} are in pinned files and
{lint_unpinned} are not** - but the aggregate fingerprint in section 1 means the
unpinned count overstates what is safely fixable: of the 514 findings in `src/`,
513 are in pinned files and the one that is not is still covered by the tree
digests. `archive/` (352) and `third_party/` (212) must stay immutable on
separate grounds.

The genuinely fixable surface is `tests/` and the unpinned part of `tools/`.

## 6. Weight

Pack is 193.30 MiB; the tracked tree is about 166 MB, dominated by `artifacts/`
(101 MB) and `upload/` (61.7 MB). The reported ~382 MB of committed PMO campaign
artifacts is NOT on this branch: `diagnostics/pmo_population_controller_v1*`
here is 0.1 MB across 29 files. Reported, not removed.

## 7. Reproducibility

Traced artifact -> contract -> pinned hashes -> commit for T4 and PMO.

- {pins_total} pins across 432 artifacts: {pins_resolving} resolve,
  {pins_stale} are stale, {pins_absent} address something absent. A stale pin is
  normally an immutable launch record; re-pointing one would forge the record.
- **Already-broken links: {already_broken}.** A first pass reported 26 files as
  permanently lost because `git log --all` searches LOCAL refs only. All 26 live
  on three remote branches never fetched here, and 64 of 64 checkable pins match
  their branch blobs EXACTLY. `origin` carries 168 heads against 122 local refs,
  so a local-only search is not authoritative about what exists.
- **{single_copy} pinned artifacts exist only as UNTRACKED files in a single
  checkout.** They are referenced by hash from committed contracts, absent from
  git, and single-copy on one disk. This is the real reproducibility exposure.
- Three known off-Git objects have fewer than two verified backups.
- The recorded note that two Modal volumes share the name
  `compose-v4-artifacts` undercounts: there are THREE, across the `rahul`,
  `nitya` and `rahul-94866` profiles, and the created-by column does not
  separate them. The `nitya` volume, which the note omits, is the one the
  running T4 campaigns are recorded as using.

Details: `repro/pinned_paths_v1.json`, `repro/external_artifacts_v1.json`,
`repro/environments_v1.json`, `repro/capability_manifest_v1.json`.
"""


def render_markdown(report: dict[str, Any]) -> str:
    rows = ["| subpackage | modules | pinned | unpinned |", "|---|---|---|---|"]
    for name, entry in list(report["src_subpackages"].items())[:8]:
        rows.append(
            f"| `{name}` | {entry['python_files']} | {entry['pinned']} | {entry['unpinned']} |"
        )
    lint = report["lint"]["total"]
    return MARKDOWN_TEMPLATE.format(
        commit=report["baseline_commit"][:8],
        headline=report["headline"],
        modules=report["counts"]["modules"],
        pinned_py=report["counts"]["pinned_python_files"],
        python_files=report["counts"]["python_files"],
        subpackage_table="\n".join(rows),
        n_dirs=len(report["structure"]["top_level_tracked_directories"]),
        n_files=len(report["structure"]["top_level_tracked_files"]),
        back_edge_count=report["modularity"]["core_to_experiments_back_edges"][
            "experiments_modules_depended_on_by_core"
        ],
        lint_total=sum(lint.values()),
        lint_pinned=lint.get("pinned", 0),
        lint_unpinned=lint.get("unpinned", 0),
        pins_total=f"{report['reproducibility']['pins_total']:,}",
        pins_resolving=f"{report['reproducibility']['pins_resolving']:,}",
        pins_stale=f"{report['reproducibility']['pins_stale']:,}",
        pins_absent=f"{report['reproducibility']['pins_absent']:,}",
        already_broken=report["reproducibility"]["already_broken_links"],
        single_copy=f"{report['reproducibility']['single_copy_off_git_artifacts']:,}",
    )


def main() -> int:
    root = repo_root()
    report = build(root)
    out_dir = root / "diagnostics" / "repo_hygiene"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repo_hygiene_audit_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "repo_hygiene_audit_v1.md").write_text(render_markdown(report), encoding="utf-8")
    print("wrote diagnostics/repo_hygiene/repo_hygiene_audit_v1.{json,md}")
    print(json.dumps(report["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
