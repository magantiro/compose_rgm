"""Build the reproducibility index under ``repro/``.

The question this answers is not "is the tree tidy" but "can a published number
still be re-run". Those are different, and only the second is expensive to lose.
Four manifests, each answering one link of the chain

    reported result -> artifact -> contract -> pinned source hashes
        -> commit/branch -> environment -> off-Git weights and ledgers

``pinned_paths_v1.json``
    Every repository file that is the subject of a sha256 pin, with the
    artifacts that pin it and whether the pin still resolves to current bytes.
    These paths are READ-ONLY. A stale pin is normally an immutable launch
    record, and re-pointing one to make a cleanup pass succeed would forge the
    record rather than fix it.

``external_artifacts_v1.json``
    The pins that address something NOT in this repository. Git cannot
    resurrect a checkpoint, corpus or Modal ledger that was never committed, so
    this is where reproducibility is actually lost. Each entry records where the
    object was found, or that it was not found.

``environments_v1.json``
    Three distinct chemistry kernels are in play and only two of them are
    production. A number computed under the wrong one is not a reproduction.

``capability_manifest_v1.json``
    Composed from the capability and entry-point baselines, plus capability
    groups keyed to the subsystem a reader would ask about (executor, proposal
    families, controllers, oracles, contract loaders). A public-symbol superset
    is necessary and NOT sufficient for "nothing was lost": an import can
    resolve while the mechanism behind it is disconnected from every caller,
    which this repository has found six times.

Usage::

    python3 tools/repro_index.py --out-dir repro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Any

# Off-Git objects whose loss would cost a published number. Each is recorded
# with where it was actually found, never with where it "should" be.
KNOWN_EXTERNAL_OBJECTS = (
    {
        "logical_name": "ringcore_a7546e2_best.pt",
        "type": "editing_checkpoint",
        "description": "Provisional 16,000-step editing checkpoint used for chemistry-prior probes.",
        "known_local_copies": ("~/compose_ckpt_backup/ringcore_a7546e2_best.pt",),
        "note": (
            "Its Modal run directory is recorded as gone from both rahul profiles; the local "
            "backup copy is the surviving instance. Selected by hazard-inclusive GM loss, which "
            "configs/ringcore_v1_checkpoint_selection.json forbids for frozen results, so it is "
            "valid for a mechanism probe and not for a published number."
        ),
    },
    {
        "logical_name": "pmo_1k_ab_molecules",
        "type": "oracle_ledger",
        "description": "Per-query molecules for the matched 1,000-call PMO A/B on celecoxib.",
        "known_local_copies": (),
        "note": (
            "Recorded as existing only on a Modal volume under "
            "pmo_population_controller_v1/<run_id>/<task>/oracle/query_*/result.json. The local "
            "artifacts carry counters only, so a post-hoc question about the molecules cannot be "
            "answered from git."
        ),
    },
    {
        "logical_name": "lineage_b_denovo_run",
        "type": "training_run_directory",
        "description": "Original Lineage B de-novo training run, recovery checkpoint and step-2500 continuation.",
        "known_local_copies": (),
        "note": (
            "Held on the compose-v4-artifacts volume of the rahul profile created 2026-07-17, NOT "
            "the rahul-94866 volume of the same name created 2026-08-09. Scanning the wrong one "
            "already produced a false 'the artifacts are gone' conclusion."
        ),
    },
)

# MEASURED 2026-09-23 by `MODAL_PROFILE=<p> modal volume list`. The recorded
# learning names two volumes sharing this name; there are three. `Created by`
# does not disambiguate them, so identity is (profile, created_at).
MODAL_VOLUME_NAMESPACE_COLLISION = {
    "volume_name": "compose-v4-artifacts",
    "distinct_volumes": [
        {"profile": "rahul", "created_at": "2026-07-17 03:16 EDT", "created_by": "rahul-94866"},
        {"profile": "nitya", "created_at": "2026-07-29 14:41 EDT", "created_by": "nitya"},
        {"profile": "rahul-94866", "created_at": "2026-08-09 00:13 EDT", "created_by": "rahul-94866"},
    ],
    "why_it_matters": (
        "Three distinct volumes share one name across three profiles, and the created-by column "
        "does not separate them. A volume name is not a volume identity: pin the profile AND the "
        "creation date. The running T4 campaigns are recorded as held on the nitya profile, which "
        "the existing note does not list at all."
    ),
}

ENVIRONMENTS = {
    "pmo_production": {
        "role": "PMO scored runs, oracle and COMPOSE executor alike",
        "python": "3.11",
        "rdkit": "2023.09.6",
        "numpy": "1.26.4",
        "pytdc": "1.1.15 (installed --no-deps)",
        "extra": "setuptools 69.5.1 plus an rdkit.six shim; PyTDC 1.1.15 imports rdkit.six",
        "image_source": "modal_apps/pmo_population_v1_app.py",
        "local_mirror": "~/compose_pmo_pinned_env",
        "env_vars": {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
    },
    "t4_editing_production": {
        "role": "T4 campaigns, editing corpus, RingCore catalog fingerprint 639ff6078c32d43c",
        "python": "3.11",
        "rdkit": "2024.03.5",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "networkx": "3.3",
        "torch": "2.4.0",
        "local_mirror": "~/compose_t4_pinned_env (torch absent there; add it for any path importing t4_fiber_campaign)",
        "env_vars": {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
    },
    "laptop_venv": {
        "role": "PRODUCTION FOR NOTHING. Convenience only.",
        "python": "3.12.9",
        "rdkit": "2026.03.6",
        "numpy": "2.5.3",
        "path": "/Users/rmaganti/compose_rgm_git/.venv",
        "warning": (
            "Measured to write different canonical SMILES from the pinned kernels on states a "
            "search constructs, and canonical SMILES is used as a cache key, so a spelling "
            "difference becomes a permanent divergence in any procedure that dedupes on it."
        ),
    },
}

# Capability groups a reader would ask about by name. Membership is derived from
# the module list, not hand-listed, so a module added later joins automatically.
CAPABILITY_GROUPS = {
    "executor_and_chemistry": ("compose_v4.chem.", "compose_v4.rewrite."),
    "rate_model_and_generator": ("compose_v4.model.", "compose_v4.gm.", "compose_v4.inference."),
    "controllers_and_proposal_families": ("compose_v4.control.", "compose_v4.policy."),
    "experiments_and_drivers": ("compose_v4.experiments.",),
    "corpus_and_data": ("compose_v4.data.",),
    "evaluation_and_oracles": (
        "compose_v4.eval.",
        "compose_v4.oracles.",
        "compose_v4.benchmark.",
        "compose_v4.baselines.",
        "compose_v4.lipids.",
    ),
    "gates": ("compose_v4.gates.",),
}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout.strip()


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pinned_paths(root: pathlib.Path, pinned: dict, resolution: dict) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    entries = {}
    for relative, entry in pinned["files"].items():
        path = root / relative
        entries[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "pinned_by": entry["pinning_artifacts"],
            "pinned_by_count": entry["pinning_artifact_count"],
            "pin_shapes": entry["shapes"],
            "process_v2_identity": entry["process_v2_identity"],
            "image_mounted_only": entry["image_mounted_only"],
        }
    return {
        "schema_version": 1,
        "policy": "READ-ONLY. Never re-pin a historical contract to make a cleanup pass succeed.",
        "baseline_tag": "pre-cleanup-2026-09-23",
        "producing_commit": head,
        "pinned_path_count": len(entries),
        "aggregate_identity_warning": {
            "statement": (
                "The per-file set UNDERSTATES the constraint. Three whole-tree fingerprints hash "
                "directories rather than named files, so a file that is individually unpinned is "
                "still covered, and ADDING a new file moves the identity just as editing one does."
            ),
            "fingerprints": [
                {
                    "name": "modal_apps/train_tracelet_gm.py::_source_fingerprint",
                    "covers": "every .py and .json under src/, scripts/, recipes/, plus that launcher",
                    "feeds": "run_identity; the immutable stage manifest refuses a changed identity",
                },
                {
                    "name": "modal_apps/t4_objective_reset_app.py::main",
                    "covers": "every .py under src/, plus configs/t4_objective_reset_runtime_v1.json",
                    "feeds": "files_sha256, compared against a stored receipt",
                },
                {
                    "name": "semantic_p50_successor_cache_implementation_sha256",
                    "covers": "every .py under src/compose_v4",
                    "feeds": "P50 successor-cache implementation identity",
                },
            ],
            "consequence": (
                "src/, scripts/ and recipes/ are read-only in AGGREGATE, not merely per file, and "
                "no shim may be added inside src/ without moving at least two identities."
            ),
        },
        "resolution_summary": resolution["counts"],
        "files": entries,
    }


def build_external_artifacts(root: pathlib.Path, resolution: dict) -> dict[str, Any]:
    unresolved = resolution["absent_distinct_paths"]
    repo_shaped = [
        p
        for p in unresolved
        if p.startswith(
            ("diagnostics/", "configs/", "src/", "tools/", "scripts/", "modal_apps/", "docs/",
             "results/", "artifacts/")
        )
    ]
    by_family: dict[str, int] = defaultdict(int)
    for path in repo_shaped:
        by_family["/".join(path.split("/")[:2])] += 1

    known = []
    for record in KNOWN_EXTERNAL_OBJECTS:
        copies = []
        for candidate in record["known_local_copies"]:
            path = pathlib.Path(candidate).expanduser()
            if path.is_file():
                copies.append(
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "verified_at": "2026-09-23",
                    }
                )
        known.append(
            {
                **{k: v for k, v in record.items() if k != "known_local_copies"},
                "verified_local_copies": copies,
                "durable_backup_locations": len(copies),
                "meets_two_backup_requirement": len(copies) >= 2,
            }
        )

    return {
        "schema_version": 1,
        "purpose": (
            "Git cannot resurrect an object that was never committed. These are the links from a "
            "published number to something outside the repository."
        ),
        "modal_volume_namespace_collision": MODAL_VOLUME_NAMESPACE_COLLISION,
        "unresolved_pin_count": resolution["counts"]["absent"],
        "unresolved_distinct_paths": len(unresolved),
        "repo_shaped_unresolved_paths": len(repo_shaped),
        "repo_shaped_unresolved_by_family": dict(
            sorted(by_family.items(), key=lambda kv: -kv[1])[:40]
        ),
        "uncommitted_run_output_finding": {
            "measured": (
                "8,828 of 8,854 repo-shaped unresolved pins resolve to files that exist ONLY as "
                "UNTRACKED files in /Users/rmaganti/compose_rgm_git. They are single-copy on one "
                "disk, referenced by hash from committed contracts, and absent from git."
            ),
            "remaining_26": (
                "The other 26 were first read as permanently lost because `git log --all` searches "
                "LOCAL refs only. All 26 are held on three remote branches that were never fetched "
                "locally, and 64 of 64 checkable pins match their branch blobs exactly."
            ),
            "recovery_refspec": 'git fetch origin "+refs/heads/<branch>:refs/remotes/origin/<branch>"',
            "recovered_from_branches": [
                "codex/compose-baseline-qualification",
                "codex/compose-constraints-hard",
                "codex/compose-multiobjective-package",
            ],
            "remote_vs_local_refs": (
                "origin carries 168 heads; this checkout had 122 refs. A local-only search is not "
                "authoritative about what exists."
            ),
        },
        "known_objects": known,
        "critical_objects_without_two_verified_backups": [
            record["logical_name"] for record in known if not record["meets_two_backup_requirement"]
        ],
    }


def build_capability_manifest(
    root: pathlib.Path, capability: dict, entry_points: dict
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, prefixes in CAPABILITY_GROUPS.items():
        members = sorted(
            module
            for module in capability["modules"]
            if any(module.startswith(prefix) for prefix in prefixes)
        )
        importable = [m for m in members if capability["modules"][m]["import_ok"]]
        symbols = sum(len(capability["modules"][m]["import_symbols"]) for m in members)
        groups[name] = {
            "module_count": len(members),
            "importable_module_count": len(importable),
            "public_symbol_count": symbols,
            "smoke_check": (
                f"python3 -c \"import importlib; "
                f"[importlib.import_module(m) for m in {importable[:3]!r}]\""
                if importable
                else None
            ),
            "modules": members,
        }
    return {
        "schema_version": 1,
        "sufficiency_warning": (
            "A public-symbol superset is NECESSARY AND NOT SUFFICIENT. An import can resolve while "
            "the mechanism behind it is reached by no production caller; this repository has found "
            "six such cases. Layers 3 and 4, golden behaviour and artifact-to-run reproduction, are "
            "separate gates and are reported separately."
        ),
        "baseline_tag": "pre-cleanup-2026-09-23",
        "producing_commit": _git(root, "rev-parse", "HEAD"),
        "environment": {
            "interpreter": capability["interpreter"],
            "library_versions": capability["library_versions"],
            "note": "Symbol tables are environment-independent; both sides of any diff use this env.",
        },
        "module_count": capability["module_count"],
        "importable_module_count": capability["importable_module_count"],
        "public_symbol_pair_count": capability["import_symbol_pair_count"],
        "entry_point_count": entry_points["entry_point_count"],
        "entry_point_import_ok": entry_points["import_ok_count"],
        "entry_points_per_directory": entry_points["per_directory"],
        "capability_groups": groups,
        "layer_coverage": {
            "1_source_api": "diagnostics/repo_hygiene/capability_baseline_061ead93.json",
            "2_entrypoints": "diagnostics/repo_hygiene/entry_point_baseline_061ead93.json",
            "3_behavioral_golden": "repro/test_fingerprint_v1.json",
            "4_artifact_to_run": "repro/external_artifacts_v1.json + repro/pinned_paths_v1.json",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="repro")
    args = parser.parse_args()
    root = repo_root()
    hygiene = root / "diagnostics" / "repo_hygiene"

    pinned = _load(hygiene / "pinned_file_set_v1.json")
    resolution = _load(hygiene / "pin_resolution_v1.json")
    capability = _load(hygiene / "capability_baseline_061ead93.json")
    entry_points = _load(hygiene / "entry_point_baseline_061ead93.json")

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "pinned_paths_v1.json": build_pinned_paths(root, pinned, resolution),
        "external_artifacts_v1.json": build_external_artifacts(root, resolution),
        "environments_v1.json": {
            "schema_version": 1,
            "warning": "Three kernels; the laptop venv is production for nothing.",
            "environments": ENVIRONMENTS,
        },
        "capability_manifest_v1.json": build_capability_manifest(root, capability, entry_points),
    }
    for name, document in written.items():
        (out_dir / name).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out_dir}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
