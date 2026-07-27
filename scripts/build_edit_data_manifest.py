#!/usr/bin/env python3
"""Versioned data manifest for the B-edit training-trace corpus (master-plan sec 0b).

Pins the exact provenance of an edit-trace corpus so a training run is reproducible and the corpus can
never silently drift from the code that produced it: the source GuacaMol partition, a standardization
hash (the graph representation / SMILES policy), an operator-registry hash (MARK_RULE_NAMES + executor +
operator payloads), a compiler-version hash (corruption + MMP compilers), split identifiers, trace
counts, and the data-driven curriculum-bin definitions (read from the characterization report).

Usage:
    PYTHONPATH=src python scripts/build_edit_data_manifest.py \
        [--characterization diagnostics/composition/edit_trace_characterization.json] \
        [--out diagnostics/composition/edit_data_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

REPO = Path(__file__).resolve().parent.parent


def _hash_sources(paths: list[str]) -> str:
    """Stable content hash over a fixed, sorted set of source files (path + bytes)."""
    digest = hashlib.sha256()
    for rel in sorted(paths):
        fp = REPO / rel
        digest.update(rel.encode())
        digest.update(fp.read_bytes() if fp.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _trace_pool_provenance() -> dict:
    """Record the GENERATED analogue trace pool by provenance (it is a derived dataset, not a committed
    fixture): row count, content SHA-256, exact generation command, input-corpus id + hash, and schema."""
    pool = REPO / "diagnostics/composition/analogue_trace_pool.jsonl"
    corpus = REPO / "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles"
    rows = sum(1 for line in pool.open() if line.strip()) if pool.exists() else 0
    schema = sorted(json.loads(pool.open().readline()).keys()) if rows else []
    return {
        "analogue_trace_pool": {
            "path": "diagnostics/composition/analogue_trace_pool.jsonl",
            "kind": "generated dataset (NOT a committed fixture) -- regenerate from the command below",
            "row_count": rows,
            "sha256": _sha256(pool),
            "generation_command": "PYTHONPATH=src:scripts python scripts/build_analogue_trace_pool.py",
            "input_corpus": {
                "id": "guacamol_heldout_val_5000_seed0",
                "path": "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles",
                "sha256": _sha256(corpus),
                "tracked_in_git": True,
            },
            "record_schema": schema,
        },
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build_manifest(characterization_path: str, out_path: str) -> dict:
    char_fp = REPO / characterization_path
    char = json.loads(char_fp.read_text()) if char_fp.exists() else {}

    def _bins(layer: str):
        return char.get(layer, {}).get("path_length", {}).get("curriculum_bins")

    manifest = {
        "manifest_version": 1,
        "git_commit": _git_commit(),
        "vocabulary": "ORGANIC_VOCABULARY (15 (element,valence) classes)",
        "source_corpus": {
            "partition": "guacamol_heldout_val_5000_seed0",
            "path": "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles",
            "note": "PILOT uses the held-out VAL partition. The production universal-prior corpus MUST "
                    "be mined from the GuacaMol TRAIN partition, with scaffold-disjoint train/val/test "
                    "and no source-target pair leakage (see splits).",
        },
        "standardization_hash": _hash_sources([
            "src/compose_v4/chem/molecular_graph.py",
            "src/compose_v4/chem/state.py",
        ]),
        "operator_registry": {
            "mark_rule_names": list(MARK_RULE_NAMES),
            "hash": _hash_sources([
                "src/compose_v4/rewrite/operators.py",
                "src/compose_v4/rewrite/kernel.py",
                "src/compose_v4/rewrite/tracelets.py",
                "src/compose_v4/rewrite/factorized_fiber.py",
            ]),
        },
        "ring_catalog": {
            "definition": "one shared versioned TypedRingCatalog used by the rewrite system, legal-action "
                          "enumeration, the executor, sampling, AND the corruption data-loader (never an "
                          "ad hoc ring perception).",
            "corruption_gate": "source_corruption threads the catalog -> enumerate_clean_ring_system_deletes, "
                               "which enables ring_system_delete (clean ring-opening) supervision.",
            "fingerprint": "recorded as checkpoint_metadata.ring_catalog_fingerprint (ring_catalog_fingerprint(), "
                           "typed_ring_catalog.py) -- the catalog is built at training time from B's ring traces, "
                           "so its fingerprint is pinned per checkpoint, not in this data-only manifest.",
        },
        "compiler_version": {
            "corruption": _hash_sources(["src/compose_v4/rewrite/source_corruption.py"]),
            "mmp_analogue": _hash_sources(["scripts/build_analogue_trace_pool.py"]),
            "analogue_prior_loader": _hash_sources(["src/compose_v4/experiments/analogue_prior.py"]),
        },
        "splits": {
            "scheme": "scaffold-disjoint train/val/test (Bemis-Murcko), no reverse-pair or MMP-core leakage",
            "status": "TO DEFINE at production scale (the pilot mined the val partition only).",
        },
        "trace_counts": char.get("meta", {}),
        "trace_pool_provenance": _trace_pool_provenance(),
        "curriculum_bins": {layer: _bins(layer) for layer in ("corruption", "analogue", "overall")},
        "characterization_report": characterization_path,
        "notes": "Percentages/curriculum are PROPOSED (see docs/PHASE0B_DATA_RECIPE.md), not locked; they "
                 "are finalized only after at-scale (GuacaMol-TRAIN) statistics exist.",
    }
    out_fp = REPO / out_path
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--characterization",
        default="diagnostics/composition/edit_trace_characterization.json",
    )
    parser.add_argument("--out", default="diagnostics/composition/edit_data_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.characterization, args.out)
    print(f"wrote manifest -> {args.out}")
    print(f"  git_commit={manifest['git_commit']}  standardization_hash={manifest['standardization_hash']}")
    print(f"  operator_registry_hash={manifest['operator_registry']['hash']}")
    print(f"  compiler_version={manifest['compiler_version']}")


if __name__ == "__main__":
    main()
