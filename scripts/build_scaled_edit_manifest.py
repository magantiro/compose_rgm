#!/usr/bin/env python3
"""Scaled broad-organic B-edit data manifest -- the mixture/curriculum LOCK from measured statistics.

Consumes the full-scale mining output (``mining_summary_full.json`` + ``edit_pool_full.jsonl`` from the
parallel fan-out) and pins a reproducible, code-bound training recipe:

  - the corpus SCOPE + scope hash + full 500k census (from the summary);
  - the compiled MMP pool provenance (row count, SHA-256, schema, source commit + corpus SHA-256);
  - measured curriculum bin EDGES from the pool's path-length distribution (never preset thresholds);
  - the realized per-family / per-layer / element-coverage statistics;
  - the LOCKED layer-mixture weights + cold-element coverage floors that drive the hierarchical sampler;
  - the standardization / operator-registry / corruption-compiler / scope-module hashes.

Percentages are LOCKED here (superseding the PROPOSED values in docs/PHASE0B_DATA_RECIPE.md) against the
at-scale statistics. The corruption layer is regenerated in memory at train time (not mined), so its share
is set by the mixture weight, not a mined count.

Usage:
    PYTHONPATH=src python scripts/build_scaled_edit_manifest.py \
        --summary /path/to/mining_summary_full.json \
        --pool    /path/to/edit_pool_full.jsonl \
        --out     diagnostics/composition/scaled_edit_data_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

REPO = Path(__file__).resolve().parent.parent
_CNOF = frozenset({"C", "N", "O", "F"})
# LOCKED layer mixture (the hierarchical-sampler weights); the corruption share dominates for operator
# coverage, MMP carries observed medicinal edits, scaffold is the longer-range curriculum (deferred until
# the A2.3 general compiler lands, so its weight is held at 0 in v1 and reallocated to MMP).
_LOCKED_MIXTURE = {"corruption": 0.55, "mmp": 0.45, "scaffold": 0.0}
_COLD_ELEMENT_FLOOR = 0.02  # >=2% of each epoch per cold non-CNOF element with supervision (measured)


# Families DISABLED under the RingCore-V1 production contract. Ring addition is compositional
# (cycle_close/cycle_open on the cycle_insert/cycle_attach slots); the legacy whole-ring grow
# macro is masked dead with its params retained only for warm-start compatibility.
_LEGACY_DISABLED_FAMILIES = ("ring_system_grow",)


def _hash_sources(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        fp = REPO / rel
        digest.update(rel.encode())
        digest.update(fp.read_bytes() if fp.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _pool_stats(pool_path: Path) -> dict:
    """Measure the compiled pool: row count, path-length distribution -> curriculum bin edges, family
    histogram, per-layer counts, and per-record cold-element coverage."""
    path_lengths: list[int] = []
    families: Counter = Counter()
    layers: Counter = Counter()
    directions: Counter = Counter()
    cold_element_records: Counter = Counter()
    schema: list[str] = []
    rows = 0
    with pool_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows += 1
            if not schema:
                schema = sorted(rec.keys())
            path_lengths.append(int(rec.get("path_length", 0)))
            for fam, n in (rec.get("operator_histogram") or {}).items():
                families[fam] += n
            layers[rec.get("layer", "unknown")] += 1
            directions[rec.get("direction", "unknown")] += 1
            # element coverage: parse the target SMILES for non-CNOF elements
            tgt = rec.get("target_smiles") or ""
            for elem in ("S", "P", "Cl", "Br", "I", "B"):
                if elem in tgt or (elem == "S" and "s" in tgt) or (elem == "P" and "p" in tgt):
                    cold_element_records[elem] += 1
    lengths = np.asarray(path_lengths) if path_lengths else np.asarray([0])
    # curriculum edges from measured quantiles (short/medium/long), integer-valued, strictly increasing.
    q = [int(np.quantile(lengths, p)) for p in (0.5, 0.8, 0.95)]
    edges = tuple(sorted(set(q))) or (1,)
    return {
        "pool_row_count": rows,
        "record_schema": schema,
        "path_length": {
            "min": int(lengths.min()), "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 3),
            "quantiles_p50_p80_p95": q,
            "curriculum_bin_edges": list(edges),
        },
        "family_histogram": dict(families.most_common()),
        "layer_counts": dict(layers),
        "direction_balance": dict(directions),
        "cold_element_record_counts": dict(cold_element_records.most_common()),
    }


def build_manifest(summary_path: Path, pool_path: Path, out_path: Path) -> dict:
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    corpus = summary.get("corpus", {})
    provenance = summary.get("provenance", {})
    pool_stats = _pool_stats(pool_path) if pool_path.exists() else {"pool_row_count": 0}

    manifest = {
        "manifest_version": 2,
        "kind": "scaled_broad_organic_lock",
        "git_commit": _git_commit(),
        "source_mining_commit": provenance.get("commit"),
        "corpus_scope": corpus.get("scope") or BROAD_ORGANIC_V1.descriptor(),
        "corpus_scope_hash": (corpus.get("scope") or {}).get("scope_hash", BROAD_ORGANIC_V1.scope_hash()),
        "corpus_census": corpus.get("census"),
        "corpus_id": provenance.get("corpus_id"),
        "corpus_sha256": provenance.get("corpus_sha256"),
        "vocabulary": "ORGANIC_VOCABULARY (15 (element,valence) classes)",
        "operator_registry": {
            "mark_rule_names": list(MARK_RULE_NAMES),
            # RingCore-V1: ring ADDITION is compositional, so cycle_insert/cycle_attach (the slots carrying
            # cycle_close/cycle_open) ARE production-enabled; the legacy whole-ring macro is the disabled
            # one. This list was previously inverted -- it excluded the two defining compositional families
            # and retained the disabled macro. That matters because operator_subtype_supervision_gate.py
            # treats production_enabled as the REQUIRED-supervision set: the inverted list would demand
            # supervision for disabled ring_system_grow (spurious NO_GO) while allowing cycle_close/
            # cycle_open to have zero supervision undetected.
            "production_enabled": [m for m in MARK_RULE_NAMES if m not in _LEGACY_DISABLED_FAMILIES],
            "legacy_disabled": list(_LEGACY_DISABLED_FAMILIES),
            "hash": _hash_sources([
                "src/compose_v4/rewrite/operators.py", "src/compose_v4/rewrite/kernel.py",
                "src/compose_v4/rewrite/factorized_fiber.py"]),
        },
        "standardization_hash": _hash_sources([
            "src/compose_v4/chem/molecular_graph.py", "src/compose_v4/chem/state.py"]),
        "compiler_version": {
            "corruption": _hash_sources(["src/compose_v4/rewrite/source_corruption.py"]),
            "mmp": _hash_sources(["scripts/build_analogue_trace_pool.py"]),
            "miner": _hash_sources(["scripts/mine_edit_traces.py"]),
            "corpus_scope_module": _hash_sources(["src/compose_v4/data/organic_corpus.py"]),
        },
        "mmp_pool": {
            "path": str(pool_path.relative_to(REPO)) if pool_path.exists()
                    and REPO in pool_path.parents else str(pool_path),
            "kind": "generated dataset (regenerate via the parallel mining fan-out); NOT a committed fixture",
            "row_count": pool_stats["pool_row_count"],
            "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest() if pool_path.exists() else None,
            "record_schema": pool_stats.get("record_schema"),
            "mmp_pairs_grouped": summary.get("mmp_pairs_grouped"),
        },
        "measured_statistics": {k: v for k, v in pool_stats.items()
                                if k not in ("pool_row_count", "record_schema")},
        "corruption_recipe": summary.get("corruption"),
        "locked_mixture": {
            "layer_weights": _LOCKED_MIXTURE,
            "note": "hierarchical-sampler layer weights (LOCKED, superseding the proposed PHASE0B values); "
                    "scaffold held at 0 until the A2.3 general compiler lands, its share folded into MMP.",
            "cold_element_floor": _COLD_ELEMENT_FLOOR,
            "cold_elements": ["S", "P", "Cl", "Br", "I", "B"],
            "curriculum_bin_edges": pool_stats.get("path_length", {}).get("curriculum_bin_edges"),
        },
        "splits": {
            "scheme": "the shared load_organic_corpus_split (seed 20260714); TRAIN-only mining -> no "
                      "val/test leakage; reverse pairs are separate directed records (balanced).",
        },
        "notes": "Mixture + curriculum are LOCKED here from at-scale statistics (supersedes the PROPOSED "
                 "docs/PHASE0B_DATA_RECIPE.md values). Load a checkpoint/data under a different "
                 "corpus_scope_hash -> fail loudly.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", default="diagnostics/composition/scaled_edit_data_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.summary), Path(args.pool), REPO / args.out)
    print(f"wrote scaled manifest -> {args.out}")
    print(f"  scope_hash={manifest['corpus_scope_hash']}  pool_rows={manifest['mmp_pool']['row_count']}")
    print(f"  curriculum_bin_edges={manifest['locked_mixture']['curriculum_bin_edges']}")
    print(f"  layer_weights={manifest['locked_mixture']['layer_weights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
