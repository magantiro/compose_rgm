"""Score one de-novo checkpoint's generated shards into a full diagnostic row.

Consumes the shard JSON files written by ``modal_apps/denovo_official_eval.py``
(each carrying per-trajectory ``smiles`` and ``event_rules``) and emits ONE row of
the checkpoint sweep.  The row answers a single question: as training continues,
does the small-ring pathology go away?

Two denominators are reported separately and must never be conflated:

``validity_over_attempts``
    Valid molecules divided by generation ATTEMPTS.  The gap to 1.0 is sampler
    efficiency -- trajectories that committed nothing -- not chemical invalidity.
``valid_state_fraction``
    The share of committed endpoints that are valid states.  This is the
    architectural guarantee and is expected to be 1.0.

The decomposition follows ``compose_v4.eval.denovo_ring_decomposition``: the raw
strained-vs-clean contrast is confounded with molecule size, so the row carries the
size-stratified table and the size-partialled regression beside it.  The per-RING
size distribution is the closure-policy shape and does not move merely because the
molecules got smaller; the per-MOLECULE prevalence does.  Report both.

Usage::

    python scripts/denovo_checkpoint_sweep.py \
        --shard-dir <dir of shard json> --label step1000 --output <path.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from compose_v4.eval.denovo_benchmark import (
    denovo_benchmark_metrics,
    strained_ring_census,
)
from compose_v4.eval.denovo_ring_decomposition import (
    ring_decomposition_report,
)

SCHEMA_VERSION = 1


# ---- Shard loading ----


def load_shard_records(shard_dir: Path) -> tuple[list[dict], list[str]]:
    """Every record from every shard in ``shard_dir``, with the shard file names.

    Two guards, both of which have a demonstrated failure mode:

    * A duplicate trajectory index means two shards claim the same trajectory.
      Shards are idempotent and index-keyed, so that is a real defect rather than
      something to silently deduplicate.  It raises.
    * All shards must share ONE sampling design ``(seed, total, horizon)``.
      Per-trajectory seeds are derived from ``(seed, total)``, so a shard sampled
      at a different ``total`` belongs to a different trajectory family entirely.
      Such a shard can sit in the same directory under a plausible name (an
      earlier run at a different N), and merging it would silently pool two
      different experiments.  It raises and names the conflicting designs.
    """

    paths = sorted(shard_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no shard json found under {shard_dir}")
    records: list[dict] = []
    seen: dict[int, str] = {}
    designs: dict[tuple, list[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            design = (payload.get("seed"), payload.get("total"), payload.get("horizon"))
            designs.setdefault(design, []).append(path.name)
        shard_records = payload.get("records", payload if isinstance(payload, list) else [])
        for record in shard_records:
            index = record.get("index")
            if index is not None:
                if index in seen:
                    raise ValueError(
                        f"trajectory index {index} appears in both {seen[index]} and {path.name}"
                    )
                seen[index] = path.name
            records.append(record)
    if len(designs) > 1:
        detail = "; ".join(
            f"(seed={seed}, total={total}, horizon={horizon}): {sorted(names)}"
            for (seed, total, horizon), names in sorted(designs.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(f"shards span more than one sampling design in {shard_dir}: {detail}")
    return records, [path.name for path in paths]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- Row construction ----


def checkpoint_row(records: list[dict], *, label: str) -> dict[str, object]:
    """The full diagnostic row for one checkpoint."""

    generated = [record.get("smiles") or "" for record in records]
    attempted = len(records)
    committed = [text for text in generated if text]

    metrics = denovo_benchmark_metrics(generated, attempted=attempted)
    census = strained_ring_census(committed)
    decomposition = ring_decomposition_report(committed, records=records)

    valid_states = sum(1 for record in records if record.get("valid_state"))
    connected = sum(1 for record in records if record.get("connected"))
    exhausted = sum(1 for record in records if record.get("exhausted_event_budget"))

    return {
        "label": label,
        "attempted": attempted,
        "committed": len(committed),
        # Denominators, stated separately and never merged.
        "validity_over_attempts": metrics["validity"],
        "valid_state_fraction": valid_states / len(committed) if committed else None,
        "connected_fraction": connected / len(committed) if committed else None,
        "exhausted_event_budget_fraction": exhausted / attempted if attempted else None,
        "published_metrics": metrics,
        "strained_ring_census": census,
        "decomposition": decomposition,
    }


def build_report(shard_dir: Path, *, label: str) -> dict[str, object]:
    records, shard_names = load_shard_records(shard_dir)
    row = checkpoint_row(records, label=label)
    row["provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "shard_dir": str(shard_dir),
        "shards": shard_names,
        "shard_count": len(shard_names),
        "python": platform.python_version(),
        "rdkit": _rdkit_version(),
        "numpy": _numpy_version(),
    }
    return row


def _rdkit_version() -> str:
    import rdkit

    return rdkit.__version__


def _numpy_version() -> str:
    import numpy

    return numpy.__version__


# ---- Entry point ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    report = build_report(arguments.shard_dir, label=arguments.label)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    published = report["published_metrics"]
    decomposition = report["decomposition"]
    distribution = decomposition["ring_size_distribution"]
    print(
        json.dumps(
            {
                "label": report["label"],
                "attempted": report["attempted"],
                "committed": report["committed"],
                "validity_over_attempts": published["validity"],
                "uniqueness": published["uniqueness"],
                "diversity": published.get("diversity"),
                "quality": published["quality"],
                "fraction_with_strained_ring": distribution["fraction_with_strained_ring"],
                "strained_ring_fraction_per_ring": distribution["strained_ring_fraction"],
                "mean_heavy_atoms": distribution["mean_heavy_atoms"],
                "output": str(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
