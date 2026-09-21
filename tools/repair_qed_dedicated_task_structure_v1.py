"""Re-derive the per-record program structure from the saved campaign snapshots.

Records written before the extraction fix carry ``primitive_depth: null`` for
every entry, because ``trace["primitive_edits"]`` is an integer COUNT and was
read as a list.  The campaign snapshots on disk are the authority, so the field
is repaired from them rather than by re-running the search.

Imports the SAME ``entry_structure`` the runner uses, so the repaired records
cannot disagree with freshly written ones.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_RUNNER = Path(__file__).with_name("run_qed_dedicated_task_v1.py")
_spec = importlib.util.spec_from_file_location("_qed_runner", _RUNNER)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
entry_structure = _runner.entry_structure


def largest_snapshot(campaign_dir: Path) -> Path | None:
    candidates = list(campaign_dir.rglob("complete.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    args = parser.parse_args()

    root = Path(args.records)
    repaired = skipped = already = 0
    for path in sorted(root.glob("[0-9]*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            skipped += 1
            continue
        structure = record.get("structure") or []
        if structure and any(s.get("primitive_depth") is not None for s in structure):
            already += 1
            continue
        snapshot_path = largest_snapshot(root / "campaign" / f"{record['index']:03d}")
        if snapshot_path is None:
            skipped += 1
            continue
        snapshot = json.loads(snapshot_path.read_text()).get("snapshot", {})
        record["structure"] = entry_structure(
            snapshot.get("entries", {}), snapshot.get("observations", {}) or {}
        )[:400]
        record["structure_repaired_from"] = str(
            snapshot_path.relative_to(root)
        )
        path.write_text(json.dumps(record))
        repaired += 1
    print(json.dumps({"repaired": repaired, "already_ok": already, "skipped": skipped}, indent=1))


if __name__ == "__main__":
    main()
