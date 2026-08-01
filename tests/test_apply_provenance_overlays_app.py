from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from modal_apps.apply_provenance_overlays_app import (
    AUDIT_PACK_COMPLETION_FILENAME,
    FROZEN_MMP_V2_ROOT,
    MMP_PACK_COMPLETION_FILENAME,
    _verified_pack_completion,
    build_overlay_completion,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _receipt(
    index: int,
    *,
    layer: str = "mmp_analogue",
    partition: str = "train",
) -> dict[str, object]:
    root = (
        Path(FROZEN_MMP_V2_ROOT) / partition
        if layer == "mmp_analogue"
        else Path("/artifacts/edit-packed-v1") / layer / partition
    )
    shard = root / f"shard_{index:04d}.jsonl.gz"
    return {
        "layer": layer,
        "partition": partition,
        "shard_path": str(shard),
        "shard_file_sha256": SHA_A,
        "manifest_path": str(shard.with_suffix(".manifest.json")),
        "manifest_file_sha256": SHA_B,
        "entries": 1,
        "states": 2,
        "overlay_path": f"{shard}.provenance.json",
        "overlay_file_sha256": SHA_C,
        "overlay_fields_sha256": SHA_A,
        "overlay_semantic_sha256": SHA_B,
        "overlay_validation_role": "historical_immutable_byte_receipt",
    }


def _complete_receipts() -> list[dict[str, object]]:
    layers = ("corruption", "cycle_ops", "mmp_analogue")
    partitions = ("train", "validation", "test")
    return [
        _receipt(index, layer=layer, partition=partition)
        for index, (layer, partition) in enumerate(
            (layer, partition) for layer in layers for partition in partitions
        )
    ]


def _completion(receipts: list[dict[str, object]]) -> dict:
    def inventory_sha256(layers: set[str]) -> str:
        inventory = sorted(
            (
                {
                    "layer": row["layer"],
                    "partition": row["partition"],
                    "shard": Path(row["shard_path"]).name,
                    "packed_sha256": row["shard_file_sha256"],
                    "manifest_sha256": row["manifest_file_sha256"],
                    "entries": row["entries"],
                    "states": row["states"],
                }
                for row in receipts
                if row["layer"] in layers
            ),
            key=lambda row: (row["layer"], row["partition"], row["shard"]),
        )
        return hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    upstream_completions = {
        "audit_pack": {
            "path": f"/artifacts/edit-packed-v1/{AUDIT_PACK_COMPLETION_FILENAME}",
            "file_sha256": SHA_A,
            "completion_flag": "PACK_COMPLETE",
            "inventory_origin": "reconciled_legacy_completion_totals",
            "expected_shards": 6,
            "shard_inventory_sha256": inventory_sha256({"corruption", "cycle_ops"}),
        },
        "mmp_pack": {
            "path": f"{FROZEN_MMP_V2_ROOT}/{MMP_PACK_COMPLETION_FILENAME}",
            "file_sha256": SHA_B,
            "completion_flag": "MMP_PACK_COMPLETE",
            "inventory_origin": "completion_shard_artifacts",
            "expected_shards": 3,
            "shard_inventory_sha256": inventory_sha256({"mmp_analogue"}),
        },
    }
    return build_overlay_completion(
        commit="1" * 40,
        launcher_source_sha256=SHA_A,
        packed_root="/artifacts/edit-packed-v1",
        mmp_root=FROZEN_MMP_V2_ROOT,
        upstream_completions=upstream_completions,
        new_overlay_fields={"trace_schema_version": 2},
        shard_receipts=receipts,
    )


def test_completion_is_order_independent_and_retry_invariant() -> None:
    receipts = _complete_receipts()
    first = _completion(list(reversed(receipts)))
    second = _completion(receipts)

    assert first == second
    assert first["training_authorized"] is False
    assert [row["shard_path"] for row in first["shards"]] == sorted(
        row["shard_path"] for row in first["shards"]
    )
    assert "overlays_written" not in first
    assert "overlays_reused" not in first
    assert len(first["completion_sha256"]) == 64


def test_completion_rejects_duplicate_physical_shards() -> None:
    receipts = _complete_receipts()
    with pytest.raises(ValueError, match="duplicate physical shards"):
        _completion([*receipts, deepcopy(receipts[0])])


def test_completion_requires_every_layer_partition_cell() -> None:
    with pytest.raises(ValueError, match="every required layer/partition cell"):
        _completion(_complete_receipts()[:-1])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(shard_file_sha256="bad"), "shard_file_sha256"),
        (lambda row: row.pop("overlay_path"), "fields disagree"),
        (
            lambda row: row.update(shard_path="/artifacts/wrong/train/shard_0000.jsonl.gz"),
            "outside its exact upstream cell",
        ),
    ],
)
def test_completion_rejects_malformed_shard_receipts(mutation, message: str) -> None:
    receipts = _complete_receipts()
    row = receipts[0]
    mutation(row)

    with pytest.raises(ValueError, match=message):
        _completion(receipts)


def test_completion_rejects_unknown_overlay_validation_role() -> None:
    receipts = _complete_receipts()
    receipts[0]["overlay_validation_role"] = "current_enough"

    with pytest.raises(ValueError, match="overlay_validation_role is invalid"):
        _completion(receipts)


def test_completion_requires_content_addressed_launch_identity() -> None:
    with pytest.raises(ValueError, match="Git SHA"):
        build_overlay_completion(
            commit="short",
            launcher_source_sha256=SHA_A,
            packed_root="/artifacts/edit-packed-v1",
            mmp_root=FROZEN_MMP_V2_ROOT,
            upstream_completions={},
            new_overlay_fields={},
            shard_receipts=_complete_receipts(),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack_fixture(root: Path, *, mmp: bool) -> tuple[Path, list[dict]]:
    rows = []
    layers = ("mmp_analogue",) if mmp else ("corruption", "cycle_ops")
    for index, (layer, partition) in enumerate(
        (layer, partition) for layer in layers for partition in ("train", "validation", "test")
    ):
        directory = root / partition if mmp else root / layer / partition
        directory.mkdir(parents=True, exist_ok=True)
        shard = directory / f"shard_{index:04d}.jsonl.gz"
        manifest = shard.with_suffix(".manifest.json")
        shard.write_bytes(f"shard-{layer}-{partition}".encode())
        manifest.write_text(json.dumps({"entries": 1, "states": 2}) + "\n")
        row = {
            "partition": partition,
            "shard": shard.name,
            "packed_sha256": _sha256(shard),
            "manifest_sha256": _sha256(manifest),
            "entries": 1,
            "states": 2,
        }
        if not mmp:
            row["layer"] = layer
        rows.append(row)
    filename = MMP_PACK_COMPLETION_FILENAME if mmp else AUDIT_PACK_COMPLETION_FILENAME
    flag = "MMP_PACK_COMPLETE" if mmp else "PACK_COMPLETE"
    completion = root / filename
    completion.write_text(
        json.dumps(
            {
                flag: True,
                "expected_shards": len(rows),
                "shard_artifacts": rows,
                **(
                    {"packed_entries": len(rows), "packed_states": 2 * len(rows)}
                    if mmp
                    else {
                        "totals": {"entries": len(rows), "states": 2 * len(rows)},
                        "entries_by_layer": {
                            layer: sum(row["layer"] == layer for row in rows)
                            for layer in ("corruption", "cycle_ops")
                        },
                    }
                ),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return completion, rows


@pytest.mark.parametrize("mmp", [False, True])
def test_pack_completion_reconciles_exact_live_inventory(
    tmp_path: Path,
    mmp: bool,
) -> None:
    root = tmp_path / ("mmp" if mmp else "audit")
    completion, rows = _write_pack_fixture(root, mmp=mmp)
    filename = MMP_PACK_COMPLETION_FILENAME if mmp else AUDIT_PACK_COMPLETION_FILENAME
    flag = "MMP_PACK_COMPLETE" if mmp else "PACK_COMPLETE"

    identity, sources = _verified_pack_completion(
        root=root,
        filename=filename,
        completion_flag=flag,
        expected_file_sha256=_sha256(completion),
        mmp=mmp,
    )

    assert identity["expected_shards"] == len(rows)
    assert len(sources) == len(rows)
    assert identity["file_sha256"] == _sha256(completion)
    assert identity["inventory_origin"] == "completion_shard_artifacts"


def test_legacy_audit_completion_derives_inventory_and_reconciles_totals(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit"
    completion, rows = _write_pack_fixture(root, mmp=False)
    payload = json.loads(completion.read_text())
    del payload["shard_artifacts"]
    completion.write_text(json.dumps(payload, sort_keys=True) + "\n")

    identity, sources = _verified_pack_completion(
        root=root,
        filename=AUDIT_PACK_COMPLETION_FILENAME,
        completion_flag="PACK_COMPLETE",
        expected_file_sha256=_sha256(completion),
        mmp=False,
    )

    assert identity["inventory_origin"] == "reconciled_legacy_completion_totals"
    assert identity["expected_shards"] == len(rows)
    assert len(sources) == len(rows)


def test_legacy_audit_completion_rejects_unreconciled_totals(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    completion, _ = _write_pack_fixture(root, mmp=False)
    payload = json.loads(completion.read_text())
    del payload["shard_artifacts"]
    payload["totals"]["entries"] += 1
    completion.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="totals disagree"):
        _verified_pack_completion(
            root=root,
            filename=AUDIT_PACK_COMPLETION_FILENAME,
            completion_flag="PACK_COMPLETE",
            expected_file_sha256=_sha256(completion),
            mmp=False,
        )


def test_pack_completion_rejects_missing_or_extra_live_shards(tmp_path: Path) -> None:
    root = tmp_path / "mmp"
    completion, rows = _write_pack_fixture(root, mmp=True)
    expected_sha256 = _sha256(completion)
    first = root / rows[0]["partition"] / rows[0]["shard"]
    first.unlink()
    with pytest.raises(RuntimeError, match="live shard count disagrees"):
        _verified_pack_completion(
            root=root,
            filename=MMP_PACK_COMPLETION_FILENAME,
            completion_flag="MMP_PACK_COMPLETE",
            expected_file_sha256=expected_sha256,
            mmp=True,
        )

    root = tmp_path / "mmp-extra"
    completion, _ = _write_pack_fixture(root, mmp=True)
    extra = root / "train" / "unexpected.jsonl.gz"
    extra.write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="live shard count disagrees"):
        _verified_pack_completion(
            root=root,
            filename=MMP_PACK_COMPLETION_FILENAME,
            completion_flag="MMP_PACK_COMPLETE",
            expected_file_sha256=_sha256(completion),
            mmp=True,
        )


def test_completion_rejects_stale_mmp_v1_root() -> None:
    receipts = _complete_receipts()
    with pytest.raises(ValueError, match="packed roots"):
        build_overlay_completion(
            commit="1" * 40,
            launcher_source_sha256=SHA_A,
            packed_root="/artifacts/edit-packed-v1",
            mmp_root="/artifacts/mmp_packed_v1",
            upstream_completions={},
            new_overlay_fields={},
            shard_receipts=receipts,
        )
