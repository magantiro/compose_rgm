"""Read-only collection of one fixed docking run; never spawn or redock."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal
from modal.volume import FileEntryType

HERE = Path(__file__).resolve().parent


def retain(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data = args.receipt.read_bytes()
    receipt = json.loads(data)
    assert receipt["schema_version"] == "t4_proposal_docking_spawn_v1"
    assert receipt["oracle_call_limit"] == 16
    prefix = receipt["volume_path"].lstrip("/")
    assert prefix == "t4_proposal_docking/" + receipt["task"]["run_id"]
    retain(HERE / "attempt_1/launch.json", data)
    volume = modal.Volume.from_name(receipt["volume"])

    def read(name):
        try:
            return b"".join(volume.read_file(prefix + "/" + name))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None

    result, failure = read("result.json"), read("failure.json")
    if result is None and failure is None:
        for name in ("progress.json", "heartbeat.json"):
            value = read(name)
            if value is not None:
                print(name, value.decode())
        return
    hashes = {}
    for entry in volume.iterdir(prefix, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/")).relative_to(prefix)
        local = HERE / "attempt_1/run" / relative
        assert local.resolve().is_relative_to((HERE / "attempt_1/run").resolve())
        content = b"".join(volume.read_file(entry.path))
        retain(local, content)
        hashes[entry.path] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": "t4_proposal_docking_download_v1",
        "volume": receipt["volume"],
        "prefix": prefix,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_sha256": dict(sorted(hashes.items())),
    }
    manifest_path = HERE / "attempt_1/download.json"
    if not manifest_path.exists():
        retain(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    if failure is not None:
        raise RuntimeError(failure.decode())
    report = json.loads(result)
    print(
        {
            k: report[k]
            for k in ("status", "new_oracle_attempts", "oracle_failures", "elapsed_seconds")
        }
    )


if __name__ == "__main__":
    main()
