"""Download a stopped T4 run, preserving physical bytes and existing files."""

import argparse
from pathlib import Path

import modal
from modal.volume import FileEntryType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prefix = args.prefix.strip("/")
    if not prefix.startswith("t4_target_recovery/") or len(prefix.split("/")) != 2:
        raise ValueError("expected one exact target-recovery run namespace")
    volume = modal.Volume.from_name("compose-v4-artifacts")
    count = 0
    for entry in volume.iterdir(prefix, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/")).relative_to(prefix)
        target = args.destination / relative
        if not target.resolve().is_relative_to(args.destination.resolve()):
            raise ValueError("remote path escapes task destination")
        data = b"".join(volume.read_file(entry.path))
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"existing download differs: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        count += 1
    print(f"Retained {count} files without modifying the remote volume")


if __name__ == "__main__":
    main()
