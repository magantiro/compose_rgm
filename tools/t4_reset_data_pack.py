"""Losslessly package the existing scored audit for collaborator transfer.

No new examples, labels, components, splits, fitting or oracle calls are made.
This is retrospective evidence, not a split-clean training dataset.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evidence pack is immutable; choose a new version")
    arms, rows = Counter(), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.source.open("rb") as source, args.output.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            for line in source:
                row = json.loads(line)
                arms[row["arm"]] += 1
                rows += 1
                compressed.write(line)
    with gzip.open(args.output, "rb") as restored:
        digest = hashlib.file_digest(restored, "sha256").hexdigest()
    source_hash = sha256_file(args.source)
    if digest != source_hash:
        raise ValueError("lossless evidence round trip failed")
    payload = {
        "schema_version": "t4_retrospective_scored_pack_v1",
        "role": "historical_audit_evidence_not_split_clean_training",
        "source_path": str(args.source),
        "source_sha256": source_hash,
        "gzip_sha256": sha256_file(args.output),
        "decompressed_sha256": digest,
        "rows": rows,
        "arm_counts": dict(sorted(arms.items())),
        "audit_path": str(args.audit),
        "audit_sha256": sha256_file(args.audit),
        "packer_sha256": sha256_file(Path(__file__)),
        "new_oracle_calls": 0,
        "new_labels": 0,
        "warning": "Freeze source/lineage group splits before deriving components, preprocessing or fitting. Repeated endpoints and adaptive sampling are not independent observations.",
    }
    seal(args.output.with_suffix(".manifest.json"), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
