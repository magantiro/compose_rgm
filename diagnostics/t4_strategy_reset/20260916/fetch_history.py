"""Read-only recovery of named historical T4 artifacts. Never starts a worker."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import time
from pathlib import Path

import modal

OUT = Path(__file__).resolve().parent
PREFIXES = {
    "v0": "t4_no_complete_routes_diagnostic_dynamic_only/afb00306dd09677f3d57c9234ec9927445e8468f2dc70003ec20ead84959d48d",
    "v1": "t4_dynamic_v1_development/b14a3c4f0551b85daec342a289ba28458e64350de08a01e4c9150055cbe09d3d",
    "v21": "t4_dynamic_v21_development/dd0c948ed8f8269e510cddadd0017e791adea3f2ef57db1e9e3f4d95a943398e",
    "full146": "t4_frozen_program_benchmark/54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da",
}
CELLS = {
    "v0": ["5ht1b_0", "braf_1", "jak2_1"],
    "v1": ["5ht1b_0", "braf_1", "jak2_1"],
    "v21": ["5ht1b_0", "braf_1", "jak2_1", "fa7_0", "parp1_0"],
    "full146": [f"{p}_{s}" for p in ("5ht1b", "braf", "jak2", "fa7", "parp1") for s in range(3)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    volume_name = "compose-t4-dynamic-v0-full-suite" if args.strict else "compose-v4-artifacts"
    volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    if args.strict:
        PREFIXES.clear()
        PREFIXES["v0d06"] = (
            "t4_dynamic_v0_full_suite_delta06/a9bae0c504e284c1066b6dd1bd5b0c3163bc2cca614d7ab5a18d068694b54302"
        )
        CELLS.clear()
        CELLS["v0d06"] = [
            f"{p}_{s}" for p in ("5ht1b", "braf", "jak2", "fa7", "parp1") for s in range(3)
        ]
    jobs = []
    for arm, cells in CELLS.items():
        for cell in cells:
            for replicate in range(3) if arm == "full146" else (0,):
                filenames = ["result.json", "checkpoint.json.gz"]
                if args.rounds:
                    if arm == "full146" and (
                        replicate or cell not in ("5ht1b_0", "braf_1", "jak2_1", "fa7_0", "parp1_0")
                    ):
                        continue
                    checkpoint = OUT / "raw" / f"{arm}_{cell}_r{replicate}_checkpoint.json.gz"
                    if not checkpoint.exists():
                        continue
                    saved = json.loads(gzip.decompress(checkpoint.read_bytes()))["payload"]
                    filenames = [
                        f"rounds/round_{r:04d}/batch.json.gz" for r in range(saved["next_round"])
                    ]
                for filename in filenames:
                    jobs.append((arm, cell, replicate, filename))

    def fetch(job):
        arm, cell, replicate, filename = job
        remote = f"{PREFIXES[arm]}/units/{cell}_r{replicate}/{filename}"
        destination = OUT / "raw" / f"{arm}_{cell}_r{replicate}_{filename.replace('/', '_')}"
        began = time.monotonic()
        record = {
            "arm": arm,
            "cell": cell,
            "replicate": replicate,
            "volume": volume_name,
            "remote": remote,
            "local": str(destination),
        }
        try:
            if destination.exists():
                blob = destination.read_bytes()
                record["status"] = "already_recovered"
            else:
                blob = b"".join(volume.read_file(remote))
                temporary = destination.with_name(destination.name + ".partial")
                temporary.write_bytes(blob)
                temporary.replace(destination)
                record["status"] = "recovered"
            record.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        except (modal.exception.NotFoundError, FileNotFoundError) as exc:
            record.update(status="missing", error=str(exc))
        record["seconds"] = time.monotonic() - began
        print(
            json.dumps(
                {k: record[k] for k in ("arm", "cell", "replicate", "status", "seconds")},
                sort_keys=True,
            ),
            flush=True,
        )
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, jobs))
    manifest = {
        "schema": "t4_strategy_reset_recovery_v1",
        "profile": "nitya",
        "new_oracle_calls": 0,
        "new_workers": 0,
        "records": records,
    }
    tag = "_rounds" if args.rounds else ""
    tag += "_strict" if args.strict else ""
    (OUT / f"recovery_manifest{tag}.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
