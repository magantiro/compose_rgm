"""Download already-paid exact option traces/laws; no remote scientific execution."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal


async def download(results, output):
    import modal

    volume = modal.Volume.from_name("compose-v4-artifacts")
    semaphore = asyncio.Semaphore(16)
    receipts, inputs, runs = {}, {}, []

    async def fetch(remote):
        target = output / remote
        receipt_path = target.with_suffix(".receipt.json")
        if target.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["remote"] != remote or receipt["cached_sha256"] != sha256_file(target):
                raise ValueError(f"invalid cached download: {target}")
        else:
            async with semaphore:
                raw = b"".join([chunk async for chunk in volume.read_file.aio(remote)])
            value = json.loads(raw)
            publish_json(target, value)
            receipt = {
                "remote": remote,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "cached_sha256": sha256_file(target),
            }
            publish_json(receipt_path, receipt)
        receipts[remote] = receipt
        return target

    for result_path in results:
        data = json.loads(result_path.read_text())
        inputs[str(result_path)] = sha256_file(result_path)
        c, run_id = data["configuration"], data["run_id"]
        spawn_path = result_path.parent / "spawn.json"
        spawn = json.loads(spawn_path.read_text())
        inputs[str(spawn_path)] = sha256_file(spawn_path)
        prefix = spawn["prefix"]
        if spawn["run_id"] != run_id or prefix.rsplit("/", 1)[-1] != run_id:
            raise ValueError("download source differs from durable launch identity")
        if data["status"] != "complete_development":
            raise ValueError(f"unfinished data source: {result_path}")
        proposals = {}
        for round_ in data.get("rounds", []):
            for arm in round_["arms"].values():
                for p in arm["proposals"]:
                    if p is not None:
                        if p["id"] in proposals and proposals[p["id"]] != p:
                            raise ValueError("shared proposal has inconsistent provenance")
                        proposals[p["id"]] = p
        for p in data.get("candidates", []):
            proposals[p["id"]] = p
        paths = await asyncio.gather(
            *(fetch(f"{prefix}/{p['id']}.json") for p in proposals.values())
        )
        law_paths = set()
        for path in paths:
            record = unseal(path)
            for event in record["events"]:
                if "mark" in event:
                    key = identity(event["source"]["graph"])
                    law_paths.add(f"{path.parent.parent.relative_to(output)}/laws/{key}.json")
        await asyncio.gather(*(fetch(p) for p in sorted(law_paths)))
        runs.append(
            {
                "prefix": prefix,
                "result_path": str(result_path),
                "configuration": c,
                "image_revision": data["image_revision"],
                "new_oracle_calls": data["new_oracle_calls"],
                "proposals": [proposals[k] for k in sorted(proposals)],
                "completed_draws": len(paths),
                "saved_laws": len(law_paths),
            }
        )
        print(f"downloaded {prefix}: {len(paths)} scored draws, {len(law_paths)} laws", flush=True)
    manifest = {
        "schema_version": "pmo_proposal_download_v1",
        "workspace": "rahul-94866",
        "volume": "compose-v4-artifacts",
        "inputs": inputs,
        "runs": runs,
        "files": receipts,
        "new_oracle_calls": 0,
        "new_neural_evaluations": 0,
    }
    publish_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(asyncio.run(download(args.results, args.output)))


if __name__ == "__main__":
    main()
