"""Fetch pinned primary comparator sampling sources, with Git-blob and SHA-256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "genmol": {
        "repo": "NVIDIA-BioNeMo/genmol",
        "commit": "add09fc83b7255bd09c797e527c0f4b51f5fb7c1",
        "paths": [
            "README.md",
            "MODEL_CARD.md",
            "configs/base.yaml",
            "src/genmol/sampler.py",
            "src/genmol/model.py",
            "src/genmol/utils/utils_chem.py",
            "src/genmol/utils/utils_data.py",
            "scripts/preprocess_data.py",
            "scripts/exps/frag/hparams.yaml",
            "scripts/exps/frag/hparams_v2.yaml",
            "scripts/exps/frag/run.py",
            "scripts/exps/denovo/hparams.yaml",
            "scripts/exps/denovo/hparams_v2.yaml",
            "scripts/exps/denovo/run.py",
            "data/len.pk",
            "LICENSE/license_code.txt",
            "LICENSE/license_weights.txt",
        ],
    },
    "ivg": {
        "repo": "invirtuolabs/InVirtuoGen_results",
        "commit": "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb",
        "paths": [
            "README.md",
            "LICENSE",
            "in_virtuo_gen/generate.py",
            "in_virtuo_gen/evaluation/downstream.py",
            "in_virtuo_gen/evaluation/denovo.py",
            "in_virtuo_gen/models/invirtuobase.py",
            "in_virtuo_gen/models/invirtuofm.py",
            "in_virtuo_gen/train_utils/sampling.py",
            "in_virtuo_gen/train_utils/metrics.py",
            "in_virtuo_gen/preprocess/create_mmap_buckets.py",
            "in_virtuo_gen/preprocess/preprocess_tokenize.py",
            "in_virtuo_gen/utils/fragments.py",
            "configs/zinc_dist.pt",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    previous = json.loads((args.cached_audit / "receipt.json").read_text())
    tasks, trees = [], {}
    for name, spec in SOURCES.items():
        path = args.cached_audit / f"{name}_tree.json"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != previous["assets"][path.name]["sha256"]:
            raise ValueError(f"cached tree changed: {path}")
        tree = json.loads(content)
        if tree["sha"] != spec["commit"] or tree.get("truncated"):
            raise ValueError("tree revision/truncation mismatch")
        entries = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
        trees[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
        for path in spec["paths"]:
            tasks.append((name, spec, path, entries[path]))

    def fetch(task):
        name, spec, path, entry = task
        cached = args.cached_audit / name / path
        url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
        if cached.is_file():
            content, origin = cached.read_bytes(), "verified_existing_cache"
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "COMPOSE-length-audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            origin = "pinned_primary_download"
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != entry["sha"]:
            raise ValueError(f"Git blob mismatch: {name}/{path}")
        return (
            f"{name}/{path}",
            content,
            {
                "url": url,
                "git_blob_sha1": blob,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "origin": origin,
            },
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, tasks))
    result = {
        "schema": "fragment_comparator_length_sources_v1",
        "role": "primary-source read-only audit; no generation, training, checkpoint loading or quality scoring",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sources": SOURCES,
        "cached_tree_identities": trees,
        "assets": {name: receipt for name, _, receipt in sorted(records)},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "binary_policy": "downloaded length assets are not deserialized or executed",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.output_dir.parent, prefix=".length-audit-"
    ) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        for name, content, _ in records:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staging / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        staging.rename(args.output_dir)
    print(json.dumps({"assets": len(records), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
