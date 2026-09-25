"""Publish train-only scaffold contexts for conditional pendant-content draws."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
from pathlib import Path

from audit_fragment_conditional_content_support_v1 import training_core_contexts
from rdkit import rdBase

from compose_v4.benchmark.training_attachment_fragments import physical_sha256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "diagnostics/fragment_training_pendant_catalog_v1/catalog.json"


def build() -> dict:
    catalog = json.loads(CATALOG.read_text())
    if catalog.get("schema") != "split_first_training_pendant_catalog_v1":
        raise ValueError("source is not the frozen pendant catalog")
    row_contexts = training_core_contexts(catalog)
    script = Path(__file__).resolve()
    dependency = ROOT / "tools/audit_fragment_conditional_content_support_v1.py"
    source = Path(catalog["source"])
    return {
        "schema": "split_first_training_decoration_content_context_v1",
        "source": str(source.resolve()),
        "source_sha256": catalog["source_sha256"],
        "catalog_sha256": physical_sha256(CATALOG),
        "split": catalog["split"],
        "training_rows": len(row_contexts),
        "row_contexts": [
            [row, core_size, interfaces] for row, (core_size, interfaces) in row_contexts.items()
        ],
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "input_sha256": {
            str(CATALOG): physical_sha256(CATALOG),
            str(source): physical_sha256(source),
            str(script): physical_sha256(script),
            str(dependency): physical_sha256(dependency),
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    prior = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".decoration-content-context-"
    ) as stage:
        temporary = Path(stage) / "complete"
        temporary.mkdir()
        (temporary / "prior.json").write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    print(json.dumps({"training_rows": prior["training_rows"], "output": str(output)}))


if __name__ == "__main__":
    main()
