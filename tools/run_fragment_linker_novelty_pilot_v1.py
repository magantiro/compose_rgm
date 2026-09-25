"""Matched linker selector pilot on fresh seed five.

The frozen and novelty4 arms use the same eight-offer proposal law and
checkpoint. The treatment changes only selection weights for endpoints not
previously emitted by that prompt and arm. A scored launch requires a separate
authorization matching the frozen contract payload hash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_fragment_attachment_library_pilot import _atomic_json
from run_fragment_linker_broaden_pilot_v1 import (
    ROOT,
    cell_support_summary,
    identity,
    preflight,
    run,
)

from compose_v4.benchmark.training_attachment_fragments import physical_sha256

CONTRACT = ROOT / "configs/fragment_linker_novelty_pilot_v1.json"
ARMS = ("frozen", "novelty4")


def load_contract(path: Path = CONTRACT) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"not a self-hashed linker novelty contract: {path}")
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"linker novelty contract hash mismatch: {path}")
    if (
        payload["schema"] != "fragment_linker_novelty_pilot_v1"
        or payload["arms"] != list(ARMS)
        or payload["task"] != "linker_design"
        or len(payload["drugs"]) != 10
        or len(set(payload["drugs"])) != 10
        or payload["seed"] != 5
        or payload["attempts_per_prompt_arm"] != 100
        or payload["candidate_draws_per_attempt"] != 8
        or payload["workers"] != 1
        or payload["threads"] != 1
        or payload["quality_selection"] is not False
        or payload["oracle_calls"] != 0
        or payload["novelty_multiplier"] != 4.0
        or payload["archive_scope"] != "same_prompt_prior_emissions"
        or payload["output_dir"] != "diagnostics/fragment_linker_novelty_pilot_v1"
    ):
        raise ValueError("linker novelty contract changes matched scientific envelope")
    return payload, envelope["payload_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorized-payload-sha256")
    args = parser.parse_args()
    contract, contract_hash = load_contract()
    if args.phase == "prepare" and args.authorized_payload_sha256 is not None:
        raise ValueError("authorization is used only for a scored launch")
    if args.phase == "run" and args.authorized_payload_sha256 != contract_hash:
        raise ValueError("scored linker novelty launch lacks matching user authorization")
    versions, prompts = preflight(contract)
    manifest = {
        "schema": "fragment_linker_novelty_pilot_manifest_v1",
        "contract": contract,
        "contract_payload_sha256": contract_hash,
        "versions": versions,
        "device": "cpu",
        "threads": 1,
        "precision": "float32; no mixed precision",
        "source_revision": contract["source_revision"],
        "score_blind_cell_support": cell_support_summary(prompts, ARMS),
    }
    output = args.output_dir.resolve()
    if output != (ROOT / contract["output_dir"]).resolve():
        raise ValueError("linker novelty output differs from frozen contract")
    path = output / "manifest.json"
    if args.phase == "prepare":
        if output.exists():
            raise FileExistsError(f"linker novelty output namespace exists: {output}")
        _atomic_json(path, manifest)
        print(json.dumps({"prepared_only": True, "manifest_sha256": physical_sha256(path)}))
        return
    if not path.is_file() or json.loads(path.read_text()) != manifest:
        raise ValueError("linker novelty launch manifest missing or changed")
    if (output / "summary.json").exists():
        raise FileExistsError("linker novelty pilot already complete")
    run(contract, path, output, prompts)


if __name__ == "__main__":
    main()
