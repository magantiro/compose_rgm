"""Post-hoc QED/SA support summary from immutable linker development panels."""

from __future__ import annotations

import json
import subprocess

from analyze_fragment_linker_quality_headroom_v1 import (
    PILOT,
    ROOT,
    immutable_json,
    physical_sha256,
    quality_flags,
)

GUIDED = ROOT / "diagnostics/fragment_linker_quality_guided_replay_v1/result.json"
GUIDED_SHA256 = "59c560b4e4071ff8a80a8dc5351ede32e5e02b67da14aac7e52e26579c2c3e7e"
HEADROOM = ROOT / "diagnostics/fragment_linker_quality_headroom_v1/result.json"
HEADROOM_SHA256 = "65e52a2835675c897dc38ed7baf9df89d21ee72d094041f6af2a40b170be3b5f"
OUTPUT = ROOT / "diagnostics/fragment_linker_quality_landscape_v1/result.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"linker quality landscape already exists: {OUTPUT}")
    if physical_sha256(GUIDED) != GUIDED_SHA256 or physical_sha256(HEADROOM) != HEADROOM_SHA256:
        raise ValueError("locked diagnostic inputs changed")
    guided = json.loads(GUIDED.read_text())
    headroom = json.loads(HEADROOM.read_text())
    properties = guided["property_values"]
    per_prompt = {}
    for drug in headroom["per_prompt"]:
        offered: set[str] = set()
        for index in range(100):
            rel = f"attempts/novelty4/{drug}_{index:03d}.json"
            path = PILOT / rel
            if physical_sha256(path) != headroom["input_attempt_sha256"][rel]:
                raise ValueError(f"saved linker panel changed: {path}")
            panel = json.loads(path.read_text())["panel"]
            offered.update(
                row["endpoint"] for row in panel["offered"] if row["status"] == "model_supported"
            )
        if not offered or not offered <= properties.keys():
            raise ValueError(f"missing property values for supported endpoints: {drug}")
        values = [(name, float(properties[name][0]), float(properties[name][1])) for name in offered]
        sa_pass = [(name, sa, qed) for name, sa, qed in values if sa <= 4.0]
        qed_pass = [(name, sa, qed) for name, sa, qed in values if qed >= 0.6]
        both = [(name, sa, qed) for name, sa, qed in values if quality_flags(sa, qed)[2]]
        near = [(name, sa, qed) for name, sa, qed in values if sa <= 4.5 and qed >= 0.55]
        per_prompt[drug] = {
            "distinct_supported_endpoints": len(values),
            "qed_pass_endpoints": len(qed_pass),
            "sa_pass_endpoints": len(sa_pass),
            "both_pass_endpoints": len(both),
            "near_both_endpoints_qed_ge_055_sa_le_45": len(near),
            "minimum_sa": min(sa for _, sa, _ in values),
            "maximum_qed": max(qed for _, _, qed in values),
            "minimum_sa_among_qed_pass": min((sa for _, sa, _ in qed_pass), default=None),
            "maximum_qed_among_sa_pass": max((qed for _, _, qed in sa_pass), default=None),
        }
    result = {
        "schema": "fragment_linker_quality_landscape_v1",
        "role": "post-hoc saved-panel support diagnosis; not a new benchmark or controller result",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "guided_result_sha256": GUIDED_SHA256,
        "headroom_result_sha256": HEADROOM_SHA256,
        "per_prompt": per_prompt,
    }
    immutable_json(OUTPUT, result)
    print(json.dumps({"result": str(OUTPUT), "per_prompt": per_prompt}))


if __name__ == "__main__":
    main()
