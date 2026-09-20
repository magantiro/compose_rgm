"""Read-only status of the 15-cell T4 delta=0.4 panel, joined to the published baseline.

Reads durable volume state directly -- it never attaches to, resumes, or otherwise
touches a running app, and it charges no oracle calls.

The comparison is joined by exact ``(target, seed_index, delta=0.4)`` identity from
``configs/t4_published_invirtuogen_baseline.json``.  The verdict is printed by this
script rather than read off by eye: docking scores are more negative = better, so
``gap = COMPOSE - IVG`` and a NEGATIVE gap means COMPOSE wins.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHES = ROOT / "diagnostics/t4_integrated_route_fiber_parp1_v1/launches"
BASELINE = "configs/t4_published_invirtuogen_baseline.json"

# Every arm of the delta=0.4 panel: the four sealed here, plus the jak2 arm that was
# already live and is REUSED rather than relaunched.
ARMS = {
    "parp1": "configs/t4_held_target_distilled_parp1_d04_250.json",
    "braf": "configs/t4_held_target_distilled_braf_d04_250.json",
    "fa7": "configs/t4_held_target_distilled_fa7_d04_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_d04_250.json",
    "jak2": "configs/t4_held_target_distilled_jak2_d06_250.json",
}
VOLUMES = {
    "parp1": "compose-t4-held-target-distilled-parp1-d04-250",
    "braf": "compose-t4-held-target-distilled-braf-d04-250",
    "fa7": "compose-t4-held-target-distilled-fa7-d04-250",
    "5ht1b": "compose-t4-held-target-distilled-5ht1b-d04-250",
    "jak2": "compose-t4-held-target-distilled-jak2-d06-250",
}


def latest_run_id(contract_sha: str) -> str | None:
    """The newest launch receipt sealed against this exact contract payload."""
    best: tuple[float, str] | None = None
    for receipt in LAUNCHES.glob("*.json"):
        payload = json.loads(receipt.read_text()).get("payload", {})
        if payload.get("task", {}).get("contract_payload_sha256") != contract_sha:
            continue
        stamp = receipt.stat().st_mtime
        if best is None or stamp > best[0]:
            best = (stamp, payload["task"]["run_id"])
    return None if best is None else best[1]


def fetch(volume: str, remote: str, into: Path) -> dict | None:
    done = subprocess.run(
        ["modal", "volume", "get", "--force", volume, remote, str(into)],
        capture_output=True, text=True, check=False,
    )
    if done.returncode != 0 or not into.exists():
        return None
    try:
        return json.loads(into.read_text())["payload"]
    except (json.JSONDecodeError, KeyError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(ARMS))
    args = parser.parse_args()
    wanted = [t for t in args.targets.split(",") if t in ARMS]

    registry = json.loads((ROOT / BASELINE).read_text())
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        for target in wanted:
            envelope = json.loads((ROOT / ARMS[target]).read_text())
            payload, sha = envelope["payload"], envelope["payload_sha256"]
            if payload["delta"] != 0.4:
                raise ValueError(f"{target} arm is at delta {payload['delta']}, not 0.4")
            run_id = latest_run_id(sha)
            for index, cell in enumerate(payload["cells"]):
                name = cell["cell"]
                best, status, calls = None, "no_launch_receipt", 0
                if run_id:
                    status = "not_started"
                    for leaf in ("result.json", "checkpoint.json"):
                        got = fetch(VOLUMES[target], f"{run_id}/{name}/{leaf}",
                                    scratch_path / f"{target}_{name}_{leaf}")
                        if got is None:
                            continue
                        status, calls = got.get("status", status), got.get("charged_calls", 0)
                        archive = got.get("archive") or {}
                        best = got.get("final_best")
                        if best is None and archive:
                            best = min(archive.values())
                        break
                ivg = registry["delta_0_4"][target][index]
                rows.append({
                    "cell": name, "target": target, "run_id": run_id, "status": status,
                    "calls": calls, "best": best, "ivg": ivg,
                    "gap": None if best is None else round(best - ivg, 3),
                })

    print(f"{'cell':<9} {'status':<22} {'calls':>5} {'COMPOSE':>8} {'IVG d0.4':>9} "
          f"{'gap':>7}  verdict")
    print("-" * 78)
    wins = scored = 0
    for row in rows:
        if row["gap"] is None:
            verdict, best, gap = "pending", "   --", "     --"
        else:
            scored += 1
            wins += row["gap"] < 0
            verdict = "COMPOSE WINS" if row["gap"] < 0 else (
                "tie" if row["gap"] == 0 else "IVG ahead")
            best, gap = f"{row['best']:8.2f}", f"{row['gap']:7.2f}"
        print(f"{row['cell']:<9} {row['status']:<22} {row['calls']:>5} {best:>8} "
              f"{row['ivg']:>9.1f} {gap:>7}  {verdict}")
    print("-" * 78)
    print("sign convention: docking score, more negative is better; "
          "gap = COMPOSE - IVG; negative gap = COMPOSE wins")
    print(f"scored cells: {scored}/{len(rows)}   COMPOSE ahead on {wins}/{scored or 1} scored")
    print("caveat: fa7 cell 2 is parenthesised in the published table at both delta "
          "values; treat its status as unresolved rather than a plain mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
