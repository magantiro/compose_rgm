"""Pre-launch gate for the T4 delta=0.4 250-call held-target panel.

Every check is recomputed from files on disk.  Nothing is transcribed: the published
InVirtuoGen comparison is joined by exact ``(target, seed_index, delta)`` identity from
the authoritative registry, and the verdict is printed by this script rather than
narrated, because the sign convention (more negative is better, so a NEGATIVE
``COMPOSE - IVG`` gap means COMPOSE wins) is easy to invert by hand.

The gate refuses the panel unless the only executable difference between each
delta=0.4 contract and its paired delta=0.6 contract is ``delta`` itself.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("parp1", "braf", "fa7", "5ht1b")
BASELINE = "configs/t4_published_invirtuogen_baseline.json"
SEEDS = "docs/GENMOL_T4_SEEDS.json"

# Contract fields the Modal app actually reads; everything else is prose or provenance.
EXECUTABLE = (
    "delta", "cells", "support", "value_penalty", "total_charged_call_ceiling",
    "parents", "parent_explore", "exploration", "expert_floor_rounds",
    "evaluator_sha256", "docking_seed", "docking_box", "charged_calls_per_cell",
    "batch", "proposal", "runtime_inputs_sha256",
)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str) -> None:
    results.append((name, bool(ok), evidence))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    registry = json.loads((ROOT / BASELINE).read_text())
    seeds = {row["idx"]: row for row in json.loads((ROOT / SEEDS).read_text())}
    table_rows = []

    for target in TARGETS:
        d04_path = f"configs/t4_held_target_distilled_{target}_d04_250.json"
        d06_path = f"configs/t4_held_target_distilled_{target}_d06_250.json"
        envelope = json.loads((ROOT / d04_path).read_text())
        payload = envelope["payload"]
        base = unseal(ROOT / d06_path)

        # 1. payload identity is self-consistent
        computed = identity(payload)
        check(f"{target}: payload_sha256 == identity(payload)",
              computed == envelope["payload_sha256"],
              f"{computed}")

        # 2. every pinned runtime input matches the file on disk
        bad = {k: sha256(ROOT / k) for k, v in payload["runtime_inputs_sha256"].items()
               if not (ROOT / k).exists() or sha256(ROOT / k) != v}
        check(f"{target}: all {len(payload['runtime_inputs_sha256'])} runtime inputs match disk",
              not bad, "all match" if not bad else f"MISMATCH {bad}")

        # 3. delta is 0.4 in the operative field
        check(f"{target}: operative delta == 0.4", payload["delta"] == 0.4,
              f"delta={payload['delta']!r} (base arm {base['delta']!r})")

        # 4. budget
        cells = payload["cells"]
        ceiling_ok = (payload["charged_calls_per_cell"] == 250
                      and payload["total_charged_call_ceiling"] == 250 * len(cells) == 750)
        check(f"{target}: 250 calls/cell, ceiling 250*{len(cells)}=750", ceiling_ok,
              f"{payload['charged_calls_per_cell']}/cell, ceiling "
              f"{payload['total_charged_call_ceiling']}, {len(cells)} cells")

        # 5. same held-target prior checkpoint as the delta=0.6 arm, by hash
        ckpt = f"diagnostics/t4_held_target_distillation_quality_v1/{target}_checkpoint.json"
        a, b, disk = (payload["runtime_inputs_sha256"].get(ckpt),
                      base["runtime_inputs_sha256"].get(ckpt), sha256(ROOT / ckpt))
        check(f"{target}: prior checkpoint identical to the delta=0.6 arm",
              a == b == disk and a is not None, f"d04={a} d06={b} disk={disk}")

        # 6. only-executable-change is delta
        differing = sorted(k for k in EXECUTABLE if payload.get(k) != base.get(k))
        check(f"{target}: sole executable change vs the delta=0.6 arm is delta",
              differing == ["delta"],
              f"differing executable fields: {differing}")

        # 7. prose agrees with the operative field.  A reference to the paired
        #    "delta=0.6 250-call contract" is legitimate provenance; a claim that THIS
        #    arm runs at 0.6 is not, and that disagreement is what put a live jak2
        #    campaign at 0.4 behind prose asserting 0.6.
        boundary = payload["claim_boundary"]
        setting = payload["experimental_setting"]
        claim = payload["central_claim_under_test"]
        provenance = "delta=0.6 250-call contract"
        check(f"{target}: claim_boundary states delta 0.4 and never 0.6",
              "delta 0.4" in boundary and "0.6" not in boundary, repr(boundary))
        check(f"{target}: experimental_setting states delta=0.4",
              "delta=0.4" in setting, repr(setting[:110] + "..."))
        check(f"{target}: central_claim_under_test states delta=0.4",
              "delta=0.4" in claim and "0.6" not in claim, repr(claim))
        check(f"{target}: the only 0.6 mention is the paired-arm provenance reference",
              setting.count("0.6") == setting.count(provenance) == 1,
              f"{setting.count('0.6')} mention(s), all provenance")

        # 8. no ancestor declared for a fresh volume
        check(f"{target}: resume_predecessor dropped (fresh volume, no resumable ancestor)",
              "resume_predecessor" not in payload, "absent")

        # 9. cells bind the published seed rows, and IVG joins by identity
        published = registry["delta_0_4"][target]
        seed_ok, join_ok = True, True
        for index, cell in enumerate(cells):
            row = seeds[cell["source_global_index"]]
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(cell["smiles"]))
            if not (cell["cell"] == f"{target}_{index}" and row["target"] == target
                    and row["smiles"] == cell["smiles"]
                    and canon == Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))):
                seed_ok = False
            if payload["reported_ivg_delta_0_4"][cell["cell"]] != published[index]:
                join_ok = False
            table_rows.append({
                "cell": cell["cell"], "target": target,
                "seed_index": index, "global_index": cell["source_global_index"],
                "seed_ds": -row["published_ds"],
                "ivg_d04": published[index],
                "ivg_d06": registry["delta_0_6"][target][index],
                "contract_sha256": envelope["payload_sha256"],
            })
        check(f"{target}: all 3 cells bind their published seed rows", seed_ok, "exact+canonical")
        check(f"{target}: reported_ivg_delta_0_4 == registry join by (target, seed_index, 0.4)",
              join_ok, f"{payload['reported_ivg_delta_0_4']}")
        check(f"{target}: IVG source is the authoritative registry",
              payload["reported_ivg_source"]["artifact"] == BASELINE
              and payload["reported_ivg_source"]["artifact_sha256"] == sha256(ROOT / BASELINE),
              BASELINE)

    # 10. namespace isolation across every wrapper in the repo
    names: dict[str, list[str]] = {}
    for wrapper in sorted((ROOT / "modal_apps").glob("*.py")):
        text = wrapper.read_text()
        for key in ("COMPOSE_HELD_VOLUME", "COMPOSE_HELD_APP", "COMPOSE_HELD_OUTPUT"):
            for line in text.splitlines():
                if key in line and '"' in line:
                    value = line.rsplit('"', 2)[1]
                    names.setdefault(f"{key}={value}", []).append(wrapper.name)
    collisions = {k: v for k, v in names.items() if len(v) > 1}
    check("namespace isolation: no volume/app/output shared by two wrappers",
          not collisions, f"{len(names)} names, collisions={collisions or 'none'}")
    for target in TARGETS:
        expected = {f"COMPOSE_HELD_VOLUME=compose-t4-held-target-distilled-{target}-d04-250",
                    f"COMPOSE_HELD_APP=compose-t4-held-target-distilled-{target}-d04-250",
                    f"COMPOSE_HELD_OUTPUT=/held_target_{target}_d04_250"}
        check(f"{target}: new arm declares its own volume/app/output",
              all(k in names for k in expected), "distinct d04 namespace")

    # 11. the app refuses untracked or dirty runtime inputs
    tracked = sorted({*(k for t in TARGETS
                        for k in unseal(ROOT / f"configs/t4_held_target_distilled_{t}_d04_250.json")
                        ["runtime_inputs_sha256"]),
                      *(f"configs/t4_held_target_distilled_{t}_d04_250.json" for t in TARGETS),
                      *(f"modal_apps/t4_integrated_route_fiber_held_{t}_d04_app.py" for t in TARGETS)})
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *tracked], cwd=ROOT, check=False
    ).returncode
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *tracked], cwd=ROOT, text=True)
    check("git: every launch input is tracked and clean at HEAD",
          dirty == 0 and not untracked.strip(),
          "clean" if dirty == 0 and not untracked.strip()
          else f"dirty={dirty} untracked={untracked.split() or 'none'}")

    width = max(len(n) for n, _, _ in results)
    print("=" * (width + 60))
    print("T4 delta=0.4 250-call panel -- PRE-LAUNCH GATE")
    print("=" * (width + 60))
    for name, ok, evidence in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {evidence}")
    failed = [n for n, ok, _ in results if not ok]
    print("=" * (width + 60))
    print(f"{len(results) - len(failed)}/{len(results)} checks PASS")

    print("\nPanel to launch (IVG values joined from the registry, never transcribed):")
    print(f"{'cell':<9} {'seed_DS':>8} {'IVG d0.4':>9} {'IVG d0.6':>9}  contract payload sha256")
    for row in table_rows:
        print(f"{row['cell']:<9} {row['seed_ds']:>8.1f} {row['ivg_d04']:>9.1f} "
              f"{row['ivg_d06']:>9.1f}  {row['contract_sha256']}")
    print(f"\ntotal charged calls at launch: {len(table_rows)} cells x 250 = {len(table_rows)*250}")

    if failed:
        print("\nGATE FAILED -- DO NOT LAUNCH:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("\nGATE GREEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
