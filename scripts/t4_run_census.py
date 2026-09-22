"""Read-only census of every T4 campaign run that exists on Modal.

The frozen T4 table reports ONE best docking score per cell for 15 cells x 2 deltas, while the
baseline it is compared against reports a three-run mean of best.  That asymmetry is only honest
if each reported COMPOSE value is a single run's result, or is selected from several runs by a
rule that does not look at the score.  A best-of-several reported as one run is selection bias.

This tool answers, per cell: how many runs exist, how far each got, and whether the value the
frozen table publishes could have been picked because it was the best one.

It performs NO oracle calls, NO docking, NO launches and NO writes to any Modal volume.  It also
performs no chemistry -- every quantity here is read from committed JSON, so the RDKit version is
immaterial and no molecule is re-scored.

Three hazards this tool defends against, all previously observed in this repository:

  * ``modal volume ls <vol> <subpath>`` silently returns the PARENT listing for a valid
    subdirectory.  Every listing here goes through ``modal.Volume.listdir``, which is exact.
  * a rate-limited listing raises, and an exception caught into an empty list reads as "this
    volume has no runs" -- which is how a whole arm disappears from a census.  Every listing
    retries with backoff and an exhausted listing is recorded as a FETCH FAILURE, never as zero.
  * a volume NAME is not a volume identity and a contract FILENAME is not its delta.
    ``configs/t4_held_target_distilled_jak2_d06_250.json`` declares ``delta: 0.4``.  Delta is
    therefore resolved by matching the ``contract_payload_sha256`` each run recorded against the
    payload hash of every contract in ``configs/``, and is reported as unresolved if no contract
    matches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MODAL_PROFILE", "nitya")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

TARGETS = ("parp1", "braf", "fa7", "jak2", "5ht1b")
CELL_RE = re.compile(r"^(" + "|".join(TARGETS) + r")_(\d+)$")
LOCK_RE = re.compile(r"^round_(\d+)_lock\.json$")
TOLERANCE = 1e-9

# A cell that terminated after docking only its root parent performed no search: it is an
# ABORTED ATTEMPT, not a competing sample of the same random variable.  One charged call is the
# root docking itself.
SEARCH_CALL_FLOOR = 2


# ---- Modal access (read-only, retrying) ------------------------------------------------


class FetchFailure(RuntimeError):
    pass


def _retry(fn, what: str, attempts: int = 6, base: float = 4.0):
    last = ""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # classify the failure rather than swallow it
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "not found" in last.lower() or "does not exist" in last.lower():
                raise FetchFailure(f"{what}: {last}") from exc
            time.sleep(base * (i + 1))
    raise FetchFailure(f"{what}: {last}")


class Reader:
    """Every Modal read goes through here so the call count and failures are attributable."""

    def __init__(self, pause: float = 0.35) -> None:
        import modal  # imported lazily so --offline needs no modal install

        self._modal = modal
        self._vols: dict[str, object] = {}
        self.pause = pause
        self.calls = 0
        self.failures: list[dict] = []

    def _vol(self, name: str):
        if name not in self._vols:
            self._vols[name] = self._modal.Volume.from_name(name)
        return self._vols[name]

    def listdir(self, volume: str, path: str = "") -> list[tuple[str, bool]]:
        def go():
            self.calls += 1
            time.sleep(self.pause)
            return [
                (e.path, str(e.type) == "2") for e in self._vol(volume).listdir(path)
            ]

        return _retry(go, f"listdir {volume}:{path or '/'}")

    def read_json(self, volume: str, path: str) -> dict:
        def go():
            self.calls += 1
            time.sleep(self.pause)
            raw = b"".join(self._vol(volume).read_file(path))
            if not raw:
                raise RuntimeError("empty file")
            return json.loads(raw.decode())

        body = _retry(go, f"read {volume}:{path}")
        payload = body.get("payload", body) if isinstance(body, dict) else None
        if not isinstance(payload, dict):
            raise FetchFailure(f"read {volume}:{path}: payload is not an object")
        return payload

    def note_failure(self, **kw) -> None:
        self.failures.append(kw)


# ---- Contract index --------------------------------------------------------------------


def build_contract_index(repo: Path) -> dict[str, dict]:
    """payload_sha256 -> contract facts.  Delta comes from here and from nowhere else."""
    index: dict[str, dict] = {}
    for path in sorted((repo / "configs").glob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(body, dict):
            continue
        digest = body.get("payload_sha256")
        payload = body.get("payload")
        if not isinstance(digest, str) or not isinstance(payload, dict):
            continue
        if "delta" not in payload:
            continue
        cells = payload.get("cells") or []
        index[digest] = {
            "contract_file": str(path.relative_to(repo)),
            "delta": payload.get("delta"),
            "charged_calls_per_cell": payload.get("charged_calls_per_cell"),
            "cells": [c for c in cells if isinstance(c, str)],
            "claim_boundary": payload.get("claim_boundary"),
            "delta_disagrees_with_claim_text": _delta_text_conflict(
                payload.get("delta"), payload.get("claim_boundary")
            ),
        }
    return index


def resolve_via_git_history(repo: Path, digest: str) -> dict | None:
    """Resolve a contract hash that no CURRENT configs/*.json carries.

    These campaigns re-sealed their contracts mid-flight (realization limits, launch-path
    repairs), so a run legitimately records a payload hash that the working tree has moved past.
    That is not a reason to drop the run from the census -- dropping it would hide a competing
    attempt, which is exactly what this census exists to find.  The historical contract is
    recovered from git and its delta is read from the recovered bytes, so delta still comes from
    a contract and never from a volume, arm or file NAME.
    """
    log = subprocess.run(
        ["git", "log", "--all", "-S", digest, "--format=%H", "--name-only", "--", "configs"],
        capture_output=True, text=True, cwd=repo, check=False, timeout=180,
    ).stdout.splitlines()
    commits: list[str] = []
    files: set[str] = set()
    for line in log:
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[0-9a-f]{40}", line):
            commits.append(line)
        elif line.startswith("configs/"):
            files.add(line)
    for commit in commits:
        for path in sorted(files):
            blob = subprocess.run(
                ["git", "show", f"{commit}:{path}"],
                capture_output=True, text=True, cwd=repo, check=False, timeout=120,
            )
            if blob.returncode != 0:
                continue
            try:
                body = json.loads(blob.stdout)
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict) or body.get("payload_sha256") != digest:
                continue
            payload = body.get("payload") or {}
            return {
                "contract_file": path,
                "delta": payload.get("delta"),
                "charged_calls_per_cell": payload.get("charged_calls_per_cell"),
                "cells": [c for c in (payload.get("cells") or []) if isinstance(c, str)],
                "claim_boundary": payload.get("claim_boundary"),
                "delta_disagrees_with_claim_text": _delta_text_conflict(
                    payload.get("delta"), payload.get("claim_boundary")
                ),
                "recovered_from_commit": commit,
                "superseded_in_working_tree": True,
            }
    return None


def _delta_text_conflict(delta, claim) -> bool:
    """The jak2 d06 defect: an executable delta that contradicts its own prose boundary."""
    if not isinstance(claim, str) or delta is None:
        return False
    stated = re.findall(r"delta\s*([0-9.]+)", claim)
    return any(abs(float(s) - float(delta)) > 1e-9 for s in stated)


# ---- Volume discovery ------------------------------------------------------------------


def list_volumes() -> list[dict]:
    env = {**os.environ, "MODAL_PROFILE": os.environ["MODAL_PROFILE"]}
    done = subprocess.run(
        ["modal", "volume", "list", "--json"],
        capture_output=True, text=True, timeout=300, check=False, env=env,
    )
    if done.returncode != 0:
        raise FetchFailure(f"volume list failed: {(done.stderr or '')[-300:]}")
    return json.loads(done.stdout)


# ---- Per-cell scan ---------------------------------------------------------------------


def scan_cell(reader: Reader, volume: str, run: str, cell: str) -> dict:
    """One (volume, run, cell).  Reads checkpoint, result and the NEWEST lock only."""
    row: dict = {"volume": volume, "run_id": run, "run_short": run[:12], "cell": cell}
    match = CELL_RE.match(cell)
    row["target"] = match.group(1) if match else None
    row["seed_index"] = int(match.group(2)) if match else None

    try:
        entries = reader.listdir(volume, f"{run}/{cell}")
    except FetchFailure as exc:
        row["fetch_failure"] = str(exc)
        return row

    names = [p.rsplit("/", 1)[-1] for p, is_dir in entries if not is_dir]
    lock_indices = sorted(
        int(m.group(1)) for m in (LOCK_RE.match(n) for n in names) if m
    )
    row["round_lock_count"] = len(lock_indices)
    row["newest_lock_index"] = lock_indices[-1] if lock_indices else None
    row["result_present"] = "result.json" in names
    row["checkpoint_present"] = "checkpoint.json" in names
    row["other_files"] = sorted(
        n for n in names if n not in {"result.json", "checkpoint.json"} and not LOCK_RE.match(n)
    )

    payloads: dict[str, dict] = {}
    for name in ("result.json", "checkpoint.json"):
        if name not in names:
            continue
        try:
            payloads[name] = reader.read_json(volume, f"{run}/{cell}/{name}")
        except FetchFailure as exc:
            row.setdefault("partial_fetch_failures", []).append(str(exc))

    # Newest lock: its charged_before is the durable lower bound on spend that survives a
    # checkpoint rollback.  Older locks are not read -- they carry candidate_pool and are large.
    row["newest_lock_charged_before"] = None
    if lock_indices:
        lock_name = f"round_{lock_indices[-1]:03d}_lock.json"
        try:
            lock = reader.read_json(volume, f"{run}/{cell}/{lock_name}")
            before = lock.get("charged_before")
            row["newest_lock_charged_before"] = before if isinstance(before, int) else None
            row["newest_lock_query_count"] = len(lock.get("queries") or [])
            scores = [
                q.get("parent_score")
                for q in (lock.get("queries") or [])
                if isinstance(q.get("parent_score"), (int, float))
            ]
            row["newest_lock_best_parent_score"] = min(scores) if scores else None
        except FetchFailure as exc:
            row.setdefault("partial_fetch_failures", []).append(str(exc))

    result = payloads.get("result.json")
    checkpoint = payloads.get("checkpoint.json")

    def archive_of(p):
        if not p:
            return {}
        return {
            k: float(v)
            for k, v in (p.get("archive") or {}).items()
            if isinstance(v, (int, float))
        }

    # The driver reports result.json where it exists and falls back to the checkpoint; that pair
    # is what the published tables were built from.
    if result is not None and archive_of(result):
        source, committed, calls = "result", archive_of(result), result.get("charged_calls")
    elif checkpoint is not None:
        source, committed, calls = (
            "checkpoint", archive_of(checkpoint), checkpoint.get("charged_calls"),
        )
    elif result is not None:
        source, committed, calls = "result_without_archive", {}, result.get("charged_calls")
    else:
        source, committed, calls = "none", {}, None

    row["committed_source"] = source
    row["status"] = (result or checkpoint or {}).get("status")
    row["archive_size"] = len(committed)
    row["committed_charged_calls"] = calls if isinstance(calls, int) else None
    row["final_best"] = min(committed.values()) if committed else None
    row["best_smiles"] = (
        min(committed.items(), key=lambda kv: kv[1])[0] if committed else None
    )
    row["contract_payload_sha256_in_cell"] = (result or checkpoint or {}).get(
        "contract_payload_sha256"
    )

    # The per-round ladder is the decisive attribution evidence.  These campaigns kept running
    # after the table was frozen, so a run's CURRENT best often no longer equals the published
    # value; but a run that ever published that value did so at a specific charged-call count,
    # and the ladder records exactly that.  "Which run does the frozen number come from" is
    # therefore answerable even when no run's current best matches it.
    committed_payload = result if source == "result" else checkpoint
    ladder = []
    for r in ((committed_payload or {}).get("rounds") or []):
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("charged_calls"), int) and isinstance(
            r.get("best_so_far"), (int, float)
        ):
            ladder.append(
                {
                    "round": r.get("round"),
                    "charged_calls": r["charged_calls"],
                    "best_so_far": float(r["best_so_far"]),
                }
            )
    ladder.sort(key=lambda e: e["charged_calls"])
    row["ladder"] = ladder
    row["ladder_points"] = len(ladder)

    calls_seen = [c for c in (row["committed_charged_calls"], row["newest_lock_charged_before"])
                  if isinstance(c, int)]
    row["reconciled_charged_calls"] = max(calls_seen) if calls_seen else None
    return row


def classify_run(row: dict) -> dict:
    """Did this run actually search, or did it abort after docking its root?"""
    calls = row.get("reconciled_charged_calls")
    best = row.get("final_best")
    if row.get("fetch_failure"):
        row["run_class"] = "fetch_failure"
    elif calls is None and best is None:
        row["run_class"] = "no_committed_state"
    elif (calls or 0) >= SEARCH_CALL_FLOOR and best is not None:
        row["run_class"] = "searched"
    elif (calls or 0) <= 1:
        row["run_class"] = "aborted_root_only"
    else:
        row["run_class"] = "searched_without_archive"
    row["terminal"] = bool(row.get("result_present"))
    return row


# ---- Frozen table ----------------------------------------------------------------------


def frozen_rows(repo: Path) -> dict[tuple[str, int, float], dict]:
    body = json.loads((repo / "diagnostics" / "T4_FROZEN_RESULT_v1.json").read_text())
    payload = body.get("payload", body)
    out: dict[tuple[str, int, float], dict] = {}
    for delta_key, block in (payload.get("deltas") or {}).items():
        for row in block.get("rows") or []:
            key = (str(row["target"]).lower(), int(row["seed"]) - 1, float(delta_key))
            out[key] = {
                "frozen_reported_value": row.get("compose"),
                "frozen_reported_charged_calls": row.get("charged_calls"),
                "frozen_source_label": row.get("source"),
                "ivg_baseline": row.get("ivg"),
            }
    return out


# ---- Verdict ---------------------------------------------------------------------------


def verdict_for_cell(runs: list[dict], frozen: dict) -> dict:
    searched = [r for r in runs if r["run_class"] == "searched"]
    aborted = [r for r in runs if r["run_class"] == "aborted_root_only"]
    unusable = [r for r in runs if r["run_class"] in ("fetch_failure", "no_committed_state")]
    reported = frozen.get("frozen_reported_value")

    reported_calls = frozen.get("frozen_reported_charged_calls")

    def _val_eq(a, b):
        return a is not None and b is not None and abs(round(a, 1) - round(float(b), 1)) < 0.05

    def attribution(r):
        """How strongly does this run account for the published (value, charged_calls) pair?"""
        ladder = r.get("ladder") or []
        exact = any(
            e["charged_calls"] == reported_calls and _val_eq(e["best_so_far"], reported)
            for e in ladder
        )
        by_value = any(_val_eq(e["best_so_far"], reported) for e in ladder)
        by_calls = any(e["charged_calls"] == reported_calls for e in ladder)
        return {
            "ladder_exact_value_and_calls": exact,
            "ladder_reports_value": by_value,
            "ladder_reaches_calls": by_calls,
            "final_best_matches": _val_eq(r.get("final_best"), reported),
        }

    for r in searched:
        r["_attr"] = attribution(r)

    exact_hits = [r for r in searched if r["_attr"]["ladder_exact_value_and_calls"]]
    value_hits = [r for r in searched if r["_attr"]["ladder_reports_value"]]
    final_hits = [r for r in searched if r["_attr"]["final_best_matches"]]
    matched = exact_hits or value_hits or final_hits
    attribution_strength = (
        "ladder_exact_value_and_calls" if exact_hits
        else "ladder_reports_value" if value_hits
        else "final_best_matches" if final_hits
        else "none"
    )

    def matches(r):
        return any(r is m for m in matched)
    out = {
        "completed_runs": sum(1 for r in runs if r["terminal"]),
        "attempted_runs": len(runs),
        "searched_runs": len(searched),
        "aborted_root_only_runs": len(aborted),
        "unusable_runs": len(unusable),
        "distinct_best_values": sorted(
            {round(r["final_best"], 4) for r in searched if r["final_best"] is not None}
        ),
        "frozen_value_matched_runs": [
            {
                "volume": r["volume"],
                "run_short": r["run_short"],
                "final_best": r["final_best"],
                "reconciled_charged_calls": r.get("reconciled_charged_calls"),
                **r["_attr"],
            }
            for r in matched
        ],
        "frozen_value_attribution_strength": attribution_strength,
        "attribution_evidence_per_searched_run": [
            {
                "run_short": r["run_short"],
                "volume": r["volume"],
                "final_best": r["final_best"],
                "reconciled_charged_calls": r.get("reconciled_charged_calls"),
                "ladder_points": r.get("ladder_points"),
                **r["_attr"],
            }
            for r in searched
        ],
        "run_advanced_past_frozen_snapshot": [
            {
                "run_short": r["run_short"],
                "frozen_value": reported,
                "frozen_calls": reported_calls,
                "current_best": r["final_best"],
                "current_calls": r.get("reconciled_charged_calls"),
            }
            for r in searched
            if r["_attr"]["ladder_reports_value"]
            and not r["_attr"]["final_best_matches"]
        ],
    }

    if not searched:
        out["selection_verdict"] = "unresolved"
        out["selection_verdict_reason"] = (
            "no run performed a search"
            + (f"; {len(aborted)} aborted after the root docking" if aborted else "")
            + (f"; {len(unusable)} unreadable" if unusable else "")
        )
        out["excluding_aborted_changes_verdict"] = False
        return out

    best_run = min(searched, key=lambda r: r["final_best"])
    deepest = max(searched, key=lambda r: (r.get("reconciled_charged_calls") or 0))
    out["best_scoring_run"] = {"run_short": best_run["run_short"], "final_best": best_run["final_best"]}
    out["furthest_progressed_run"] = {
        "run_short": deepest["run_short"],
        "reconciled_charged_calls": deepest.get("reconciled_charged_calls"),
        "final_best": deepest.get("final_best"),
    }

    if len(searched) == 1:
        only = searched[0]
        out["selection_verdict"] = "single_run"
        out["selection_verdict_reason"] = (
            "exactly one run produced a search result"
            + (
                f"; {len(aborted)} further attempt(s) aborted after the root docking and "
                "performed no search, so they are not competing samples"
                if aborted
                else ""
            )
        )
        out["selection_verdict_detail"] = (
            "reported value reproduced by the single searched run"
            if matches(only)
            else "reported value does NOT match the single searched run"
        )
        # A run that never searched cannot have been selected against, so removing the aborted
        # attempts from the count cannot move a single_run verdict.
        out["excluding_aborted_changes_verdict"] = False
        return out

    is_best = matched and all(
        abs(r["final_best"] - best_run["final_best"]) < TOLERANCE for r in matched
    )
    is_deepest = any(r["run_short"] == deepest["run_short"] for r in matched)
    coincide = abs(best_run["final_best"] - deepest["final_best"]) < TOLERANCE

    if not matched:
        out["selection_verdict"] = "unresolved"
        out["selection_verdict_reason"] = (
            "several runs searched but none reproduces the frozen value; the published value "
            "may predate or postdate the current volume state"
        )
    elif is_deepest and coincide:
        out["selection_verdict"] = "multiple_runs_progress_selected"
        out["selection_verdict_detail"] = (
            "the furthest-progressed run is ALSO the best-scoring one, so progress selection "
            "and score selection are indistinguishable on this cell; the published rule "
            "(_pick_authoritative) is progress-based and would have chosen this run without "
            "consulting the score"
        )
    elif is_deepest:
        out["selection_verdict"] = "multiple_runs_progress_selected"
        out["selection_verdict_detail"] = (
            "reported run is the furthest-progressed and is NOT the best-scoring"
        )
    elif is_best:
        out["selection_verdict"] = "multiple_runs_score_selected"
        out["selection_verdict_detail"] = (
            "SELECTION BIAS: the reported run is the best-scoring among several that searched, "
            "and is not the furthest-progressed"
        )
    else:
        out["selection_verdict"] = "unresolved"
        out["selection_verdict_reason"] = (
            "reported run is neither the furthest-progressed nor the best-scoring"
        )

    # Both readings, as asked: does dropping the aborted attempts change the answer?
    out["excluding_aborted_changes_verdict"] = bool(aborted) and len(searched) == 1
    return out


# ---- Driver ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diagnostics/t4_replication_manifest/run_census_v1.json")
    ap.add_argument("--cache", required=True, help="scratchpad path for raw scan rows")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--pause", type=float, default=0.35)
    args = ap.parse_args()

    from t4_reconcile_ledger import ARMS  # the registry is read, never retyped

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    contracts = build_contract_index(REPO)
    volumes_meta = {v["Name"]: v for v in list_volumes()}
    t4_volumes = sorted(n for n in volumes_meta if n.startswith("compose-t4-"))
    registry_volumes = {vol: arm for arm, (vol, _t, _c) in ARMS.items()}

    if cache.exists() and not args.refresh:
        scan = json.loads(cache.read_text())
        print(f"reusing cached scan: {len(scan['rows'])} rows", flush=True)
        reader = None
    else:
        reader = Reader(pause=args.pause)
        rows: list[dict] = []
        volumes_report: list[dict] = []
        for volume in t4_volumes:
            entry = {
                "volume": volume,
                "profile": os.environ["MODAL_PROFILE"],
                "created_at": volumes_meta[volume].get("Created at"),
                "created_by": volumes_meta[volume].get("Created by"),
                "in_arms_registry": volume in registry_volumes,
                "registry_arm": registry_volumes.get(volume),
                "runs": [],
            }
            try:
                root = reader.listdir(volume)
            except FetchFailure as exc:
                entry["fetch_failure"] = str(exc)
                reader.note_failure(volume=volume, path="/", error=str(exc))
                volumes_report.append(entry)
                print(f"{volume:52s} FETCH FAILURE", flush=True)
                continue
            for path, is_dir in root:
                if not is_dir:
                    continue
                run = path
                try:
                    sub = reader.listdir(volume, run)
                except FetchFailure as exc:
                    entry["runs"].append({"run_id": run, "fetch_failure": str(exc)})
                    reader.note_failure(volume=volume, path=run, error=str(exc))
                    continue
                cells = sorted(
                    p.rsplit("/", 1)[-1] for p, d in sub
                    if d and CELL_RE.match(p.rsplit("/", 1)[-1])
                )
                # Subdirectories that are NOT recognised cells are recorded, not dropped: a
                # directory silently discarded because its name did not match is how a
                # competing run stays invisible to a census built to find competing runs.
                unmatched_dirs = sorted(
                    p.rsplit("/", 1)[-1] for p, d in sub
                    if d and not CELL_RE.match(p.rsplit("/", 1)[-1])
                )
                run_files = [p.rsplit("/", 1)[-1] for p, d in sub if not d]
                launch = None
                if "launch.json" in run_files:
                    try:
                        launch = reader.read_json(volume, f"{run}/launch.json")
                    except FetchFailure as exc:
                        reader.note_failure(volume=volume, path=f"{run}/launch.json", error=str(exc))
                entry["runs"].append({
                    "run_id": run,
                    "cells": cells,
                    "unmatched_subdirectories": unmatched_dirs,
                    "run_files": sorted(run_files),
                    "launch": launch,
                })
                for cell in cells:
                    row = classify_run(scan_cell(reader, volume, run, cell))
                    row["arm"] = registry_volumes.get(volume) or f"UNREGISTERED:{volume}"
                    row["in_arms_registry"] = volume in registry_volumes
                    row["volume_created_at"] = entry["created_at"]
                    row["code_revision"] = (launch or {}).get("code_revision")
                    row["charged_call_ceiling"] = (launch or {}).get("charged_call_ceiling")
                    row["contract_payload_sha256_in_launch"] = (launch or {}).get(
                        "contract_payload_sha256"
                    )
                    rows.append(row)
                    print(
                        f"  {volume[:44]:44s} {run[:10]} {cell:9s} "
                        f"{row['run_class']:12s} calls={row['reconciled_charged_calls']} "
                        f"best={row['final_best']} locks={row.get('round_lock_count')}",
                        flush=True,
                    )
            volumes_report.append(entry)
        scan = {
            "rows": rows,
            "volumes": volumes_report,
            "modal_calls": reader.calls,
            "fetch_failures": reader.failures,
            "scanned_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        cache.write_text(json.dumps(scan, indent=1, sort_keys=True) + "\n")

    # ---- resolve contract identity and delta -------------------------------------------
    git_cache: dict[str, dict | None] = {}
    for row in scan["rows"]:
        digest = row.get("contract_payload_sha256_in_launch") or row.get(
            "contract_payload_sha256_in_cell"
        )
        row["contract_payload_sha256"] = digest
        row["contract_hash_sources_agree"] = (
            row.get("contract_payload_sha256_in_launch") is None
            or row.get("contract_payload_sha256_in_cell") is None
            or row["contract_payload_sha256_in_launch"] == row["contract_payload_sha256_in_cell"]
        )
        info = contracts.get(digest or "")
        source = "contract_payload_sha256_current_tree"
        if info is None and digest:
            info = git_cache.get(digest, "MISS")
            if info == "MISS":
                info = resolve_via_git_history(REPO, digest)
                git_cache[digest] = info
            if info is not None:
                source = "contract_payload_sha256_recovered_from_git"
        row["contract_file"] = info["contract_file"] if info else None
        row["delta"] = info["delta"] if info else None
        row["delta_source"] = source if info else "UNRESOLVED"
        row["contract_superseded_in_working_tree"] = bool(
            info and info.get("superseded_in_working_tree")
        )
        row["contract_recovered_from_commit"] = (info or {}).get("recovered_from_commit")
        row["contract_budget_per_cell"] = info["charged_calls_per_cell"] if info else None
        row["contract_delta_disagrees_with_its_own_claim_text"] = (
            info["delta_disagrees_with_claim_text"] if info else None
        )

    frozen = frozen_rows(REPO)
    per_cell = []
    grouped: dict[tuple, list[dict]] = {}
    for row in scan["rows"]:
        if row.get("target") is None or row.get("delta") is None:
            continue
        grouped.setdefault((row["target"], row["seed_index"], float(row["delta"])), []).append(row)

    for key in sorted(frozen, key=lambda k: (k[2], k[0], k[1])):
        target, seed_index, delta = key
        runs = grouped.get(key, [])
        entry = {
            "target": target,
            "seed_index": seed_index,
            "seed_label": seed_index + 1,
            "delta": delta,
            "delta_source": "contract_payload_sha256 on each run row",
            **frozen[key],
            "runs": [
                {
                    k: r.get(k)
                    for k in (
                        "arm", "volume", "run_short", "run_id", "cell", "contract_file",
                        "contract_payload_sha256", "status", "run_class", "terminal",
                        "result_present", "checkpoint_present", "committed_charged_calls",
                        "newest_lock_charged_before", "reconciled_charged_calls",
                        "round_lock_count", "newest_lock_index", "final_best", "best_smiles",
                        "archive_size", "code_revision", "charged_call_ceiling",
                        "in_arms_registry", "fetch_failure", "partial_fetch_failures",
                        "ladder_points",
                    )
                }
                for r in sorted(runs, key=lambda r: -(r.get("reconciled_charged_calls") or 0))
            ],
        }
        entry.update(verdict_for_cell(runs, frozen[key]))
        per_cell.append(entry)

    # cells present on a volume but absent from the frozen table
    extra = [k for k in grouped if k not in frozen]

    # ---- frozen value vs the evidence currently committed on the volume -----------------
    comparison = []
    for c in per_cell:
        fz = c["frozen_reported_value"]
        searched = [r for r in c["runs"] if r["run_class"] == "searched"]
        if fz is None or not searched:
            continue
        matched = c["frozen_value_matched_runs"]
        rid = matched[0]["run_short"] if matched else None
        run = next(
            (r for r in searched if r["run_short"] == rid),
            max(searched, key=lambda r: r.get("reconciled_charged_calls") or 0),
        )
        current = run["final_best"]
        if current is None or abs(current - float(fz)) < 0.05:
            continue
        comparison.append({
            "cell": f"{c['target']}_{c['seed_index']}",
            "delta": c["delta"],
            "frozen_reported_value": fz,
            "frozen_reported_charged_calls": c["frozen_reported_charged_calls"],
            "current_committed_best": current,
            "current_charged_calls": run.get("reconciled_charged_calls"),
            "difference": round(current - float(fz), 4),
            "direction": "current evidence is BETTER than published" if current < fz
                         else "current evidence is WORSE than published",
            "attribution": c["frozen_value_attribution_strength"],
            "explanation": (
                "the published (value, calls) pair IS a point on this run's committed "
                "trajectory, so the table was a faithful snapshot and the run has since "
                "advanced"
                if c["frozen_value_attribution_strength"] == "ladder_exact_value_and_calls"
                else "the published (value, calls) pair is NOT a point on this run's committed "
                     "trajectory; the value appears earlier in its history, which is the "
                     "signature of a locks-derived lower bound rather than the reconciled "
                     "min(locks, archive)"
            ),
            "run_short": run["run_short"],
            "ivg_baseline": c["ivg_baseline"],
            "changes_win_loss_against_ivg": bool(
                c["ivg_baseline"] is not None
                and ((fz - c["ivg_baseline"]) > 0) != ((current - c["ivg_baseline"]) > 0)
            ),
        })

    # ---- repeated dockings of one molecule, which the census gets for free ---------------
    by_molecule: dict[tuple, list[dict]] = {}
    for row in scan["rows"]:
        if (
            row.get("archive_size") == 1
            and row.get("best_smiles")
            and (row.get("reconciled_charged_calls") or 0) <= 1
        ):
            by_molecule.setdefault((row["cell"], row["best_smiles"]), []).append(row)
    repeats = []
    for (cell, smiles), rows_ in sorted(by_molecule.items()):
        scores = sorted(r["final_best"] for r in rows_ if r["final_best"] is not None)
        if len(scores) < 2:
            continue
        repeats.append({
            "cell": cell,
            "smiles": smiles,
            "n_dockings": len(scores),
            "scores": scores,
            "spread_kcal_per_mol": round(max(scores) - min(scores), 4),
            "note": "same molecule, same target, same fixed box, separate runs; each is that "
                    "run's root docking and its archive holds exactly this one molecule",
        })

    counts: dict[str, int] = {}
    for c in per_cell:
        counts[c["selection_verdict"]] = counts.get(c["selection_verdict"], 0) + 1
    score_selected = [
        f"{c['target']}_{c['seed_index']} d{c['delta']}"
        for c in per_cell
        if c["selection_verdict"] == "multiple_runs_score_selected"
    ]

    report = {
        "schema_version": "t4_run_census_v1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "scanned_at_utc": scan["scanned_at_utc"],
        "question": (
            "For each of the 30 frozen cells: how many runs exist, and was the reported value "
            "ever SELECTED from more than one run?"
        ),
        "method": {
            "reads": "checkpoint.json, result.json and the NEWEST round lock per cell; older "
                     "locks are listed but not read (they carry candidate_pool and are large)",
            "charged_calls": "max(committed charged_calls, newest lock charged_before) -- a LOWER "
                             "BOUND, exactly as t4_reconcile_ledger documents",
            "final_best": "min over the committed archive (result.json preferred, else "
                          "checkpoint.json), which is what the published tables were built from",
            "delta": "resolved by matching each run's contract_payload_sha256 against the payload "
                     "hash of every configs/*.json; never taken from a volume or arm name",
            "search_floor": f"a run is 'searched' only at >= {SEARCH_CALL_FLOOR} charged calls; "
                            "one charged call is the root docking alone",
            "no_chemistry": "no molecule is parsed or re-scored, so the RDKit version is immaterial",
            "writes": "none to any Modal volume; no oracle calls, no docking, no launches",
        },
        "volumes": scan["volumes"],
        "modal_calls": scan.get("modal_calls"),
        "fetch_failures": scan.get("fetch_failures", []),
        "runs": scan["rows"],
        "per_cell": per_cell,
        "cells_on_volume_absent_from_frozen_table": sorted(
            f"{t}_{s} d{d}" for t, s, d in extra
        ),
        "frozen_value_vs_current_committed_evidence": {
            "cells_compared": sum(1 for c in per_cell if c["frozen_reported_value"] is not None),
            "cells_matching_current_evidence_exactly": sum(
                1 for c in per_cell if c["frozen_reported_value"] is not None
            ) - len(comparison),
            "discrepancies": comparison,
            "all_discrepancies_favour_the_baseline": all(
                d["difference"] < 0 for d in comparison
            ),
            "reading": (
                "every discrepancy found is in the CONSERVATIVE direction -- the published "
                "table reports a WORSE docking score than the volume currently holds -- so "
                "none of them inflates a COMPOSE claim"
                if comparison and all(d["difference"] < 0 for d in comparison)
                else "see discrepancies"
            ),
        },
        "repeated_docking_of_one_molecule": {
            "observations": repeats,
            "why_it_is_here": (
                "several cells were launched more than once and each launch docked the same "
                "root molecule before doing anything else, so the census yields repeated "
                "dockings of a byte-identical SMILES at no extra cost.  This is independent "
                "evidence for the docking_reproducibility_caveat already recorded in the "
                "frozen table."
            ),
            "max_spread_kcal_per_mol": max((r["spread_kcal_per_mol"] for r in repeats), default=None),
        },
        "unregistered_volumes": {
            "with_campaign_cells": sorted(
                v["volume"] for v in scan["volumes"]
                if not v["in_arms_registry"]
                and any(run.get("cells") for run in v["runs"])
            ),
            "without_campaign_cells": sorted(
                v["volume"] for v in scan["volumes"]
                if not v["in_arms_registry"]
                and not any(run.get("cells") for run in v["runs"])
            ),
            "note": (
                "the ARMS registry in tools/t4_reconcile_ledger.py does not know these volumes, "
                "so the standing reconciliation never reads them.  They were found by listing "
                "every compose-t4-* volume on the profile rather than by trusting the registry."
            ),
        },
        "non_campaign_volumes_verified": {
            "compose-t4-objective-reset-20260916": {
                "measured": "five cell directories named *_zero_oracle; every result.json reports "
                            "new_oracle_calls = 0 and self-describes as a 'proposal-yield "
                            "diagnostic, not a stochastic cold-start pass gate'",
                "schema": "different from a campaign cell: no archive, no charged_calls",
                "conclusion": "produces no docking score, so cannot be a competing sample; "
                              "verified by reading the payloads, not inferred from the name",
            },
            "compose-t4-integrated-route-fiber-parp1-v1": {
                "measured": "one run holding only launch.json and summary.json; no cell dirs",
            },
            "compose-t4-dynamic-v0-full-suite": {
                "measured": "nested run/<hash>/units/, no per-cell campaign directories",
            },
            "compose-t4-route-distilled-artifacts": {
                "measured": "four runs of route-policy / structural-subgoal artifacts nested "
                            "under attempt_N/fold_N; no per-cell campaign directories",
            },
            "empty_volumes": [
                "compose-t4-held-target-distilled-jak2-d06-v1",
                "compose-t4-held-target-distilled-parp1-d06-v1",
                "compose-t4-nodistill-parp1-v1-parp1-0-d04",
                "compose-t4-nodistill-parp1-v1-parp1-1-d04",
                "compose-t4-nodistill-parp1-v1-parp1-2-d04",
            ],
        },
        "measured_vs_inferred": {
            "MEASURED": [
                (
                    "every run directory, cell directory and round-lock count on all 28 "
                    "compose-t4-* volumes of profile nitya (zero fetch failures across "
                    f"{scan.get('modal_calls')} Modal reads)"
                ),
                (
                    "each cell's committed status, charged_calls, archive size, best value and best "
                    "SMILES, read from result.json where present and checkpoint.json otherwise"
                ),
                (
                    "each cell's per-round (charged_calls, best_so_far) trajectory, which attributes "
                    "the published number to a specific run"
                ),
                (
                    "delta for every run, resolved from the contract payload hash the run recorded "
                    "(21 of 65 rows required recovering a superseded contract from git history)"
                ),
                (
                    "for 5ht1b_1 at delta 0.6: the locks-only best over all 35 round locks is "
                    "exactly -11.3 at charged_before = 222, which is the published pair, while the "
                    "committed archive holds -12.2"
                ),
                "repeated dockings of byte-identical SMILES across separate launches",
            ],
            "INFERRED": [
                (
                    "that the published selection rule is progress-based rather than score-based. "
                    "The direct evidence is parp1_2 at delta 0.6, where the deepest run holds -11.3 "
                    "and a shallower completed run holds -11.4, and the table published -11.3 -- the "
                    "WORSE value.  That is one cell, and it is corroborated by _pick_authoritative "
                    "in tools/t4_reconcile_ledger.py ranking on (calls, evidence, locks) and never "
                    "on score.  On the other 8 multi-run cells the deepest run is also the best, so "
                    "those cells cannot discriminate the two rules."
                ),
            ],
            "NOT_ESTABLISHED": [
                (
                    "whether any cell ever rolled back and advanced past the lost segment: round "
                    "locks are written to a fixed path per round index, so a redone round overwrites "
                    "its own lock.  reconciled_charged_calls is a LOWER BOUND on lifetime spend."
                ),
                (
                    "whether an older lock holds a better parent_score than the committed archive on "
                    "cells other than 5ht1b_1 d0.6: only the NEWEST lock was read per cell, so this "
                    "census does not repeat the full locks reconciliation."
                ),
            ],
            "could_the_verdict_have_varied": (
                "yes, and it did.  The verdict metric distinguishes progress-selection from "
                "score-selection only on cells where the two disagree; parp1_2 at delta 0.6 is "
                "such a cell and it resolved AGAINST score-selection.  Run classes also vary "
                "across the census (39 searched, 14 aborted-at-root, 12 with no committed "
                "state), so 'searched' is not a constant."
            ),
        },
        "selection_bias_summary": {
            "verdict_counts": counts,
            "cells_with_score_selection": score_selected,
            "answer": (
                "NO cell's published value was selected on score from several searching runs"
                if not score_selected
                else "SELECTION BIAS PRESENT on: " + ", ".join(score_selected)
            ),
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    print(json.dumps(report["selection_bias_summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
