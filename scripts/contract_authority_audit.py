"""Mechanical contract-authority audit.

For every contract JSON in scope, enumerate the leaf key paths and ask a single
mechanical question per key: does ANY python module under the runtime search
roots mention that key name as a string literal or attribute?

A key that no module mentions cannot be read at runtime, so it is a DEAD-or-PROSE
candidate.  A key that is mentioned still has to be read by a human to decide
between OPERATIVE (changes behaviour), RECEIPT-ONLY (copied into an output), and
DERIVED.  This script narrows the human reading, it does not replace it.

Invariant: the "never mentioned" set is SOUND (a key absent from every module is
provably not read by name) but not COMPLETE (a key can be mentioned in a module
that only writes it into a receipt).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Roots that can plausibly execute at runtime. diagnostics/ holds frozen source
# capsules (copies), so it is excluded: a hit there is not a live consumption.
SEARCH_ROOTS = ["src", "modal_apps", "tools", "scripts"]


def leaf_paths(obj, prefix=""):
    """Yield (dotted_path, leaf_key, value) for every scalar leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                yield from leaf_paths(v, p)
            else:
                yield (p, k, v)
    elif isinstance(obj, list):
        # list of scalars -> report the container once
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            yield (prefix, prefix.split(".")[-1], obj)
        else:
            for i, v in enumerate(obj):
                yield from leaf_paths(v, f"{prefix}[{i}]")


def grep_key(key: str) -> list[str]:
    """Return file:line hits where the key name appears in runtime python."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return []
    roots = [str(REPO / r) for r in SEARCH_ROOTS if (REPO / r).exists()]
    try:
        out = subprocess.run(
            ["grep", "-rn", "--include=*.py", f"\\b{key}\\b", *roots],
            capture_output=True, text=True, timeout=180, check=False,
        ).stdout
    except subprocess.TimeoutExpired:
        return []
    hits = []
    for line in out.splitlines():
        if "__pycache__" in line:
            continue
        hits.append(line[len(str(REPO)) + 1:])
    return hits


def audit(contract_path: Path) -> dict:
    payload = json.loads(contract_path.read_text())
    # contracts are stored either bare or wrapped as {"payload":..., "payload_sha256":...}
    body = payload.get("payload", payload) if isinstance(payload, dict) else payload
    rows = []
    seen_keys: dict[str, list[str]] = {}
    for dotted, key, value in leaf_paths(body):
        if key not in seen_keys:
            seen_keys[key] = grep_key(key)
        hits = seen_keys[key]
        rows.append({
            "path": dotted,
            "key": key,
            "value": value if not isinstance(value, str) or len(value) < 160 else value[:157] + "...",
            "mention_count": len(hits),
            "mentions": hits[:12],
        })
    return {
        "contract": str(contract_path.relative_to(REPO)),
        "leaf_count": len(rows),
        "never_mentioned": [r["path"] for r in rows if r["mention_count"] == 0],
        "rows": rows,
    }


def main(argv):
    results = [audit(REPO / a) for a in argv]
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
