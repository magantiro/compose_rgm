"""Client for the production-pinned shared canonicalizer/evaluator.

    baseline-native environment
            |  raw SMILES
    shared evaluator, pinned to production RDKit 2024.3.5
            |  canonical key + validity + oracle score

The alternative -- forcing every baseline into one RDKit -- breaks the old
implementations outright: MARS's stack tops out well below the production pin,
and GraphXForm asks for a different one again. So baselines keep their native
environment, molecules cross the boundary as raw SMILES, and canonicalization
and scoring happen exactly once in a pinned subprocess. That makes
``unique_valid_canonical_evaluations`` genuinely comparable and stops a
canonicalization difference from silently changing a budget.

The client speaks line-delimited JSON to
``scripts/compose_shared_evaluator_server.py`` running under the pinned
interpreter. It is deliberately a subprocess and not an import: the whole point
is that the evaluator's RDKit is *not* the caller's RDKit.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SERVER = REPO / "scripts" / "compose_shared_evaluator_server.py"

#: Default location of the pinned interpreter. Overridable, because the path is
#: machine-specific; the VERSION it must report is not.
DEFAULT_PINNED_PYTHON = Path("/tmp/baseline_envs/rdkit-prod-2024_3_5/bin/python")
REQUIRED_RDKIT = "2024.03.5"


class SharedEvaluatorError(RuntimeError):
    """The shared evaluator could not be started or spoke unexpectedly."""


@dataclass
class SharedEvaluator:
    """A pinned-RDKit canonicalizer and objective, as a subprocess.

    Use as a context manager, or call :meth:`close` when finished.
    """

    objective: str = "developability"
    python: Path = DEFAULT_PINNED_PYTHON
    allow_unpinned: bool = False
    identity: dict[str, Any] = field(default_factory=dict, init=False)
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not Path(self.python).exists():
            raise SharedEvaluatorError(
                f"pinned interpreter not found at {self.python}. Build it with:\n"
                f"  python3.11 -m venv {Path(self.python).parents[1]}\n"
                f"  {Path(self.python).parents[0]}/pip install "
                f"'rdkit=={REQUIRED_RDKIT}' 'numpy==1.26.4'"
            )
        cmd = [str(self.python), str(SERVER), "--objective", self.objective]
        if self.allow_unpinned:
            cmd.append("--allow-unpinned")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        handshake = self._proc.stdout.readline()
        if not handshake:
            stderr = self._proc.stderr.read()
            raise SharedEvaluatorError(f"evaluator did not start: {stderr.strip()}")
        self.identity = json.loads(handshake)
        if not self.identity.get("ready"):
            raise SharedEvaluatorError(f"unexpected handshake: {self.identity}")
        if not self.allow_unpinned and self.identity.get("rdkit") != REQUIRED_RDKIT:
            raise SharedEvaluatorError(
                f"evaluator reports RDKit {self.identity.get('rdkit')}, "
                f"required {REQUIRED_RDKIT}"
            )

    # ---- Lifecycle ----

    def __enter__(self) -> SharedEvaluator:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - closing must not raise
            self._proc.kill()
        finally:
            self._proc = None

    # ---- Requests ----

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            raise SharedEvaluatorError("evaluator is closed")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read()
            raise SharedEvaluatorError(f"evaluator died: {stderr.strip()}")
        return json.loads(line)

    def evaluate(self, smiles: str) -> dict[str, Any]:
        """Canonical key, validity and score for one raw SMILES."""
        return self._request({"smiles": smiles})

    def canonical(self, smiles: str) -> str | None:
        return self.evaluate(smiles)["canonical"]

    def score(self, smiles: str) -> float:
        """Score, with an invalid molecule returning 0.0 as PMO's oracle does."""
        reply = self.evaluate(smiles)
        value = reply.get("score")
        return 0.0 if value is None else float(value)


def compare_canonicalization(
    smiles: list[str],
    evaluator: SharedEvaluator,
    local_canonical,
) -> dict[str, Any]:
    """Push a fixed panel through both RDKits and record every disagreement.

    Disagreements are *recorded*, never smoothed: a canonical-key difference
    between the baseline's RDKit and the pinned one is exactly the failure mode
    the shared evaluator exists to make visible, and the count of them is a
    reportable property of the pairing.
    """
    agreements = 0
    disagreements: list[dict[str, Any]] = []
    local_only_invalid: list[str] = []
    pinned_only_invalid: list[str] = []

    for smi in smiles:
        pinned = evaluator.canonical(smi)
        local = local_canonical(smi)
        if pinned is None and local is None:
            agreements += 1
        elif pinned is None:
            pinned_only_invalid.append(smi)
        elif local is None:
            local_only_invalid.append(smi)
        elif pinned == local:
            agreements += 1
        else:
            disagreements.append({"input": smi, "pinned": pinned, "local": local})

    return {
        "n": len(smiles),
        "agreements": agreements,
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
        "valid_in_local_only": local_only_invalid,
        "valid_in_pinned_only": pinned_only_invalid,
        "pinned_rdkit": evaluator.identity.get("rdkit"),
    }
