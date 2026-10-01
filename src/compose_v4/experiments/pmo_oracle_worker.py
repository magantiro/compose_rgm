"""Keep the pinned PyTDC oracle in its own RDKit 2023.9.6 process."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import TextIO

from compose_v4.experiments.pmo_oracle_assets import (
    AssetPinnedOracle,
    assert_positive_control,
    pinned_working_directory,
    requires_positive_control,
)


def construct_oracle(factory, task: str, asset_root: Path | None):
    """Resolve eager and lazy PyTDC assets against the same pinned root."""
    if asset_root is None:
        return factory(name=task)
    with pinned_working_directory(asset_root):
        return factory(name=task)


def serve(oracle, task: str, source: TextIO, sink: TextIO, identity: dict) -> None:
    """Emit one handshake, then answer numbered score requests in order."""
    sink.write(json.dumps({"status": "ready", "task": task, "identity": identity}) + "\n")
    sink.flush()
    for expected, line in enumerate(source):
        request = json.loads(line)
        if request == {"command": "close"}:
            return
        if (
            not isinstance(request, dict)
            or set(request) != {"index", "smiles"}
            or type(request["index"]) is not int
            or request["index"] != expected
            or not isinstance(request["smiles"], str)
            or not request["smiles"]
        ):
            raise ValueError(f"invalid PMO oracle request at index {expected}")
        with contextlib.redirect_stdout(sys.stderr):
            score = float(oracle(request["smiles"]))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"PMO oracle returned a value outside [0, 1] at {expected}")
        sink.write(json.dumps({"index": expected, "score": score}) + "\n")
        sink.flush()


def check_oracle_environment():
    """Verify the isolated chemistry kernel without constructing or calling an oracle."""
    if not sys.version_info[:2] == (3, 11):
        raise ValueError("PMO oracle requires Python 3.11")
    try:
        versions = {name: importlib.metadata.version(name) for name in ("PyTDC", "rdkit", "numpy")}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f"PMO oracle environment is missing {error.name}") from error
    if versions != {"PyTDC": "1.1.15", "rdkit": "2023.9.6", "numpy": "1.26.4"}:
        raise ValueError(f"PMO oracle environment is not pinned: {versions}")
    shim = types.ModuleType("rdkit.six")
    shim.string_types = (str,)
    shim.iteritems = lambda mapping: iter(mapping.items())
    import rdkit

    sys.modules["rdkit.six"] = shim
    rdkit.six = shim
    from tdc import Oracle

    return Oracle, versions


def _real_oracle(task: str, asset_root: Path | None):
    Oracle, versions = check_oracle_environment()

    manifest_path = Path(__file__).resolve().parents[3] / "experiments/pmo/assets.json"
    manifest = json.loads(manifest_path.read_text())
    audit = manifest["oracle_audit"]
    audit_path = manifest_path.parent / audit["path"]
    with audit_path.open("rb") as stream:
        audit_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if audit_sha256 != audit["sha256"]:
        raise ValueError(f"PMO oracle task registry changed: {audit_path}")
    if task not in json.loads(audit_path.read_text())["tasks"]:
        raise ValueError(f"PMO oracle task is outside the declared suite: {task}")
    if task in manifest["oracle_assets"]:
        if asset_root is None:
            raise ValueError(f"{task} requires the pinned PyTDC oracle asset directory")
        asset = manifest["oracle_assets"][task]
        path = (asset_root / asset["relative_path"]).resolve(strict=True)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != asset["sha256"]:
            raise ValueError(f"PMO oracle asset identity mismatch: {path}")
    else:
        actual = None
    with contextlib.redirect_stdout(sys.stderr):
        oracle = construct_oracle(Oracle, task, asset_root)
        control = None
        if requires_positive_control(task):
            assert asset_root is not None
            oracle = AssetPinnedOracle(oracle, asset_root, name=task)
            control = assert_positive_control(oracle, task)
    return oracle, {"versions": versions, "positive_control": control, "asset_sha256": actual}


class PmoOracleClient:
    """Sequential JSON-lines bridge to the pinned oracle environment."""

    def __init__(
        self,
        *,
        python: Path,
        task: str,
        asset_root: Path | None,
        source_root: Path,
        stderr_path: Path,
    ) -> None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = stderr_path.open("a")
        env = dict(os.environ)
        env["PYTHONPATH"] = str((source_root / "src").resolve())
        command = [str(Path(python).resolve()), "-m", __name__, "--task", task]
        if asset_root is not None:
            command.extend(("--asset-root", str(Path(asset_root).resolve())))
        try:
            self._process = subprocess.Popen(
                command,
                cwd=source_root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
            )
            self._index = 0
            assert self._process.stdout is not None
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"PMO oracle worker exited before handshake: {stderr_path}")
            handshake = json.loads(line)
            if handshake.get("status") != "ready" or handshake.get("task") != task:
                raise ValueError(f"unexpected PMO oracle handshake: {handshake}")
            self.identity = handshake["identity"]
        except BaseException:
            if hasattr(self, "_process") and self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=10)
            self._stderr.close()
            raise

    def __call__(self, smiles: str) -> float:
        if self._process.poll() is not None:
            raise RuntimeError("PMO oracle worker is no longer running")
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps({"index": self._index, "smiles": smiles}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("PMO oracle worker stopped before returning a score")
        response = json.loads(line)
        if response.get("index") != self._index or not isinstance(
            response.get("score"), (int, float)
        ):
            raise ValueError(f"PMO oracle response does not match query {self._index}")
        self._index += 1
        return float(response["score"])

    def close(self) -> None:
        try:
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write('{"command":"close"}\n')
                    self._process.stdin.flush()
                except BrokenPipeError:
                    pass
                self._process.wait(timeout=10)
        finally:
            self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--check-env", action="store_true", help="verify packages without an oracle call"
    )
    args = parser.parse_args()
    if args.check_env:
        if args.task is not None or args.asset_root is not None:
            parser.error("--check-env cannot be combined with --task or --asset-root")
        try:
            _, versions = check_oracle_environment()
        except (ImportError, ValueError) as error:
            parser.exit(1, f"PMO oracle preflight failed: {error}\n")
        print(json.dumps({"status": "ready", "versions": versions}, sort_keys=True))
        return
    if args.task is None:
        parser.error("--task is required unless --check-env is set")
    oracle, identity = _real_oracle(args.task, args.asset_root)
    serve(oracle, args.task, sys.stdin, sys.stdout, identity)


if __name__ == "__main__":
    main()
