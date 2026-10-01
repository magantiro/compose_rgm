"""Explicit local Open Babel / QuickVina2 adapter with pinned file inputs.

No downloads and no implicit working-directory changes. Each evaluation retains
its own input, command logs and pose files. Conformer generation is unseeded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from compose_v4.data.immutable_artifact import write_bytes_if_absent


def verified_file(path: Path, expected_sha256: str, *, executable: bool = False) -> Path:
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"a lowercase SHA-256 is required for {path}")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"asset must be a regular file: {resolved}")
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"asset SHA-256 mismatch: {resolved}, expected {expected_sha256}, got {digest}"
        )
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"tool is not executable: {resolved}")
    return resolved


@dataclass(frozen=True)
class DockingConfig:
    obabel: Path
    obabel_sha256: str
    qvina: Path
    qvina_sha256: str
    receptor: Path
    receptor_sha256: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    seed: int
    cpu: int = 1

    def __post_init__(self) -> None:
        for name in ("center", "size"):
            values = getattr(self, name)
            if len(values) != 3 or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError(f"docking {name} must contain three finite coordinates")
        if any(v <= 0 for v in self.size):
            raise ValueError("docking box size must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("docking seed must be a nonnegative signed 32-bit integer")
        if type(self.cpu) is not int or self.cpu < 1:
            raise ValueError("docking cpu must be a positive integer")


class QuickVinaEvaluator:
    """Chargeable evaluator. Constructing it verifies assets but runs no tools."""

    def __init__(self, config: DockingConfig, output: Path):
        self.config, self.output = config, Path(output)
        self.obabel = verified_file(config.obabel, config.obabel_sha256, executable=True)
        self.qvina = verified_file(config.qvina, config.qvina_sha256, executable=True)
        self.receptor = verified_file(config.receptor, config.receptor_sha256)

    def identity(self) -> dict:
        return {
            "schema": "compose.quickvina2.v1",
            "obabel": {"path": str(self.obabel), "sha256": self.config.obabel_sha256},
            "qvina": {"path": str(self.qvina), "sha256": self.config.qvina_sha256},
            "receptor": {"path": str(self.receptor), "sha256": self.config.receptor_sha256},
            "center": self.config.center,
            "size": self.config.size,
            "seed": self.config.seed,
            "cpu": self.config.cpu,
            "num_modes": 10,
            "exhaustiveness": 1,
            "conformer": "obabel --gen3D, unseeded",
            "timeouts_seconds": [120, 60, 300],
        }

    def __call__(self, smiles: str) -> float:
        # Recheck physical assets before every call. A changed binary or receptor
        # must not produce a result under the old run identity.
        verified_file(self.obabel, self.config.obabel_sha256, executable=True)
        verified_file(self.qvina, self.config.qvina_sha256, executable=True)
        verified_file(self.receptor, self.config.receptor_sha256)
        self.output.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="query-", dir=self.output))
        mol, ligand, pose = (folder / name for name in ("ligand.mol", "ligand.pdbqt", "pose.pdbqt"))
        command = [
            str(self.qvina),
            "--receptor",
            str(self.receptor),
            "--ligand",
            str(ligand),
            "--out",
            str(pose),
        ]
        for prefix, values in (("center", self.config.center), ("size", self.config.size)):
            for axis, number in zip("xyz", values, strict=True):
                command.extend([f"--{prefix}_{axis}", str(number)])
        command.extend(
            [
                "--cpu",
                str(self.config.cpu),
                "--num_modes",
                "10",
                "--exhaustiveness",
                "1",
                "--seed",
                str(self.config.seed),
            ]
        )
        commands = [
            ([str(self.obabel), f"-:{smiles}", "--gen3D", "-O", str(mol)], 120),
            ([str(self.obabel), str(mol), "-O", str(ligand)], 60),
            (command, 300),
        ]
        write_bytes_if_absent(
            folder / "input.json",
            json.dumps(
                {
                    "smiles": smiles,
                    "protocol": self.identity(),
                    "commands": commands,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode(),
        )
        for index, (argv, timeout) in enumerate(commands):
            try:
                process = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as error:
                write_bytes_if_absent(folder / f"stage_{index}.stdout", error.stdout or b"")
                write_bytes_if_absent(folder / f"stage_{index}.stderr", error.stderr or b"")
                raise RuntimeError(
                    f"docking stage {index} timed out, retained files at {folder}"
                ) from error
            write_bytes_if_absent(folder / f"stage_{index}.stdout", process.stdout)
            write_bytes_if_absent(folder / f"stage_{index}.stderr", process.stderr)
            if process.returncode:
                raise RuntimeError(
                    f"docking stage {index} exited {process.returncode}, retained files at {folder}"
                )
        with pose.open() as stream:
            for line in stream:
                if line.startswith("REMARK VINA RESULT"):
                    score = float(line.split()[3])
                    if not math.isfinite(score):
                        raise ValueError(f"nonfinite docking result in {pose}")
                    write_bytes_if_absent(
                        folder / "score.json", json.dumps({"score": score}).encode()
                    )
                    return score
        raise ValueError(f"no VINA score in {pose}")
