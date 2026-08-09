"""Materialize the Process-V2 P50 scorer state where its seal reproduces.

WHY THIS EXISTS
---------------
``_scratch_runtime_for_score_revision`` reconstructs the predecessor scorer from
a seed and refuses to continue unless the rebuilt base state hashes to the
frozen ``initial_model_state_sha256``.  That check is correct and must not be
weakened -- but it is only *satisfiable* on the platform that produced the
constant.  The frozen value was produced on this Modal image (debian/linux
x86-64); a macOS arm64 host running the identical pinned versions rebuilds the
same architecture -- 118 parameters, zero shape mismatches, same seed, same ring
catalog -- with different numbers, because PyTorch's CPU ``normal_`` fill is
vectorized per architecture and is only reproducible on the same platform.

So the local compile cannot re-derive the frozen state, and pinning versions
cannot fix that: it is not a version difference.

This app closes the gap the honest way.  It builds the scorer HERE, where the
seal passes, and returns the authenticated bytes.  A local consumer then loads
them and asserts the same hash instead of re-deriving it, which is a STRICTER
check than the seed reconstruction: reconstruction proves an environment can
regenerate the state, loading-and-verifying proves the state actually in use is
the frozen one.  Nothing about the invariant is relaxed.

The alternative -- overwriting the expected constant with whatever the local
machine produces -- is the ``neutralize_catalog_drift()`` antipattern and would
have produced a split-brain corpus, since the already-compiled rows were scored
under the Modal-initialized model.  See CLAUDE.md, "Fail closed on catalog
drift".

CPU ONLY.  No GPU is requested or needed: this constructs a model and hashes
its state, it does not train.  The whole call is a few CPU-seconds.
"""

from __future__ import annotations

import io
import json
import platform
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/root/compose")
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

# Pinned to the production environment exactly (CLAUDE.md). The whole point of
# this app is that the environment is load-bearing, so it may not float.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
)
for _source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / _source_directory,
        str(REMOTE_ROOT / _source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )

app = modal.App("compose-v4-materialize-process-v2-predecessor-state")


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def materialize_predecessor_state(descriptor: dict) -> dict:
    """Build the revised scorer under the frozen seal and return its bytes."""

    import torch

    from compose_v4.experiments import editing_v2_process_v2_t1_runtime as t1
    from compose_v4.experiments.editing_gate_zero_runtime import (
        PRODUCTION_RINGCORE_CATALOG_FINGERPRINT,
        build_production_ringcore_catalog,
    )

    # Fail closed here too: a drifted catalog would silently change the geometry
    # the state is built on, and the bytes would then be wrong rather than absent.
    build_production_ringcore_catalog(max_atoms=40)

    repo_root = Path(str(REMOTE_ROOT))

    # This is the real seal. If it raises, the state is NOT materialized -- we
    # export only bytes that already satisfied every production check.
    runtime, binding, bridge = t1._scratch_runtime_for_score_revision(
        descriptor, repo_root=repo_root
    )

    config = runtime.config
    semantic = t1.load_gate_zero_semantic_contract(
        repo_root / t1.GATE_ZERO_MODEL_PROCESS_V2
    )
    predecessor_state = t1.build_semantic_scratch_runtime(
        config, semantic
    ).model.state_dict()
    revised_state = runtime.model.state_dict()

    base_state = {
        name: value
        for name, value in predecessor_state.items()
        if name != "graft_relation_head.weight"
    }
    base_state_sha256 = t1.state_dict_semantic_sha256(base_state)
    frozen = t1._SCORE_REVISION_PREDECESSOR["initial_model_state_sha256"]
    if base_state_sha256 != frozen:
        raise RuntimeError(
            f"remote base state {base_state_sha256} != frozen {frozen}; refusing to export"
        )

    def _dump(state: dict) -> bytes:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()

    return {
        "predecessor_state_bytes": _dump(predecessor_state),
        "revised_state_bytes": _dump(revised_state),
        "base_state_sha256": base_state_sha256,
        "predecessor_state_sha256": t1.state_dict_semantic_sha256(predecessor_state),
        "revised_state_sha256": t1.state_dict_semantic_sha256(revised_state),
        "runtime_initial_model_state_sha256": runtime.initial_model_state_sha256,
        "process_identity_sha256": runtime.process_identity_sha256,
        "catalog_fingerprint": PRODUCTION_RINGCORE_CATALOG_FINGERPRINT,
        "score_revision_containment": dict(bridge["score_revision_containment"]),
        "binding": binding,
        "predecessor_parameter_names": sorted(predecessor_state),
        "revised_parameter_names": sorted(revised_state),
        "software": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "torch": torch.__version__,
        },
    }


@app.local_entrypoint()
def main(descriptor_path: str, output_dir: str) -> None:
    """Fetch the authenticated scorer bytes and write them beside a receipt."""

    descriptor = json.loads(Path(descriptor_path).read_text())
    result = materialize_predecessor_state.remote(descriptor)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "predecessor_state.pt").write_bytes(result.pop("predecessor_state_bytes"))
    (output / "revised_state.pt").write_bytes(result.pop("revised_state_bytes"))
    (output / "MATERIALIZATION_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    print(f"base_state_sha256   {result['base_state_sha256']}")
    print(f"revised_state_sha256 {result['revised_state_sha256']}")
    print(f"built on            {result['software']}")
    print(f"wrote               {output}")
