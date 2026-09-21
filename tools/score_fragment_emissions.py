#!/usr/bin/env python3
"""Score a captured emission list with the official metrics, on THIS interpreter.

Run it twice -- once on the laptop kernel that generated the molecules, once on
the production-pinned kernel -- and diff the two outputs.  That isolates the
question "do the reported numbers depend on the RDKit build" from the much
larger question "does the GENERATOR depend on it", which it does, because the
sampler calls RDKit inside its own accept/reject loop.  Only the first question
is answerable by re-scoring fixed strings, and conflating the two would overstate
what the check proves.

``in_virtuo_gen.train_utils.metrics`` imports ``torch.distributed`` at module
scope but never calls it on the ``already_smiles=True`` path, so a torch-free
pinned environment gets a stub rather than a 200MB install.  The stub raises on
attribute access other than the import itself, so if upstream ever started
using it the check would fail loudly instead of silently diverging.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _stub_torch_if_absent() -> bool:
    try:
        import torch  # noqa: F401

        return False
    except ImportError:
        pass

    class _Forbidden(types.ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(
                f"the torch stub was asked for {name!r}; the official metric path "
                "was supposed never to touch torch"
            )

    torch_mod = _Forbidden("torch")
    dist_mod = _Forbidden("torch.distributed")
    # ``import torch.distributed as dist`` binds the submodule as an ATTRIBUTE of
    # the parent, so the attribute has to exist or __getattr__ fires on the
    # import itself rather than on a real use.  Anything BEYOND the import --
    # an actual dist.* call -- still raises, which is the point.
    torch_mod.distributed = dist_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = dist_mod
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stubbed = _stub_torch_if_absent()
    import rdkit
    from rdkit import Chem

    from compose_v4.benchmark.fragment_official_metrics import (
        official_distance,
        official_prompt_metrics,
        official_unique_valid,
    )

    payload = json.loads(args.emissions.read_text())
    emitted = payload["emitted"]
    metrics = official_prompt_metrics(emitted, expected_samples=payload["attempts"])

    # The distance reference is the dummy-stripped prompt, rebuilt here so the
    # two kernels each canonicalize it with their own perception.
    dummy = Chem.MolFromSmarts("[#0]")
    parts = []
    for fragment in payload["prompt_fragments"]:
        mol = Chem.MolFromSmiles(fragment)
        stripped = Chem.DeleteSubstructs(mol, dummy)
        stripped.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(stripped)
        parts.append(Chem.MolToSmiles(stripped, canonical=True))
    reference = ".".join(p for p in parts if p)

    metrics["distance"] = official_distance(emitted, reference)
    metrics["distance_to_original_drug"] = official_distance(
        emitted, payload["original_smiles"]
    )

    result = {
        "schema": "compose_fragment_emission_score_v1",
        "scoring_rdkit": rdkit.__version__,
        "generating_rdkit": payload.get("generating_rdkit"),
        "torch_stubbed": stubbed,
        "task": payload["task"],
        "drug": payload["drug"],
        "seed": payload["seed"],
        "attempts": payload["attempts"],
        "emitted_nonempty": sum(1 for e in emitted if e),
        "unique_valid": len(official_unique_valid(emitted)),
        "distance_reference": reference,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
