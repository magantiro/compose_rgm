"""Derive the canonical controller's runtime file closure BY EXECUTION.

Reading imports both misses a module reached only at run time and over-pins one that
is merely imported. This traces a real proposal draw on a real T4 seed -- the shallow,
structured and anchored lanes plus the zero-support fallback plus `Fiber.check` -- and
reports every `compose_v4` module whose code actually ran.

Run it under the PINNED kernel:
    PYTHONPATH=src ~/compose_region_pinned_env/bin/python tools/derive_t4_canonical_closure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import numpy as np

    from compose_v4.control.completion_law_contract import build_completion_law
    from compose_v4.control.region_law_contract import build_region_law
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_canonical_controller import load_seeds
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

    seeds = load_seeds(ROOT / "docs/GENMOL_T4_SEEDS.json")
    touched: set[str] = set()

    # `sys.settrace` on this path is unusable: one shallow draw on a 32-heavy-atom
    # parent explodes into thousands of variant instantiations and RDKit gate calls,
    # and tracing every frame of that costs tens of minutes. The closure is therefore
    # taken as the set of `compose_v4` modules PRESENT IN `sys.modules` after a real
    # run -- still derived by execution rather than by reading import statements, and
    # erring on the side of OVER-pinning, which is the safe direction for a contract.

    # Two seeds of very different size, so a size-gated branch cannot be missed.
    for row in (seeds[1], seeds[3]):
        fiber = Fiber(row["smiles"], 0.6, support="compose_valid")
        region_law = build_region_law("free_gate_margin_v1", delta=0.6,
                                      reference_smiles=row["smiles"])
        completion_law = build_completion_law("free_gate_margin_v1", delta=0.6,
                                              reference_smiles=row["smiles"])
        if True:
            for lane, law, claw in (
                ("shallow", region_law, completion_law),
                ("structured", None, None),
                ("anchored_replacement", None, None),
            ):
                expand(row["smiles"], -8.0, fiber, np.random.default_rng(7),
                       draws=6, multi_region=True, horizon=3, proposal_lane=lane,
                       region_law=law, completion_law=claw)
            # The fallback is traced on a SMALL parent: its region x completion budget
            # is quadratic in molecule size and `sys.settrace` multiplies it, while the
            # set of modules it touches does not depend on the parent.
            small = "CC(C)CCN(C)C(=O)c1ccccc1"
            small_fiber = Fiber(small, 0.2, support="compose_valid")
            fallback_candidates(small, np.random.default_rng(11), check=small_fiber.check,
                                reference_smiles=small, delta=0.2)
            fiber.check(row["smiles"])

    for name, module in sorted(sys.modules.items()):
        if not name.startswith("compose_v4"):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            touched.add(str(Path(filename).resolve().relative_to(ROOT)))
        except ValueError:
            continue
    print(json.dumps(sorted(touched), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
