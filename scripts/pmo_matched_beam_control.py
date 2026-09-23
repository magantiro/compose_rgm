"""The embarrassing control: a greedy beam given the SAME 16 starts and the SAME 250 calls.

The beam's earlier 0.5172 used 9,521 counted evaluations from ONE start and is useless as a
comparator. This run matches the canary exactly -- same initialization bank, same production
proposal stack, same charged budget -- and asks whether reward-greedy allocation alone already
does what the learned controller is meant to do at equal cost.

If the beam wins here, that is the finding, and it is worth more than a favourable A/B.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np

OUT = "diagnostics/pmo_matched_beam_v1/matched_beam_v1.json"


def _oracle():
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name="celecoxib_rediscovery")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = int(args[0]) if args else 250
    width = int(args[1]) if len(args) > 1 else 6

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.program_task import archive_top_k
    from compose_v4.control.region_replacement_option import RegionReplacementOption
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw
    from compose_v4.experiments import pmo_population_v1 as production

    score = _oracle()
    contract = production.load_contract(pathlib.Path("."))
    initialized = production._load_initialization(
        pathlib.Path("."), {"initialization": contract["initialization"]}
    )
    # The SAME 16 molecules the canary initializes from, charged the same way.
    starts = [c["endpoint"] for c in initialized["candidates"]][:16]
    option = RegionReplacementOption(region_law=ScaleBalancedRegionLaw())
    rng = np.random.default_rng(contract["controller"]["seed"])

    charged = []
    for smiles in starts:
        charged.append((smiles, float(score(smiles))))
    frontier = sorted(charged, key=lambda r: -r[1])[:width]
    curve = []

    while len(charged) < budget:
        children = []
        for parent, parent_score in frontier:
            graph = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
            for _ in range(8):
                if len(charged) + len(children) >= budget:
                    break
                try:
                    _s, _p, _a, trace, _m = synthesize_dynamic_program(
                        graph, rng, max_modules=3,
                        region_law=ScaleBalancedRegionLaw(),
                        replacement_option=option, replacement_option_rate=0.5,
                    )
                except (ValueError, RuntimeError):
                    continue
                endpoint = trace.get("endpoint")
                if endpoint and endpoint not in {c[0] for c in charged}:
                    children.append((endpoint, parent_score))
        if not children:
            break
        for endpoint, _ps in children:
            if len(charged) >= budget:
                break
            charged.append((endpoint, float(score(endpoint))))
            curve.append(
                {"calls": len(charged), "best": max(c[1] for c in charged),
                 "top10": archive_top_k(charged, k=10)}
            )
        frontier = sorted(charged, key=lambda r: -r[1])[:width]
        print(f"  calls {len(charged):3d} best {max(c[1] for c in charged):.4f} "
              f"top10 {archive_top_k(charged, k=10):.4f}", flush=True)

    report = {
        "schema_version": "pmo_matched_beam_v1",
        "evidence_role": "matched_budget_control",
        "charged_oracle_calls": len(charged),
        "beam_width": width,
        "initialization": "the canary's own 16-molecule bank",
        "best_score": max(c[1] for c in charged),
        "final_top10": archive_top_k(charged, k=10),
        "auc_top10": float(np.mean([c["top10"] for c in curve])) if curve else None,
        "curve": curve,
    }
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nMATCHED BEAM: charged {len(charged)} best {report['best_score']:.4f} "
          f"top10 {report['final_top10']:.4f} auc {report['auc_top10']:.4f}")
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
