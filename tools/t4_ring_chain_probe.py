"""Eight prescribed ring/local/ring trajectories, not learned-controller search."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.control.docking_value import DockingValue, molecular_features
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    OptionState,
)
from compose_v4.control.option_selector import option_horizon
from compose_v4.control.region import enumerate_regions
from compose_v4.control.region_rewrite import Lineage
from compose_v4.control.ring_program import (
    RingProgress,
    RingSpec,
    completed_construction,
    real_slots,
    ring_spec,
)
from compose_v4.experiments.continuation_profile import encode_action
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_warm_continuation import exact_context
from compose_v4.gates.med_chem_gate import validity_reasons
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "803741324ac0e30bff96f9ad36809c6d08ff8334a76a1cc1927a7296a91daf30"
LOCK_SHA = "27ee5f6824a53c6f9aa8896a7cd666c43feb27eb36129a1d32c3cf355ebf58d3"
SEQUENCES = (
    (
        RingSpec("fused", 6, (6, 0, 0), "nonaromatic").option,
        "local",
        RingSpec("pendant", 5, (4, 1, 0), "aromatic").option,
    ),
    (
        RingSpec("pendant", 6, (5, 0, 1), "nonaromatic").option,
        "local",
        RingSpec("fused", 5, (4, 1, 0), "aromatic").option,
    ),
)


def diagnostic_law(graph):
    marks = []
    for family, action in _factorized_candidates(
        graph, allow_bond_reroute=False, vocabulary=ORGANIC_VOCABULARY
    ):
        if family in ("atom_insert", "bond_reorder"):
            marks.append((family, action))
        elif family == "bond_insert":
            marks.append(("cycle_close", CycleCloseEdge(action.a, action.b, action.order)))
    return tuple(f for f, _ in marks), tuple(a for _, a in marks), (1 / len(marks),) * len(marks)


def run(archive_path: Path, lock_path: Path, output: Path):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("use a clean committed source worktree")
    if sha(archive_path) != ARCHIVE_SHA or sha(lock_path) != LOCK_SHA:
        raise ValueError("frozen archive or predictor-lock identity mismatch")
    if rdBase.rdkitVersion != "2024.03.5" or np.__version__ != "1.26.4":
        raise ValueError("probe requires the pinned T4 chemistry runtime")
    envelope = json.loads(lock_path.read_bytes())
    if digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("corrupt predictor lock envelope")
    lock = envelope["payload"]["lock"]
    envelope = json.loads(archive_path.read_bytes())
    if digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("corrupt archive envelope")
    archive = envelope["payload"]["archive"]
    model = DockingValue.from_payload(lock["value_snapshot"])
    if model.payload["source_sha256"] != ARCHIVE_SHA:
        raise ValueError("predictor belongs to a different archive")
    for row, features in zip(
        model.payload["training_rows"], model.payload["features"], strict=True
    ):
        if molecular_features(row["smiles"])[1] != tuple(map(tuple, features)):
            raise ValueError("local predictor features differ from saved production features")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(archive[0]["smiles"]))

    def properties(smiles):
        return {
            "smiles": smiles,
            **calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            ),
        }

    for candidate in lock["pool"]:
        actual = properties(candidate["smiles"])
        for field in ("qed", "sa", "sim", "v"):
            if not np.isclose(actual[field], candidate[field], rtol=0, atol=1e-12):
                raise ValueError(f"pinned property parity failure: {field}")
        if not np.isclose(
            model.predict([candidate["smiles"]])[0],
            candidate["predicted_docking"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("pinned predictor parity failure")
    best = min(
        (
            r
            for r in archive
            if r["ds"] is not None and acceptable_endpoint(properties(r["smiles"]))
        ),
        key=lambda r: (r["ds"], r["smiles"]),
    )
    inputs = {str(archive_path): ARCHIVE_SHA, str(lock_path): LOCK_SHA}
    dependencies = implementation_closure(Path(__file__))
    configuration = {
        "sequences": SEQUENCES,
        "seeds": [0, 1],
        "source_roles": ["original", "best_feasible"],
        "reference": "uniform diagnostic insertion/reorder/semantic-closure descriptors, NOT R_theta",
        "where": "largest enumerated region <=24, deterministic key tie break",
        "max_active": 40,
        "slots": 48,
        "stop_seconds": 120,
        "new_oracle_calls": 0,
        "fitting": False,
        "winner_inputs": [],
    }
    run_id = digest(
        {"inputs": inputs, "implementation": dependencies, "configuration": configuration}
    )
    started, rows = time.perf_counter(), []
    for source_index, source in enumerate((archive[0], best)):
        for sequence_index, sequence in enumerate(SEQUENCES):
            for seed in (0, 1):
                unit_path = output / "units" / f"{source_index}-{sequence_index}-{seed}.json"
                if unit_path.exists():
                    unit = json.loads(unit_path.read_bytes())
                    if (
                        unit["run_id"] != run_id
                        or digest(unit["payload"]) != unit["payload_sha256"]
                    ):
                        raise ValueError(
                            "completed unit has incompatible identity or corrupt payload"
                        )
                    rows.append(unit["payload"])
                    continue
                graph = decode_state(source["state"])
                initial, lineage = graph, Lineage.initial(real_slots(graph))
                initial_lineage = lineage
                rng, system = np.random.default_rng(seed), editing_v2_semantic_rewrite_system()
                kernel = OptionContinuationKernel(
                    diagnostic_law,
                    system,
                    max_executor_applications=None,
                    product_gate=EXECUTABLE_PRODUCT_GATE,
                )
                unit_start, trace, endpoints, status = time.perf_counter(), [], [], "complete"
                for stage, option in enumerate(sequence):
                    spec = ring_spec(option)
                    smiles = canonical_state_key(graph)
                    region = max(
                        (r for r in enumerate_regions(smiles) if 1 <= r.size <= 24),
                        key=lambda r: (r.size, repr(r.key())),
                    )
                    context = exact_context(graph, smiles, region)
                    node = OptionState(
                        graph,
                        graph,
                        context,
                        lineage,
                        option,
                        0,
                        option_horizon(option, 1),
                        f"chain:{source_index}:{sequence_index}:{seed}:{stage}",
                        ring_progress=RingProgress() if spec else None,
                    )
                    while node.remaining:
                        if time.perf_counter() - started > configuration["stop_seconds"]:
                            status = "administrative_elapsed_stop"
                            break
                        row = kernel.row(node)
                        if not row.successors:
                            status = "no_admissible_action"
                            break
                        index = int(rng.choice(len(row.successors), p=row.probabilities))
                        following = row.successors[index]
                        mark = encode_action(*kernel.marks(node)[index])
                        # Replay selected actions only; do not regenerate any molecular law.
                        replayed = system.apply(node.graph, *decode_action(mark))
                        if encode_state(replayed) != encode_state(following.graph):
                            raise ValueError("selected edit failed exact semantic replay")
                        trace.append(
                            {
                                "stage": stage,
                                "option": option,
                                "mark": mark,
                                "source": encode_state(node.graph),
                                "product": encode_state(following.graph),
                            }
                        )
                        node = following
                    if status != "complete":
                        break
                    graph, lineage = node.graph, node.lineage
                    if spec and not completed_construction(
                        node.origin, graph, node.ring_progress, spec
                    ):
                        raise ValueError("completed ring lacks its construction witness")
                    smiles, mol = (
                        canonical_state_key(graph),
                        Chem.MolFromSmiles(canonical_state_key(graph)),
                    )
                    prop = properties(smiles)
                    endpoints.append(
                        {
                            **prop,
                            "stage": stage,
                            "option": option,
                            "state": encode_state(graph),
                            "accepted": acceptable_endpoint(prop),
                            "med_chem_reasons": validity_reasons(smiles),
                            "predicted_docking": float(model.predict([smiles])[0]),
                            "intended_release": region.released_fraction,
                            "cumulative_change": structural_displacement(
                                initial, graph, initial_lineage, lineage
                            ),
                            "topology": topology(graph),
                            "ring_sizes": sorted(map(len, mol.GetRingInfo().AtomRings())),
                            "bridgeheads": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                            "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
                        }
                    )
                result = {
                    "source_index": source_index,
                    "source": source["smiles"],
                    "source_state": source["state"],
                    "source_actual_docking": source["ds"],
                    "sequence_index": sequence_index,
                    "seed": seed,
                    "status": status,
                    "endpoints": endpoints,
                    "trace": trace,
                    "kernel_work": asdict(kernel.work),
                    "selected_replay_calls": len(trace),
                    "seconds": time.perf_counter() - unit_start,
                }
                publish(
                    unit_path,
                    {"run_id": run_id, "payload": result, "payload_sha256": digest(result)},
                )
                rows.append(result)
                print(
                    json.dumps(
                        {
                            "unit": unit_path.name,
                            "status": status,
                            "completed_options": len(endpoints),
                            "seconds": result["seconds"],
                        }
                    ),
                    flush=True,
                )
    result = {
        "schema_version": "t4_prescribed_ring_chain_probe_v1",
        "run_id": run_id,
        "input_sha256": inputs,
        "implementation_sha256": dependencies,
        "configuration": configuration,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "workers": 1,
            "accelerator": "none",
            "precision": "float64/integer graph arrays",
        },
        "predictor_parity": {
            "training_feature_rows": len(model.payload["training_rows"]),
            "saved_candidates": len(lock["pool"]),
            "status": "matched",
            "tolerance": 1e-12,
        },
        "sa_scorer_sha256": sha(Path(sascorer.__file__)),
        "sa_fragment_scores_sha256": sha(Path(sascorer.__file__).with_name("fpscores.pkl.gz")),
        "units": rows,
        "seconds_this_invocation": time.perf_counter() - started,
        "evidence_role": "prescribed option capability and frozen predictor diagnostic, not learned-guided or autonomous discovery",
    }
    publish(output / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.lock, args.output)
