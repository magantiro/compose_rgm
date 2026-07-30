from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "modal_apps"
    / "audit_mmp_semantic_coverage.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_mmp_semantic_coverage",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_editing_scale_is_distinct_from_production_bins() -> None:
    assert _MODULE._editing_scale(2) == "precise_local_1_2"
    assert _MODULE._editing_scale(3) == "ordinary_lead_optimization_3_6"
    assert _MODULE._editing_scale(6) == "ordinary_lead_optimization_3_6"
    assert _MODULE._editing_scale(7) == "longer_compositional_gt_6"
    assert _MODULE._path_bin(5) == "bin_0:length<=5"
    assert _MODULE._path_bin(6) == "bin_1:length<=9"
    assert _MODULE._path_bin(14) == "bin_3:length>13"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "[0.0,0.2)"),
        (0.2, "[0.2,0.4)"),
        (0.4, "[0.4,0.6)"),
        (0.6, "[0.6,0.8)"),
        (0.8, "[0.8,1.0)"),
        (1.0, "1.0"),
    ],
)
def test_similarity_bins_have_explicit_boundaries(
    value: float,
    expected: str,
) -> None:
    assert _MODULE._similarity_bin(value) == expected


def test_selected_mark_coefficient_is_linear_in_path_length() -> None:
    one = _MODULE.selected_mark_coefficient(1)
    assert one > 0.0
    assert _MODULE.selected_mark_coefficient(8) == pytest.approx(8.0 * one)
    with pytest.raises(ValueError, match="positive"):
        _MODULE.selected_mark_coefficient(0)


def test_source_pool_row_and_effective_partition_counts_are_frozen() -> None:
    assert _MODULE.EXPECTED_POOL_RECORDS == 363_456
    assert sum(_MODULE.EXPECTED_EFFECTIVE_COUNTS.values()) == 361_019
    assert _MODULE.EXPECTED_EFFECTIVE_COUNTS["train"] == 330_991


def _state(
    atom_types: list[int],
    bonds: list[list[int]],
) -> dict:
    return {
        "n_slots": len(atom_types),
        "atom_types": atom_types,
        "formal_charges": [0] * len(atom_types),
        "implicit_h_counts": [0] * len(atom_types),
        "bonds": bonds,
    }


def test_graph_cycle_rank_uses_exact_beta_one() -> None:
    chain = _state([2, 2, 2, 0], [[0, 1, 1], [1, 2, 1]])
    triangle = _state(
        [2, 2, 2, 0],
        [[0, 1, 1], [1, 2, 1], [0, 2, 1]],
    )
    disconnected_edge_and_atom = _state([2, 2, 2, 0], [[0, 1, 1]])
    null = _state([0, 0], [])
    assert _MODULE._graph_cycle_rank(chain) == 0
    assert _MODULE._graph_cycle_rank(triangle) == 1
    assert _MODULE._graph_cycle_rank(disconnected_edge_and_atom) == 0
    assert _MODULE._graph_cycle_rank(null) == 0


def test_reconstruct_one_cut_semantics_from_exact_states_and_actions() -> None:
    source = _state([2, 2, 2, 0], [[0, 1, 1], [1, 2, 1]])
    midpoint = _state([2, 2, 0, 0], [[0, 1, 1]])
    target = _state([2, 2, 0, 3], [[0, 1, 1], [1, 3, 1]])
    entry = {
        "trace": {
            "steps": [
                {
                    "action": {
                        "executor_rule": "atom_delete",
                        "payload_type": "AtomDelete",
                        "payload": {"v": 2},
                    }
                },
                {
                    "action": {
                        "executor_rule": "atom_insert",
                        "payload_type": "AtomInsert",
                        "payload": {
                            "slot": 3,
                            "atom_type": 3,
                            "formal_charge": 0,
                            "implicit_h_count": 2,
                            "neighbors": [[1, 1]],
                        },
                    }
                },
            ]
        },
        "states": [source, midpoint, target],
    }
    got = _MODULE.reconstruct_one_cut_semantics(entry)
    assert got["pair_type"] == "atom_swap"
    assert got["constant_core_heavy"] == 2
    assert got["source_variable_heavy"] == 1
    assert got["target_variable_heavy"] == 1
    assert got["source_attachment_count"] == 1
    assert got["target_attachment_count"] == 1
    assert got["variable_ring_atoms"] == 0
    assert got["source_graph_cycle_rank"] == 0
    assert got["target_graph_cycle_rank"] == 0
    assert got["graph_cycle_rank_delta"] == 0


def test_packed_paths_accepts_read_only_local_root_override(tmp_path: Path) -> None:
    root = tmp_path / "mmp"
    partition = root / "train"
    partition.mkdir(parents=True)
    expected = partition / "shard_0000.jsonl.gz"
    expected.touch()
    manifest = {
        "roots": {"mmp_layer": "/remote/not-mounted"},
        "layers": {
            "mmp_analogue": {
                "train": ["train/shard_0000.jsonl.gz"],
            }
        },
    }
    assert _MODULE._packed_paths(
        manifest,
        "train",
        root_override=root,
    ) == (expected,)
