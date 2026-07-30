from __future__ import annotations

from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, SequenceNode


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "experiment_registry.yaml"


def _duplicate_mapping_keys(node: Node, path: str = "$") -> list[str]:
    duplicates: list[str] = []
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(key_node.value)
            if key in seen:
                duplicates.append(f"{path}.{key}")
            seen.add(key)
            duplicates.extend(_duplicate_mapping_keys(value_node, f"{path}.{key}"))
    elif isinstance(node, SequenceNode):
        for index, item in enumerate(node.value):
            duplicates.extend(_duplicate_mapping_keys(item, f"{path}[{index}]"))
    return duplicates


def test_experiment_registry_has_no_silent_duplicate_yaml_keys() -> None:
    root = yaml.compose(REGISTRY.read_text())
    assert root is not None
    assert _duplicate_mapping_keys(root) == []


def test_experiment_registry_has_one_complete_v2_document() -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    assert registry["schema"] == "compose.experiments.registry"
    assert registry["schema_version"] == 2
    assert set(registry["experiments"]) == {
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
    }
