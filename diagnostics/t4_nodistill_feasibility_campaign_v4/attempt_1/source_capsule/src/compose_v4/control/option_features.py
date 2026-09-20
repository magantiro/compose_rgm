"""Compositional features for structural options and their selected regions.

These descriptors encode transformation semantics, not molecular fragments or
endpoint templates.  Parameterized ring programs remain open over every valid
C/N/O count vector admitted by :class:`RingSpec`; the feature map does not turn
the current default menu into a vocabulary boundary.
"""

from __future__ import annotations

import numpy as np

from compose_v4.control.carbonyl_option import (
    ADD_CARBONYL_OPTION,
    INSERT_RING_CARBONYL_OPTION,
)
from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION
from compose_v4.control.macro_engine import MACRO_FAMILIES
from compose_v4.control.option_selector import (
    BUILD_RING_SYSTEM_OPTION,
    GENERIC_OPTION,
    OPTION_GROUPS,
    option_horizon,
)
from compose_v4.control.region_replacement import replacement_spec
from compose_v4.control.ring_expansion import EXPAND_RING_OPTION
from compose_v4.control.ring_program import ring_spec

PURPOSES = tuple(OPTION_GROUPS)
PRIMITIVE_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
    "bond_insert",
)
ELECTRONIC_STATES = ("saturated", "nonaromatic", "aromatic")
SPECIALS = (
    "parameterized_ring",
    "replacement",
    "legacy_ring_program",
    "ring_expansion",
    "add_carbonyl",
    "insert_ring_carbonyl",
)
VALUE_CONTEXT_NAMES = (
    "task:current_utility",
    "task:incumbent_utility",
    "clock:remaining_options_scaled",
)
REGION_CONTEXT_NAMES = (
    "region:released_fraction",
    "region:size_fraction",
    "region:boundary_bonds_scaled",
    "region:context_components_scaled",
    "region:has_ring",
    "region:ring_boundary_fraction",
)


def option_feature_names() -> tuple[str, ...]:
    return (
        *(f"purpose:{name}" for name in PURPOSES),
        *(f"primitive:{name}" for name in PRIMITIVE_FAMILIES),
        *(f"special:{name}" for name in SPECIALS),
        "topology:pendant",
        "topology:fused",
        "ring_size_scaled",
        "composition:C",
        "composition:N",
        "composition:O",
        *(f"electronic:{name}" for name in ELECTRONIC_STATES),
        "refinement_scaled",
        "primitive_horizon_scaled",
    )


def boundary_value_features(
    molecular_features,
    *,
    current_utility: float,
    incumbent_utility: float,
    remaining_options: int,
) -> np.ndarray:
    """Features available before the next WHERE decision.

    Utilities must already be oriented so larger is better. This function does
    not call an oracle or silently reverse task-specific score conventions.
    """

    if (
        isinstance(remaining_options, bool)
        or not isinstance(remaining_options, (int, np.integer))
        or not 0 <= remaining_options <= 16
    ):
        raise ValueError("remaining option clock must be an integer in 0..16")
    base = np.asarray(molecular_features, dtype=np.float32)
    context = np.asarray(
        (current_utility, incumbent_utility, remaining_options / 16), dtype=np.float32
    )
    if (
        base.ndim != 1
        or not base.size
        or not np.isfinite(base).all()
        or not np.isfinite(context).all()
    ):
        raise ValueError("option-boundary value features must be finite vectors")
    return np.concatenate((base, context))


def _families(option: str, spec) -> set[str]:
    if spec is not None:
        families = set(MACRO_FAMILIES["scaffold_extend"] + MACRO_FAMILIES["cyclize"])
        if spec.refine:
            families.update(MACRO_FAMILIES["restate"])
        if replacement_spec(option) is not None:
            families.update(MACRO_FAMILIES["shrink"])
        return families
    if option in MACRO_FAMILIES:
        return set(MACRO_FAMILIES[option])
    if option in (BUILD_RING_SYSTEM_OPTION, BUILD_FUSED_RING_OPTION):
        return set(
            MACRO_FAMILIES["scaffold_extend"]
            + MACRO_FAMILIES["cyclize"]
            + MACRO_FAMILIES["restate"]
        )
    if option == EXPAND_RING_OPTION:
        return set(
            MACRO_FAMILIES["open"] + MACRO_FAMILIES["scaffold_extend"] + MACRO_FAMILIES["cyclize"]
        )
    if option == ADD_CARBONYL_OPTION:
        return {"atom_insert"}
    if option == INSERT_RING_CARBONYL_OPTION:
        return {"cycle_open", "atom_insert", "cycle_close", "bond_insert"}
    if option == GENERIC_OPTION:
        return set(PRIMITIVE_FAMILIES)
    raise KeyError(f"unsupported structural option {option!r}")


def structural_option_features(option: str, *, generic_horizon: int = 3) -> np.ndarray:
    """Encode option semantics without inspecting a target molecule or score."""

    replacement = replacement_spec(option)
    spec = replacement if replacement is not None else ring_spec(option)
    unwrapped = option[len("replace_region:") :] if replacement is not None else option
    values = []
    for purpose in PURPOSES:
        members = OPTION_GROUPS[purpose]
        is_member = unwrapped in members
        if spec is not None:
            is_member = purpose == ("restructure" if replacement is not None else "ring_topology")
        elif option == BUILD_FUSED_RING_OPTION:
            is_member = purpose == "ring_topology"
        elif option in (EXPAND_RING_OPTION, INSERT_RING_CARBONYL_OPTION):
            is_member = purpose == "restructure"
        elif option == ADD_CARBONYL_OPTION:
            is_member = purpose == "material"
        values.append(float(is_member))
    families = _families(option, spec)
    values.extend(float(name in families) for name in PRIMITIVE_FAMILIES)
    values.extend(
        (
            float(spec is not None),
            float(replacement is not None),
            float(option in (BUILD_RING_SYSTEM_OPTION, BUILD_FUSED_RING_OPTION)),
            float(option == EXPAND_RING_OPTION),
            float(option == ADD_CARBONYL_OPTION),
            float(option == INSERT_RING_CARBONYL_OPTION),
        )
    )
    values.extend(
        (
            float(spec is not None and spec.topology == "pendant"),
            float(spec is not None and spec.topology == "fused"),
            0.0 if spec is None else spec.size / 6,
        )
    )
    if spec is None:
        values.extend((0.0, 0.0, 0.0))
    else:
        values.extend(count / spec.size for count in spec.counts)
    values.extend(float(spec is not None and spec.electronic == name) for name in ELECTRONIC_STATES)
    values.append(0.0 if spec is None else spec.refine / 2)
    # Replacement horizons depend on the state and are supplied elsewhere; its
    # construction horizon is the stable semantic quantity encoded here.
    horizon_option = unwrapped if replacement is not None else option
    values.append(option_horizon(horizon_option, generic_horizon) / 16)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(option_feature_names()),) or not np.isfinite(result).all():
        raise RuntimeError("structural option feature registry is inconsistent")
    return result


def option_state_features(
    molecular_features,
    *,
    current_utility: float,
    incumbent_utility: float,
    remaining_options: int,
    released_fraction: float,
    region_size: int,
    total_atoms: int,
    boundary_bonds: int,
    context_components: int,
    region_has_ring: bool,
    ring_boundary_bonds: int,
) -> np.ndarray:
    """Append task, option-clock and numeric region context to fixed molecular features."""

    if (
        total_atoms < 1
        or not 0 <= region_size <= total_atoms
        or boundary_bonds < 0
        or context_components < 0
        or ring_boundary_bonds < 0
        or ring_boundary_bonds > boundary_bonds
        or not 0 <= released_fraction <= 1
    ):
        raise ValueError("invalid option clock or region cardinality")
    value = boundary_value_features(
        molecular_features,
        current_utility=current_utility,
        incumbent_utility=incumbent_utility,
        remaining_options=remaining_options,
    )
    region = np.asarray(
        (
            released_fraction,
            region_size / total_atoms,
            boundary_bonds / max(total_atoms, 1),
            context_components / max(total_atoms, 1),
            float(region_has_ring),
            ring_boundary_bonds / max(boundary_bonds, 1),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(region).all():
        raise ValueError("option-boundary state features must be finite vectors")
    return np.concatenate((value, region))
