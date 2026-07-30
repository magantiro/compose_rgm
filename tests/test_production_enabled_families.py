"""The manifest operator set must be DERIVED from the authoritative capability flags, not handwritten.

C18: `build_scaled_edit_manifest` computed `production_enabled` by excluding cycle_insert/cycle_attach.
That is exactly the DE-NOVO (base-B) family set -- correct when cycle ops were off and the grow macro was
on, and never updated for RingCore-V1, which inverts both flags. Because
`operator_subtype_supervision_gate` consumes `production_enabled` as the REQUIRED-supervision set, the
stale list would demand supervision for the disabled macro (spurious NO_GO) while letting the two defining
compositional families go unsupervised undetected.

The fix is a single semantic derivation from the capability flags that gate those families, so no builder
can drift again.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    MARK_RULE_NAMES,
    production_enabled_families,
)


def test_ringcore_v1_enables_cycle_ops_and_disables_grow():
    enabled = production_enabled_families(enable_cycle_ops=True, enable_ring_grow_macro=False)
    assert "ring_system_grow" not in enabled
    assert "cycle_insert" in enabled and "cycle_attach" in enabled
    assert set(enabled) == set(MARK_RULE_NAMES) - {"ring_system_grow"}


def test_de_novo_base_b_is_the_mirror_image():
    """The historical buggy list was precisely this -- correct for base B, stale for RingCore."""
    enabled = production_enabled_families(enable_cycle_ops=False, enable_ring_grow_macro=True)
    assert "ring_system_grow" in enabled
    assert "cycle_insert" not in enabled and "cycle_attach" not in enabled


def test_ordering_follows_mark_rule_names():
    enabled = production_enabled_families(enable_cycle_ops=True, enable_ring_grow_macro=False)
    assert enabled == [m for m in MARK_RULE_NAMES if m in set(enabled)]


def test_ring_system_delete_is_an_explicit_optional_accelerator():
    enabled = production_enabled_families(
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
    )
    assert "ring_system_delete" not in enabled
    assert "cycle_insert" in enabled and "cycle_attach" in enabled
    assert set(enabled) == set(MARK_RULE_NAMES) - {
        "ring_system_grow",
        "ring_system_delete",
    }


def test_scaled_manifest_builder_uses_the_derivation():
    """The builder must call the shared derivation, not re-filter names locally."""
    src = (Path(__file__).resolve().parent.parent / "scripts" / "build_scaled_edit_manifest.py").read_text()
    assert "production_enabled_families(" in src, "builder must use the shared capability derivation"
    assert 'not in ("cycle_insert", "cycle_attach")' not in src, "the stale de-novo filter must be gone"


def test_ring_core_manifest_builder_uses_the_derivation():
    src = (Path(__file__).resolve().parent.parent / "scripts" / "build_ring_core_v1_manifest.py").read_text()
    assert "production_enabled_families(" in src, "builder must use the shared capability derivation"
