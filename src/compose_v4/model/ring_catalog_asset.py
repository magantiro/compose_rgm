"""Load a frozen, JSON-encoded ring catalog without a training checkpoint."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from compose_v4.rewrite.typed_ring_catalog import (
    AttachTemplate,
    CycleTemplate,
    EarTemplate,
    RingSystemElectronicAlias,
    RingSystemTemplate,
    TypedRingCatalog,
    ring_catalog_fingerprint,
)


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value


def _record(cls, payload: object):
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} must be an object")
    names = {field.name for field in fields(cls)}
    if set(payload) != names:
        raise ValueError(f"{cls.__name__} fields differ: {sorted(set(payload) ^ names)}")
    return cls(**{name: _tupleize(payload[name]) for name in names})


def load_ring_catalog_asset(path: Path, *, expected_fingerprint: str) -> TypedRingCatalog:
    """Restore the training catalog and verify its model-bound fingerprint."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "compose.ring_catalog_asset.v1":
        raise ValueError(f"unsupported ring catalog asset: {path}")
    raw = payload.get("catalog")
    if not isinstance(raw, dict):
        raise TypeError(f"ring catalog is missing or invalid: {path}")
    names = {field.name for field in fields(TypedRingCatalog)}
    if set(raw) != names:
        raise ValueError(f"ring catalog fields differ: {sorted(set(raw) ^ names)}")
    aliases = tuple(
        RingSystemElectronicAlias(
            pattern=_record(RingSystemTemplate, item["pattern"]),
            target_atoms=_tupleize(item["target_atoms"]),
        )
        for item in raw["ring_system_electronic_aliases"]
    )
    catalog = TypedRingCatalog(
        cycle_templates=tuple(_record(CycleTemplate, item) for item in raw["cycle_templates"]),
        attach_templates=tuple(_record(AttachTemplate, item) for item in raw["attach_templates"]),
        ear_templates=tuple(_record(EarTemplate, item) for item in raw["ear_templates"]),
        zero_span_ring_buckets=_tupleize(raw["zero_span_ring_buckets"]),
        ring_system_templates=tuple(
            _record(RingSystemTemplate, item) for item in raw["ring_system_templates"]
        ),
        ring_system_template_counts=_tupleize(raw["ring_system_template_counts"]),
        ring_system_electronic_alias_version=raw["ring_system_electronic_alias_version"],
        ring_system_electronic_aliases=aliases,
        ring_system_electronic_alias_counts=_tupleize(raw["ring_system_electronic_alias_counts"]),
    )
    actual = ring_catalog_fingerprint(catalog)
    if actual != expected_fingerprint or payload.get("fingerprint") != actual:
        raise ValueError(f"ring catalog fingerprint mismatch: {path}: {actual}")
    return catalog
