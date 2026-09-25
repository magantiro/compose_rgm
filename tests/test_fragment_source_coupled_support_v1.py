"""The support census is deterministic and does not read property labels."""

from __future__ import annotations

import json

from audit_fragment_source_coupled_support_v1 import audit
from test_source_coupled_pendant_policy import _inputs


def test_support_audit_counts_shared_sources_and_records_inputs(tmp_path):
    catalog, mass = _inputs()
    catalog_path = tmp_path / "catalog.json"
    mass_path = tmp_path / "mass.json"
    attempts_path = tmp_path / "attempts"
    attempts_path.mkdir()
    catalog_path.write_text(json.dumps(catalog))
    mass_path.write_text(json.dumps(mass))
    (attempts_path / "TEST_000.json").write_text(
        json.dumps(
            {
                "drug": "TEST",
                "panel": {
                    "offered": [
                        {
                            "status": "model_supported",
                            "provenance": {
                                "pendant_plan": {
                                    "core_heavy_atoms": 38,
                                    "draws": [
                                        {"context": "C:0:0", "heavy_atoms": 1, "rings": 0},
                                        {"context": "N:0:0", "heavy_atoms": 1, "rings": 0},
                                    ],
                                }
                            },
                        }
                    ]
                },
            }
        )
    )
    first = audit(catalog_path, mass_path, attempts_path, draws=20)
    second = audit(catalog_path, mass_path, attempts_path, draws=20)
    assert first == second
    assert first["role"] == "zero_quality_support_only"
    assert first["input_sha256"]["attempt_files"]["TEST_000.json"]
    assert first["prompts"][0]["model_supported_offers"] == 1
    assert first["prompts"][0]["offers_with_shared_source_row"] == 1
    assert first["prompts"][0]["coupled_draws"] > 0
    assert first["prompts"][0]["independent_draws"] > 0
