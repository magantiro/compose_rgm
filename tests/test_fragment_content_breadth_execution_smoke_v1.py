"""Input identity and seed checks for the exact-execution preflight."""

import pytest

from tools.smoke_fragment_content_breadth_execution_v1 import run, seed


def test_paired_seed_is_deterministic_and_prompt_specific():
    assert seed("BARICITINIB", 0) == seed("BARICITINIB", 0)
    assert len({seed("BARICITINIB", 0), seed("BARICITINIB", 1), seed("ERLOTINIB", 0)}) == 3


def test_input_hash_gate_fails_before_molecular_work(tmp_path):
    catalog = tmp_path / "catalog.json"
    prior = tmp_path / "prior.json"
    catalog.write_text("{}")
    prior.write_text("{}")
    with pytest.raises(ValueError, match="input hash mismatch"):
        run(catalog, prior)
