"""Frozen envelope checks for the source-coupled execution smoke."""

from run_fragment_source_coupled_execution_smoke_v1 import (
    ARMS,
    ATTEMPTS_PER_PROMPT_ARM,
    DRUGS,
    MATERIAL_HASHES,
    _seed,
)


def test_smoke_has_complete_small_panel() -> None:
    assert ARMS == ("frozen", "source_coupled")
    assert len(DRUGS) == 10
    assert ATTEMPTS_PER_PROMPT_ARM == 3
    assert len(MATERIAL_HASHES) == 4


def test_seed_is_deterministic_and_arm_specific() -> None:
    assert _seed("BARICITINIB", "frozen") == _seed("BARICITINIB", "frozen")
    assert _seed("BARICITINIB", "frozen") != _seed("BARICITINIB", "source_coupled")
    assert _seed("BARICITINIB", "frozen") != _seed("CYCLOTHIAZIDE", "frozen")
