import json

import pytest
from finalize_fragment_superstructure_reference_v1 import analyze, load


def test_completed_reference_comparison_uses_ten_prompts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    learned = (
        root.parent
        / "fragment-interface-ablation-20260923/diagnostics/fragment_superstructure_official_v2/result.json"
    )
    uniform = root / "diagnostics/fragment_superstructure_reference_ablation_v1/result.json"
    if not learned.is_file() or not uniform.is_file():
        pytest.skip("documented source result assets are absent")
    result = analyze(learned, uniform)
    assert len(result["per_prompt"]) == 10
    assert len(result["per_seed"]) == 30
    assert result["attempts_per_arm"] == 3000
    assert result["estimates"]["quality"]["prompts_uniform_higher"] == 2
    assert result["estimates"]["quality"]["paired_prompt_mean_difference"] == pytest.approx(
        13.633333333333335
    )


def test_modified_input_is_rejected(tmp_path):
    bad = tmp_path / "result.json"
    bad.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load(bad, "0" * 64)
