"""BEAE fine-tune substrate: the propiolate aza-Michael reconstructs measured leads.

The novel-linker (Fig-6) substrate qualifies with the propiolate transform
reconstructing both measured BEAE leads (RM-60, Example-2) from their real
components -- in both addition orders, with the correct E-enamine ester -- and
rejecting non-propiolate acceptors.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUAL = REPO_ROOT / "configs/lipid_reactions/beae_finetune_substrate_v1.json"


def _load() -> dict:
    return json.loads(QUAL.read_text())


def test_both_leads_reconstruct_with_e_enamine() -> None:
    d = _load()
    assert set(d["leads"]) == {"RM-60", "Example-2"}
    for name, lead in d["leads"].items():
        assert lead["reconstructed_forward"], name
        assert lead["reconstructed_reverse_order"], name   # order-independent assembly
        assert lead["has_E_enamine_ester"], name
        assert lead["qualified"], name


def test_propiolate_transform_is_chemoselective() -> None:
    d = _load()
    assert d["negatives"], "no negative controls recorded"
    for neg in d["negatives"]:
        assert neg["rejected"], neg["reason"]


def test_substrate_is_finetune_scoped_not_pretraining() -> None:
    d = _load()
    # the substrate is explicitly the linker fine-tune, kept out of the general corpus
    assert "fine-tune" in d["role"].lower() and "not general" in d["role"].lower()
    assert d["all_qualified"] is True
