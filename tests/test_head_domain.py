"""Head-aware applicability-domain gate: rank within known heads, abstain outside."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from compose_v4.oracles.head_domain import domain_from_head_smiles

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts/oracles/head_domain_v1"


def _domain():
    manifest = json.loads((ART / "manifest.json").read_text())
    heads = [row["head_region_smiles"] for row in csv.DictReader((ART / "known_head_regions.csv").open())]
    lo, hi = manifest["head_pka_range"]
    return domain_from_head_smiles(heads, lo, hi, manifest["similarity_threshold"]), manifest


def test_known_head_domain_is_large() -> None:
    domain, manifest = _domain()
    # the known-head domain is a large, useful space (the milestone-4 point).
    assert domain.n_known_heads >= 800
    assert manifest["known_head_regions"] >= 800


def test_known_head_candidates_admitted() -> None:
    domain, _ = _domain()
    # DMA / piperazine heads (known ionizable classes) with intrinsic pKa ~9 admit.
    dmapa = domain.admission(
        "CCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)N(C)CCN(C)C", head_pka=9.5)
    piperazine = domain.admission("CCCCCCCCCCCCC(=O)N1CCN(CCCCCCCCCCC)CC1", head_pka=8.5)
    assert dmapa["admitted"] and piperazine["admitted"]


def test_novel_head_abstains_to_active_learning() -> None:
    domain, _ = _domain()
    # guanidine head (novel scaffold + out-of-range intrinsic pKa) -> abstain.
    r = domain.admission("CCCCCCCCCCCCCCCCNC(=N)N", head_pka=12.5)
    assert not r["admitted"]
    assert "active-learning" in r["recommendation"]


def test_no_ionizable_head_rejected() -> None:
    domain, _ = _domain()
    r = domain.admission("CCCCCCCCCCCCCCCCCC(=O)OCC")
    assert not r["admitted"]
    assert any("no ionizable head" in reason for reason in r["reasons"])


def test_frozen_validation_all_correct() -> None:
    manifest = json.loads((ART / "manifest.json").read_text())
    assert manifest["all_validation_correct"] is True
