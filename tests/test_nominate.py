"""End-to-end oracle nomination: candidate -> admit / rank / abstain."""

from __future__ import annotations

from pathlib import Path

from compose_v4.oracles.nominate import OracleNominator

REPO_ROOT = Path(__file__).resolve().parents[1]

# a known ionizable lipid whose head is in-domain (polyamine/piperazine core)
KNOWN = "CCCCCCCCCCC(O)CN(CCN1CCN(CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC1)CC(O)CCCCCCCCCC"
NOVEL_HEAD = "CCCCCCCCCCCCCCCCNC(=N)N"        # guanidine head (novel + OOD pKa)
NO_HEAD = "CCCCCCCCCCCCCCCCCC(=O)OCC"          # no ionizable head


def _nom() -> OracleNominator:
    return OracleNominator.load(REPO_ROOT, ranking_domain_id="a549")


def test_novel_head_abstains_to_active_learning() -> None:
    r = _nom().nominate(NOVEL_HEAD, head_pka=12.5)
    assert r["decision"] == "abstain_novel_head"
    assert r["ranked"] is False and r["filtering_score"] is None
    assert "active-learning" in r["recommendation"]


def test_non_ionizable_is_rejected() -> None:
    r = _nom().nominate(NO_HEAD)
    assert r["decision"] == "reject_no_ionizable_head"
    assert r["ranked"] is False


def test_known_head_lipid_is_ranked_or_off_domain() -> None:
    # a known-head lipid passes the head gate; it then ranks (in molecular AD)
    # or abstains off-domain -- but never abstains as a novel head.
    r = _nom().nominate(KNOWN, head_pka=9.5)
    assert r["decision"] in {"rank", "abstain_off_domain"}
    assert r["head_admission"]["admitted"] is True
    if r["decision"] == "rank":
        assert r["filtering_score"] is not None


def test_rank_batch_orders_ranked_before_abstained() -> None:
    results = _nom().rank([(KNOWN, 9.5), (NOVEL_HEAD, 12.5), (NO_HEAD, None)])
    # ranked candidates (score not None) come first, sorted best-first.
    ranked = [r for r in results if r["ranked"]]
    abstained = [r for r in results if not r["ranked"]]
    assert results[:len(ranked)] == ranked
    assert len(abstained) >= 2
    scores = [r["filtering_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
