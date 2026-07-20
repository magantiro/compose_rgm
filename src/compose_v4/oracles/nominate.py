"""End-to-end oracle nomination: candidate -> admit / rank / abstain.

Wires the three qualified components into one Fig-4 nomination call:
  1. head-aware AD gate (rank within known heads, abstain on novel ones);
  2. molecular AD (the filtering bundle's whole-molecule domain gate);
  3. calibrated filtering score (conservative pessimistic ensemble score) for
     admitted candidates.

A candidate is RANKED only if its head is in the known-head domain AND its whole
molecule is in the bundle's applicability domain; otherwise it ABSTAINS with the
reason (novel head -> active-learning; OOD molecule -> off-domain). Filtering
only: no reward fine-tuning (that stays gated behind the preregistration guard).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .head_domain import HeadDomain, domain_from_head_smiles
from .pan_lung_filtering import score_molecular_filter


def load_head_domain(head_domain_dir: Path) -> HeadDomain:
    manifest = json.loads((head_domain_dir / "manifest.json").read_text())
    heads = [row["head_region_smiles"]
             for row in csv.DictReader((head_domain_dir / "known_head_regions.csv").open())]
    lo, hi = manifest["head_pka_range"]
    return domain_from_head_smiles(heads, lo, hi, manifest["similarity_threshold"])


@dataclass(frozen=True)
class OracleNominator:
    """Frozen, filtering-only pan-lung candidate nominator."""

    head_domain: HeadDomain
    filtering_manifest: Path
    ranking_domain_id: str = "a549"  # the qualified molecular head used for ranking

    @classmethod
    def load(cls, repo_root: Path, ranking_domain_id: str = "a549") -> "OracleNominator":
        return cls(
            head_domain=load_head_domain(repo_root / "artifacts/oracles/head_domain_v1"),
            filtering_manifest=repo_root / "artifacts/oracles/pan_lung_filtering_v1/manifest.json",
            ranking_domain_id=ranking_domain_id,
        )

    def nominate(self, smiles: str, head_pka: float | None = None) -> dict:
        head = self.head_domain.admission(smiles, head_pka=head_pka)
        if not head["admitted"]:
            no_head = any("no ionizable head" in r for r in head["reasons"])
            return {
                "format": "compose_pan_lung_nomination_v1",
                "smiles": smiles,
                "decision": "reject_no_ionizable_head" if no_head else "abstain_novel_head",
                "ranked": False,
                "filtering_score": None, "head_admission": head, "molecular_admission": None,
                "recommendation": ("not an ionizable lipid (no basic head)" if no_head
                                   else "novel head outside domain -> small active-learning calibration round"),
            }
        score = score_molecular_filter(self.filtering_manifest, self.ranking_domain_id, smiles)
        mol_adm = score["admission"]
        if not mol_adm["admitted"]:
            return {
                "format": "compose_pan_lung_nomination_v1",
                "smiles": smiles, "decision": "abstain_off_domain", "ranked": False,
                "filtering_score": None, "head_admission": head, "molecular_admission": mol_adm,
                "recommendation": "molecule outside oracle applicability domain -> abstain",
            }
        return {
            "format": "compose_pan_lung_nomination_v1",
            "smiles": smiles, "decision": "rank", "ranked": True,
            "filtering_score": score["conservative_pessimistic_score"],
            "head_admission": head, "molecular_admission": mol_adm,
            "ensemble_scores": score["scores"],
            "recommendation": "in-domain -> rank by conservative pessimistic score",
        }

    def rank(self, candidates: list[tuple[str, float | None]]) -> list[dict]:
        """Nominate a batch; ranked candidates are returned sorted best-first,
        followed by abstained ones (score None)."""
        results = [self.nominate(smi, pk) for smi, pk in candidates]
        ranked = [r for r in results if r["ranked"]]
        abstained = [r for r in results if not r["ranked"]]
        ranked.sort(key=lambda r: r["filtering_score"], reverse=True)
        return ranked + abstained
