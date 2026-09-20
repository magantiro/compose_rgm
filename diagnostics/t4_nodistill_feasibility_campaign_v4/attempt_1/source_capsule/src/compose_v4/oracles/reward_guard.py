"""Fail-closed reward wrapper around the filtering-only pan-lung bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .pan_lung_filtering import score_lut_filter, score_molecular_filter


class RewardFineTuningDisabledError(RuntimeError):
    """Raised when an admitted candidate requests a reward without authorization."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FilteringRewardGuard:
    """Expose filtering scores while making reward generation fail closed.

    Reward fine-tuning requires two independent explicit switches: the bundle
    manifest must authorize it and a frozen preregistration file must bind its
    candidate budget and thresholds to the exact manifest hash.  The released
    filtering bundle leaves the manifest switch false.
    """

    def __init__(
        self, manifest_path: Path, preregistration_path: Path | None = None
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.preregistration_path = (
            Path(preregistration_path).resolve() if preregistration_path is not None else None
        )
        self.preregistration: dict[str, Any] | None = None
        self.preregistration_sha256: str | None = None
        self.authorization_errors: list[str] = []
        self._load_authorization()

    def _load_authorization(self) -> None:
        if not bool(self.manifest.get("reward_fine_tuning_enabled", False)):
            self.authorization_errors.append(
                "filtering bundle does not authorize reward fine-tuning"
            )
        if self.preregistration_path is None:
            self.authorization_errors.append("no preregistration file was supplied")
            return
        if not self.preregistration_path.exists():
            self.authorization_errors.append("preregistration file does not exist")
            return
        self.preregistration_sha256 = _sha256(self.preregistration_path)
        try:
            registration = json.loads(self.preregistration_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            self.authorization_errors.append(f"invalid preregistration file: {exc}")
            return
        self.preregistration = registration
        required = {
            "format": "compose_pan_lung_reward_preregistration_v1",
            "enable_reward_fine_tuning": True,
            "frozen_before_optimization": True,
            "bundle_manifest_sha256": self.manifest_sha256,
        }
        for key, expected in required.items():
            if registration.get(key) != expected:
                self.authorization_errors.append(
                    f"preregistration field {key!r} must equal {expected!r}"
                )
        if not str(registration.get("registration_id", "")).strip():
            self.authorization_errors.append("preregistration registration_id is missing")
        if not str(registration.get("registered_at_utc", "")).strip():
            self.authorization_errors.append("preregistration timestamp is missing")
        budget = registration.get("candidate_budget")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            self.authorization_errors.append(
                "preregistration candidate_budget must be a positive integer"
            )
        thresholds = registration.get("minimum_pessimistic_scores")
        if not isinstance(thresholds, dict) or not thresholds:
            self.authorization_errors.append(
                "preregistration minimum_pessimistic_scores must be a non-empty object"
            )

    @property
    def reward_fine_tuning_authorized(self) -> bool:
        return not self.authorization_errors

    def authorization_status(self) -> dict[str, object]:
        return {
            "reward_fine_tuning_authorized": self.reward_fine_tuning_authorized,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "preregistration": (
                str(self.preregistration_path)
                if self.preregistration_path is not None
                else None
            ),
            "preregistration_sha256": self.preregistration_sha256,
            "blocking_reasons": list(self.authorization_errors),
        }

    @staticmethod
    def _filtering_decision(score: dict[str, object]) -> dict[str, object]:
        return {
            "mode": "filtering_only",
            "admitted": bool(score["admission"]["admitted"]),
            "filtering_score": score["conservative_pessimistic_score"],
            "reward": None,
            "oracle": score,
        }

    def filter_molecular(self, domain_id: str, smiles: str) -> dict[str, object]:
        return self._filtering_decision(
            score_molecular_filter(self.manifest_path, domain_id, smiles)
        )

    def filter_lut(self, head: str, tail: str, round_id: int) -> dict[str, object]:
        return self._filtering_decision(
            score_lut_filter(self.manifest_path, head, tail, round_id)
        )

    def _reward_decision(self, score: dict[str, object]) -> dict[str, object]:
        if not bool(score["admission"]["admitted"]):
            return {
                "mode": "reward_request_rejected_ood",
                "admitted": False,
                "filtering_score": None,
                "reward": None,
                "oracle": score,
                "authorization": self.authorization_status(),
            }
        if not self.reward_fine_tuning_authorized:
            raise RewardFineTuningDisabledError(
                "reward fine-tuning is disabled: " + "; ".join(self.authorization_errors)
            )
        reward = float(score["conservative_pessimistic_score"])
        return {
            "mode": "preregistered_reward",
            "admitted": True,
            "filtering_score": reward,
            "reward": reward,
            "oracle": score,
            "authorization": self.authorization_status(),
        }

    def reward_molecular(self, domain_id: str, smiles: str) -> dict[str, object]:
        return self._reward_decision(
            score_molecular_filter(self.manifest_path, domain_id, smiles)
        )

    def reward_lut(self, head: str, tail: str, round_id: int) -> dict[str, object]:
        return self._reward_decision(
            score_lut_filter(self.manifest_path, head, tail, round_id)
        )
