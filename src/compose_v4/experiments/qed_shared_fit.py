"""Fit a source-balanced QED value head from frozen-reference feature shards."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from compose_v4.experiments.hphi_region_features import input_dim


@dataclass(frozen=True)
class FeatureShard:
    path: Path
    sha256: str
    role: str
    index: int
    examples: int
    positive_labels: int
    bellman_pairs: int


@dataclass(frozen=True)
class QEDFitConfig:
    budget_max: int = 24
    epochs: int = 40
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    bellman_weight: float = 0.3
    batch_size: int = 4096
    seed: int = 0
    device: str = "cpu"
    checkpoint_selection: str = "overall_bce"

    def validate(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.budget_max, self.epochs, self.patience, self.batch_size)
        ):
            raise ValueError("QED value fit needs positive budget, epochs, and patience")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("QED value fit needs a nonnegative integer seed")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1:
            raise ValueError("QED value fit needs a finite positive learning rate below one")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("QED value fit needs finite nonnegative weight decay")
        if not math.isfinite(self.bellman_weight) or self.bellman_weight < 0:
            raise ValueError("QED value fit needs finite nonnegative Bellman weight")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("QED value fit device must be cpu or cuda")
        if self.checkpoint_selection not in ("overall_bce", "guidance_brier"):
            raise ValueError("QED value fit checkpoint selection is unsupported")


def _read_shard(
    path: Path, expected_sha256: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, str]:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"QED feature shard changed after inventory: {path}")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != {
            "features",
            "labels",
            "region_indices",
            "next_row_indices",
            "next_terminal_targets",
            "metadata",
        }:
            raise ValueError(f"QED feature shard has unexpected fields: {path}")
        features = archive["features"]
        labels = archive["labels"]
        region_indices = archive["region_indices"]
        next_row_indices = archive["next_row_indices"]
        next_terminal_targets = archive["next_terminal_targets"]
        metadata = json.loads(str(archive["metadata"]))
    if not isinstance(metadata, dict):
        raise TypeError(f"QED feature shard metadata must be an object: {path}")
    return (
        features,
        labels,
        region_indices,
        next_row_indices,
        next_terminal_targets,
        metadata,
        observed,
    )


def inspect_feature_corpus(
    directory: Path,
    *,
    reference_identity: dict,
    source_split_sha256: str,
    train_input_indices: tuple[int, ...],
    validation_count: int,
    budget_max: int,
    builder_sha256: str | None = None,
    goal_regions: tuple[tuple[float, float], ...] | None = None,
) -> tuple[FeatureShard, ...]:
    """Require exactly one valid shard for every frozen train/validation source."""
    expected = {
        f"train_{index:04d}.npz": ("train", index, input_index)
        for index, input_index in enumerate(train_input_indices)
    }
    expected.update(
        {
            f"validation_{index:04d}.npz": ("validation", index, index)
            for index in range(validation_count)
        }
    )
    observed = {path.name for path in directory.glob("*.npz")}
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ValueError(
            f"QED feature corpus incomplete or mixed: missing={missing[:5]}, extra={extra[:5]}"
        )
    width = input_dim(budget_max)
    shards = []
    for name, (role, index, input_index) in sorted(expected.items()):
        path = directory / name
        (
            features,
            labels,
            region_indices,
            next_row_indices,
            next_terminal_targets,
            metadata,
            digest,
        ) = _read_shard(path)
        if metadata.get("schema_version") != "compose.qed.shared_features.v4":
            raise ValueError(f"QED feature schema mismatch: {path}")
        if (
            features.dtype != np.float32
            or labels.dtype != np.float32
            or features.ndim != 2
            or labels.ndim != 1
            or region_indices.dtype != np.int16
            or region_indices.shape != labels.shape
            or next_row_indices.dtype != np.int32
            or next_row_indices.shape != labels.shape
            or next_terminal_targets.dtype != np.int8
            or next_terminal_targets.shape != labels.shape
            or len(labels) == 0
            or features.shape != (len(labels), width)
            or not np.isfinite(features).all()
            or not np.isfinite(labels).all()
            or not np.isin(labels, (0.0, 1.0)).all()
        ):
            raise ValueError(f"QED feature shard has invalid arrays: {path}")
        if (
            np.any(next_row_indices < -1)
            or np.any(next_row_indices >= len(labels))
            or np.any(~np.isin(next_terminal_targets, (-1, 0, 1)))
            or np.any((next_row_indices >= 0) == (next_terminal_targets >= 0))
            or np.any((next_row_indices >= 0) & (next_row_indices <= np.arange(len(labels))))
            or np.any(
                (next_terminal_targets >= 0) & (next_terminal_targets.astype(np.float32) != labels)
            )
            or np.any(
                region_indices[next_row_indices[next_row_indices >= 0]]
                != region_indices[next_row_indices >= 0]
            )
        ):
            raise ValueError(f"QED feature shard has invalid Bellman successor links: {path}")
        recorded_regions = metadata.get("goal_regions")
        if (
            not isinstance(recorded_regions, list)
            or not recorded_regions
            or np.any(region_indices < 0)
            or np.any(region_indices >= len(recorded_regions))
        ):
            raise ValueError(f"QED feature shard has invalid goal indices: {path}")
        region_examples = [int(np.sum(region_indices == i)) for i in range(len(recorded_regions))]
        region_positive_labels = [
            int(np.sum(labels[region_indices == i])) for i in range(len(recorded_regions))
        ]
        bellman_pairs = int(np.sum((next_row_indices >= 0) | (next_terminal_targets >= 0)))
        required = {
            "role": role,
            "source_index": index,
            "source_input_row_index": input_index,
            "source_split_sha256": source_split_sha256,
            "reference": reference_identity,
            "budget_max": budget_max,
            "feature_schema": "region_features_v1",
            "target_semantics": "terminal_region",
            "examples": len(labels),
            "positive_labels": int(np.sum(labels)),
            "region_examples": region_examples,
            "region_positive_labels": region_positive_labels,
            "bellman_pairs": bellman_pairs,
            "bellman_boundary_pairs": int(np.sum(next_terminal_targets >= 0)),
        }
        for field, value in required.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"QED feature shard {path}: {field} does not match the frozen corpus"
                )
        if builder_sha256 is not None and metadata.get("builder_sha256") != builder_sha256:
            raise ValueError(f"QED feature shard used another feature builder: {path}")
        if goal_regions is not None and metadata.get("goal_regions") != [
            list(region) for region in goal_regions
        ]:
            raise ValueError(f"QED feature shard used another goal grid: {path}")
        shards.append(
            FeatureShard(path, digest, role, index, len(labels), int(labels.sum()), bellman_pairs)
        )
    return tuple(shards)


def _arrays(
    shard: FeatureShard,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, region_indices, next_row_indices, next_terminal_targets, _, _ = _read_shard(
        shard.path, shard.sha256
    )
    return features, labels, region_indices, next_row_indices, next_terminal_targets


def training_normalization(
    shards: tuple[FeatureShard, ...], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming population moments fitted only on training source shards."""
    count = 0
    mean = np.zeros(width, dtype=np.float64)
    m2 = np.zeros(width, dtype=np.float64)
    for shard in shards:
        if shard.role != "train":
            continue
        features, _, _, _, _ = _arrays(shard)
        local_count = len(features)
        local_mean = features.mean(axis=0, dtype=np.float64)
        local_m2 = np.square(features.astype(np.float64) - local_mean).sum(axis=0)
        delta = local_mean - mean
        next_count = count + local_count
        mean += delta * local_count / next_count
        m2 += local_m2 + delta * delta * count * local_count / next_count
        count = next_count
    if count == 0:
        raise ValueError("QED value fit has no training examples")
    scale = np.sqrt(m2 / count)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _model(width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def _normalized(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: str
) -> torch.Tensor:
    return torch.from_numpy(((features - mean) / scale).astype(np.float32)).to(device)


def fit_value_head(
    shards: tuple[FeatureShard, ...],
    config: QEDFitConfig,
    *,
    guidance_region_index: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict]:
    """Train with one equal-weight optimizer update per source and source-level validation."""
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("QED value fit requested CUDA but no CUDA device is available")
    train = tuple(shard for shard in shards if shard.role == "train")
    validation = tuple(shard for shard in shards if shard.role == "validation")
    if not train or not validation:
        raise ValueError("QED value fit needs both frozen training and validation sources")
    if type(guidance_region_index) is not int or guidance_region_index < 0:
        raise ValueError("QED value fit needs a valid guidance goal index")
    width = input_dim(config.budget_max)
    mean, scale = training_normalization(shards, width)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = _model(width).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    order_rng = np.random.default_rng(config.seed)
    history = []
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for source_position in order_rng.permutation(len(train)):
            shard = train[int(source_position)]
            features, labels, _, next_rows, next_terminal_targets = _arrays(shard)
            x = _normalized(features, mean, scale, config.device)
            y = torch.from_numpy(labels).to(config.device).unsqueeze(1)
            successor_rows = torch.from_numpy(next_rows.astype(np.int64)).to(config.device)
            boundary_values = torch.from_numpy(next_terminal_targets.astype(np.float32)).to(
                config.device
            )
            bellman_mask = (successor_rows >= 0) | (boundary_values >= 0)
            bellman_count = int(bellman_mask.sum())
            targets = boundary_values.clamp(min=0).unsqueeze(1)
            if config.bellman_weight > 0 and bellman_count:
                model.eval()
                with torch.no_grad():
                    non_boundary = successor_rows >= 0
                    targets[non_boundary] = torch.sigmoid(model(x[successor_rows[non_boundary]]))
                model.train()
            row_order = torch.randperm(len(x), device=config.device)
            optimizer.zero_grad(set_to_none=True)
            source_loss = 0.0
            for start in range(0, len(x), config.batch_size):
                indices = row_order[start : start + config.batch_size]
                logits = model(x[indices])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y[indices], reduction="sum"
                ) / len(x)
                if config.bellman_weight > 0 and bellman_count:
                    paired = bellman_mask[indices]
                    if bool(paired.any()):
                        loss = loss + config.bellman_weight * (
                            nn.functional.binary_cross_entropy_with_logits(
                                logits[paired], targets[indices][paired], reduction="sum"
                            )
                            / bellman_count
                        )
                loss.backward()
                source_loss += float(loss.detach())
            optimizer.step()
            train_losses.append(source_loss)
        model.eval()
        validation_losses = []
        brier_losses = []
        guidance_brier_losses = []
        with torch.no_grad():
            for shard in validation:
                features, labels, region_indices, _, _ = _arrays(shard)
                x = _normalized(features, mean, scale, config.device)
                y = torch.from_numpy(labels).to(config.device).unsqueeze(1)
                logits = model(x)
                validation_losses.append(
                    float(nn.functional.binary_cross_entropy_with_logits(logits, y))
                )
                brier_losses.append(float(torch.mean((torch.sigmoid(logits) - y) ** 2)))
                guidance_rows = torch.from_numpy(region_indices == guidance_region_index).to(
                    config.device
                )
                if bool(guidance_rows.any()):
                    guidance_brier_losses.append(
                        float(
                            torch.mean(
                                (torch.sigmoid(logits[guidance_rows]) - y[guidance_rows]) ** 2
                            )
                        )
                    )
        val_loss = float(np.mean(validation_losses))
        guidance_brier = (
            float(np.mean(guidance_brier_losses)) if guidance_brier_losses else math.inf
        )
        history.append(
            {
                "epoch": epoch,
                "train_source_mean_objective": float(np.mean(train_losses)),
                "validation_source_mean_bce": val_loss,
                "validation_source_mean_brier": float(np.mean(brier_losses)),
                "validation_guidance_source_mean_brier": guidance_brier,
            }
        )
        selection_loss = (
            guidance_brier if config.checkpoint_selection == "guidance_brier" else val_loss
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise ValueError("QED value fit did not produce a finite validation checkpoint")
    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    train_prevalence = float(np.mean([shard.positive_labels / shard.examples for shard in train]))
    constant_brier = float(
        np.mean(
            [
                (1 - 2 * train_prevalence) * shard.positive_labels / shard.examples
                + train_prevalence**2
                for shard in validation
            ]
        )
    )
    selected_brier = history[best_epoch]["validation_source_mean_brier"]
    target_train_rates = []
    target_train_examples = 0
    target_train_positive = 0
    for shard in train:
        _, labels, indices, _, _ = _arrays(shard)
        target = labels[indices == guidance_region_index]
        if len(target):
            target_train_rates.append(float(target.mean()))
            target_train_examples += len(target)
            target_train_positive += int(target.sum())
    target_prior = float(np.mean(target_train_rates)) if target_train_rates else None
    target_validation_briers = []
    target_constant_briers = []
    target_validation_examples = 0
    target_validation_positive = 0
    with torch.no_grad():
        for shard in validation:
            features, labels, indices, _, _ = _arrays(shard)
            target = indices == guidance_region_index
            if not np.any(target):
                continue
            target_features = _normalized(features[target], mean, scale, "cpu")
            predictions = torch.sigmoid(model(target_features)).cpu().numpy().reshape(-1)
            actual = labels[target]
            target_validation_briers.append(float(np.mean((predictions - actual) ** 2)))
            if target_prior is not None:
                target_constant_briers.append(float(np.mean((target_prior - actual) ** 2)))
            target_validation_examples += len(actual)
            target_validation_positive += int(actual.sum())
    target_brier = float(np.mean(target_validation_briers)) if target_validation_briers else None
    target_constant_brier = (
        float(np.mean(target_constant_briers)) if target_constant_briers else None
    )
    target_qualified = bool(
        target_train_positive > 0
        and target_train_positive < target_train_examples
        and target_validation_positive > 0
        and target_validation_positive < target_validation_examples
        and target_brier is not None
        and target_constant_brier is not None
        and target_brier < target_constant_brier
    )
    return (
        model,
        mean,
        scale,
        {
            "selected_epoch": best_epoch,
            "checkpoint_selection": config.checkpoint_selection,
            "selected_validation_selection_loss": best_loss,
            "bellman_weight": config.bellman_weight,
            "train_bellman_pairs": sum(shard.bellman_pairs for shard in train),
            "validation_bellman_pairs": sum(shard.bellman_pairs for shard in validation),
            "selected_validation_source_mean_bce": history[best_epoch][
                "validation_source_mean_bce"
            ],
            "selected_validation_source_mean_brier": selected_brier,
            "train_source_mean_positive_rate": train_prevalence,
            "validation_constant_brier": constant_brier,
            "beats_training_constant_on_validation": selected_brier < constant_brier,
            "guidance_region_index": guidance_region_index,
            "guidance_train_sources_with_examples": len(target_train_rates),
            "guidance_train_examples": target_train_examples,
            "guidance_train_positive_labels": target_train_positive,
            "guidance_validation_sources_with_examples": len(target_validation_briers),
            "guidance_validation_examples": target_validation_examples,
            "guidance_validation_positive_labels": target_validation_positive,
            "guidance_validation_source_mean_brier": target_brier,
            "guidance_validation_constant_brier": target_constant_brier,
            "qualified_for_guidance": target_qualified,
            "epochs_run": len(history),
            "history": history,
        },
    )
