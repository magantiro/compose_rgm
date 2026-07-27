"""Hierarchical training sampler for the B-edit mixture: layer -> curriculum bin -> example.

The training records are a broad-organic mixture of three layers (synthetic corruption, real MMP analogue
paths, sparse scaffold-series paths). Uniform sampling over the record tuple makes the realized layer mix
the raw COUNT ratio and lets the naturally frequent chemistry (sulfur-/chlorine-heavy molecules) dominate
while rare-element edit targets (P, I, B) go unseen. This sampler instead draws:

    layer (by configured weight) -> curriculum bin (stratified over non-empty path-length bins) -> example

and enforces measured MINIMUM COVERAGE floors for cold vocabulary: a fraction of every epoch is reserved,
round-robin, for records whose edit target touches each cold (non-CNOF) element, so the widened output-head
slots always receive positive supervision. It is a plain index sampler (``__iter__``/``__len__``) usable
directly as a ``torch.utils.data.DataLoader`` sampler; it holds no tensors and is fully seeded.

Design notes:
- Curriculum bin EDGES are set from MEASURED statistics (the compiled path-length distribution), never
  preset easy/medium/hard thresholds -- pass them in from the mining/manifest stats.
- Layer weights are the ablation knob; passing weights proportional to counts + no floors reproduces the
  uniform baseline exactly (so the sampler is a strict generalization).
- Floors are a REQUESTED minimum, capped so their sum never exceeds the epoch; the remainder follows the
  layer/bin hierarchy. No arbitrary equal-per-element lock.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ---- per-record tags (attach when the record tuple is built) ------------------------------------------
CORRUPTION = "corruption"
MMP = "mmp"
SCAFFOLD = "scaffold"
_CNOF = frozenset({"C", "N", "O", "F"})


@dataclass(frozen=True)
class RecordTag:
    """What the sampler needs to know about one training record."""
    layer: str                       # CORRUPTION | MMP | SCAFFOLD
    path_length: int                 # for the curriculum bin
    target_elements: frozenset[str] = field(default_factory=frozenset)  # elements the edits PRODUCE

    @property
    def cold_elements(self) -> frozenset[str]:
        return frozenset(self.target_elements) - _CNOF


def _bin_index(path_length: int, edges: tuple[int, ...]) -> int:
    """Bin a path length into [0, len(edges)] half-open buckets by ascending edges."""
    idx = 0
    for edge in edges:
        if path_length <= edge:
            return idx
        idx += 1
    return idx


class HierarchicalMarkSampler:
    """Index sampler over a tagged record tuple. Deterministic given ``seed``."""

    def __init__(
        self,
        tags: list[RecordTag],
        *,
        layer_weights: dict[str, float],
        path_length_bins: tuple[int, ...] = (1, 2, 4),
        cold_element_floor: float = 0.0,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        if not tags:
            raise ValueError("no records to sample")
        if cold_element_floor < 0.0:
            raise ValueError("cold_element_floor must be non-negative")
        self.tags = tags
        self.path_length_bins = tuple(path_length_bins)
        self.cold_element_floor = float(cold_element_floor)
        self.samples_per_epoch = int(samples_per_epoch or len(tags))
        self.seed = int(seed)

        # (layer, bin) -> record indices; and cold element -> record indices.
        self._groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        self._cold_groups: dict[str, list[int]] = defaultdict(list)
        present_layers: set[str] = set()
        for i, tag in enumerate(tags):
            self._groups[(tag.layer, _bin_index(tag.path_length, self.path_length_bins))].append(i)
            present_layers.add(tag.layer)
            for elem in tag.cold_elements:
                self._cold_groups[elem].append(i)

        # keep only layers that both have records AND a positive requested weight.
        weights = {ly: float(w) for ly, w in layer_weights.items() if w > 0 and ly in present_layers}
        if not weights:
            raise ValueError("no layer has both records and positive weight")
        total = sum(weights.values())
        self._layers = sorted(weights)
        self._layer_probs = np.array([weights[ly] / total for ly in self._layers], dtype=float)
        self._layer_bins = {
            ly: sorted(b for (lyr, b) in self._groups if lyr == ly and self._groups[(lyr, b)])
            for ly in self._layers
        }

    def realized_layer_fractions(self) -> dict[str, float]:
        """The layer mix a long uniform-count draw would give (diagnostic; == count ratio)."""
        counts: dict[str, int] = defaultdict(int)
        for tag in self.tags:
            counts[tag.layer] += 1
        n = len(self.tags)
        return {ly: counts[ly] / n for ly in sorted(counts)}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        picks: list[int] = []

        # 1) coverage floors: reserve floor*N draws (per cold element), capped so the sum <= epoch.
        if self.cold_element_floor > 0.0 and self._cold_groups:
            per_element = int(self.cold_element_floor * self.samples_per_epoch)
            budget = self.samples_per_epoch // 2  # floors never consume more than half the epoch
            for elem in sorted(self._cold_groups):
                idxs = self._cold_groups[elem]
                take = min(per_element, max(0, budget - len(picks)))
                if take and idxs:
                    picks.extend(int(j) for j in rng.choice(idxs, size=take, replace=True))

        # 2) fill the remainder by layer (weighted) -> bin (stratified) -> record (uniform).
        while len(picks) < self.samples_per_epoch:
            layer = self._layers[int(rng.choice(len(self._layers), p=self._layer_probs))]
            bins = self._layer_bins[layer]
            if not bins:
                continue
            b = int(rng.choice(bins))  # stratified: each non-empty bin equally likely within the layer
            idxs = self._groups[(layer, b)]
            picks.append(int(rng.choice(idxs)))

        rng.shuffle(picks)
        return iter(picks[: self.samples_per_epoch])
