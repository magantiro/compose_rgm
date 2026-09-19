"""Target-free proposal imitation, separate from rollout or future-value learning.

Saved exact-slot witnesses provide action labels. They do not provide a behavior
policy likelihood, an advantage, or an expected future outcome. Recognition uses
the existing program contracts; no molecular endpoint is installed in a menu.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compose_v4.control.carbonyl_option import (
    CARBONYL_OPTIONS,
    CarbonylProgress,
    carbonyl_indices,
    completed_core_carbonyl,
    core_descriptor_indices,
)
from compose_v4.control.macro_engine import ELEMENT_CODE, MACRO_FAMILIES
from compose_v4.control.option_features import structural_option_features
from compose_v4.control.option_policy import AdvantageWeightedOptionActor
from compose_v4.control.option_selector import GENERIC_OPTION, MACRO_OPTIONS
from compose_v4.control.ring_program import (
    RingProgress,
    RingSpec,
    completed_construction,
    construction_branches,
    construction_indices,
    real_slots,
)
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state


@dataclass(frozen=True)
class DemonstratedOption:
    start: int
    stop: int
    option: str
    compound: bool


def descriptor_menu() -> tuple[str, ...]:
    """Data-independent diagnostic menu over existing ring parameter support.

    All C/N/O quotas are included, not just quotas observed in winners. This is
    a distractor set for offline classification, not a production applicability
    row. The production caller continues to supply its own applicable options.
    """
    rings = []
    for topology in ("pendant", "fused"):
        for size in (5, 6):
            for carbon in range(size + 1):
                for nitrogen in range(size - carbon + 1):
                    counts = (carbon, nitrogen, size - carbon - nitrogen)
                    for electronic in ("nonaromatic", "aromatic"):
                        rings.append(RingSpec(topology, size, counts, electronic).option)
    return (GENERIC_OPTION, *MACRO_OPTIONS, *CARBONYL_OPTIONS, *rings)


def _ring_segment(graphs, actions, start):
    stop = start
    while stop < len(actions) and actions[stop][0] == "atom_insert" and stop - start < 7:
        stop += 1
    growth = stop - start
    if growth not in (3, 4, 5, 6) or stop == len(actions):
        return None
    family, closure = actions[stop]
    if family not in MACRO_FAMILIES["cyclize"]:
        return None
    births = [action for _, action in actions[start:stop]]
    if any(len(birth.neighbors) != 1 for birth in births):
        return None
    path = tuple(int(birth.slot) for birth in births)
    if any(birth.neighbors[0][0] != previous for previous, birth in zip(path, births[1:])):
        return None
    ends = {int(closure.a), int(closure.b)}
    if path[-1] not in ends:
        return None
    end = (ends - {path[-1]}).pop()
    first_anchor = int(births[0].neighbors[0][0])
    topology = "pendant" if end == path[0] else "fused"
    anchors = (first_anchor,) if topology == "pendant" else (first_anchor, end)
    if len(set(anchors + path)) != len(anchors + path):
        return None
    size = growth + (2 if topology == "fused" else 0)
    if size not in (5, 6):
        return None
    origin, product = graphs[start], graphs[stop + 1]
    cycle = path if topology == "pendant" else (first_anchor, *path, end)
    elements = tuple(ELEMENT_CODE[e] for e in ("C", "N", "O"))
    types = [int(product.atom_types[slot]) for slot in cycle]
    if any(element not in elements for element in types):
        return None
    counts = tuple(types.count(element) for element in elements)
    for electronic in ("aromatic", "nonaromatic"):
        spec = RingSpec(topology, size, counts, electronic)
        for branch in construction_branches(origin, real_slots(origin), spec):
            if branch[0] != anchors:
                continue
            progress = RingProgress()
            matched = True
            for step, (rule, action) in enumerate(actions[start : stop + 1]):
                if construction_indices(
                    (rule,),
                    (action,),
                    (1.0,),
                    graphs[start + step],
                    spec,
                    progress,
                    branch,
                    step,
                ) != [0]:
                    matched = False
                    break
                if step < growth:
                    progress = RingProgress(anchors, branch[1], path[: step + 1])
                # Descriptor membership plus exact replay is necessary; the
                # final semantic contract also rejects unintended attachments.
            if matched and completed_construction(origin, product, progress, spec):
                return DemonstratedOption(start, stop + 1, spec.option, True)
    return None


def _carbonyl_segment(graphs, actions, start):
    if start + 4 > len(actions) or actions[start][0] != "cycle_open":
        return None
    opened = actions[start][1]
    for edge in ((int(opened.a), int(opened.b)), (int(opened.b), int(opened.a))):
        progress = CarbonylProgress()
        for step, (family, action) in enumerate(actions[start : start + 4]):
            if core_descriptor_indices((family,), (action,), (1.0,), progress, edge, step) != [0]:
                break
            progress = progress.advance(edge, int(action.slot) if step in (1, 2) else None, step)
        else:
            if completed_core_carbonyl(graphs[start], graphs[start + 4], progress):
                return DemonstratedOption(start, start + 4, CARBONYL_OPTIONS[1], True)
    return None


def recognize_trace(states: list[dict], records: list[dict]) -> tuple[DemonstratedOption, ...]:
    """Replay a trace exactly, then greedily compress existing compound options.

    Compression is a deterministic demonstration-label convention, not a
    shortest-route proof. Unmatched compounds retain their primitive decisions.
    Never reconstruct persistent slots from a canonical molecular string.
    """
    if len(states) != len(records) + 1 or not records:
        raise ValueError("trace must have one more exact state than nonempty action records")
    graphs = [decode_state(state) for state in states]
    actions = [decode_action(record) for record in records]
    system = editing_v2_semantic_rewrite_system()
    for index, (family, action) in enumerate(actions):
        product = system.apply(graphs[index], family, action)
        if encode_state(product) != encode_state(graphs[index + 1]):
            raise ValueError(f"trace state {index + 1} differs from exact executor replay")
        if not 1 <= product.n_real_atoms <= 40 or product.n_atoms != 48:
            raise ValueError(f"trace state {index + 1} exceeds the declared molecular support")
        if canonical_state_key(graphs[index]) == canonical_state_key(product):
            raise ValueError(f"trace action {index} is a canonical self-event")
    ordinary = {
        "atom_insert": "grow",
        "atom_delete": "shrink",
        "atom_restate_semantic": "local",
        "bond_reorder": "local",
        "bond_reroute": "rebuild",
        "cycle_open": "open",
        "cycle_close": "cyclize",
        "bond_insert": "cyclize",
        "ring_system_restate": "restate",
    }
    result, start = [], 0
    while start < len(actions):
        segment = _ring_segment(graphs, actions, start) or _carbonyl_segment(graphs, actions, start)
        if segment is None:
            family, action = actions[start]
            option = ordinary.get(family, GENERIC_OPTION)
            if carbonyl_indices(graphs[start], (family,), (action,)):
                option = CARBONYL_OPTIONS[0]
            segment = DemonstratedOption(start, start + 1, option, False)
        result.append(segment)
        start = segment.stop
    return tuple(result)


@dataclass(frozen=True)
class DemonstrationFitConfig:
    hidden: int = 32
    updates: int = 250
    batch_size: int = 32
    learning_rate: float = 0.003
    base_floor: float = 0.1
    kl_penalty: float = 0.02
    seed: int = 20260912

    def __post_init__(self):
        for name in ("hidden", "updates", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.base_floor <= 1 or not math.isfinite(self.base_floor):
            raise ValueError("base_floor must be finite in (0,1]")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.kl_penalty < 0 or not math.isfinite(self.kl_penalty):
            raise ValueError("kl_penalty must be finite and nonnegative")


def batch_option_scores(actor, states, options):
    """Exactly factor the first affine layer, without B x K feature replication."""
    if states.ndim != 2 or states.shape[1] != actor.state_dim:
        raise ValueError("demonstration state matrix has the wrong shape")
    if options.ndim != 2 or options.shape[1] != actor.option_dim:
        raise ValueError("demonstration option matrix has the wrong shape")
    first = actor.network[0]
    hidden = F.linear(states, first.weight[:, : actor.state_dim])[:, None, :]
    hidden = hidden + F.linear(options, first.weight[:, actor.state_dim :], first.bias)[None, :, :]
    for layer in actor.network[1:]:
        hidden = layer(hidden)
    return hidden.squeeze(-1)


def source_weights(source_ids) -> np.ndarray:
    counts = Counter(source_ids)
    if not counts or any(not isinstance(source, str) or not source for source in counts):
        raise ValueError("nonempty source identities are required")
    return np.asarray([1 / (len(counts) * counts[source]) for source in source_ids])


def fit_demonstration_actor(features, labels, source_ids, options, reference, *, config):
    """Supervised, source-balanced imitation. No invented behavior probabilities."""
    x, y = np.asarray(features, dtype=np.float32), np.asarray(labels)
    names, base = tuple(options), np.asarray(reference, dtype=np.float32)
    if (
        x.ndim != 2
        or not x.size
        or not np.isfinite(x).all()
        or y.shape != (len(x),)
        or not np.issubdtype(y.dtype, np.integer)
        or len(source_ids) != len(x)
        or np.any(y < 0)
        or np.any(y >= len(names))
        or len(set(names)) != len(names)
        or GENERIC_OPTION not in names
        or base.shape != (len(names),)
        or not np.isfinite(base).all()
        or np.any(base <= 0)
        or not np.isclose(base.sum(), 1)
    ):
        raise ValueError("invalid demonstration features, labels, options or reference")
    weights = source_weights(source_ids)
    option = torch.from_numpy(np.stack([structural_option_features(name) for name in names]))
    x, y, base = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)), torch.from_numpy(base)
    # Restore the caller's torch RNG; sampling uses a separate generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        actor = AdvantageWeightedOptionActor(x.shape[1], option.shape[1], config.hidden)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    history = []
    for step in range(config.updates):
        indices = torch.multinomial(
            torch.from_numpy(weights), config.batch_size, replacement=True, generator=generator
        )
        scores = batch_option_scores(actor, x[indices], option)
        q = config.base_floor * base + (1 - config.base_floor) * torch.softmax(
            scores + base.log(), dim=1
        )
        loss = -q[torch.arange(len(indices)), y[indices]].log().mean()
        loss += config.kl_penalty * (q * (q.log() - base.log())).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite demonstration-actor loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, config.updates - 1):
            history.append({"update": step + 1, "loss": float(loss.detach())})
    return actor.eval(), history


def append_ignored_context(actor: AdvantageWeightedOptionActor, dimensions: int):
    """Adapt to the existing option runtime, ignoring unseen utility/region inputs.

    The new columns are exactly zero. This does not fabricate utility labels or
    learn WHERE. It allows a separately qualified value model to use real context
    while the demonstration actor only uses the current molecular prefix.
    """
    if type(dimensions) is not int or dimensions < 0:
        raise ValueError("context dimensions must be a nonnegative integer")
    result = copy.deepcopy(actor)
    old = result.network[0]
    weight = old.weight.detach()
    zeros = weight.new_zeros((weight.shape[0], dimensions))
    old.weight = nn.Parameter(
        torch.cat((weight[:, : actor.state_dim], zeros, weight[:, actor.state_dim :]), dim=1)
    )
    old.in_features += dimensions
    result.state_dim += dimensions
    return result.eval()
