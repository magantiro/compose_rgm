"""Generic CPU node descriptors survive batching without changing support."""

import io
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
    _concatenate_factorized_mark_batches,
)
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


@dataclass(frozen=True)
class ExampleContext:
    schema: str = "test_hydrogen_and_ports_v1"
    feature_dim: int = 2

    def __call__(self, state, scaffold_context, neural_bonds):
        flags = scaffold_context.node_flags()
        return np.stack((state.implicit_h_counts, flags[:, 1]), axis=-1).astype(np.float32)


@pytest.fixture
def setup():
    records = build_tracelet_path_records(("CCO", "C1CC1"), n_slots=8)
    catalog = build_typed_ring_catalog(tuple(r.path.trace for r in records))
    kwargs = {
        "hidden_dim": 16,
        "message_passing_steps": 2,
        "mark_dim": 8,
        "scaffold_conditioning": True,
        "rate_factorization": "superposed",
    }
    torch.manual_seed(911)
    base = FactorizedTraceletRateModel(catalog, **kwargs).eval()
    torch.manual_seed(911)
    model = FactorizedTraceletRateModel(
        catalog, node_context_provider=ExampleContext(), **kwargs
    ).eval()
    state = pad_molecular_graph(smiles_to_molecular_graph("NCC"), 8)
    source = pad_molecular_graph(smiles_to_molecular_graph("NC"), 8)
    context = ScaffoldContext.from_source(source, (1,))
    teacher = AtomInsert(3, 2, 0, 3, ((2, 1),))
    example = FactorizedMarkExample(
        state, 0.4, teacher, "atom_insert", 1.0, 1.0, scaffold_context=context
    )
    collator = FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=catalog,
        node_context_provider=model.node_context_provider,
    )
    return base, model, collator, example


def test_initialization_default_support_and_learning_channel(setup):
    base, model, collator, example = setup
    for name, parameter in base.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
    assert "node_context_encoder.weight" not in base.state_dict()
    batch = collator([example])
    bare = replace(
        batch, node_context_features=None, node_context_schema=None, node_context_keys=None
    )
    masks = model._action_tables(batch, *model._encode_batch(batch))[0]
    base_masks = base._action_tables(bare, *base._encode_batch(bare))[0]
    assert masks.keys() == base_masks.keys()
    for name in masks:
        torch.testing.assert_close(masks[name], base_masks[name], rtol=0, atol=0)
    result = model.forward_mark_batch(batch)
    assert torch.isfinite(result.selected_mark_log_probability).all()
    (-result.selected_mark_log_probability.sum()).backward()
    gradient = model.node_context_encoder.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # A zero feature projection recovers the identical base law.
    with torch.no_grad():
        model.node_context_encoder.weight.zero_()
    restored, original = model.forward_mark_batch(batch), base.forward_mark_batch(bare)
    for name in ("total_hazard", "family_log_probabilities", "selected_mark_log_probability"):
        torch.testing.assert_close(getattr(restored, name), getattr(original, name), rtol=0, atol=0)


def test_features_survive_collation_split_concat_serialization_and_cpu_movement(setup, monkeypatch):
    _, model, collator, example = setup
    other_context = ScaffoldContext.from_source(
        pad_molecular_graph(smiles_to_molecular_graph("NC"), 8), (0, 1)
    )
    batch = collator([example, replace(example, scaffold_context=other_context)])
    original = model.forward_mark_batch(batch)
    rebuilt = _concatenate_factorized_mark_batches((batch.subbatch(0, 1), batch.subbatch(1, 2)))
    buffer = io.BytesIO()
    torch.save(rebuilt, buffer)
    buffer.seek(0)
    decoded = torch.load(buffer, weights_only=False).to(torch.device("cpu"))
    assert decoded.node_context_keys == batch.node_context_keys
    assert decoded.node_context_schema == batch.node_context_schema
    torch.testing.assert_close(
        decoded.node_context_features, batch.node_context_features, rtol=0, atol=0
    )

    # Forward consumes prepared features; it must not call the provider again.
    def forbidden(*_):
        raise AssertionError("feature recomputation inside training forward")

    monkeypatch.setattr(ExampleContext, "__call__", forbidden)
    result = model.forward_mark_batch(decoded)
    for name in ("total_hazard", "family_log_probabilities", "selected_mark_log_probability"):
        torch.testing.assert_close(getattr(original, name), getattr(result, name), rtol=0, atol=0)


def test_missing_stale_and_wrong_schema_features_fail_closed(setup):
    base, model, collator, example = setup
    batch = collator([example])
    for field, value, match in (
        ("node_context_features", None, "missing"),
        ("node_context_features", torch.zeros(1, 8, 1), "shape"),
        ("node_context_keys", ("stale",), "bound"),
        ("node_context_schema", "different", "schema"),
    ):
        with pytest.raises(ValueError, match=match):
            model.forward_mark_batch(replace(batch, **{field: value}))
    with pytest.raises(ValueError, match="silently ignore"):
        base.forward_mark_batch(batch)
    with pytest.raises(ValueError, match="schemas"):
        _concatenate_factorized_mark_batches(
            (batch, replace(batch, node_context_schema="different"))
        )
    other = ScaffoldContext.from_source(
        pad_molecular_graph(smiles_to_molecular_graph("NC"), 8), (0, 1)
    )
    with pytest.raises(ValueError, match="bound"):
        model.forward_mark_batch(
            replace(
                batch,
                scaffold_contexts=(other,),
                scaffold_node_flags=torch.from_numpy(other.node_flags()[None]),
            )
        )


def test_sampling_and_prepared_training_share_features_and_hazard(setup):
    _, model, collator, example = setup
    batch = collator([example])
    reference = model.forward_mark_batch(batch)
    rng = np.random.default_rng(911)
    for _ in range(8):
        mark = model.sample_rewrite_mark_conditioned(
            example.state,
            example.time,
            rng,
            property_values=None,
            scaffold_context=example.scaffold_context,
        )
        assert mark.total_hazard == pytest.approx(reference.total_hazard.item(), rel=1e-6)
        scored = model.forward_mark_batch(
            collator(
                [replace(example, teacher_action=mark.action, teacher_rule_name=mark.rule_name)]
            )
        )
        assert torch.isfinite(scored.selected_mark_log_probability).all()
    (cached,) = model._sampling_state_cache.values()
    torch.testing.assert_close(
        cached.node_context_features, batch.node_context_features, rtol=0, atol=0
    )


def test_provider_can_require_the_same_aromatic_view_as_sampling(setup, monkeypatch):
    _, _, collator, example = setup
    monkeypatch.setattr(ExampleContext, "requires_aromatic_bond_view", True, raising=False)
    with pytest.raises(ValueError, match="requires the aromatic bond view"):
        replace(collator, use_aromatic_bond_view=False)([example])


@pytest.mark.parametrize("bad", (np.full((8, 2), np.nan), np.ones((8, 2)), np.zeros((8, 3))))
def test_provider_must_emit_finite_correct_shape_zero_padding(setup, monkeypatch, bad):
    _, _, collator, example = setup
    monkeypatch.setattr(ExampleContext, "__call__", lambda *_: bad)
    with pytest.raises(ValueError, match="node context features"):
        collator([example])
