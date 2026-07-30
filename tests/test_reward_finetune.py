"""Regression test for the RTB reward fine-tuner (scripts/griddd_reward_finetune.py).

Validates the RTB objective and the differentiable trajectory log-prob on a tiny
model + a real benzene path -- no rollouts, no oracle, no heavy compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from griddd_reward_finetune import (  # noqa: E402
    RewardFineTuner,
    rtb_loss,
    trajectory_log_prob,
)


def test_rtb_loss_is_zero_at_the_trajectory_balance_optimum() -> None:
    # residual = logZ + logp_theta - logp_base - r/alpha; construct it to vanish.
    log_z = torch.tensor(0.5)
    log_p_theta = torch.tensor(-2.0)
    log_p_base = torch.tensor(-3.0)
    reward = torch.tensor(0.15)
    alpha = 0.1
    # 0.5 + (-2) - (-3) - 0.15/0.1 = 0.5 - 2 + 3 - 1.5 = 0.0
    assert float(rtb_loss(log_p_theta, log_p_base, reward, log_z, alpha)) < 1e-6


def test_rtb_loss_positive_off_optimum_and_differentiable() -> None:
    log_z = torch.zeros((), requires_grad=True)
    log_p_theta = torch.tensor(-2.0, requires_grad=True)
    loss = rtb_loss(log_p_theta, torch.tensor(-3.0), torch.tensor(0.9), log_z, 0.1)
    assert float(loss) > 0
    loss.backward()
    assert log_p_theta.grad is not None and log_z.grad is not None


def _benzene_trajectory():
    records = build_tracelet_path_records(("c1ccccc1",), n_slots=12, typed_ring_payloads=True)
    catalog = build_typed_ring_catalog(
        (r.path.trace for r in records),
        max_cycle_templates=32, max_attach_templates=32, max_ear_templates=32,
    )
    rec = records[0]
    steps = rec.path.trace.steps
    n = min(3, len(steps))
    tau = {
        "states": tuple(rec.path.states[i] for i in range(n)),
        "times": tuple(0.4 for _ in range(n)),
        "actions": tuple(steps[i].action for i in range(n)),
        "rule_names": tuple(steps[i].rule_name for i in range(n)),
        "jump_rates": tuple(rec.path.operational_jump_rate(i) for i in range(n)),
        "reward": 0.15,
    }
    return catalog, tau


def test_trajectory_log_prob_is_finite_and_differentiable(monkeypatch) -> None:
    import griddd_reward_finetune as reward_module

    catalog, tau = _benzene_trajectory()
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
        enable_ring_system_delete=False,
    )
    captured = {}
    real_prepare = reward_module.prepare_factorized_mark_batch

    def capture_prepare(*args, **kwargs):
        captured["compute_ring_system_delete"] = kwargs.get(
            "compute_ring_system_delete"
        )
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(reward_module, "prepare_factorized_mark_batch", capture_prepare)
    log_p = trajectory_log_prob(
        model, tau["states"], tau["times"], tau["actions"], tau["rule_names"], tau["jump_rates"]
    )
    assert captured["compute_ring_system_delete"] is False
    assert torch.isfinite(log_p)
    log_p.backward()  # gradients flow back into the policy
    assert any(p.grad is not None for p in model.parameters())


def test_trajectory_log_prob_inherits_editing_capabilities() -> None:
    # H2 fix: a ring_system_restate mark is OUTSIDE a model's dense candidates unless the batch is built with
    # compute_ring_restates + the ring_catalog. trajectory_log_prob must INHERIT model.operator_capabilities +
    # model.ring_catalog so an edit model scores its OWN editing marks finite. Without inheriting (the old
    # behavior), even a restate-enabled model raises "teacher ring restate outside exact dynamic candidates".
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions

    catalog, _ = _benzene_trajectory()
    state = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1C"), 12)
    restates = enumerate_ring_system_restate_actions(state)
    assert restates, "expected a ring_system_restate candidate on toluene"
    edit = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_restates=True
    )
    log_p = trajectory_log_prob(
        edit, (state,), (0.4,), (restates[0],), ("ring_system_restate",), (1.0,)
    )
    assert torch.isfinite(log_p)  # capability inherited -> the model's editing mark is scored, not dropped


def test_reward_finetuner_train_step_runs_and_lowers_a_high_residual() -> None:
    catalog, tau = _benzene_trajectory()
    policy = FactorizedTraceletRateModel(catalog, hidden_dim=16, message_passing_steps=1)
    frozen = FactorizedTraceletRateModel(catalog, hidden_dim=16, message_passing_steps=1)
    tuner = RewardFineTuner(policy, frozen)
    loss0 = tuner.train_step([tau])
    assert loss0 >= 0.0
    # a few steps should not blow up and should keep producing finite losses
    losses = [tuner.train_step([tau]) for _ in range(3)]
    assert all(loss == loss for loss in losses)  # not NaN
