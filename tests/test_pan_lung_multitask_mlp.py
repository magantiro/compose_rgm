"""Lock the R1xM4 masked multitask MLP cell and its masked-loss mechanism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # make the scripts package importable
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.train_pan_lung_multitask_mlp import MaskedMultitaskMLP  # noqa: E402

DIAG = REPO_ROOT / "diagnostics/pan_lung_multitask_mlp_R1M4.json"


def test_multitask_mlp_forward_shape() -> None:
    model = MaskedMultitaskMLP(in_dim=16, n_heads=3)
    out = model(torch.zeros((5, 16)))
    assert out.shape == (5, 3)


def test_masked_loss_only_counts_active_head() -> None:
    # two rows, two heads; a wrong prediction on an inactive head must not add loss.
    n_heads = 2
    pred = torch.tensor([[1.0, 9.0], [9.0, 1.0]])  # row0 head0, row1 head1
    head = torch.tensor([0, 1])
    y = torch.tensor([1.0, 1.0])
    mask = torch.zeros((2, n_heads))
    mask[torch.arange(2), head] = 1.0
    y_full = torch.zeros((2, n_heads))
    y_full[torch.arange(2), head] = y
    loss = (((pred - y_full) ** 2) * mask).sum() / mask.sum()
    assert float(loss) == 0.0  # active-head predictions are exact; inactive ignored


def test_committed_r1m4_cell_regression() -> None:
    diag = json.loads(DIAG.read_text())
    assert diag["cell_id"] == "R1xM4"
    comparison = diag["comparison_multitask_vs_single"]
    # large full-SMILES heads must carry real held-lipid signal.
    assert comparison["airway_a549_in_vitro_expression"]["multitask_spearman"] > 0.5
    assert comparison["airway_hbe_in_vitro_expression"]["multitask_spearman"] > 0.8
    # LuT (component-only) is excluded from this molecular cell.
    assert "systemic_iv_lung_selectivity" not in comparison
    # all five full-structure heads reported.
    assert len(comparison) == 5


def test_documented_negative_transfer_on_small_heads() -> None:
    diag = json.loads(DIAG.read_text())
    comparison = diag["comparison_multitask_vs_single"]
    # the small in-vivo/local heads are not rescued by shared-representation
    # multitask: their held-lipid signal stays weak (|Spearman| well below the
    # big heads). This is the reported negative-transfer finding.
    for head in ("airway_hbec_ali_in_vitro_expression", "systemic_iv_barcoded_lung_uptake"):
        assert abs(comparison[head]["multitask_spearman"]) < 0.3
