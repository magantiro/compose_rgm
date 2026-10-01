"""The public task guides must not point to missing local files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "README.md",
    "docs/METHODS.md",
    "docs/DEVELOPMENT.md",
    "docs/reference_guidance.md",
    "experiments/README.md",
    "experiments/ASSETS.md",
    "experiments/ABLATIONS.md",
    "experiments/fragments/README.md",
    "experiments/fragments/GENERATION.md",
    "experiments/qed/README.md",
    "experiments/pmo/README.md",
    "experiments/t4/README.md",
)
CONTRIBUTOR_GUIDES = ("CLAUDE.md",)


@pytest.mark.parametrize(
    "guide",
    GUIDES + tuple(name for name in CONTRIBUTOR_GUIDES if (ROOT / name).is_file()),
)
def test_local_links_exist(guide: str) -> None:
    source = ROOT / guide
    assert source.is_file()
    for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
        if target.startswith(("http:", "https:", "mailto:", "#")):
            continue
        local_path = (source.parent / target.split("#", 1)[0]).resolve()
        assert local_path.exists(), f"{guide}: missing local link {target}"
