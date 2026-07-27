"""Regression: the B-edit launch documentation must not revert to the strict warm-start flag.

The strict `--initialize-from-source-checkpoint` crashes on the CNOF(4)->ORGANIC(15) head-width mismatch; the
production B->B-edit warm-start must use `--initialize-compatible-from-source-checkpoint` (semantic partial
transfer). This test fails if any *command-line* usage of the strict flag (flag followed by a shell
line-continuation) reappears in the launch doc, and requires the compatible flag to be present.
"""
from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent / "docs" / "BEDIT_TRAINING_LAUNCH.md"
_STRICT_CMD = "--initialize-from-source-checkpoint \\"          # command-continuation form
_COMPATIBLE = "--initialize-compatible-from-source-checkpoint"


def test_launch_doc_uses_compatible_flag_in_command():
    text = _DOC.read_text()
    assert _COMPATIBLE in text, "launch doc must document the compatible warm-start flag"
    assert _STRICT_CMD not in text, (
        "launch doc reverted to the strict --initialize-from-source-checkpoint in a command block; "
        "it crashes on the CNOF->ORGANIC head-width mismatch. Use "
        "--initialize-compatible-from-source-checkpoint."
    )


def test_launch_doc_flags_the_strict_crash():
    # the correction must explicitly document why the strict flag is wrong (prose mention is allowed)
    text = _DOC.read_text().lower()
    assert "crash" in text and "compatible" in text
