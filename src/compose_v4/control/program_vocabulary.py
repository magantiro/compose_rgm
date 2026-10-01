"""Task-independent rebuild names shared by proposal and value features.

The vocabulary follows the generic compiler, not a manually shortened feature
list. Anchored composites retain their existing identities and order.
"""

from __future__ import annotations

from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES

ANCHORED_OPTIONS = ("regrow", "append_ring", "fuse_ring", "ring_then_grow")
REPLACEMENT_OPTIONS = (
    *ANCHORED_OPTIONS,
    *(family for family in GENERIC_MODULES if family not in ANCHORED_OPTIONS),
)
