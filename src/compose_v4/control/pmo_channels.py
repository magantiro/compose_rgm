"""The production PMO proposal channel names, in one place.

`pmo_population_controller` owns these, but importing it from `pmo_contextual_macro` would
close an import cycle (the controller imports the value model). The names are duplicated
here and a test asserts the two agree, so a rename cannot silently split them.
"""

from __future__ import annotations

SHALLOW_CHANNEL = "shallow_program_channel"
STRUCTURED_CHANNEL = "structured_program_channel"
JUMP_CHANNEL = "joint_dependency_region_jump"

CHANNELS: tuple[str, ...] = (SHALLOW_CHANNEL, STRUCTURED_CHANNEL, JUMP_CHANNEL)
