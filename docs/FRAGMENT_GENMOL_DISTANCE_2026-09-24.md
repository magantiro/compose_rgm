# GenMol-compatible distance on the completed development pilot

The previous local helper measures 2048-bit fingerprint distance from a capped
conditioning fragment. GenMol's source instead measures 1024-bit Morgan
radius-2 Tanimoto distance from the **original drug**, over distinct valid
outputs. The two definitions must not share a headline label.

The post-generation tool `score_saved_fragment_genmol_distance.py` isolates
and calls the exact SHA-256-pinned GenMol `get_distance` function. It consumes
the already sealed 400-attempt pilot and its matched saved baseline, verifies
attempt hashes, and does not change either sampler or use references for
proposal selection. Every prompt has outputs; no empty cell was dropped.

| Task, ten prompts and 200 attempts per arm | Local baseline | Full-program arm |
| --- | ---: | ---: |
| Motif extension | 0.687638 | 0.683374 |
| Scaffold decoration | 0.589707 | 0.648044 |

These are equal-prompt macro means, not a pooled molecule mean. They describe
structural distance, not quality improvement or constraint failure. In
particular, the decoration arm's higher distance accompanied much worse
quality, as recorded separately in the complete census.

Artifact: `diagnostics/fragment_complete_program_genmol_distance_v1/result.json`.
It records every prompt/reference, attempt/output/unique denominator, source
and input hashes, environment and code revision. This is still a smaller,
single-seed development panel, not a formal comparison to a published row.

Verification includes an exact reference-distance-zero fixture, duplicate
and failed-attempt accounting, explicit empty-cell behavior, and refusal to
execute any metric source except the inspected pinned bytes. The existing
frozen 2048-bit diagnostic values remain unchanged and differently named.
