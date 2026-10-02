# QED value-head build operations

This branch preserves the exact zero-oracle Modal entry points used to build a
source-disjoint value head for the shared molecular reference. These scripts are
operational records and are not part of the reviewer-facing package.

The rollout image reads source revision
`9c6db0b8dc16a9cbde93b21cba080f0c72c3b648`. It uses the NLL reference
checkpoint and catalog pinned by `experiments/fragments/assets.json`, horizon
24, and 16 derived seeds per training source. The split has 1,023 training,
128 validation, and 800 unopened test sources. No objective oracle is called.

The feature and fit images read revision
`98d8345dab6fe404786305adcfc9a1801e71ddee`. The only change relevant to
the feature arrays is that graph embeddings no longer build legal-action
tables. The original and new builders produced identical feature, label,
goal-index, and Bellman-link arrays for training source 0 and validation
source 0. The focused QED suite passed 37 tests and Linux CI run
`37043415646` passed 395 tests, lint, formatting, and the isolated PMO
environment check.

`original_features_app.py` and `original_fit_watch_app.py` preserve the
superseded, stopped feature pipeline. Its completed shards remain on the
volume, but the current fit uses only `qed_shared_features_fast_v1/`.

Use Modal profile `rahul-94866` and volume `compose-v4-artifacts`.

| Function | Deployed app | Artifact or call |
| --- | --- | --- |
| `train16_rollouts_app.py` | `ap-2EAhnB6CeWlTycac6CmNxI` | `qed_shared_train16_v1/`; initial driver `fc-01M3YT8V2N12749X11ZW6E4N7X` |
| `fast_features_app.py` | `ap-DrAAJlSg8tswdvgQPeuhL2` | `qed_shared_features_fast_v1/` |
| `fit_watch_app.py` | `ap-J4hzkLk3MbdqCkfFIRIGYJ` | `qed_shared_value_attempt_v4/` only after all 1,151 feature shards exist |
| `rollout_supervisor_app.py` | `ap-jHitWZ5RXNAsWZGYeWXLuA` | `qed_shared_train16_supervisor_v1.json` |

The rollout worker cap was set to 32 at 13:22 EDT, then raised to 48 at
14:10 EDT after the first run's hard-source gaps limited throughput. The
source code and output identity did not change. The
supervisor may resume it at most three times, only after the prior call is
terminal and no committed output is still arriving. It refuses another run if
the last one made no progress. The feature app checks every ten minutes for
newly committed rollouts, builds at most 64 shards per trigger, and preserves
each source's frozen role and input identity. The fit watcher checks every 30
minutes and launches only once after all required shards exist. A failed fit
is not retried automatically.

The fit's target-specific validation check remains unchanged. A head is
eligible for guidance only if it beats the train-only constant on the frozen
validation sources for QED at least 0.9 and similarity at least 0.4, with
nondegenerate labels in both roles. A failed head remains a recorded negative
result and must not be added to the reviewer-facing asset manifest.

Stop the two scheduled apps and the rollout supervisor after the fit is
complete or a fail-closed condition is recorded. Do not delete their volume
artifacts or overwrite a published shard.
