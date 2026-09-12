# Macro lookahead development episode

Completed: 40 new docking calls in four rounds, 36.17 minutes. Best observed
score improved from -10.6 to -11.0 by a six-membered oxygen-containing fused-ring
construction. This is one unreplicated warm-start episode, not evidence that
lookahead caused the improvement. No job is still running or automatically
continuing. See `DIAGNOSIS.md` for the post-run structural and allocation audit.

Launched 2026-09-10 in session `compose_iclr` from clean, committed source
`6f413638cb4c` on local branch `t4-macro-lookahead`, worktree
`/private/tmp/compose-macro-lookahead`. Main-worktree unrelated edits were not
serialized or modified. No branch was pushed.

Modal app `genmol-t4-opt`; call `fc-01M25AC0YX3HXVAAB4ZM327DS6`.
Volume `compose-v4-artifacts`; run
`t4_macro_lookahead/8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28`.
The local run directory holds the spawn receipt and collected immutable inputs;
the full primitive witnesses and caches remain on the volume.

Four rounds, sixteen proposal workers, at most 384 macro attempts and forty new
docking attempts. Starts from all 94 prior attempted calls, the saved candidate
pool and a best observed score of -10.6. This is warm-start PARP1 seed0, delta=0.4
development, not an independent benchmark or a matched causal comparison.

The source worktree's `docs/T4_MACRO_LOOKAHEAD.md` and self-hashed configuration
record the algorithm, frozen inputs, limits and acceptance checks. Twelve focused
tests passed. No broad regression suite or model training was performed.
The previous episode's outcome remains immutable in `../t4_macro_feedback/`.

Read status without starting work:

```sh
.venv/bin/python tools/t4_feedback_status.py \
  diagnostics/t4_macro_lookahead/8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28/spawn.json
```

Add `--collect --details` to retain saved result, round, proposal and value
ledgers and their physical hashes. No automatic episode continuation or oracle
retry is authorized. The collected result and four complete round ledgers are
now available locally.
