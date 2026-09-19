# 5HT1B-2 strict protonation-support probe

## Outcome

The versioned Editing-V3 support compiles and exactly replays complete programs
to all three released strict 5HT1B seed-2 endpoints. Complete-program support is
3/3, exact execution precision is 3/3, and the programs contain 23, 21, and 13
primitive edits. Each program uses exactly one public
`atom_protonation_restate` action and remains within the unchanged 32-primitive
runtime limit. No residual support blocker was observed.

The new action is deliberately narrow. It atomically and reversibly maps a real
nitrogen with exactly three heavy-atom single bonds between the named states
`(+1, implicit H=1)` and `(0, implicit H=0)`. Persistent slot, element, the
complete heavy-atom bond matrix, and all non-addressed atom fields remain fixed.
The historical V4 codec and Editing-V2 runtime remain unchanged and reject this
action.

## Claim boundary

This is an answer-known, teacher-forced, zero-oracle support certificate. The
known endpoints were compiler targets only. They were not injected into an
autonomous proposal pool, and autonomous proposal recovery, docking utility,
route-policy probability, and FiberControl selection were not evaluated.

## Authoritative artifact

`result.json` payload SHA-256:
`78da8092e1789d615675adec2bc3d57913ae3bffa6a2e1d6ba2cdb04510ad1ca`.
The byte-level file SHA-256 reproduced identically across two runs:
`2394615ab013ab8a60d0af437d0d38cd2ec98398bfd1651cbe99b64b7d6168b7`.
The probe used implementation revision
`85f8197395dc93e9dfeff6b3e43ce6f75f7e66da` and made zero docking, oracle, and
Modal calls.
