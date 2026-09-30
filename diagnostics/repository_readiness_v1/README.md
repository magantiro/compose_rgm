# Repository readiness checks

These records concern local software verification, not new benchmark outcomes.
The task and acceptance criteria are in
`docs/REPOSITORY_READINESS_20260929.md`.

The baseline is this checkout under the documented Python 3.11/RDKit 2024.3.5
core environment. Terminal logs and JUnit XML retain actual failures and exit
status. The other agent's partial suite on another branch is context only.

The final clean-clone suite record is
`clean_clone_verification_20260929.json`. It identifies commit `4ca47afa`,
the environment and command, the 5,357/18/1 pass/skip/expected-failure result,
and SHA-256 hashes for the local log, JUnit XML, and captured exit-code file.
The large logs are ignored by Git; this compact record is tracked. The
read-only Process-V2 hash-chain check returned status `AGREES`. A passing core
suite does not supply the external experiment assets or settle paper-row
provenance described in the readiness assessment.

The initial collection check reproduced a Python-version-dependent test defect:
the protocol test accessed `typing.Protocol.__protocol_attrs__`, an internal
attribute absent in Python 3.11. The repair checks the declared public members
and retains the per-member refusal tests.
