# COMPOSE Parallel-Agent Handoff Template

Copy this file to:

```text
docs/workstreams/<workstream>/HANDOFF.md
```

and complete every section. Do not delete sections; write `NONE` when appropriate.

---

# Workstream

- **Name:**
- **Claim ID:**
- **Branch:**
- **Base commit:**
- **HEAD commit:**
- **Working tree clean:** yes/no
- **Status:** `DESIGN_ONLY | SMOKE_HELD_IN | DEVELOPMENT | CONFIRMATORY_HELD_OUT | SUPERSEDED | INVALID_INSTRUMENT`
- **Held-out data opened:** yes/no

# One-sentence scientific question

> 

# Claim this work can support

> 

# Claims this work cannot support

- 

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | | | |
| `R_theta` checkpoint | | | |
| split / panel | | | |
| sampling law | | | |
| goal/oracle | | | |
| other | | | |

# Protocol

- **Panel construction:**
- **Arms:**
- **Primary metric:**
- **Secondary metrics:**
- **Independent statistical unit:**
- **Allowed calibration:**
- **Stop rules:**
- **Forbidden adaptations:**

# What was implemented

- 

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| | | |

# Results

State clearly whether these are smoke, development, or confirmatory.

| Metric | Arm / condition | Value | Uncertainty / denominator |
|---|---|---:|---|
| | | | |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| | | |

# Bugs, invalid instruments, and superseded runs

For every invalid run:

- what was wrong;
- how it was detected;
- whether any conclusion depended on it;
- artifact status (`INVALID_INSTRUMENT` or `SUPERSEDED`);
- corrective commit.

# Known limitations

- 

# Exact reproduction commands

```bash
# environment/setup

# smoke

# analysis
```

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| | | | |

# Files changed

```text
path/to/file
path/to/other
```

# Recommended next action

One bounded action only:

> 

# Actions explicitly not recommended

- 

# Main-session pickup checklist

- [ ] Read protocol before results.
- [ ] Verify all frozen-input hashes.
- [ ] Confirm held-out-open status.
- [ ] Reproduce one smoke.
- [ ] Inspect known-invalid runs.
- [ ] Decide explicitly whether to merge, authorize held-out evaluation, or stop.

