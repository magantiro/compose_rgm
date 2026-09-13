# PMO IVG-oracle parity audit

## Outcome being tested

The completed PMO development waves used PyTDC 0.3.6 and RDKit 2024.03.5.
The official IVG repository instead installs PyTDC 1.1.15 and pins RDKit
2023.9.6. This audit rescored the exact already locked COMPOSE molecules under
the IVG-pinned core oracle versions before making arithmetic comparisons.

The generated object does not change. It remains a complete, replay-verified
COMPOSE molecular edit-program endpoint within the original 40-active-atom,
48-slot, neutral graph support. The primary output of this audit is the ordered
reward sequence for each task under the parity evaluator, followed by the
official 10,000-query top-ten area under the curve (AUC).

The central claim under test is narrow: do the previously locked, answer-known
development candidates exceed the two repository-reported IVG AUC values when
both are interpreted under IVG's reported core oracle environment? This is not
a held-out search, autonomous-discovery, or general PMO claim.

## Frozen inputs and accounting

The query lock is built from four completed, content-addressed result artifacts:

- Perindopril curriculum: 15 molecules.
- Five exact-target tasks: 80 molecules.
- QED, GSK3B and JNK3 panel refinement: 33 molecules.
- C7 isomer and Median 1 panel wave: 22 molecules.

The audit therefore contains exactly 150 authoritative task calls. It preserves
each task's molecule strings and order. It does not replace, regenerate,
deduplicate across tasks, or rank candidates using parity scores.

Before this contract was written, one C7 isomer was evaluated once as a
diagnostic and scored 1.0. That call is preserved in
`diagnostics/pmo_ivg_oracle_parity/precontract_probe_0001.json` and is not
silently reused. The authoritative lock scores it again. Total new
parity-environment calls are therefore 151, consisting of one disclosed probe
plus 150 locked calls. Automatic retries and replacement queries are forbidden.

## Environment gate

The runner requires PyTDC 1.1.15, RDKit 2023.09.6, NumPy 1.26.4,
scikit-learn 1.2.2, pandas 2.1.4, SciPy 1.15.0, seaborn 0.13.2,
Requests 2.32.4 and setuptools 75.6.0. It hashes the PyTDC distribution
metadata, `tdc/oracles.py`, and the molecular-oracle implementation. It also
downloads, then verifies, the exact `gsk3b_current.pkl` and
`jnk3_current.pkl` assets before the first authoritative score.

The first launch stopped at this gate with zero authoritative calls. Its
expected hashes belonged to legacy HN-GFN copies named `gsk3b.pkl` and
`jnk3.pkl`, not the distinct current assets selected by PyTDC 1.1.15 with
scikit-learn 1.2.2. The failure and invalidated first lock are preserved. The
repaired contract binds the downloaded current assets to Dataverse file IDs
6413412 and 6413420, their byte sizes and SHA-256 values, and uses
`query_lock_v2.json`. No molecule, order, comparator, metric or call ceiling
changed during this pre-score repair.

The v2 launch then stopped at the byte-size gate, also before any authoritative
query. The current-file hashes and Dataverse IDs were correct, but the v2
contract had copied the legacy byte sizes rather than reading the downloaded
files. That failure and lock remain immutable. The v3 repair changes only the
two measured byte counts and the lock name; it does not change a molecule or
score.

The official Dockerfile leaves some non-oracle packages and its base Python
runtime indirectly versioned. The contract therefore calls this an
IVG-pinned core-oracle parity audit, not a byte-identical reproduction of the
entire IVG training container.

## Commands and acceptance

Prepare from clean committed source:

```bash
.venv/bin/python tools/pmo_ivg_oracle_parity.py prepare
```

Run in the pinned environment:

```bash
UV_CACHE_DIR=.uv-cache uv run --isolated --no-project --python 3.11 \
  --with-editable . \
  --with 'rdkit==2023.9.6' \
  --with 'PyTDC==1.1.15' \
  --with 'numpy==1.26.4' \
  --with 'scikit-learn==1.2.2' \
  --with 'pandas==2.1.4' \
  --with 'scipy==1.15.0' \
  --with 'seaborn==0.13.2' \
  --with 'requests==2.32.4' \
  --with 'setuptools==75.6.0' \
  --with fuzzywuzzy \
  --with huggingface-hub \
  python tools/pmo_ivg_oracle_parity.py run
```

Acceptance requires:

1. The immutable `query_lock_v3.json` contains all 150 source observations across exactly 11
   tasks, with input hashes and original scores.
2. Every environment, PyTDC-source and learned-oracle asset identity passes
   before scoring.
3. Exactly 150 started and 150 completed authoritative receipts exist, with no
   retries or replacements.
4. Each result records the parity score, prior-protocol score, ordered curve,
   top-ten AUC, both IVG comparators and arithmetic margins.
5. Negative and null margins are retained. No candidate is removed because it
   scores poorly.
6. Focused tests, lint, formatting, diff checks and the required repository
   verification are reported exactly as run.

## Decision rule

Only parity AUC values may support arithmetic comparison with the published IVG
table. Prior COMPOSE AUC values remain useful within their original evaluator
but are not cross-system evidence. Even a positive parity margin remains
answer-known winner/panel-informed development evidence because public PMO
structures informed the program locks.
