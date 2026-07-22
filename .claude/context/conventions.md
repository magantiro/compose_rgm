# Conventions

How code here is written and checked. "Match the existing style of the file you're
editing" overrides anything below when they conflict. Adopted from `KoshaTx/compose`.

## Tooling
- **Python 3.10+.** Lint `ruff check .` (line-length 100, py310 — in `pyproject.toml`).
- Tests `pytest tests/` — `pythonpath=["src"]`, so imports are `from compose_v4.…`. No
  `conftest.py`; tests use local underscore helpers, not fixtures. Experiment scripts also
  need `scripts` on the path: `PYTHONPATH=src:scripts`.
- Install `uv sync` (or `pip install -e ".[dev]"`); run in the `.venv`.
- macOS: export `KMP_DUPLICATE_LIB_OK=TRUE`; SMC runs single-process (fork deadlock — see learnings).

## Style
- `from __future__ import annotations` at module top.
- Imports grouped stdlib → third-party (numpy, rdkit, torch, networkx, scipy) → local (`from compose_v4.*`).
- `snake_case` funcs/vars, `PascalCase` classes, `UPPER_CASE` constants.
- Rich module-level docstrings stating the data structure + invariants maintained/tested.
  `# ---- Section ----` markers; inline `#` comments. No `__all__`.
- `src/` is the core model and is kept ruff-clean; keep it that way. Scripts are drivers.

## Workflow
- Feature branches → PR to `main`; both gates (`pytest tests/` + `scripts/sanity_check.py`) green before merge.
- **Commits atomic**: one self-contained change; subject `<module>: <one-line>`, imperative,
  lowercase, no trailing period. **Never mention Claude / AI** (no `Co-Authored-By` / "generated
  with"), in commits or PR bodies.
- `.claude/` is code: edit context files in the same PR as the change that makes them true.
