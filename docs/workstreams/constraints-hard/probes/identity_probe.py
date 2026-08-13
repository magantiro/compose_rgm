"""Stage-0 executor semantics probe: does atom identity survive the pipeline?

Answers the question `docs/workstreams/constraints-hard/CONSTRAINT_SEMANTICS.md`
calls critical: may we claim IDENTITY_INVARIANT, or only
LABELED_SUBGRAPH_PRESENCE_INVARIANT?

Pure local. CPU only. No model checkpoint, no Gate-0 decision, no network.
Runs in under two seconds.

    python3 docs/workstreams/constraints-hard/probes/identity_probe.py

Repo root is derived from this file's location; override with COMPOSE_REPO_ROOT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(
    os.environ.get(
        "COMPOSE_REPO_ROOT",
        Path(__file__).resolve().parents[4],
    )
)
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from compose_v4.chem.molecular_graph import (  # noqa: E402
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.rewrite import operators as ops  # noqa: E402
from compose_v4.rewrite.kernel import (  # noqa: E402
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)

FAILURES: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: got {actual!r}, expected {expected!r}")


def occupied(mg) -> tuple[int, ...]:
    """Slot addresses currently holding a real chemical element."""
    return tuple(int(i) for i in np.flatnonzero(is_element(mg.atom_types)))


def probe_1_operators_preserve_slots() -> None:
    print("=" * 78)
    print("PROBE 1  micro-operators are slot-preserving within one rewrite")
    print("=" * 78)
    mg = pad_molecular_graph(smiles_to_molecular_graph("CCCO"), 12)
    print(f"  source            {canonical_state_key(mg)}  occupied={occupied(mg)}")
    succ = ops.apply_atom_delete(mg, ops.AtomDelete(v=3))
    print(f"  after delete v=3  {canonical_state_key(succ)}  occupied={occupied(succ)}")
    check(
        "surviving atoms keep their original slot addresses",
        occupied(succ),
        tuple(i for i in occupied(mg) if i != 3),
    )
    print("  => identity survives the OPERATOR. Probes 2-3 show it does not")
    print("     survive the PIPELINE.")


def probe_2_alias_collapse_merges_disjoint_atom_sets() -> None:
    print()
    print("=" * 78)
    print("PROBE 2  alias collapse merges successors that retain DIFFERENT atoms")
    print("=" * 78)
    prop = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 8)
    print(f"  source            {canonical_state_key(prop)}  occupied={occupied(prop)}")
    a = ops.apply_atom_delete(prop, ops.AtomDelete(v=0))
    b = ops.apply_atom_delete(prop, ops.AtomDelete(v=2))
    ka, kb = canonical_state_key(a), canonical_state_key(b)
    print(f"  delete slot 0     key={ka!r}  retained slots={occupied(a)}")
    print(f"  delete slot 2     key={kb!r}  retained slots={occupied(b)}")
    check("the two successors share one canonical key", ka == kb, True)
    check("...but retain different atom sets", occupied(a) == occupied(b), False)
    print("  => production_successor_kernel.py:573 `setdefault(key, successor)`")
    print("     keeps whichever mark was enumerated first. A controller that")
    print("     protects 'the atom at slot 0' cannot distinguish these, because")
    print("     the object it sees is the key.")


def probe_3_key_roundtrip_renumbers() -> None:
    print()
    print("=" * 78)
    print("PROBE 3  re-parsing a state from its own canonical key RENUMBERS it")
    print("=" * 78)
    src = pad_molecular_graph(smiles_to_molecular_graph("OCCN"), 10)
    key = canonical_state_key(src)
    rt = pad_molecular_graph(smiles_to_molecular_graph(key), 10)
    print(f"  built from 'OCCN'   atom_types={src.atom_types.tolist()}  key={key!r}")
    print(f"  re-parsed from key  atom_types={rt.atom_types.tolist()}  key={key!r}")
    check(
        "slot labelling survives one round-trip through the canonical key",
        bool(np.array_equal(src.atom_types, rt.atom_types)),
        False,
    )
    check(
        "the molecule itself is unchanged (same key)",
        canonical_state_key(rt) == key,
        True,
    )
    print("  => the production controllers carry the trajectory as a list[str] of")
    print("     canonical SMILES and re-parse at every step")
    print("     (retarget_intervention_app.py:243-256, 276-282;")
    print("      editing_v2_bridge_control.py:99-101), so this renumbering is on")
    print("     the live path, not a hypothetical.")


def probe_4_constraints_hook_raises() -> None:
    print()
    print("=" * 78)
    print("PROBE 4  RewriteSystem(constraints=) cannot be used to MASK the fiber")
    print("=" * 78)

    def forbid_everything(_state, _action, _successor) -> bool:
        return False

    prop = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 8)
    system = editing_v2_semantic_rewrite_system(constraints=(forbid_everything,))
    raised: str | None = None
    try:
        system.apply(prop, "atom_delete", ops.AtomDelete(v=0))
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
        print(f"  RewriteSystem.apply raised {raised}: {exc}")
    check("a failing constraint RAISES rather than filtering", raised, "InvalidRewrite")
    print("  => canonical_successor_result:557-567 converts ANY exception from")
    print("     runtime.apply into a fatal ProductionSuccessorKernelError. A")
    print("     protected-core constraint installed there would crash the kernel")
    print("     on the first core-violating mark, not filter it. The mask must")
    print("     filter the enumerated successor list and renormalize instead.")


def main() -> int:
    probe_1_operators_preserve_slots()
    probe_2_alias_collapse_merges_disjoint_atom_sets()
    probe_3_key_roundtrip_renumbers()
    probe_4_constraints_hook_raises()
    print()
    print("=" * 78)
    if FAILURES:
        print("PROBE SUITE FAILED — the Stage-0 verdict may no longer hold:")
        for line in FAILURES:
            print("  -", line)
        return 1
    print("VERDICT: LABELED_SUBGRAPH_PRESENCE_INVARIANT")
    print("         IDENTITY_INVARIANT is NOT provable and must not be claimed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
