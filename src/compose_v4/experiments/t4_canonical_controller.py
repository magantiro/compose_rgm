"""The canonical T4 controller: ONE frozen configuration across every target and seed.

This module holds the parts of the T4 experiment that must be IDENTICAL on every
cell, and nothing that could differ between them.  A cell supplies three inputs --
the source molecule, the receptor/oracle, and the similarity radius ``delta`` -- and
nothing else.  Everything that decides *how* the controller searches lives here as a
module constant, so "the same controller ran on all thirty cells" is a property of
the code rather than a claim about how three contracts happened to be written.

Structure
---------
``T4_RECEPTORS`` is the ONLY place a target name appears in this package's runtime.
It is a pure data table (receptor file name, docking box, receptor digest) keyed by
the name the benchmark itself uses.  ``assert_no_target_name_routing`` walks the AST
of every runtime module and FAILS if a target name reaches a branch condition
anywhere, so a ``if target == "braf"`` cannot be added later without a red test.
Routing on molecular STATE is unrestricted; routing on identity is refused.

The three arms
--------------
Exactly one thing differs between them, and it is declared as a vocabulary:

``LOCAL_EXPERTS``
    ``("shallow",)`` -- local executable programs only.  Arm A.

``COORDINATED_EXPERTS``
    ``("shallow", "anchored_replacement", "structured")`` -- the same local lane plus
    the two EXISTING coordinated structural program families, at a fixed shared
    allocation.  Arms B and C.

Arm C adds the generic adaptive support-expansion rule (``support_expansion`` in the
contract, ``compose_v4.experiments.t4_support_expansion``) and changes nothing else.
So A vs B isolates coordinated structural programs, and B vs C isolates adaptive use
of structural support.

Why the route-distilled expert is NOT here
------------------------------------------
Earlier T4 arms carried a fourth lane, ``route_complete_region``, driven by a
``RouteDistilledGoalExpert`` checkpoint.  Every such checkpoint in this repository is
a LEAVE-ONE-TARGET-OUT fit over the 77 locked T4 routes
(``split_audit.split == "leave_one_target_out"``).  Using one of them across the whole
panel is training on the benchmark's own answers for four targets of five; using a
different one per target IS target-name routing, which is the thing this experiment
exists to abolish.  There is no leave-ALL-targets-out checkpoint, because removing all
five targets removes the corpus.  The coordinated role is therefore filled by
``structured`` (``synthesize_progressive_program``), a task-independent progressive
module composer that reads nothing but the parent state.

This is a deliberate narrowing of the controller, and it is stated here rather than in
a claim boundary so that a reader of the runtime sees it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = "t4_canonical_shared_controller_v1"

# ---- The one place a target name appears -------------------------------------------
# Receptor file name, qvina02 search box [[center], [size]], and the sha256 of the
# receptor .pdbqt the production image downloads. Every value is transcribed from the
# frozen per-target contracts this panel supersedes; none of them is a controller
# setting, and none of them is read by anything that decides how to search.
T4_RECEPTORS = {
    "parp1": {
        "receptor": "parp1",
        "docking_box": [[26.413, 11.282, 27.238], [18.521, 17.479, 19.995]],
        "receptor_sha256": "8d0891ddf915f51cf3108f39dd9dc01dfcf86be342c566021956afdd79eb4ad9",
    },
    "fa7": {
        "receptor": "fa7",
        "docking_box": [[10.131, 41.879, 32.097], [20.673, 20.198, 21.362]],
        "receptor_sha256": "bfd705fb8220c52e1b447afce23a3810bc6deddb06983e98a2b0934b24ba092d",
    },
    "5ht1b": {
        "receptor": "5ht1b",
        "docking_box": [[-26.602, 5.277, 17.898], [22.5, 22.5, 22.5]],
        "receptor_sha256": "6ab5ac2d63be05c4d7992f96175258c2b253ccc41539833af4fb12465ee3a59f",
    },
    "braf": {
        "receptor": "braf",
        "docking_box": [[84.194, 6.949, -7.081], [22.032, 19.211, 14.106]],
        "receptor_sha256": "707b21bfb654321cb14d33ea07e9accab4cb9a884378285cd562c2324a31213a",
    },
    "jak2": {
        "receptor": "jak2",
        "docking_box": [[114.758, 65.496, 11.345], [19.033, 17.929, 20.283]],
        "receptor_sha256": "cb900f3f75dcf2ac3efe6538528ed1e87ec934a527b3b060c2b09dfe3a11d9b6",
    },
}

QVINA02_SHA256 = "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0"

# ---- Arms -------------------------------------------------------------------------
LOCAL_EXPERTS = ("shallow",)
COORDINATED_EXPERTS = ("shallow", "anchored_replacement", "structured")

ARMS = {
    "A": {
        "arm": "A",
        "arm_name": "local",
        "experts": LOCAL_EXPERTS,
        "adaptive_support_expansion": False,
        "deltas": (0.6,),
        "isolates": "the local executable program lane on its own",
    },
    "B": {
        "arm": "B",
        "arm_name": "coordinated",
        "experts": COORDINATED_EXPERTS,
        "adaptive_support_expansion": False,
        "deltas": (0.6,),
        "isolates": "A plus the existing coordinated structural programs at a fixed shared allocation",
    },
    "C": {
        "arm": "C",
        "arm_name": "adaptive",
        "experts": COORDINATED_EXPERTS,
        "adaptive_support_expansion": True,
        "deltas": (0.6, 0.4),
        "isolates": "B plus the frozen generic adaptive support-expansion rule",
    },
}

# ---- The frozen shared controller configuration ------------------------------------
# Every field below is the SAME on every cell of every arm. `experts` and
# `support_expansion` are the only per-arm fields and they live in ARMS.
SHARED_CONTROLLER = {
    "support": "compose_valid",
    "charged_calls_per_cell": 250,
    "batch": 8,
    "parents": 4,
    "parent_explore": 0.3,
    "exploration": 2,
    "expert_floor_rounds": 2,
    "value_penalty": 1.0,
    "docking_seed": 20260922,
    "proposal": {
        "shallow": {
            "draws": 480,
            "horizon": 3,
            "region_law": "free_gate_margin_v1",
            "completion_law": "free_gate_margin_v1",
        },
        "anchored_replacement": {"draws": 512, "horizon": 3},
        "structured": {"draws": 480, "horizon": 3},
    },
}

# The adaptive rule, arm C only. Every field is a BOUND; the trigger reads only
# generic search statistics (this round's distinct eligible-candidate count) and
# never a cell identity.
ADAPTIVE_SUPPORT_EXPANSION = {
    "draw_ladder": [960, 1920, 3840],
    "lanes": ["shallow", "anchored_replacement"],
    "zero_support_fallback": True,
    "stop_at_distinct_eligible": 4,
    "max_extra_draws_per_event": 6720,
    "wall_seconds": 5400.0,
}

#: Arm C expands whenever a round's own eligible yield is below this many distinct
#: endpoints. It is a fixed constant applied identically on every cell. At 1 it is
#: exactly the shipped "the pool came back empty" condition; at 4 it also covers the
#: marginal rounds that survive on one unlucky draw (a control cell has been measured
#: surviving round one on 3 eligible endpoints out of 7,420 produced).
EXPANSION_TRIGGER_MIN_ELIGIBLE = 4

CONTROLLER_SEED_BASE = 2026092200


def controller_seed(source_global_index: int, delta: float) -> int:
    """The per-cell RNG seed: a pure function of the two benchmark INPUTS.

    It does not depend on the arm, so A, B and C start every cell from the same
    stream, and it does not depend on the target name, so no cell can be given a
    luckier seed than its siblings.
    """

    return int(CONTROLLER_SEED_BASE + 100 * int(source_global_index) + round(delta * 10))


def canonical_cells(seeds: Sequence[dict], deltas: Iterable[float]) -> list[dict]:
    """Every (seed, delta) pair as a cell record, in a fixed deterministic order.

    The cell label follows the panel convention this table has to be read beside:
    ``<target>_<index within that target>_d<10*delta>``, e.g. ``fa7_0_d06``. The
    within-target index is the seed's RANK inside its own target group, computed
    from the registry rather than assumed to be ``global_index % 3``.
    """

    ranks: dict[str, int] = {}
    ordered = []
    for row in sorted(seeds, key=lambda item: int(item["idx"])):
        target = str(row["target"])
        rank = ranks.get(target, 0)
        ranks[target] = rank + 1
        ordered.append((row, target, rank))

    cells: list[dict] = []
    for delta in deltas:
        for row, target, rank in ordered:
            receptor = T4_RECEPTORS[target]
            index = int(row["idx"])
            cells.append(
                {
                    "cell": f"{target}_{rank}_d{round(float(delta) * 10):02d}",
                    "target": target,
                    "target_seed_index": rank,
                    "source_global_index": index,
                    "smiles": str(row["smiles"]),
                    "delta": float(delta),
                    "controller_seed": controller_seed(index, float(delta)),
                    "receptor": receptor["receptor"],
                    "docking_box": receptor["docking_box"],
                    "receptor_sha256": receptor["receptor_sha256"],
                }
            )
    return cells


def load_seeds(path: Path) -> list[dict]:
    """The 15 published T4 held-target seeds, ordered by their global index."""

    rows = json.loads(Path(path).read_text())
    return sorted(rows, key=lambda row: int(row["idx"]))


def jsonable(value):
    """Round-lock-safe serialization for the values a round produces.

    It lives here rather than in the Modal app so that a test can exercise the SAME
    function the app writes locks with, without importing `modal`.
    """

    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def controller_identity(contract: dict) -> str:
    """A digest over the SHARED controller configuration of one contract.

    It deliberately EXCLUDES the cell list, the arm label and the per-arm expert
    vocabulary, so the three arms must agree on it exactly. Anything that would make
    one arm search differently for a reason other than its declared vocabulary --
    a different batch size, a different draw count, a different law, a different
    docking seed -- moves this value and the preflight fails.
    """

    from compose_v4.control.docking_value import identity

    shared = {key: contract[key] for key in sorted(SHARED_CONTROLLER) if key in contract}
    return identity({"schema_version": SCHEMA_VERSION, "shared_controller": shared})


def cell_identity_digest(cells: Sequence[dict]) -> str:
    """A digest over the benchmark INPUTS of a cell list, ignoring order-free noise."""

    from compose_v4.control.docking_value import identity

    return identity(
        [
            {
                "cell": row["cell"],
                "smiles": row["smiles"],
                "delta": row["delta"],
                "controller_seed": row["controller_seed"],
                "receptor": row["receptor"],
            }
            for row in cells
        ]
    )


# ---- The structural guarantee: no target-name routing -------------------------------

#: Every token that identifies a T4 target or one of its cells. Matching is done on
#: the lowercased string constant, so "BRAF", "braf_0" and "Braf" are all caught.
TARGET_TOKENS = tuple(sorted(T4_RECEPTORS))


class TargetNameRoutingError(AssertionError):
    """A runtime module branched on a target identity."""


def _is_docstring(node: ast.AST, parents: dict) -> bool:
    parent = parents.get(id(node))
    return isinstance(parent, ast.Expr) and isinstance(
        parents.get(id(parent)), (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    )


def target_name_routing_findings(
    source: str, *, filename: str, allow_data_assignment_to: Sequence[str] = ()
) -> list[dict]:
    """Every place a target name reaches a BRANCH CONDITION in one module.

    A target name is allowed to appear as data (a table key, a cell label built by
    `canonical_cells`, a docstring). It is NOT allowed inside the test of an `if`,
    an `ifexp`, a `while`, a comparison, a boolean operator, an `assert` test or a
    `match` case -- those are the constructs through which a controller could learn
    which protein it is looking at.
    """

    tree = ast.parse(source, filename=filename)
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    allowed_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = []
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
            if any(name in allow_data_assignment_to for name in names):
                for inner in ast.walk(node):
                    allowed_nodes.add(id(inner))

    def _in_condition(node: ast.AST) -> str | None:
        current = node
        while True:
            parent = parents.get(id(current))
            if parent is None:
                return None
            if isinstance(parent, (ast.If, ast.While)) and parent.test is current:
                return type(parent).__name__
            if isinstance(parent, ast.IfExp) and parent.test is current:
                return "IfExp"
            if isinstance(parent, ast.Assert) and parent.test is current:
                return "Assert"
            if isinstance(parent, (ast.Compare, ast.BoolOp)):
                return type(parent).__name__
            if isinstance(parent, ast.comprehension) and current in parent.ifs:
                return "comprehension"
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                return None
            current = parent

    findings: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in allowed_nodes or _is_docstring(node, parents):
            continue
        lowered = node.value.lower()
        hit = [token for token in TARGET_TOKENS if token in lowered]
        if not hit:
            continue
        where = _in_condition(node)
        if where is None:
            continue
        findings.append(
            {"file": filename, "line": node.lineno, "value": node.value,
             "tokens": hit, "construct": where}
        )
    return findings


def assert_no_target_name_routing(
    paths: Iterable[Path], *, allow_data_assignment_to: Sequence[str] = ("T4_RECEPTORS",)
) -> list[dict]:
    """Fail if any runtime module branches on a target identity. Returns the scan."""

    scanned = []
    findings: list[dict] = []
    for path in paths:
        path = Path(path)
        source = path.read_text()
        scanned.append(str(path))
        findings.extend(
            target_name_routing_findings(
                source, filename=str(path), allow_data_assignment_to=allow_data_assignment_to
            )
        )
    if findings:
        raise TargetNameRoutingError(
            "target-name routing found in the canonical runtime: "
            + json.dumps(findings, sort_keys=True)
        )
    return scanned


def should_expand_support(
    *, distinct_eligible: int, minimum: int = EXPANSION_TRIGGER_MIN_ELIGIBLE
) -> bool:
    """The adaptive trigger: a generic search statistic, identical on every cell.

    `distinct_eligible` is the number of distinct eligible endpoints this round's
    ordinary proposal produced. Nothing about the cell, the target, the round index
    or the incumbent score enters.
    """

    return int(distinct_eligible) < int(minimum)
