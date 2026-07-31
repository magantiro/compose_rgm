from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "build_active8_trace_inventory_app.py"


def test_local_entrypoint_waits_for_driver_completion() -> None:
    """A detached driver is cancelled when `modal run` tears down its app."""

    tree = ast.parse(APP_PATH.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert any(
        isinstance(call.func.value, ast.Name)
        and call.func.value.id == "driver"
        and call.func.attr == "remote"
        for call in calls
    )
    assert not any(
        isinstance(call.func.value, ast.Name)
        and call.func.value.id == "driver"
        and call.func.attr == "spawn"
        for call in calls
    )
