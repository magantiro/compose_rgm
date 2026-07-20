"""Inspect one detached Modal function call without launching remote work."""

from __future__ import annotations

import json
from dataclasses import asdict

import modal


app = modal.App("compose-v4-inspect-function-call")


@app.local_entrypoint()
def main(function_call_id: str, timeout_seconds: float = 5.0) -> None:
    call = modal.FunctionCall.from_id(function_call_id)
    try:
        result = call.get(timeout=timeout_seconds)
    except TimeoutError:
        print(
            json.dumps(
                {
                    "function_call_id": function_call_id,
                    "phase": "running_or_pending",
                    "timeout_seconds": timeout_seconds,
                },
                sort_keys=True,
            )
        )
        return
    except Exception as error:
        print(
            json.dumps(
                {
                    "dashboard_url": call.get_dashboard_url(),
                    "error": {
                        "cause": repr(error.__cause__),
                        "context": repr(error.__context__),
                        "message": str(error),
                        "repr": repr(error),
                        "type": type(error).__name__,
                    },
                    "function_call_id": function_call_id,
                    "phase": "failed",
                    "call_graph": [
                        {
                            **asdict(item),
                            "status": str(item.status),
                        }
                        for item in call.get_call_graph()
                    ],
                },
                default=str,
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            {
                "function_call_id": function_call_id,
                "phase": "complete",
                "result": result,
            },
            default=str,
            sort_keys=True,
        )
    )
