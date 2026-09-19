from dataclasses import dataclass

from tools.audit_t4_parp_route_support import stage_positions


@dataclass(frozen=True)
class _Template:
    template_id: str


@dataclass(frozen=True)
class _Endpoint:
    key: str


@dataclass(frozen=True)
class _Proposal:
    endpoint: _Endpoint
    templates: tuple[_Template, ...]


def test_stage_positions_requires_both_exact_endpoint_and_exact_program(monkeypatch):
    monkeypatch.setattr(
        "tools.audit_t4_parp_route_support.canonical_state_key",
        lambda endpoint: endpoint.key,
    )
    rows = [
        _Proposal(_Endpoint("winner"), (_Template("other"),)),
        _Proposal(_Endpoint("other"), (_Template("teacher"),)),
        _Proposal(_Endpoint("winner"), (_Template("teacher"),)),
        _Proposal(_Endpoint("winner"), (_Template("teacher"), _Template("extra"))),
    ]
    assert stage_positions(rows, "teacher", "winner") == [3]
