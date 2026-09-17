"""Can a COMPOSE program be held fixed while exactly one decision is resampled?

Everything proposed for a contrastive controller rests on this and nothing else. A
matched bundle is only matched if `P` and `P^(j<-d'_j)` differ in one meaningful
structural decision and agree everywhere else; if the compiler cannot hold the rest
fixed, the bundle degrades into four loosely related programs and the contrast that was
supposed to identify "six-membered beat five-membered" identifies nothing.

The substrate already exists and is not rebuilt here. `encode_patch_stream` turns a
complete structural patch into a canonical typed token sequence, `token_domain` returns
the frozen legal value fiber for each token given its prefix, and `decode_patch_stream`
reads a stream back into a patch. An intervention is therefore literally: encode,
replace one token's value with an alternative from its own domain, decode.

Each token already carries a `factor` naming its semantic coordinate -- `control`,
`atom_attributes`, `target_topology`, `attachments`, `bond_attributes` -- so the
coordinate system is inherited rather than invented. Binding is separate again:
`attachment_bindings` enumerates where a patch may attach without changing what the
patch is, which is an independent coordinate by construction.

What this measures, per coordinate:

- OFFERED -- the token has more than one legal value, so an intervention exists at all.
- DECODES -- substituting it yields a readable patch rather than a malformed stream.
- DISTINCT -- the decoded patch actually differs from the original.

A coordinate that is offered but does not decode is COUPLED: it cannot move alone and
belongs in an intervention block with whatever it drags along. `output_count` is the
expected case, since the number of output atoms determines how many later tokens the
stream contains. Coupling is reported, never forced into a fake factorial design.

Zero oracle calls.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from compose_v4.control.complete_region_patch_policy import (
    PatchToken,
    SourceRegionContext,
    decode_patch_stream,
    encode_patch_stream,
    token_domain,
)

SCHEMA_VERSION = "t4_program_intervention_v1"


def intervene(tokens, index: int, value: int) -> tuple[PatchToken, ...]:
    """The stream with exactly one token's value replaced."""
    token = tokens[index]
    return (*tokens[:index], PatchToken(token.kind, value, token.factor), *tokens[index + 1 :])


def alternatives(context: SourceRegionContext, tokens, index: int) -> tuple[int, ...]:
    """Legal values this one decision could have taken, from the frozen fiber."""
    token = tokens[index]
    domain = token_domain(context, tokens[:index], kind=token.kind, factor=token.factor)
    return tuple(value for value in domain if value != token.value)


def probe_subgoal(subgoal) -> dict:
    """Try every single-token intervention on one complete patch."""
    context = SourceRegionContext.from_subgoal(subgoal)
    tokens = encode_patch_stream(subgoal)
    per_factor = defaultdict(
        lambda: {"tokens": 0, "offered": 0, "decodes": 0, "distinct": 0, "alternatives": 0}
    )
    coupled = Counter()

    for index, token in enumerate(tokens):
        record = per_factor[token.factor]
        record["tokens"] += 1
        options = alternatives(context, tokens, index)
        if not options:
            continue
        record["offered"] += 1
        record["alternatives"] += len(options)
        decoded_any = distinct_any = False
        for value in options:
            try:
                candidate = decode_patch_stream(context, intervene(tokens, index, value))
            except (ValueError, IndexError, KeyError):
                continue
            decoded_any = True
            if candidate != subgoal:
                distinct_any = True
        record["decodes"] += int(decoded_any)
        record["distinct"] += int(distinct_any)
        if not decoded_any:
            coupled[f"{token.factor}:{token.kind}"] += 1

    return {
        "stream_length": len(tokens),
        "per_factor": {k: dict(v) for k, v in per_factor.items()},
        "coupled_tokens": dict(coupled),
    }


def aggregate(reports) -> dict:
    """Pool per-patch probes into a per-coordinate verdict."""
    totals = defaultdict(
        lambda: {"tokens": 0, "offered": 0, "decodes": 0, "distinct": 0, "alternatives": 0}
    )
    coupled = Counter()
    for report in reports:
        for factor, record in report["per_factor"].items():
            for key, value in record.items():
                totals[factor][key] += value
        coupled.update(report["coupled_tokens"])

    verdict = {}
    for factor, record in sorted(totals.items()):
        offered = record["offered"]
        verdict[factor] = {
            **record,
            "independent_rate": record["distinct"] / offered if offered else 0.0,
            "mean_alternatives_per_decision": (
                record["alternatives"] / offered if offered else 0.0
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "patches": len(reports),
        "coordinates": verdict,
        "coupled_token_kinds": dict(coupled.most_common()),
        "new_oracle_calls": 0,
    }
