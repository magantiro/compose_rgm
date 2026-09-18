"""Read a paired campaign and report what it actually shows.

The headline of a search run is normally best-so-far, which is one number per arm and
therefore n=1: with single-call docking reproducibility measured at about half a unit on
this target (the benchmark root returned -8.00 and -8.50 under identical requests), a
best-so-far gap under that is not an ordering.

The lockstep design buys something much stronger for free. Both arms choose from the
SAME pool in the same round, so each round is a MATCHED PAIR: two batches of eight drawn
from one candidate list, differing only in whether reward information was used to pick
them. Comparing the arms' realized batch quality round by round gives one observation per
round rather than one per campaign, and a sign test over those rounds is a real test of
the selection mechanism even when the best-so-far curves are close.

Three diagnostics separate the three failure modes the plan names:

  generation   -- was anything better than the incumbent present in the pool at all?
                  Bounded below by what was docked, which is a sample of the pool.
  reward model -- where did the arm's picks land in the realized ranking of that round's
                  docked molecules? A model that ranks well puts its picks near the top.
  acquisition  -- did the arm's own predicted endpoint score correlate with what came back?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _finite(values):
    return [v for v in values if v is not None]


def _sign_test(differences, *, favourable) -> dict:
    """Two-sided sign test over matched rounds. Exact, no distributional assumption."""
    from math import comb

    wins = sum(1 for d in differences if favourable(d))
    losses = sum(1 for d in differences if favourable(-d))
    n = wins + losses
    if n == 0:
        return {"rounds": 0, "wins": 0, "losses": 0, "ties": len(differences), "p": None}
    extreme = max(wins, losses)
    tail = sum(comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return {
        "rounds": n, "wins": wins, "losses": losses,
        "ties": len(differences) - n, "p": min(1.0, 2 * tail),
    }


def report(payload: dict) -> str:
    options, rounds = payload["options"], payload["rounds"]
    names = ["adaptive", "blind"]
    lines = [
        "# Adaptive versus reward-blind selection on identical pools",
        "",
        (
            f"`{options['target']}` delta={options['delta']}, {options['budget']} charged "
            f"calls per arm, batch {options['batch']}, {options['draws']} raw programs per "
            f"parent, program depth {options['horizon']}, seed {options['seed']}."
        ),
        f"Shared root `{options['root']}` at {payload['root_score']:.2f}.",
        "",
        "## Best so far against charged calls",
        "",
        "| calls | pool | multi | adaptive | blind |",
        "| ----: | ---: | ----: | -------: | ----: |",
    ]
    for entry in rounds:
        arms = entry["arms"]
        calls = arms.get("adaptive", {}).get("calls") or arms.get("blind", {}).get("calls")
        cells = []
        for name in names:
            arm = arms.get(name, {})
            best = arm.get("best")
            mark = "*" if arm.get("improved") else ""
            cells.append(f"{best:.2f}{mark}" if best is not None else "-")
        lines.append(
            f"| {calls} | {entry['pool']} | {entry['multi_region_in_pool']} | "
            + " | ".join(cells) + " |"
        )

    # ---- the matched-pair statistic ----
    per_round = {name: {"best": [], "mean": []} for name in names}
    for entry in rounds:
        for name in names:
            scored = _finite(d["score"] for d in entry["arms"].get(name, {}).get("docked", []))
            per_round[name]["best"].append(min(scored) if scored else None)
            per_round[name]["mean"].append(sum(scored) / len(scored) if scored else None)
    paired = [
        (a, b)
        for a, b in zip(per_round["adaptive"]["best"], per_round["blind"]["best"])
        if a is not None and b is not None
    ]
    lines += [
        "",
        "## Matched rounds: both arms picked eight from the same list",
        "",
        "| round | adaptive batch best | blind batch best | adaptive - blind |",
        "| ----: | ------------------: | ---------------: | ---------------: |",
    ]
    for index, (a, b) in enumerate(paired, start=1):
        lines.append(f"| {index} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
    if paired:
        differences = [a - b for a, b in paired]
        test = _sign_test(differences, favourable=lambda d: d < 0)
        mean = sum(differences) / len(differences)
        lines += [
            "",
            (
                f"Mean paired difference **{mean:+.3f}** (negative favours adaptive, "
                f"because a lower docking score is better)."
            ),
            f"Sign test over {test['rounds']} decided rounds: adaptive better in "
            f"**{test['wins']}**, blind better in {test['losses']}, ties {test['ties']}, "
            f"two-sided p = {test['p']:.4f}" if test["p"] is not None else "No decided rounds.",
        ]

    # ---- the three failure-mode diagnostics ----
    lines += ["", "## Which stage is limiting", "", "| round | pool best observed | adaptive pick rank | blind pick rank |", "| ----: | -----------------: | -----------------: | --------------: |"]
    for index, entry in enumerate(rounds, start=1):
        everything = []
        for name in names:
            everything += [
                d for d in entry["arms"].get(name, {}).get("docked", []) if d["score"] is not None
            ]
        if not everything:
            continue
        ordering = sorted({d["smiles"]: d["score"] for d in everything}.items(), key=lambda kv: kv[1])
        rank_of = {smiles: position for position, (smiles, _) in enumerate(ordering, start=1)}
        cells = []
        for name in names:
            picks = [
                rank_of[d["smiles"]]
                for d in entry["arms"].get(name, {}).get("docked", [])
                if d["score"] is not None
            ]
            cells.append(f"{min(picks)} of {len(ordering)}" if picks else "-")
        lines.append(f"| {index} | {ordering[0][1]:.2f} | " + " | ".join(cells) + " |")

    lines += ["", "## Calibration of the adaptive arm's own predictions", ""]
    predicted, realized = [], []
    for entry in rounds:
        for record in entry["arms"].get("adaptive", {}).get("docked", []):
            if record["score"] is not None and record.get("predicted_endpoint") is not None:
                predicted.append(record["predicted_endpoint"])
                realized.append(record["score"])
    if len(predicted) >= 8:
        import numpy as np

        rho = float(np.corrcoef(predicted, realized)[0, 1])
        lines.append(
            f"Over {len(predicted)} counted calls, corr(predicted endpoint, realized) = "
            f"**{rho:+.3f}**. Positive means the model orders candidates in the right "
            f"direction; near zero means selection is effectively blind whatever the rule says."
        )
    else:
        lines.append("Too few counted calls to say anything about calibration.")

    # ---- attribution ----
    lines += ["", "## Where improvements came from", ""]
    for name in names:
        wins = [
            entry["arms"][name]
            for entry in rounds
            if entry["arms"].get(name, {}).get("improved")
        ]
        single = sum(1 for w in wins if w.get("improving_regions") == 1)
        multi = sum(1 for w in wins if (w.get("improving_regions") or 0) > 1)
        families: dict[str, int] = {}
        for win in wins:
            for family in win.get("improving_families") or []:
                families[family] = families.get(family, 0) + 1
        ordered = ", ".join(f"{k} {v}" for k, v in sorted(families.items(), key=lambda kv: -kv[1]))
        lines.append(
            f"- **{name}**: {len(wins)} improving rounds, {single} single-region, "
            f"{multi} multi-region{('; families ' + ordered) if ordered else ''}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="diagnostics/t4_fiber_control/paired.json")
    parser.add_argument("--out", default="diagnostics/t4_fiber_control/PAIRED.md")
    options = parser.parse_args()
    payload = json.loads(Path(options.result).read_text())
    text = report(payload)
    Path(options.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
