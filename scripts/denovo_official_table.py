"""Render the official de-novo generation table from per-seed report JSONs.

Reads the ``seed_reports/<horizon>/seed<N>.json`` files written by
``modal_apps/denovo_official_eval.py::score_seed`` and prints:

  1. the per-seed rows, so a single seed is never mistaken for the aggregate;
  2. the 3-seed mean +- sample standard deviation;
  3. the published GenMol V1/V2 reference rows, for comparison only;
  4. when several horizons are present, the quality-diversity frontier.

The published rows are transcribed constants, NOT values this repo computed.
They are printed in a separately labelled block so they can never be confused
with a measured COMPOSE number.

Usage
-----
    python scripts/denovo_official_table.py <report-dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from compose_v4.eval.denovo_benchmark import aggregate_seed_metrics

# Transcribed from the official repos.  Reference only -- never recomputed here.
PUBLISHED = {
    "GenMol V2": {"validity": 100.0, "uniqueness": 97.8, "quality": 89.7, "diversity": 0.830},
    "GenMol V1": {"validity": 100.0, "uniqueness": 99.7, "quality": 84.6, "diversity": 0.818},
}


def load_reports(root: Path) -> dict[float, list[dict]]:
    by_horizon: dict[float, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("seed*.json")):
        payload = json.loads(path.read_text())
        by_horizon[float(payload["horizon"])].append(payload)
    for entries in by_horizon.values():
        entries.sort(key=lambda item: int(item["seed"]))
    return dict(by_horizon)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def render(by_horizon: dict[float, list[dict]]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("COMPOSE unconditional de-novo generation")
    lines.append("=" * 78)

    for horizon in sorted(by_horizon):
        entries = by_horizon[horizon]
        lines.append("")
        lines.append(f"-- operational_horizon = {horizon} ({len(entries)} seed(s)) --")
        lines.append(
            f"{'seed':>10} {'valid%':>8} {'uniq%':>8} {'qual%':>8} "
            f"{'qual|VU%':>9} {'divers':>8} {'heavy':>7} {'events':>7}"
        )
        for entry in entries:
            lines.append(
                f"{int(entry['seed']):>10} "
                f"{_pct(entry['validity']):>8} "
                f"{_pct(entry['uniqueness']):>8} "
                f"{_pct(entry['quality']):>8} "
                f"{_pct(entry['quality_given_valid_unique']):>9} "
                f"{float(entry['diversity']):>8.3f} "
                f"{float(entry['mean_heavy_atoms']):>7.1f} "
                f"{float(entry['mean_events']):>7.1f}"
            )
        summary = aggregate_seed_metrics(entries)
        std = summary.get("quality_std")
        spread = "" if std is None else f" +- {100.0 * float(std):.1f}"
        lines.append(
            f"{'MEAN':>10} "
            f"{_pct(summary['validity_mean']):>8} "
            f"{_pct(summary['uniqueness_mean']):>8} "
            f"{_pct(summary['quality_mean']):>8} "
            f"{_pct(summary['quality_given_valid_unique_mean']):>9} "
            f"{float(summary['diversity_mean']):>8.3f}"
            f"{spread}"
        )

    lines.append("")
    lines.append("-- published reference rows (transcribed, NOT measured here) --")
    lines.append(
        f"{'system':>10} {'valid%':>8} {'uniq%':>8} {'qual%':>8} {'divers':>17}"
    )
    for name, row in PUBLISHED.items():
        lines.append(
            f"{name:>10} {row['validity']:>8.1f} {row['uniqueness']:>8.1f} "
            f"{row['quality']:>8.1f} {row['diversity']:>17.3f}"
        )

    if len(by_horizon) > 1:
        lines.append("")
        lines.append("-- quality-diversity frontier (mean over seeds) --")
        lines.append(f"{'horizon':>9} {'quality%':>9} {'diversity':>10} {'heavy':>7}")
        for horizon in sorted(by_horizon):
            entries = by_horizon[horizon]
            summary = aggregate_seed_metrics(entries)
            heavy = sum(float(e["mean_heavy_atoms"]) for e in entries) / len(entries)
            lines.append(
                f"{horizon:>9.1f} "
                f"{_pct(summary['quality_mean']):>9} "
                f"{float(summary['diversity_mean']):>10.3f} "
                f"{heavy:>7.1f}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    by_horizon = load_reports(args.report_dir)
    if not by_horizon:
        raise SystemExit(f"no seed*.json reports under {args.report_dir}")

    print(render(by_horizon))

    if args.json is not None:
        payload = {
            "published_reference": PUBLISHED,
            "horizons": {
                str(horizon): {
                    "per_seed": entries,
                    "aggregate": aggregate_seed_metrics(entries),
                }
                for horizon, entries in by_horizon.items()
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
