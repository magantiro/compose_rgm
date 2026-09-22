"""Render the three-arm ring-marginal comparison from one report JSON.

The decisive check is NOT "did small rings fall below some threshold".  It is
whether the planned arm reproduces the TRAINING ring-system distribution while
preserving what already works -- validity, uniqueness, diversity and QED.  So
the corpus row is printed first, every arm is printed against it, and the
preserved-quantity block is printed beside the ring block rather than below it,
because a ring repair bought by regressing diversity or QED is not a repair.

Usage::

    python3 scripts/denovo_ring_marginal_table.py <report.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RING_SIZES = ("3", "4", "5", "6", "7", "8")


def _fraction(table: dict, key: str) -> str:
    value = table.get(key)
    return "  .   " if value is None else f"{float(value) * 100:5.1f}%"


def _interval(block: dict | None, key: str) -> str:
    if not block or key not in block:
        return ""
    row = block[key]
    return f" [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"


def _total_variation(left: dict, right: dict) -> float:
    """TV between two categorical laws given as {label: fraction}."""

    return 0.5 * sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
        for key in set(left) | set(right)
    )


def _system_level_block(report: dict, corpus_census: dict | None) -> list[str]:
    """Ring SIZE, ring-system COUNT and ring-system SIGNATURE are three laws.

    An arm can match the size law and still deliver the wrong number of ring
    systems, which is exactly what a plan whose systems the host refuses does.
    The signature fractions live only in the corpus census, so this block is
    printed when that file is supplied and skipped otherwise rather than
    silently comparing against a missing key.
    """

    if corpus_census is None:
        return []
    lines = [
        "RING-SYSTEM LAWS (a per-ring size match is NOT a ring-system match)",
        f"{'row':<26}{'TV ring-SIZE':>15}{'TV system-COUNT':>18}{'TV system-SIGNATURE':>22}",
    ]
    lines.append("-" * len(lines[-1]))
    for arm, row in sorted(report["arms"].items()):
        census = row.get("ring_signature_census")
        if census is None:
            continue
        lines.append(
            f"{'arm ' + arm:<26}"
            f"{row.get('ring_size_total_variation_vs_corpus', float('nan')):15.3f}"
            + f"{_total_variation(census['ring_system_count_fraction'], corpus_census['ring_system_count_fraction']):18.3f}"
            + f"{_total_variation(census['system_signature_fraction'], corpus_census['system_signature_fraction']):22.3f}"
        )
    lines.append("")
    return lines


def render(report: dict, corpus_census: dict | None = None) -> str:
    lines: list[str] = []
    corpus = report["corpus_reference"]
    lines.append(f"design            {report['design']}")
    # `total_per_arm` is the DESIGN target; the realized n differs per arm whenever a
    # shard was lost, so print it as a target and let the `n` column carry the truth.
    lines.append(
        f"model             {report['model']}  design target n/arm {report['total_per_arm']}"
        "  (realized n per arm below)"
    )
    lines.append(
        f"corpus            {corpus['source']} ({corpus['molecules']:,} molecules)"
    )
    lines.append("")

    header = (
        f"{'row':<26}"
        + "".join(f"{f'{size}-ring':>8}" for size in RING_SIZES)
        + f"{'TV':>8}{'strain/ring':>13}{'mols w/ 3-4':>13}{'sys/mol':>9}"
    )
    lines.append("RING DISTRIBUTION (per-ring size law; TV against the corpus)")
    lines.append(header)
    lines.append("-" * len(header))
    sizes = corpus["ring_size_fraction"]
    lines.append(
        f"{'GuacaMol train':<26}"
        + "".join(_fraction(sizes, size) + "  " for size in RING_SIZES)
        + f"{'-':>8}"
        + f"{corpus['strained_ring_fraction'] * 100:12.1f}%"
        + f"{corpus['fraction_with_strained_ring'] * 100:12.1f}%"
        + f"{corpus['ring_systems_per_molecule']:9.2f}"
    )
    for arm, row in sorted(report["arms"].items()):
        census = row.get("ring_signature_census")
        if census is None:
            continue
        boot = row.get("ring_bootstrap")
        lines.append(
            f"{'arm ' + arm:<26}"
            + "".join(_fraction(census["ring_size_fraction"], size) + "  " for size in RING_SIZES)
            + f"{row.get('ring_size_total_variation_vs_corpus', float('nan')):8.3f}"
            + f"{census['strained_ring_fraction'] * 100:12.1f}%"
            + f"{census['fraction_with_strained_ring'] * 100:12.1f}%"
            + f"{census['ring_systems_per_molecule']:9.2f}"
        )
        if boot:
            lines.append(
                f"{'  95% CI':<26}"
                + " " * (8 * len(RING_SIZES) + 2 * len(RING_SIZES))
                + f"{_interval(boot, 'ring_size_total_variation').strip():>8}"
                f"  strain {_interval(boot, 'strained_ring_fraction').strip()}"
                f"  mols {_interval(boot, 'fraction_with_strained_ring').strip()}"
            )
    lines.append("")

    lines.append("PRESERVED QUANTITIES (a ring repair bought by regressing these is not one)")
    preserved = (
        f"{'row':<26}{'n':>6}{'validity':>10}{'uniqueness':>12}{'quality':>16}"
        f"{'diversity':>11}{'mean QED':>10}{'mean SA':>9}{'heavy':>8}{'events':>8}"
    )
    lines.append(preserved)
    lines.append("-" * len(preserved))
    for arm, row in sorted(report["arms"].items()):
        metrics = row["published_metrics"]
        decomposition = row["decomposition"]["ring_size_distribution"]
        stderr = row.get("published_metrics_stderr", {})
        quality = f"{metrics['quality']:.3f}"
        if "quality" in stderr:
            quality += f"+-{stderr['quality']:.3f}"
        lines.append(
            f"{'arm ' + arm:<26}"
            f"{row['attempted']:6d}"
            f"{metrics['validity']:10.3f}"
            f"{metrics['uniqueness']:12.3f}"
            f"{quality:>16}"
            f"{metrics['diversity']:11.4f}"
            f"{metrics['mean_qed']:10.3f}"
            f"{metrics['mean_sa']:9.3f}"
            f"{decomposition['mean_heavy_atoms']:8.1f}"
            f"{row['mean_events']:8.1f}"
        )
    lines.append("")

    lines.append(
        "NOTE: ring-size TV is CONDITIONAL on rings existing; compare sys/mol separately."
    )
    lines.append("")
    lines.extend(_system_level_block(report, corpus_census))
    lines.append("PLAN REALIZATION (planned arms only)")
    for arm, row in sorted(report["arms"].items()):
        plan = row.get("plan_realization")
        if not plan:
            continue
        lines.append(
            f"  arm {arm}: {plan['realized_systems']}/{plan['requested_systems']} systems "
            f"({plan['realization_rate'] * 100:.1f}%), "
            f"{plan['fully_realized_fraction'] * 100:.1f}% of plans complete, "
            f"reasons {plan['unrealized_reasons']}, "
            f"bin fallback {plan['bin_fallback_fraction'] * 100:.1f}%"
        )
        survived = plan.get("endpoint_matches_installed_skeleton")
        trigger = plan.get("plan_trigger_fired_fraction")
        lines.append(
            "          endpoint skeleton == installed skeleton: "
            + ("n/a" if survived is None else f"{survived * 100:.1f}%")
            + ("" if trigger is None else f"   trigger fired {trigger * 100:.1f}%")
        )
    lines.append("")

    lines.append("STRATIFIED SA AND QED (strained vs clean, within heavy-atom bin)")
    for arm, row in sorted(report["arms"].items()):
        strata = row["decomposition"]["strain_size_strata"]["strata"]
        lines.append(f"  arm {arm}")
        lines.append(
            f"    {'bin':<8}{'n clean':>9}{'n strained':>12}"
            f"{'dSA':>9}{'dQED':>9}"
        )
        for label, cell in strata.items():
            # A cell below the minimum stratum reports None, never a value.
            sa_difference = cell.get("within_bin_sa_difference")
            qed_difference = cell.get("within_bin_qed_difference")
            lines.append(
                f"    {label:<8}{cell['clean']['n']:>9}{cell['strained']['n']:>12}"
                + (f"{sa_difference:9.3f}" if sa_difference is not None else f"{'-':>9}")
                + (f"{qed_difference:9.3f}" if qed_difference is not None else f"{'-':>9}")
            )
        regression = row["decomposition"]["strain_size_regression"]
        for metric in ("sa", "qed"):
            fit = regression.get(metric)
            if not fit or fit.get("strained_coefficient") is None:
                lines.append(f"    {metric.upper()} ~ strain  (too few strained molecules)")
                continue
            lines.append(
                f"    {metric.upper()} ~ strain {fit['strained_coefficient']:+.3f}"
                f"+-{fit['strained_stderr']:.3f}"
                f"   heavy {fit['heavy_atom_coefficient']:+.4f}"
                f"+-{fit['heavy_atom_stderr']:.4f}"
                + (
                    f"   (raw contrast {fit['raw_strained_difference']:+.3f})"
                    if fit.get("raw_strained_difference") is not None
                    else "   (raw contrast undefined: one arm is empty)"
                )
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument(
        "--corpus-census",
        help="corpus_ring_census_v1.json; adds the system-COUNT and SIGNATURE comparisons",
    )
    arguments = parser.parse_args()
    census = (
        json.loads(Path(arguments.corpus_census).read_text())
        if arguments.corpus_census
        else None
    )
    print(render(json.loads(Path(arguments.report).read_text()), census))


if __name__ == "__main__":
    main()
