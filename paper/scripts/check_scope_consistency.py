#!/usr/bin/env python3
"""Fail manuscript configuration generation if the paper registry drifts from the
committed max_atoms=40 production artifacts or the live scope.

The paper's `config_registry.yaml` must not carry a hand-copied number that
disagrees with the authoritative production state-space contract. This gate
triangulates three sources:

  1. paper/config_registry.yaml                                  (the paper's claims)
  2. diagnostics/composition/scaled_edit_data_manifest_40.json   (the locked manifest)
     + diagnostics/composition/broad_characterization_40.json    (the characterization)
  3. the LIVE scope compose_v4.data.organic_corpus.BROAD_ORGANIC_V1 (when importable)

It fails (nonzero) if any of the following hold:
  * config SCOPE_MAX_ATOMS != the live scope's max_atoms, the manifest's, or 40;
  * config SCOPE_HASH != the live scope hash or the manifest's corpus_scope_hash;
  * a retained-corpus / rejection / pool / hash value in the registry differs from
    the manifest (and, where present, the characterization artifact);
  * any registry value re-introduces the superseded max_atoms=48 scope (the value
    "48" for SCOPE_MAX_ATOMS, the legacy scope hash, or the 490,466 retained count).

Registry values are authored LaTeX-ready (e.g. "466{,}483", "93.3\\%",
"\\texttt{70526d92f1f08d8e}"); they are normalized to a bare token before
comparison, so display formatting never causes a false mismatch. The live-scope
axis is enforced when compose_v4 imports and is otherwise reported SKIPPED (the
manifest is itself live-derived, so config-vs-manifest still guards drift); run
under the project venv to enforce all three axes.

Exit 0 = consistent; nonzero = drift (with a report). Safe to run any time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
MANIFEST = REPO / "diagnostics" / "composition" / "scaled_edit_data_manifest_40.json"
CHARACTERIZATION = REPO / "diagnostics" / "composition" / "broad_characterization_40.json"

PROD_MAX_ATOMS = 40
PROD_SCOPE_HASH = "3721d69851110fdd"
LEGACY_SCOPE_HASH = "e59fb09801459470"  # the retired max_atoms=48 scope -- must not reappear
LEGACY_RETAINED = "490466"              # the max_atoms=48 retained count -- must not reappear


def _norm(value: object) -> str:
    """Strip LaTeX display formatting to a bare comparison token: drop \\texttt{},
    thousands separators ({,}/\\,/,), percent signs, braces, backslashes, spaces."""
    s = str(value)
    for token in ("\\texttt{", "\\%", "%", "{,}", "\\,", ",", "{", "}", "\\", " "):
        s = s.replace(token, "")
    return s.strip()


def _load_config() -> dict:
    cfg = yaml.safe_load((PAPER / "config_registry.yaml").read_text()) or {}
    return {k: v["value"] for k, v in cfg.items() if isinstance(v, dict) and "value" in v}


def _live_scope() -> tuple[dict | None, str | None]:
    """Return (facts, error). facts = {max_atoms, scope_hash, vocab_hash} from the
    live registry scope, or (None, reason) if compose_v4 cannot be imported."""
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import hashlib

        from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
        from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1
    except Exception as exc:  # noqa: BLE001 -- any import failure => skip the live axis
        return None, f"{type(exc).__name__}: {exc}"
    classes = list(ORGANIC_VOCABULARY.classes)
    vocab_hash = hashlib.sha256(json.dumps(classes, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "max_atoms": BROAD_ORGANIC_V1.max_atoms,
        "scope_hash": BROAD_ORGANIC_V1.scope_hash(),
        "vocab_hash": vocab_hash,
        "n_classes": len(classes),
    }, None


def main() -> int:
    failures: list[str] = []
    cfg = _load_config()
    manifest = json.loads(MANIFEST.read_text())
    charac = json.loads(CHARACTERIZATION.read_text())

    census = manifest["corpus_census"]
    pool = manifest["mmp_pool"]
    dirbal = manifest["measured_statistics"]["direction_balance"]
    edges = manifest["measured_statistics"]["path_length"]["curriculum_bin_edges"]

    def check(key: str, expected: object, label: str) -> None:
        if key not in cfg:
            failures.append(f"registry missing key {key} ({label})")
            return
        got, exp = _norm(cfg[key]), _norm(expected)
        if got != exp:
            failures.append(f"{key}: registry {cfg[key]!r} (->{got}) != {label} {expected!r} (->{exp})")

    # ---- 1. the two load-bearing contract values, triangulated -----------------
    check("SCOPE_MAX_ATOMS", PROD_MAX_ATOMS, "production contract 40")
    check("SCOPE_MAX_ATOMS", census["scope"]["max_atoms"], "manifest census max_atoms")
    check("SCOPE_MAX_ATOMS", charac["scope"]["max_atoms"], "characterization max_atoms")
    check("SCOPE_HASH", PROD_SCOPE_HASH, "production scope hash")
    check("SCOPE_HASH", manifest["corpus_scope_hash"], "manifest corpus_scope_hash")
    check("SCOPE_HASH", charac["scope"]["scope_hash"], "characterization scope_hash")

    # ---- 2. corpus census (registry == manifest == characterization) -----------
    check("CORPUS_ELIGIBLE_COUNT_BROAD", census["retained"], "manifest retained")
    check("CORPUS_ELIGIBLE_COUNT_BROAD", charac["corpus_census_500k"]["retained"], "characterization retained")
    check("CORPUS_ELIGIBLE_FRACTION_BROAD", round(census["retained_fraction"] * 100, 1), "manifest retained_fraction%")
    check("CORPUS_ELIGIBLE_COUNT_CNOF", census["non_cnof_element_combination_count"]["(cnof_only)"], "manifest (cnof_only)")
    check("CORPUS_REJECTED_TOTAL", census["rejected"], "manifest rejected")
    check("CORPUS_REJECT_TOOBIG", census["rejection_reasons"]["too_big"], "manifest too_big")
    check("CORPUS_REJECT_UNSUPPORTED_ELEMENT", census["rejection_reasons"]["unsupported_element"], "manifest unsupported_element")
    check("CORPUS_REJECT_UNPARSEABLE", census["rejection_reasons"]["unparseable"], "manifest unparseable")
    check("CORPUS_CHARGED_ATOM_MOLECULES", census["contains_charged_atom"], "manifest contains_charged_atom")
    check("CORPUS_SHA256", manifest["corpus_sha256"][:16], "manifest corpus_sha256[:16]")

    # ---- 2b. corpus accounting identity (the 33-record delta, machine-checked) --
    check("CORPUS_CANONICAL_DUPLICATES", census["canonical_duplicates"], "manifest canonical_duplicates")
    check("CORPUS_SOURCE_LINES", census["total"], "manifest census total")
    ident = census["retained"] + census["canonical_duplicates"] + census["rejected"]
    if ident != census["total"]:
        failures.append(f"accounting identity broken: retained+dup+rejected={ident} != total {census['total']}")
    reasons_sum = sum(census["rejection_reasons"].values())
    if reasons_sum != census["rejected"]:
        failures.append(f"rejection reasons sum {reasons_sum} != rejected {census['rejected']}")
    chain = charac.get("corpus_accounting", {}).get("exclusive_chain", {})
    if chain.get("retained_unique") != census["retained"] or chain.get("nonempty_source_lines") != census["total"]:
        failures.append("characterization corpus_accounting.exclusive_chain endpoints disagree with the manifest census")

    # ---- 3. production pool + curriculum + versioned hashes ---------------------
    check("PROD_MMP_PAIRS", pool["mmp_pairs_grouped"], "manifest mmp_pairs_grouped")
    check("PROD_RECORDS", pool["row_count"], "manifest pool row_count")
    check("PROD_RECORDS_FWD", dirbal["forward"], "manifest direction forward")
    check("PROD_RECORDS_REV", dirbal["reverse"], "manifest direction reverse")
    check("CURRICULUM_BIN_EDGES", "/".join(str(e) for e in edges), "manifest curriculum_bin_edges")
    check("TRACE_POOL_SHA256", pool["sha256"][:16], "manifest pool sha256[:16]")
    check("OPERATOR_REGISTRY_HASH", manifest["operator_registry"]["hash"], "manifest operator_registry hash")
    check("RING_CATALOG_HASH", manifest["corruption_recipe"]["ring_catalog_fingerprint"], "manifest ring_catalog_fingerprint")
    check("STANDARDIZATION_HASH", manifest["standardization_hash"], "manifest standardization_hash")
    check("PROD_MINING_COMMIT", manifest["source_mining_commit"], "manifest source_mining_commit")

    # ---- 4. the superseded max_atoms=48 scope must not reappear -----------------
    for key, val in cfg.items():
        norm = _norm(val)
        if LEGACY_SCOPE_HASH in norm:
            failures.append(f"{key}: re-introduces the retired 48-atom scope hash {LEGACY_SCOPE_HASH}")
        if key == "SCOPE_MAX_ATOMS" and norm == "48":
            failures.append(f"{key}: max_atoms=48 is INVALID_FOR_BEDIT40_PRODUCTION")
        if key.startswith("CORPUS_ELIGIBLE_COUNT") and norm == LEGACY_RETAINED:
            failures.append(f"{key}: {LEGACY_RETAINED} is the retired 48-atom retained count")

    # ---- 5. live scope (enforced when compose_v4 imports) ----------------------
    live, err = _live_scope()
    if live is None:
        print(f"check_scope_consistency: NOTE -- live-scope axis SKIPPED ({err}); "
              "config-vs-manifest still enforced. Run under the project venv to enforce it.")
    else:
        if live["max_atoms"] != PROD_MAX_ATOMS:
            failures.append(f"LIVE scope max_atoms {live['max_atoms']} != production 40")
        check("SCOPE_MAX_ATOMS", live["max_atoms"], "LIVE scope max_atoms")
        check("SCOPE_HASH", live["scope_hash"], "LIVE scope_hash")
        check("VOCAB_HASH", live["vocab_hash"], "LIVE sha256(ORGANIC_VOCABULARY.classes)[:16]")
        if live["scope_hash"] != PROD_SCOPE_HASH:
            failures.append(f"LIVE scope_hash {live['scope_hash']} != production {PROD_SCOPE_HASH} "
                            "(live code drifted from the mined manifest)")

    if failures:
        print("check_scope_consistency: INCONSISTENT (paper registry drifts from the max_atoms=40 contract)\n")
        for msg in failures:
            print("  - " + msg)
        return 1
    axis = "config+manifest+characterization+live" if live else "config+manifest+characterization"
    print(f"check_scope_consistency: OK -- registry matches the max_atoms=40 production contract [{axis}].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
