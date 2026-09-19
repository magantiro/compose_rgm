"""A secondary medicinal-chemistry screen, defined against a reference population.

This is NOT the T4 endpoint criterion and must never be substituted for it. The benchmark
defines validity as QED > 0.6, SA < 4 and similarity >= delta on a valid connected
molecule; a comparison against a published method has to run on that fiber and no
narrower, or the comparison is handicapped and the published number is no longer the thing
being beaten.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
`queryable_fiber.instability` currently bundles three different kinds of judgement into
the search's support: genuine validity (radicals), medicinal-chemistry preference (sixteen
motifs) and a net-charge rule. Measured, that bundle rejects **7.0% of the benchmark's own
Jin QED lead set (56 of 800)** and **13.8% of held-out GuacaMol (690 of 5,000)**. The
single largest contributor is `michael_acceptor_ketone`, a plain enone, at 3.1% of Jin
leads and 9.0% of GuacaMol. A rule that refuses one in eleven drug-like molecules is a
preference, not a stability law, and applying it inside the search costs real score.

THE DISCIPLINE THIS ENCODES
---------------------------
A chemistry screen invented after seeing a molecule we dislike is goalpost-moving, and
this project has already paid for that lesson twice -- once when an acyl-nitroso topped a
32-call experiment at -10.6 and had to be retracted, and once when four of 118 autonomous
endpoints turned out to carry a free primary imine. So:

  * a motif is admitted to this screen only with a stated mechanism of instability, never
    because a high-scoring molecule happened to contain it;
  * every rule is measured against a reference population BEFORE use, and a rule that
    rejects the benchmark's own leads at a material rate is rejected instead;
  * the screen is reported as a SECOND number alongside the official one.

`FALSE_POSITIVE_CEILING` is the admission test. It is deliberately strict: these motifs
are supposed to be things no chemist would carry forward, and such things are rare in a
curated drug-like set. A rule that fires on 1% of real leads is describing ordinary
chemistry.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "chemistry_screen_v1"

#: A candidate rule may not fire on more than this fraction of a curated drug-like
#: reference set. Measured base rates that motivated the number: primary imine 0.12% of
#: the Jin QED leads, enone 3.1% (rejected on exactly this test).
FALSE_POSITIVE_CEILING = 0.005

#: (name, SMARTS, mechanism). The mechanism field is required: it is what distinguishes a
#: stability law from a preference, and it is what a reviewer will ask for.
SCREEN_MOTIFS: tuple[tuple[str, str, str], ...] = (
    (
        "acyl_nitroso_or_N_O_acyl",
        "[CX3](=O)[OX2][NX3]",
        (
            "N-O acyl bond is hydrolytically and thermally labile; also an electrophilic "
            "nitroso donor. Found topping a 32-call experiment at -10.6 and retracted."
        ),
    ),
    (
        "peroxide",
        "[OX2][OX2]",
        (
            "The O-O bond homolyses to two radicals; an explosive hazard and "
            "incompatible with an aqueous assay."
        ),
    ),
    (
        "acyl_halide",
        "[CX3](=O)[F,Cl,Br,I]",
        (
            "Hydrolyses to the carboxylic acid within seconds in water, so the docked "
            "species is not the species present in the assay."
        ),
    ),
    (
        "anhydride",
        "[CX3](=O)[OX2][CX3]=O",
        (
            "Hydrolyses to two carboxylic acids in water and acylates protein "
            "nucleophiles indiscriminately on the way."
        ),
    ),
    (
        "isocyanate_isothiocyanate",
        "[NX2]=[CX2]=[OX1,SX1]",
        (
            "An indiscriminate electrophile that carbamoylates lysine and cysteine "
            "non-specifically rather than binding reversibly."
        ),
    ),
    (
        "diazo_or_diazonium",
        "[#6]=[NX2+]=[NX1-,NX1]",
        (
            "Extrudes dinitrogen under ambient conditions, so the docked species is not "
            "a persistent one."
        ),
    ),
    (
        "primary_ketimine_or_aldimine",
        "[CX3;!$([CX3][NX3]);!$([CX3][OX2]);!$([CX3][SX2])]=[NX2;H1;!$([NX2][#7,#8])]",
        (
            "An unstabilised C=NH hydrolyses to the carbonyl plus ammonia. Deliberately "
            "EXCLUDES amidines, imidates, isothioamides and hydrazones/oximes, where the "
            "adjacent heteroatom lone pair stabilises the C=N and which are ordinary "
            "drug chemistry. This rule is the narrow form, not a ban on C=N."
        ),
    ),
)

_COMPILED = tuple(
    (name, Chem.MolFromSmarts(smarts), mechanism) for name, smarts, mechanism in SCREEN_MOTIFS
)


def screen(smiles: str) -> list[str]:
    """Names of the screened motifs this molecule carries. Empty means it passes."""
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return ["unparseable"]
    return [name for name, pattern, _ in _COMPILED if pattern is not None
            and molecule.HasSubstructMatch(pattern)]


def false_positive_rates(reference_smiles) -> dict[str, float]:
    """Per-motif firing rate on a reference population, for the admission test."""
    molecules = [Chem.MolFromSmiles(s) for s in reference_smiles]
    molecules = [m for m in molecules if m is not None]
    if not molecules:
        raise ValueError("the reference population is empty; the admission test is vacuous")
    return {
        name: sum(1 for m in molecules if m.HasSubstructMatch(pattern)) / len(molecules)
        for name, pattern, _ in _COMPILED
        if pattern is not None
    }
