"""Molecular quality and distribution diagnostics for target-free samples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer
from scipy import linalg
from scipy.stats import wasserstein_distance


Descriptor = Callable[[Chem.Mol], float]

DESCRIPTORS: dict[str, Descriptor] = {
    "qed": lambda mol: float(QED.qed(mol)),
    "sa_score": lambda mol: float(sascorer.calculateScore(mol)),
    "molecular_weight": lambda mol: float(Descriptors.MolWt(mol)),
    "logp": lambda mol: float(Crippen.MolLogP(mol)),
    "tpsa": lambda mol: float(rdMolDescriptors.CalcTPSA(mol)),
    "h_bond_donors": lambda mol: float(Lipinski.NumHDonors(mol)),
    "h_bond_acceptors": lambda mol: float(Lipinski.NumHAcceptors(mol)),
    "rotatable_bonds": lambda mol: float(Lipinski.NumRotatableBonds(mol)),
    "fraction_csp3": lambda mol: float(rdMolDescriptors.CalcFractionCSP3(mol)),
    "aromatic_atom_fraction": lambda mol: float(
        sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
        / max(mol.GetNumAtoms(), 1)
    ),
}


def molecular_quality_report(
    generated_smiles: tuple[str, ...],
    *,
    reference_smiles: tuple[str, ...],
    train_smiles: tuple[str, ...] = (),
    include_fcd: bool = False,
    fcd_reference_limit: int | None = 5000,
    fcd_generated_limit: int | None = None,
    fcd_device: str = "cpu",
    max_similarity_molecules: int = 1000,
    seed: int = 20260714,
) -> dict[str, object]:
    """Compare generated molecules with a reference molecular distribution."""

    if not generated_smiles or not reference_smiles:
        raise ValueError("generated_smiles and reference_smiles must be non-empty")
    generated_molecules, generated_canonical = _parse(generated_smiles)
    reference_molecules, reference_canonical = _parse(reference_smiles)
    train_molecules, train_canonical = _parse(train_smiles)
    if not generated_molecules or not reference_molecules:
        raise ValueError("generated and reference sets must contain valid molecules")

    generated_unique = set(generated_canonical)
    train_unique = set(train_canonical)
    report: dict[str, object] = {
        "requested_samples": len(generated_smiles),
        "valid_fraction": len(generated_molecules) / len(generated_smiles),
        "unique_fraction": len(generated_unique) / len(generated_molecules),
        "novel_to_train_fraction": float(
            np.mean([smiles not in train_unique for smiles in generated_canonical])
        ),
        "descriptor_distributions": _descriptor_report(
            generated_molecules,
            reference_molecules,
        ),
        "ring_systems": _ring_report(generated_molecules, reference_molecules),
        "scaffolds": _scaffold_report(
            generated_molecules,
            train_molecules,
        ),
        "similarity": _similarity_report(
            generated_molecules,
            train_molecules,
            max_molecules=max_similarity_molecules,
            seed=seed,
        ),
    }
    if include_fcd:
        from fcd_torch import FCD

        if fcd_reference_limit is not None and fcd_reference_limit < 2:
            raise ValueError("fcd_reference_limit must be at least two")
        if fcd_generated_limit is not None and fcd_generated_limit < 2:
            raise ValueError("fcd_generated_limit must be at least two")

        # fcd-torch<=1.0.7 still calls NumPy's historical ``row_stack``
        # alias, which was removed in NumPy 2.0.  The operation is exactly
        # ``vstack``; restoring the alias locally keeps the evaluator working
        # without changing its numerical result.
        if not hasattr(np, "row_stack"):
            np.row_stack = np.vstack  # type: ignore[attr-defined]
        # SciPy 1.16 removed the deprecated ``disp`` argument and now returns
        # only the matrix square root.  fcd-torch requests ``disp=False`` and
        # unpacks the legacy ``(root, error)`` pair; adapt that API while
        # preserving the same square-root calculation.
        try:
            linalg.sqrtm(np.eye(1), disp=False)
        except TypeError:
            scipy_sqrtm = linalg.sqrtm

            def legacy_sqrtm(matrix, disp=True, blocksize=64):
                del blocksize
                root = scipy_sqrtm(matrix)
                return root if disp else (root, np.nan)

            linalg.sqrtm = legacy_sqrtm  # type: ignore[assignment]
        generated_for_fcd = generated_canonical[
            : fcd_generated_limit or len(generated_canonical)
        ]
        reference_for_fcd = reference_canonical[
            : fcd_reference_limit or len(reference_canonical)
        ]
        if min(len(generated_for_fcd), len(reference_for_fcd)) < 2:
            raise ValueError("FCD requires at least two valid molecules per distribution")
        evaluator = FCD(device=fcd_device, n_jobs=1, batch_size=512)
        generated_activations = evaluator.get_predictions(generated_for_fcd)
        reference_activations = evaluator.get_predictions(reference_for_fcd)
        components = _frechet_activation_components(
            generated_activations,
            reference_activations,
        )
        report["frechet_chemnet_distance"] = components["total"]
        report["frechet_chemnet"] = {
            **components,
            "generated_count": len(generated_for_fcd),
            "reference_count": len(reference_for_fcd),
            "device": fcd_device,
        }
    return report


def _frechet_activation_components(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Decompose FCD into ChemNet mean shift and covariance coverage.

    The total is the usual Fréchet distance.  Reporting the two additive
    components distinguishes a misplaced average molecule from a generator
    that covers too little of the corpus distribution—the decisive diagnostic
    in the FCD-9.96 predecessor run.
    """

    generated = np.asarray(generated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("ChemNet activations must be two-dimensional")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("generated/reference activation dimensions must agree")
    if min(generated.shape[0], reference.shape[0]) < 2:
        raise ValueError("at least two activations are required per distribution")

    generated_mean = generated.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    generated_covariance = np.atleast_2d(np.cov(generated, rowvar=False))
    reference_covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    product = generated_covariance.dot(reference_covariance)
    covariance_mean = _matrix_square_root(product)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(generated_covariance.shape[0]) * eps
        covariance_mean = _matrix_square_root(
            (generated_covariance + offset).dot(reference_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0.0, atol=1e-3):
            raise ValueError(
                f"FCD covariance square root has imaginary magnitude "
                f"{np.max(np.abs(covariance_mean.imag))}"
            )
        covariance_mean = covariance_mean.real

    mean_component = float(np.square(generated_mean - reference_mean).sum())
    generated_trace = float(np.trace(generated_covariance))
    reference_trace = float(np.trace(reference_covariance))
    covariance_component = float(
        generated_trace
        + reference_trace
        - 2.0 * float(np.trace(covariance_mean))
    )
    total = mean_component + covariance_component
    return {
        "total": total,
        "mean_component": mean_component,
        "covariance_component": covariance_component,
        "mean_fraction": mean_component / max(total, eps),
        "covariance_fraction": covariance_component / max(total, eps),
        "generated_covariance_trace": generated_trace,
        "reference_covariance_trace": reference_trace,
        "covariance_trace_ratio": generated_trace / max(reference_trace, eps),
    }


def _matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    """Call SciPy's pre/post-1.16 sqrtm APIs without changing the result."""

    try:
        result = linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg.sqrtm(matrix)
    return result[0] if isinstance(result, tuple) else result


def _parse(smiles: tuple[str, ...]) -> tuple[list[Chem.Mol], list[str]]:
    molecules = []
    canonical = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            continue
        molecules.append(molecule)
        canonical.append(Chem.MolToSmiles(molecule, canonical=True))
    return molecules, canonical


def _descriptor_report(
    generated: list[Chem.Mol],
    reference: list[Chem.Mol],
) -> dict[str, dict[str, float]]:
    result = {}
    for name, function in DESCRIPTORS.items():
        generated_values = np.asarray([function(mol) for mol in generated])
        reference_values = np.asarray([function(mol) for mol in reference])
        reference_scale = max(float(reference_values.std()), 1e-8)
        distance = float(wasserstein_distance(generated_values, reference_values))
        result[name] = {
            "generated_mean": float(generated_values.mean()),
            "reference_mean": float(reference_values.mean()),
            "wasserstein_distance": distance,
            "standardized_wasserstein": distance / reference_scale,
        }
    return result


def _ring_report(
    generated: list[Chem.Mol],
    reference: list[Chem.Mol],
) -> dict[str, float]:
    generated_stats = [_ring_statistics(mol) for mol in generated]
    reference_stats = [_ring_statistics(mol) for mol in reference]
    generated_sizes = sum((item[4] for item in generated_stats), Counter())
    reference_sizes = sum((item[4] for item in reference_stats), Counter())
    size_keys = tuple(range(3, 13)) + (13,)
    generated_hist = np.asarray([generated_sizes[key] for key in size_keys], dtype=float)
    reference_hist = np.asarray([reference_sizes[key] for key in size_keys], dtype=float)
    ring_size_tv = _count_total_variation(generated_hist, reference_hist)
    generated_aromatic = np.asarray(
        [rdMolDescriptors.CalcNumAromaticRings(mol) for mol in generated],
        dtype=float,
    )
    reference_aromatic = np.asarray(
        [rdMolDescriptors.CalcNumAromaticRings(mol) for mol in reference],
        dtype=float,
    )
    generated_ring_count = np.asarray([item[0] for item in generated_stats], dtype=float)
    reference_ring_count = np.asarray([item[0] for item in reference_stats], dtype=float)
    return {
        "generated_mean_rings": float(np.mean([item[0] for item in generated_stats])),
        "reference_mean_rings": float(np.mean([item[0] for item in reference_stats])),
        "generated_fused_fraction": float(np.mean([item[1] for item in generated_stats])),
        "reference_fused_fraction": float(np.mean([item[1] for item in reference_stats])),
        "generated_spiro_fraction": float(np.mean([item[2] for item in generated_stats])),
        "reference_spiro_fraction": float(np.mean([item[2] for item in reference_stats])),
        "generated_bridged_fraction": float(np.mean([item[3] for item in generated_stats])),
        "reference_bridged_fraction": float(np.mean([item[3] for item in reference_stats])),
        "generated_mean_aromatic_rings": float(generated_aromatic.mean()),
        "reference_mean_aromatic_rings": float(reference_aromatic.mean()),
        "generated_aromatic_ring_fraction": float(
            generated_aromatic.sum() / max(generated_ring_count.sum(), 1.0)
        ),
        "reference_aromatic_ring_fraction": float(
            reference_aromatic.sum() / max(reference_ring_count.sum(), 1.0)
        ),
        "ring_size_total_variation": ring_size_tv,
    }


def _ring_statistics(molecule: Chem.Mol) -> tuple[int, bool, bool, bool, Counter]:
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    fused = any(
        len(left & right) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    )
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0
    bridged = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0
    sizes = Counter(min(len(ring), 13) for ring in rings)
    return len(rings), fused, spiro, bridged, sizes


def _scaffold_report(
    generated: list[Chem.Mol],
    train: list[Chem.Mol],
) -> dict[str, float]:
    generated_scaffolds = [_scaffold(mol) for mol in generated]
    train_scaffolds = {_scaffold(mol) for mol in train}
    return {
        "unique_scaffold_fraction": len(set(generated_scaffolds)) / len(generated),
        "novel_scaffold_fraction": float(
            np.mean([scaffold not in train_scaffolds for scaffold in generated_scaffolds])
        )
        if train_scaffolds
        else 1.0,
        "acyclic_fraction": float(
            np.mean([scaffold == "<ACYCLIC>" for scaffold in generated_scaffolds])
        ),
    }


def _scaffold(molecule: Chem.Mol) -> str:
    return (
        MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        or "<ACYCLIC>"
    )


def _similarity_report(
    generated: list[Chem.Mol],
    train: list[Chem.Mol],
    *,
    max_molecules: int,
    seed: int,
) -> dict[str, float | None]:
    if max_molecules <= 1:
        raise ValueError("max_similarity_molecules must exceed one")
    rng = np.random.default_rng(seed)
    generated = _subsample(generated, max_molecules, rng)
    train = _subsample(train, max_molecules, rng)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    generated_fingerprints = [generator.GetFingerprint(mol) for mol in generated]

    similarities = []
    for index, fingerprint in enumerate(generated_fingerprints):
        similarities.extend(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint,
                generated_fingerprints[index + 1 :],
            )
        )
    internal_diversity = 1.0 - float(np.mean(similarities)) if similarities else 0.0

    nearest_train = None
    if train:
        train_fingerprints = [generator.GetFingerprint(mol) for mol in train]
        nearest_train = float(
            np.mean(
                [
                    max(DataStructs.BulkTanimotoSimilarity(fp, train_fingerprints))
                    for fp in generated_fingerprints
                ]
            )
        )
    return {
        "internal_diversity": internal_diversity,
        "mean_nearest_train_tanimoto": nearest_train,
    }


def _subsample(items: list, limit: int, rng: np.random.Generator) -> list:
    if len(items) <= limit:
        return items
    indices = rng.choice(len(items), size=limit, replace=False)
    return [items[int(index)] for index in indices]


def _count_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left_total = float(left.sum())
    right_total = float(right.sum())
    if left_total == 0.0 or right_total == 0.0:
        return float(left_total != right_total)
    return 0.5 * float(np.abs(left / left_total - right / right_total).sum())
