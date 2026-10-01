"""Load pinned upstream fragment metrics from explicit local files.

The upstream sources retain their own license. They are not downloaded or copied
into this package. Relative imports use a private namespace so an installed
InVirtuoGen package cannot substitute different metric implementations.
"""

from __future__ import annotations

import hashlib
import sys
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FragmentEvaluator:
    evaluate_smiles: Callable
    average_tanimoto: Callable
    source_sha256: Mapping[str, str]


def _unsupported(*args, **kwargs):
    raise NotImplementedError("only already_smiles=True fragment evaluation is supported")


def load_evaluator(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> FragmentEvaluator:
    """Verify both executable files before importing their exact bytes.

    Hashes must come from the independently pinned asset manifest, not from the
    supplied source files. Generation workers call this before evaluating output.
    """
    modules = {"evaluator_mol": "utils.mol", "evaluator_metrics": "train_utils.metrics"}
    contents = {}
    for name in modules:
        if name not in hashes or name not in paths:
            raise ValueError(f"missing evaluator identity: {name}")
        path = Path(paths[name])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise ValueError(f"evaluator SHA-256 mismatch: {path}, expected {hashes[name]}")
        contents[name] = data
    identity = hashlib.sha256(
        "|".join(hashes[name] for name in sorted(modules)).encode()
    ).hexdigest()
    namespace = f"_compose_fragment_evaluator_{identity}"
    cached = sys.modules.get(namespace)
    if cached is not None:
        result = getattr(cached, "evaluator", None)
        if not isinstance(result, FragmentEvaluator):
            raise RuntimeError("fragment evaluator namespace is occupied by an unknown module")
        return result

    created = []

    def new_module(suffix: str, *, package: bool = False):
        name = namespace + ("." + suffix if suffix else "")
        module = types.ModuleType(name)
        module.__package__ = name if package else name.rpartition(".")[0]
        if package:
            module.__path__ = []
        sys.modules[name] = module
        created.append(name)
        return module

    try:
        root = new_module("", package=True)
        for suffix in ("utils", "train_utils", "preprocess"):
            new_module(suffix, package=True)
        tokenizer = new_module("preprocess.preprocess_tokenize")
        tokenizer.custom_decode_sequence = _unsupported
        fragments = new_module("utils.fragments")
        fragments.bridge_smiles_fragments = _unsupported
        fragments.bridge_smiles_fragments_fix = _unsupported
        loaded = {}
        for key, suffix in modules.items():
            module = new_module(suffix)
            module.__file__ = str(Path(paths[key]).resolve())
            # Import only the bytes checked above, with no second unverified read.
            exec(compile(contents[key], module.__file__, "exec"), module.__dict__)  # noqa: S102
            loaded[key] = module
        result = FragmentEvaluator(
            loaded["evaluator_metrics"].evaluate_smiles,
            loaded["evaluator_mol"].calculate_average_tanimoto,
            {key: hashes[key] for key in modules},
        )
        root.evaluator = result
        return result
    except BaseException:
        # Cleanup only. Import errors and interruptions remain failures.
        for name in reversed(created):
            sys.modules.pop(name, None)
        raise
