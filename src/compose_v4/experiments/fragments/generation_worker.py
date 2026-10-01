"""Offline CPU worker for the fragment library.

The parent supplies explicit source and asset paths. This worker records the
imported library files and never imports samplers from a source archive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def deny_network(event: str, args: tuple) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError(f"fragment worker is offline: {event}")


def seed_for(prompt: str, task: str, seed: int) -> int:
    key = f"{prompt}|{task}|{seed}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


def environment(expected: dict) -> dict:
    actual = {name: version(name) for name in expected if name != "python"}
    actual["python"] = platform.python_version()
    for name, required in expected.items():
        got = ".".join(actual[name].split(".")[:2]) if name == "python" else actual[name]
        if got != required:
            raise ValueError(f"wrong {name} version: {actual[name]}; required {required}")
    return actual


def panel_sampler(task: str, assets: dict):
    if task == "motif_extension":
        from compose_v4.benchmark.joint_completion_prior import (
            JointCompletionPrior,
            JointCompletionSampler,
        )

        catalog = json.loads(Path(assets["region_catalog"]).read_text())
        prior = json.loads(Path(assets["joint_prior"]).read_text())
        return JointCompletionSampler(catalog["entries"], JointCompletionPrior.from_dict(prior))
    if task == "scaffold_decoration":
        from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler

        return JointMassPendantSampler(
            json.loads(Path(assets["pendant_catalog"]).read_text()),
            json.loads(Path(assets["mass_prior"]).read_text()),
        )
    from compose_v4.benchmark.fragment_linker_sampler import load_linker_catalog
    from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior

    path = Path(assets["region_catalog"])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return (
        load_linker_catalog(path, expected_sha256=digest),
        JointCompletionPrior.from_dict(json.loads(Path(assets["joint_prior"]).read_text())),
    )


def sample_panel(task, prompt, context, sampler, model, rng, emitted):
    from compose_v4.chem.state import is_connected_or_null, is_valid_state

    if task == "motif_extension":
        from compose_v4.benchmark.fragment_motif_focused_programs import sample_motif_panel

        panel = sample_motif_panel(context, sampler, model, rng)
    elif task == "scaffold_decoration":
        from compose_v4.benchmark.fragment_pendant_programs import sample_pendant_panel

        panel = sample_pendant_panel(context, sampler, model, rng)
    else:
        from compose_v4.benchmark.fragment_linker_sampler import sample_linker_panel

        catalog, prior = sampler
        panel = sample_linker_panel(
            prompt,
            catalog,
            prior,
            model,
            rng,
            cell_allocation="frozen",
            prior_emitted=frozenset(emitted),
        )
    selected = panel.selected
    valid = bool(
        selected and is_valid_state(selected.endpoint) and is_connected_or_null(selected.endpoint)
    )
    faithful = False
    if selected and task == "linker_design":
        from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity

        faithful = bool(
            linker_fidelity(prompt, selected.smiles)["satisfied"]
            and selected.provenance["exact_mapped_core_identity_checked"]
            and selected.provenance["source_core_locked_all_states"]
        )
    elif selected:
        from compose_v4.benchmark.fragment_program_adapter import (
            ProgramConstraint,
            verify_prompt_endpoint,
        )
        from compose_v4.rewrite.trace_shard import decode_state

        constraint = ProgramConstraint.from_context(context)
        lock = constraint.lock(context.start_state)
        faithful = constraint.complete(selected.endpoint) and all(
            lock.permits(decode_state(state)) for state in selected.trace["states"]
        )
        verify_prompt_endpoint(context, selected)
    if selected and not (valid and faithful):
        raise ValueError(f"selected {task} output violates chemistry/fragment invariants")
    return {
        "panel": panel.receipt,
        "selected_valid_connected": valid,
        "selected_constraint_fidelity": faithful,
    }


EXPECTED_CATALOG_FINGERPRINT = "639ff6078c32d43c"


def generate(config: dict) -> dict:
    versions = environment(config["environment"])
    import numpy as np
    import torch

    from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
    from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
    from compose_v4.model.reference_checkpoint import load_frozen_reference
    from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = load_frozen_reference(
        Path(config["assets"]["checkpoint"]),
        expected_sha256=config["asset_sha256"]["checkpoint"],
        expected_catalog_fingerprint=EXPECTED_CATALOG_FINGERPRINT,
        catalog_path=Path(config["assets"]["catalog"]),
        expected_catalog_sha256=config["asset_sha256"]["catalog"],
    )
    model = reference.model
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("fragment reference must load as CPU float32")
    catalog_fingerprint = ring_catalog_fingerprint(model.ring_catalog)
    if catalog_fingerprint != EXPECTED_CATALOG_FINGERPRINT:
        raise ValueError(f"frozen reference catalog drift: {catalog_fingerprint}")
    task = config["task"]
    superstructure = task.startswith("superstructure_")
    prompt_task = "superstructure_generation" if superstructure else task
    prompts = [
        p
        for p in load_genmol_prompts(Path(config["assets"]["prompts"]))
        if p.task is FragmentTask(prompt_task) and p.drug_name in config["prompts"]
    ]
    if {p.drug_name for p in prompts} != set(config["prompts"]):
        raise ValueError("requested prompt absent from frozen prompt file")
    evaluator = None
    if config["evaluate"]:
        from compose_v4.experiments.fragments.evaluator import load_evaluator

        evaluator = load_evaluator(config["assets"], config["asset_sha256"])
    if superstructure:
        from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
        from compose_v4.benchmark.fragment_conditioned_sampler import (
            SamplerConfig,
            SamplingReceipt,
            sample_completion,
        )
        from compose_v4.rewrite.kernel import de_novo_rewrite_system

        sampler_config = SamplerConfig(**config["sampler_config"])
        control = AttachmentControlConfig(**config["attachment_control"])
        system = de_novo_rewrite_system()
        if task == "superstructure_uniform":
            from compose_v4.benchmark.uniform_native_mark_law import UniformNativeMarkLaw

            model = UniformNativeMarkLaw(model)
    else:
        sampler = panel_sampler(task, config["assets"])
    cells = []
    for seed in config["seeds"]:
        for prompt in prompts:
            rng_seed = seed_for(prompt.drug_name, prompt_task, seed)
            rng = np.random.default_rng(rng_seed)
            context = (
                build_prompt_context(prompt, config=sampler_config, control=control)
                if superstructure
                else build_prompt_context(prompt)
            )
            attempts, samples, emitted = [], [], set()
            if superstructure:
                receipt = SamplingReceipt()
            for index in range(config["attempts"]):
                before = copy.deepcopy(rng.bit_generator.state)
                if superstructure:
                    start_commits = len(receipt.committed_endpoints)
                    out = sample_completion(
                        model,
                        system,
                        context,
                        rng,
                        config=sampler_config,
                        receipt=receipt,
                        control=control,
                    )
                    new = receipt.committed_endpoints[start_commits:]
                    if len(new) > 1:
                        raise ValueError("more than one chemical commit per attempt")
                    item = {
                        "emitted_smiles": out,
                        "committed_smiles": new[0] if new else None,
                        "accepted_actions": receipt.action_traces[-1],
                        "events": receipt.events[-1],
                    }
                    samples.append(new[0] if new else "")
                else:
                    item = sample_panel(task, prompt, context, sampler, model, rng, emitted)
                    out = item["panel"]["selected_smiles"]
                    samples.append(out or "")
                    if out:
                        emitted.add(out)
                item.update(
                    attempt_index=index,
                    rng_state_before=before,
                    rng_state_after=copy.deepcopy(rng.bit_generator.state),
                )
                attempts.append(item)
                print(
                    f"{task} seed={seed} prompt={prompt.drug_name} attempt={index + 1}/{config['attempts']}",
                    flush=True,
                )
            cell = {
                "seed": seed,
                "prompt": prompt.drug_name,
                "rng_seed": rng_seed,
                "attempts": attempts,
                "chemical_samples": samples,
            }
            if superstructure:
                cell["sampling_receipt"] = asdict(receipt)
            if config["evaluate"]:
                from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

                cell["metrics"] = official_prompt_metrics(
                    samples, evaluator=evaluator, expected_samples=config["attempts"]
                )
            cells.append(cell)
    return {
        "schema": "compose_fragment_generation_v1",
        "task": task,
        "cells": cells,
        "software": versions,
        "hardware": {"machine": platform.machine(), "processor": platform.processor()},
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "KMP_DUPLICATE_LIB_OK",
            )
        },
        "ringcore_catalog_fingerprint": catalog_fingerprint,
        "checkpoint_sha256": reference.checkpoint_sha256,
        "seed_derivation": "BLAKE2b-8 of drug|task|seed, big-endian modulo 2**32; NumPy default_rng",
        "metrics_evaluated": config["evaluate"],
    }


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    runtime = Path(config["runtime"])
    sys.path.insert(0, str(runtime / "src"))
    sys.addaudithook(deny_network)
    started = time.monotonic()
    result = generate(config)
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name != "compose_v4" and not name.startswith("compose_v4."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(runtime.resolve() / "src"):
            raise RuntimeError(f"fragment dependency loaded outside the source tree: {name}")
        sources[str(path.relative_to(runtime.resolve()))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    worker = Path(__file__).resolve()
    sources[str(worker.relative_to(runtime.resolve()))] = hashlib.sha256(
        worker.read_bytes()
    ).hexdigest()
    result["source_sha256"] = dict(sorted(sources.items()))
    result["elapsed_seconds"] = time.monotonic() - started
    Path(config["result_path"]).write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
