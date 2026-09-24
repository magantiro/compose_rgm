"""One PMO target per container: the frozen FiberControl controller, oracle swapped.

ARCHITECTURE. Each target is a stateful sequential campaign, so parallelism is ACROSS
targets and never within one. Each invocation gets one CPU, one input at a time, its own
volume namespace and a completely fresh controller. Two campaigns sharing a process would
share the controller's class-level state; two campaigns sharing a volume path would race on
the same mutable checkpoint. Both are prevented structurally here rather than by convention.

THE ONLY EXPERIMENTAL VARIABLE IS THE ORACLE. Every controller constant -- reward
definition, parent-offset acquisition, both value heads, the fitting threshold, temperature,
the model/random split, family and lineage floors, pool target, macro vocabulary, region
replacement and its rate, the region law, intent-versus-realized credit, endpoint dedup and
decision-only clipping -- stays exactly as frozen on Celecoxib. Nothing is tuned per target;
a per-target constant would make this benchmark engineering rather than one controller.

PROPOSAL BREADTH IS A COUNT, NOT A DURATION, and that is what makes this safe to run on
heterogeneous hardware at all. The campaign's stopping rule was `attempts OR wall_seconds`,
under which a faster container fits more proposals into the same window and searches
differently from the same seed. Containers are not speed-matched, so that would have
confounded hardware with algorithm across every target simultaneously.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
# Overridable so a new arm can deploy WITHOUT rebaking the image an in-flight campaign
# would pick up on a preemption retry.  Absent -> the historical name, unchanged.
RUN_APP = os.environ.get("PMO_FIBERCONTROL_APP") or "compose-pmo-fibercontrol-targets"
ARTIFACT_VOLUME = os.environ.get("PMO_FIBERCONTROL_VOLUME") or "compose-pmo-fibercontrol"

ASSET_DIR = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"
# Overridable so a revised runtime runs under its OWN versioned contract instead of
# re-pointing the authorization the previous runtime was granted.
CONTRACT = (os.environ.get("PMO_FIBERCONTROL_CONTRACT")
            or "configs/pmo_population_controller_v1_scored_contract_corrected.json")

#: Deterministic seed derivation. NEVER Python's `hash()`: `str` hashing is
#: PYTHONHASHSEED-salted, so the same key yields a different seed in every process and no
#: row produced that way is reproducible by anyone, including us. A fixed index table is
#: stable across processes, machines and releases.
TARGET_INDEX = {
    "celecoxib_rediscovery": 0,
    "albuterol_similarity": 1,
    "mestranol_similarity": 2,
    "thiothixene_rediscovery": 3,
    "troglitazone_rediscovery": 4,
    "median1": 5,
    "isomers_c7h8n2o2": 6,
    "perindopril_mpo": 7,
    "gsk3b": 8,
    "jnk3": 9,
    "qed": 10,
    # The remaining six drug-MPO tasks.  Appended, never renumbered: `derived_seed` is
    # `base_seed + 1000 * index`, so reusing or reordering an index would silently give
    # two different tasks the same seed or move a task already launched under one.
    # All seven resolve as exact `tdc.metadata.oracle_names` members, so none is exposed
    # to `fuzzy_search`'s 0.8-threshold fallback onto a neighbouring oracle.
    "amlodipine_mpo": 11,
    "fexofenadine_mpo": 12,
    "osimertinib_mpo": 13,
    "ranolazine_mpo": 14,
    "sitagliptin_mpo": 15,
    "zaleplon_mpo": 16,
    # The remaining deterministic PMO objectives.  Wired now so the registry stops being
    # discovered one launch failure at a time; launching stays a separate decision.
    # Chosen for what they stress: deco_hop and valsartan_smarts exercise retained
    # substructure and decoration change (the anchored-replacement lane); scaffold_hop
    # exercises scaffold remodeling (the ring-topology question); the isomer tasks force
    # an elemental/topological assembly rather than fingerprint hill-climbing, and the
    # second one exercises P/F/Cl; median1/median2 test interpolation between two
    # neighbourhoods rather than copying one target; qed is a sanity control.
    "valsartan_smarts": 17,
    "deco_hop": 18,
    "scaffold_hop": 19,
    "isomers_c9h10n2o2pf2cl": 20,
    "median2": 21,
    # drd2 joins gsk3b/jnk3 as the learned-predictor trio.  These three are the ONLY
    # asset-backed oracles in the suite (measured: the other 20 are pure RDKit and open
    # no file), and gsk3b/drd2 load their pickle LAZILY on first call from a RELATIVE
    # path, so they need the working directory pinned for the oracle's whole lifetime,
    # not just its constructor.  Registered here, launched only after that check.
    "drd2": 22,
}


def derived_seed(task: str, base_seed: int, replicate: int = 0) -> int:
    """`base_seed + index` -- stable, inspectable, and independent of process hashing."""
    if task not in TARGET_INDEX:
        raise ValueError(f"no stable seed index for {task}; add it deliberately")
    return int(base_seed) + 1000 * TARGET_INDEX[task] + replicate


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.5.31")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "uv pip install --system --no-deps 'PyTDC==1.1.15'",
        "uv pip install --system 'numpy==1.26.4' 'pandas==2.1.4' "
        "'rdkit==2023.9.6' 'requests==2.32.4' 'scikit-learn==1.2.2' "
        "'scipy==1.15.0' 'seaborn==0.13.2' 'setuptools==75.6.0' "
        "fuzzywuzzy huggingface-hub networkx packaging tqdm",
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE_ROOT / "src"), copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    # The WHOLE configs directory: load_contract reads a base contract that the scored
    # contract references, and guessing the closure file-by-file already cost one launch.
    .add_local_dir(
        ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_file(
        ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json",
        str(REMOTE_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json",
        str(REMOTE_ROOT
            / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"),
        copy=True,
    )
    # The contract VERIFIES the sha256 of each file in implementation_sha256, and one of
    # them lives outside src/. Baked so the hash chain resolves inside the container.
    .add_local_file(
        ROOT / "modal_apps/pmo_population_v1_app.py",
        str(REMOTE_ROOT / "modal_apps/pmo_population_v1_app.py"), copy=True,
    )
    .add_local_dir(ROOT / ASSET_DIR, str(REMOTE_ROOT / ASSET_DIR), copy=True)
    # Per-task prescreen initialization banks.  Baked so the prescreen arm is a per-call
    # FLAG on the same deployment rather than a second image: both arms then provably
    # share one controller build, which is what makes the comparison matched.
    .add_local_dir(
        ROOT / "diagnostics/pmo_prescreen_v1/initialization",
        str(REMOTE_ROOT / "diagnostics/pmo_prescreen_v1/initialization"), copy=True,
    )
    # Graft-seeded initialization: puts a target-blind retrieved ring-swap basin in play
    # from call 0, which is what AUC rewards. Selected per call by `graft_init`.
    .add_local_dir(
        ROOT / "diagnostics/pmo_prescreen_v1/graft_initialization",
        str(REMOTE_ROOT / "diagnostics/pmo_prescreen_v1/graft_initialization"), copy=True,
    )
    # Scaffold-deduplicated donor banks for the transplant lane. Gated zero-oracle:
    # 36->40 distinct scaffolds on thiothixene and the thioxanthene tricycle admitted,
    # consulting no deeper than oracle rank 44.
    .add_local_dir(
        ROOT / "diagnostics/pmo_prescreen_v1/reservoir_donors",
        str(REMOTE_ROOT / "diagnostics/pmo_prescreen_v1/reservoir_donors"), copy=True,
    )
    .env({
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}/scripts:{REMOTE_ROOT}",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Pinned so dict/set iteration cannot differ between containers. Nothing in the
        # controller derives a seed from hashing, and this keeps it that way by construction.
        "PYTHONHASHSEED": "0",
        # BAKED at build time.  `CONTRACT` is a module global, and the module is
        # re-imported INSIDE the container where the launcher's environment does not
        # exist -- so an override that is only exported at deploy time silently reverts
        # to the default there.  Putting the resolved value in the image env is what
        # makes the deployed runtime and its contract agree.
        "PMO_POPULATION_CONTRACT": CONTRACT,
    })
)

# ONE mounted volume, and a path namespace per target inside it. A volume cannot be chosen
# per call -- the mount is declared on the function -- so isolation is enforced by the label,
# which the launcher refuses to reuse. No two concurrent targets touch the same file.
volume = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
app = modal.App(RUN_APP)


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=8192,
    timeout=24 * 60 * 60,
    # Preemption restarts with the same input, so a retry is only useful once the campaign
    # resumes from committed state. Set deliberately with that dependency in mind.
    retries=modal.Retries(max_retries=3),
    # ONE campaign per container. A campaign is stateful and must process its rounds
    # serially; two inputs in one container would interleave two campaigns over shared
    # module-level controller state.
    single_use_containers=True,
    scaledown_window=60,
    volumes={str(ARTIFACT_ROOT): volume},
)
def run_target(spec: dict) -> dict:
    """One task, one fresh controller, one volume namespace."""
    task = str(spec["task"])
    budget = int(spec["budget"])
    rounds = int(spec["rounds"])
    queries = int(spec.get("queries", 16))
    seed = int(spec["seed"])
    label = str(spec["label"])

    out = ARTIFACT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CANARY_OUT": str(out),
        "CANARY_SEED": str(seed),
        "CANARY_SMILES_CACHE": str(spec.get("smiles_cache", 512)),
        "CANARY_PROPOSAL_WALL": str(spec.get("proposal_wall", 1e9)),
        **({"PMO_PRESCREEN_INIT": str(
            REMOTE_ROOT / "diagnostics/pmo_prescreen_v1"
            / ("graft_initialization" if spec.get("graft_init") else "initialization"))}
           if spec.get("prescreen") else {}),
        # Per-call, and read per call on the other side, so one deployment serves the
        # donor and no-donor arms and they provably share a controller build.
        **({"PMO_TRANSPLANT_SHARE": str(spec["transplant_share"])}
           if spec.get("transplant_share") else {}),
        # Per-call so the parent-weighting arm and its control provably share ONE
        # controller build; absent -> "uniform" -> byte-identical to the frozen runs.
        **({"PMO_PARENT_WEIGHTING": str(spec["parent_weighting"])}
           if spec.get("parent_weighting") else {}),
        # The dedicated transplant lane and the donor bank it draws from. Both absent ->
        # byte-identical to every run so far.
        **({"PMO_TRANSPLANT_LANE": "1"} if spec.get("transplant_lane") else {}),
        **({"PMO_DONOR_RESERVOIR_FILE": str(
            REMOTE_ROOT / "diagnostics/pmo_prescreen_v1/reservoir_donors" / f"{task}.json")}
           if spec.get("donor_reservoir") else {}),
        # NOTE: PMO_POPULATION_CONTRACT is deliberately NOT set here.  `**os.environ`
        # already carries the value baked into the image at build time, and re-deriving
        # it from CONTRACT would re-evaluate that global INSIDE the container, where the
        # launcher's override does not exist -- silently replacing the deployed
        # contract with the default.  That cost two launches to find.
    }
    completed = subprocess.run(
        [
            "python", "-u",
            str(REMOTE_ROOT / "scripts/pmo_reward_adaptive_canary.py"),
            str(budget), str(rounds), str(queries), task,
        ],
        cwd=str(REMOTE_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "stdout.log").write_text(completed.stdout[-500_000:])
    (out / "stderr.log").write_text(completed.stderr[-500_000:])
    provenance = {
        "task": task,
        "label": label,
        "base_seed": spec.get("base_seed"),
        "derived_seed": seed,
        "replicate": spec.get("replicate", 0),
        "budget": budget,
        "git_commit": spec.get("git_commit"),
        "modal_call_id": os.environ.get("MODAL_TASK_ID"),
        "returncode": completed.returncode,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))
    volume.commit()
    return provenance
