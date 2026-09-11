"""Prepare, durably spawn, and inspect bounded complete-option particles."""

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument(
        "--experiment",
        choices=("particles", "replacement", "continuation", "learned", "donor"),
        default="particles",
    )
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--replicate", type=int, choices=(0, 1), default=0)
    parser.add_argument("--reuse-result", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file

    if args.experiment == "donor":
        from compose_v4.experiments import pmo_donor_comparison as experiment
    elif args.experiment == "learned":
        from compose_v4.experiments import pmo_learned_proposal as experiment
    elif args.experiment == "continuation":
        from compose_v4.experiments import pmo_replacement_continuation as experiment
    elif args.experiment == "replacement":
        from compose_v4.experiments import pmo_region_replacement as experiment
    else:
        from compose_v4.experiments import pmo_option_particles as experiment
    APP, APP_NAME, KIND = experiment.APP, experiment.APP_NAME, experiment.KIND
    PREPARED, PROTOCOL = experiment.PREPARED, experiment.PROTOCOL
    load_contract, prepare = experiment.load_contract, experiment.prepare
    from compose_v4.experiments.t4_matched_pilot import _stamp

    if args.mode == "prepare":
        if args.experiment == "replacement":
            print(json.dumps(prepare(ROOT)))
            return
        if args.prior is None:
            parser.error("prepare requires the prior trajectory result")
        if args.experiment == "donor":
            print(
                json.dumps(
                    prepare(
                        ROOT, args.prior, replicate=args.replicate, reuse_result=args.reuse_result
                    )
                )
            )
            return
        if args.replicate:
            parser.error("--replicate applies only to the donor comparison")
        if args.experiment == "learned":
            if any(p is None for p in (args.model, args.training_manifest, args.training_data)):
                parser.error("learned prepare requires model, training-manifest and training-data")
            print(
                json.dumps(
                    prepare(
                        ROOT,
                        args.prior,
                        model_path=args.model,
                        manifest_path=args.training_manifest,
                        decisions_path=args.training_data,
                    )
                )
            )
            return
        print(json.dumps(prepare(ROOT, args.prior)))
        return
    if args.output is None or args.output.resolve().is_relative_to(ROOT):
        parser.error("launch/status output must be outside the serialized tree")
    import modal

    if args.mode == "launch":
        from tools.preflight import assert_synced

        preflight = assert_synced(strict=True)
        c = load_contract(ROOT)
        from modal_apps.run_process_v2_p50_app import local_image_revision

        revision = local_image_revision(expected_commit=preflight["commit"], repo_root=ROOT)
        body = {
            "contract_sha256": c["contract_sha256"],
            "image_revision": revision,
            "app_sha256": sha256_file(ROOT / APP),
        }
        task = {**body, "run_id": identity(body), "started_at": _stamp()}
        folder = args.output / task["run_id"]
        path = folder / "spawn.json"
        if path.exists():
            raise SystemExit("existing spawn receipt: inspect it, do not duplicate")
        receipt = {
            "schema_version": "option_particles_spawn_v1",
            "task": task,
            "run_id": task["run_id"],
            "status": "spawn_started",
            "prefix": f"{KIND}/{task['run_id']}",
            "volume": "compose-v4-artifacts",
        }
        publish_json(path, receipt)
        # Preserve exact source bytes as well as the source-revision identity.
        paths = sorted(
            set(body["image_revision"]["serialized_sources"])
            | {c[k]["path"] for k in ("model", "training") if k in c}
            | {
                APP,
                PROTOCOL,
                PREPARED,
                "docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md",
                "modal_apps/pmo_online_policy_app.py",
                "modal_apps/run_process_v2_p50_app.py",
            }
        )
        archive = folder / "source_snapshot.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for p in paths:
                tar.add(ROOT / p, arcname=p, recursive=False)
        publish_json(
            folder / "source_snapshot.json",
            {
                "schema_version": "development_source_archive_v1",
                "run_id": task["run_id"],
                "archive_sha256": sha256_file(archive),
                "files": {p: sha256_file(ROOT / p) for p in paths},
            },
        )
        call = modal.Function.from_name(APP_NAME, "run").spawn(task)
        receipt.update(status="spawned", call_id=call.object_id)
        publish_json(path, receipt)
        print(path)
        print(call.object_id)
        return
    receipt = json.loads(args.output.read_text())
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    except TimeoutError:
        volume = modal.Volume.from_name(receipt["volume"])
        prefix = receipt["prefix"]
        try:
            entries = {e.path.rsplit("/", 1)[-1] for e in volume.listdir(prefix)}
        except modal.exception.NotFoundError:
            print("Parent call live; waiting for first durable receipt")
            return
        if "heartbeat.json" in entries:
            print(json.loads(b"".join(volume.read_file(f"{prefix}/heartbeat.json"))))
        else:
            print("Parent call live; initializing")
    else:
        path = args.output.parent / "result.json"
        publish_json(path, result)
        print(path)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "arms",
                        "archive_metrics",
                        "attempted_options",
                        "new_oracle_calls",
                        "seconds",
                    )
                    if k in result
                }
            )
        )


if __name__ == "__main__":
    main()
