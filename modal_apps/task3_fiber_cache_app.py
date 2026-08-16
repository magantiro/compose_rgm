"""Persist R_theta successor fibers once, so acquisition ideas can be tried for free.

WHY THIS EXISTS NOW AND NOT BEFORE
----------------------------------
Inside a single run, caching fibers was worthless: every arm recorded
`expansions == steps`, so no start was ever revisited. BETWEEN development
iterations the opposite holds -- the same fixture states recur every time a new
acquisition is tried -- so the cache finally has something to hit. It is the
difference between a five-minute offline smoke and a ten-minute charged run per
idea.

EXACT FIBER EQUALITY IS THE GATE, AND IT INCLUDES PROBABILITIES
---------------------------------------------------------------
A cache that returned a slightly different fiber would silently change the
experiment it is meant to accelerate. So a sample of states is expanded TWICE in
the same container and the two fibers must agree on the canonical successor
SMILES AND on the R_theta probabilities, exactly. If they do not, the cache is
not written and the job fails loudly. Determinism is asserted, not assumed.

Probabilities are stored even though the first acquisition does not use them:
they are what a later policy would steer on, and re-expanding 400 states to add
a column would cost more than storing it now.

CPU only. Memory is requested near measured peak rather than generously, since
it is billed on the greater of requested and actual.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image


image = (
    _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(ROOT / "artifacts" / "benchmarks"
                    / "task3_mechanism_archives_v1" / "archives.json",
                    str(REMOTE_ROOT / "archives.json"), copy=True)
)
app = modal.App("compose-v4-task3-fiber-cache")

CACHE_ROOT = "/artifacts/editing_v2/task3_fiber_cache"
#: States re-expanded to prove determinism before anything is written.
EQUALITY_SAMPLE = 3


@app.function(
    image=image,
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 60,
    max_containers=4,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def cache_fibers(spec: tuple[int, int, int]) -> dict[str, Any]:
    """Expand a slice of one seed's fixture archive and persist the fibers."""

    seed, offset, count = spec
    import resource

    from compose_v4.experiments.task3_rtheta_runtime import (
        build_r_theta,
        expand_fiber,
    )

    started = time.perf_counter()
    fixture = json.loads((REMOTE_ROOT / "archives.json").read_text())
    entry = fixture["archives"][str(seed)]
    # Deterministic slice: sorted molecules, so slices never overlap or gap.
    molecules = sorted(entry["molecules"])[offset:offset + count]
    model = build_r_theta(ARTIFACT_ROOT, REMOTE_ROOT)

    def expand(smiles: str):
        return expand_fiber(model, smiles)

    # ---- determinism gate, before anything is written --------------------
    for smiles in molecules[:EQUALITY_SAMPLE]:
        try:
            first, second = expand(smiles), expand(smiles)
        except Exception as error:  # noqa: BLE001
            print({"equality_check_skipped": smiles[:50],
                   "error": f"{type(error).__name__}"}, flush=True)
            continue
        if first != second:
            raise RuntimeError(
                f"R_theta fiber is not reproducible for {smiles!r}: a cache would "
                f"silently change the experiments it is meant to accelerate")
    print({"determinism": "exact fiber equality held, successors and "
                          "probabilities", "checked": EQUALITY_SAMPLE}, flush=True)

    cache: dict[str, list] = {}
    failures = 0
    for position, smiles in enumerate(molecules):
        try:
            cache[smiles] = expand(smiles)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print({"expansion_failed": smiles[:50],
                   "error": f"{type(error).__name__}"}, flush=True)
        if position % 10 == 0:
            print({"seed": seed, "offset": offset, "done": position,
                   "of": len(molecules),
                   "elapsed_s": round(time.perf_counter() - started, 1)}, flush=True)

    out = Path(CACHE_ROOT) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fibers_{offset:04d}.json"
    path.write_text(json.dumps({
        "seed": seed, "offset": offset,
        "archive_sha256": entry["sha256"],
        "determinism": "exact fiber equality verified in-container",
        "fibers": cache}))
    artifact_volume.commit()

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    report = {"seed": seed, "offset": offset, "cached": len(cache),
              "failed": failures,
              "median_fiber": sorted(len(v) for v in cache.values())[len(cache) // 2]
              if cache else 0,
              "seconds": time.perf_counter() - started,
              "peak_rss_mib": round(peak_mib, 1)}
    print(json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(seed: int = 100, states: int = 80, slices: int = 2) -> None:
    per = states // slices
    specs = [(seed, i * per, per) for i in range(slices)]
    core_hours = states * 12.5 * 2 / 3600
    print(json.dumps({
        "phase": "launching_fiber_cache", "seed": seed, "states": states,
        "containers": slices,
        "estimated_core_hours": round(core_hours, 3),
        "estimated_usd": round(core_hours * 0.04716
                               + (states * 12.5 / 3600) * 4 * 0.007992, 4),
    }, indent=1))
    for report in cache_fibers.map(specs):
        print(report)
