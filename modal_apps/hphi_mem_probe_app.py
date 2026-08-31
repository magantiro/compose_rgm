"""Measure TRUE peak memory of one hphi source run, to right-size MEM_MIB.

MEM_MIB=4.5GiB in hphi_h40head_ab_app.py is an unexamined constant with no
comment justifying it. Modal bills the GREATER of requested and actual, so at
1 CPU + 4.5 GiB memory is 43% of the container cost -- on the official 800x20
that is ~$22 of a ~$52 bill, possibly for memory nothing ever touches.

This calls the SAME run_source via get_raw_f(), so the measured path is exactly
production. It changes nothing in that file. Generous 8 GiB request so the
probe itself can never OOM and truncate the measurement.

Reports VmHWM (kernel-tracked high-water mark) at each stage, so a one-off
load spike is distinguishable from steady growth across slots.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import modal

from pathlib import Path

from modal_apps.hphi_h40head_ab_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
    run_source,
)
from modal_apps.hphi_h40head_ab_app import image as _app_image

ROOT = Path(__file__).resolve().parents[1]

# modal_apps is a NAMESPACE package on the container (no __init__.py); the base
# image ships only run_process_v2_p50_app.py into REMOTE_ROOT/modal_apps. The
# entrypoint file is automounted flat to /root/, so importing the h40head module
# fails unless it is also placed inside that namespace dir. Adding it here keeps
# the probe measuring the REAL run_source instead of a reimplementation.
image = _app_image.add_local_file(
    ROOT / "modal_apps/hphi_h40head_ab_app.py",
    str(REMOTE_ROOT / "modal_apps/hphi_h40head_ab_app.py"),
    copy=True,
)

app = modal.App("hphi-mem-probe")


def _kb(field: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field):
                return int(line.split()[1])
    return -1


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(task: dict[str, Any]) -> dict[str, Any]:
    trace: list[tuple[float, int]] = []
    stop = threading.Event()

    def sample() -> None:
        t0 = time.time()
        while not stop.is_set():
            trace.append((round(time.time() - t0, 1), _kb("VmRSS:")))
            time.sleep(0.5)

    th = threading.Thread(target=sample, daemon=True)
    baseline = _kb("VmRSS:")
    th.start()
    t0 = time.time()
    try:
        run_source.get_raw_f()(task)
    finally:
        stop.set()
        th.join(timeout=5)
    elapsed = time.time() - t0
    peak_rss = max((r for _, r in trace), default=-1)
    return {
        "baseline_mib": round(baseline / 1024, 1),
        "vmhwm_mib": round(_kb("VmHWM:") / 1024, 1),
        "peak_sampled_rss_mib": round(peak_rss / 1024, 1),
        "final_rss_mib": round(_kb("VmRSS:") / 1024, 1),
        "seconds": round(elapsed, 1),
        "n_samples": len(trace),
        "trace_max_at_s": max(trace, key=lambda x: x[1])[0] if trace else None,
        # every 30th sample (~15s apart): enough to see whether growth
        # plateaus or keeps climbing across slots, without 2k rows.
        "trace_mib": [(t, round(r / 1024)) for t, r in trace[::30]],
    }


@app.local_entrypoint()
def main(index: int = 0, k_end: int = 2) -> None:
    root = ROOT
    vs = [x.strip() for x in
          (root / "data/jin/hphi_valid_128.txt").read_text().split("\n")
          if x.strip()]
    task = {"index": int(index), "source": vs[int(index)],
            "stratum": "memprobe", "arm": "restart", "horizon": 40,
            "out_dir": "hphi_memprobe", "head_dir": "hphi_v2",
            "budget_max": 24, "k_start": 0, "k_end": int(k_end)}
    print(f"probing source {index} for {k_end} slot(s), 8 GiB ceiling, 1 CPU")
    r = probe.remote(task)
    for k, v in r.items():
        print(f"  {k}: {v}")
    hwm = r["vmhwm_mib"]
    print(f"\nPEAK = {hwm:.0f} MiB")
    for mult, name in ((1.25, "1.25x"), (1.5, "1.5x")):
        print(f"  {name} headroom -> request {int(hwm * mult / 128 + 1) * 128} MiB")
