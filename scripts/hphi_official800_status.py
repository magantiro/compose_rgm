"""Read-only progress probe: how many official records have landed."""
from __future__ import annotations
import subprocess, sys

VOL = "compose-v4-artifacts"
REMOTE = "editing_v2/r_theta_run"


def count(sub: str) -> int:
    r = subprocess.run(["modal", "volume", "ls", VOL, f"{REMOTE}/{sub}"],
                       capture_output=True, text=True)
    return sum(1 for line in r.stdout.splitlines() if line.strip().endswith(".json"))


if __name__ == "__main__":
    for sub in (sys.argv[1:] or ["hphi_official800_k8", "hphi_official_parity_check"]):
        print(f"{sub}: {count(sub)} records")
