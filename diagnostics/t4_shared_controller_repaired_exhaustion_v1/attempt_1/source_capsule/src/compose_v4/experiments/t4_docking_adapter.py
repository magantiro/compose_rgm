"""Thin T4 subprocess adapter with caller-supplied, hash-bound box/protocol.

Commands/timeouts match modal_apps/genmol_t4_opt_app.py::_dock. No model, archive,
teacher endpoint, published score, or proposal selection is available here.
"""

import subprocess
from pathlib import Path


def dock_t4(smiles, tag, seed, *, box, cpu=1):
    directory = Path("/tmp") / tag
    directory.mkdir(parents=True, exist_ok=True)
    mol, ligand, output = (directory / name for name in ("l.mol", "l.pdbqt", "o.pdbqt"))
    # The durable caller forbids an ambiguous scored retry. Refuse stale files
    # rather than removing a possibly valuable existing pose.
    if any(p.exists() for p in (mol, ligand, output)):
        raise RuntimeError("docking scratch collision; refusing an unaccounted retry")
    try:
        subprocess.run(
            ["obabel", f"-:{smiles}", "--gen3D", "-O", str(mol)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        subprocess.run(
            ["obabel", str(mol), "-O", str(ligand)], capture_output=True, timeout=60, check=True
        )
        (cx, cy, cz), (sx, sy, sz) = box["coordinates"]
        subprocess.run(
            [
                "/opt/dock/qvina02",
                "--receptor",
                box["receptor"],
                "--ligand",
                str(ligand),
                "--out",
                str(output),
                "--center_x",
                str(cx),
                "--center_y",
                str(cy),
                "--center_z",
                str(cz),
                "--size_x",
                str(sx),
                "--size_y",
                str(sy),
                "--size_z",
                str(sz),
                "--cpu",
                str(int(cpu)),
                "--num_modes",
                "10",
                "--exhaustiveness",
                "1",
                "--seed",
                str(int(seed)),
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )
        with output.open() as handle:
            for line in handle:
                if line.startswith("REMARK VINA RESULT"):
                    return float(line.split()[3])
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return None
