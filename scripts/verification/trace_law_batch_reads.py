"""Ground truth for what a law call actually READS off a FactorizedMarkBatch.

WHY THIS IS NOT A STATIC ANALYSIS
---------------------------------
A first pass enumerated `batch.*` accesses inside `_encode_batch` and
`_action_tables` by AST and concluded 19 of 43 fields go unread. That reasoning
is unsound as a deletion criterion for two separate reasons, and both are the
kind that would produce a silently different law rather than a crash:

  1. Those are not the only consumers. `enumerate_factorized_marked_law` also
     calls `_family_base_logits`, iterates the family/table loop, and builds the
     marks -- each of which may read the batch. A field unread by the two
     functions I happened to inspect may still be read by the law.
  2. "Unread" does not imply "not needed". A producer can be load-bearing
     through a SIDE EFFECT rather than its return value: the chemistry feature
     cache is mutated (`move_to_end`, `popitem`, `pop`) inside the same loop,
     and the ring/macro branches raise on inconsistent configuration, which is
     a correctness guard rather than dead work.

So this records, at runtime, every attribute read on the real batch during a
real law call, ATTRIBUTED TO THE READING FRAME. Frame attribution matters
because `_one_state_batch` calls `dataclasses.replace` and `.to(device)`, both
of which touch every field; counting those would report all 43 as live and hide
the answer entirely.

The output is the only defensible input to a "build less" change: the set of
fields the law demonstrably consumes, and who consumed each one.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Frames that read the batch as plumbing, not as consumers. Reads made here
# carry no information about what the law needs.
PLUMBING = {"replace", "to", "__init__", "__repr__", "__eq__", "_asdict",
            "trace_reads", "<module>", "main", "_trace_getattr"}


def main() -> None:
    import torch
    from pareto_local_runtime import build_local_model

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedMarkBatch

    torch.set_num_threads(1)
    print("building local runtime ...", flush=True)
    model = build_local_model()

    srcs = [s.strip() for s in
            (REPO / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()]

    # Record (field -> {frame: count}) only while tracing is armed.
    reads: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    armed = {"on": False}
    base = FactorizedMarkBatch.__getattribute__

    def _trace_getattr(self, name):
        if armed["on"] and not name.startswith("__"):
            f = sys._getframe(1)
            fname = f.f_code.co_name
            # Walk out of plumbing frames to find the real consumer.
            depth = 0
            while fname in PLUMBING and depth < 8:
                f = f.f_back
                if f is None:
                    break
                fname = f.f_code.co_name
                depth += 1
            if fname not in PLUMBING:
                reads[name][fname] += 1
        return base(self, name)

    FactorizedMarkBatch.__getattribute__ = _trace_getattr

    fields = sorted(f.name for f in FactorizedMarkBatch.__dataclass_fields__.values())
    n_states = 6
    armed["on"] = True
    for smi in srcs[:n_states]:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), 48)
        law = enumerate_factorized_marked_law(model, st, 0.5)
        print(f"  {smi[:44]:<44} -> {len(law.marks):>4} marks", flush=True)
    armed["on"] = False
    FactorizedMarkBatch.__getattribute__ = base

    read = {k for k in reads}
    unread = [f for f in fields if f not in read]

    print(f"\n{'='*74}\nBATCH FIELDS: {len(fields)}   "
          f"READ BY THE LAW: {len(read)}   NEVER READ: {len(unread)}\n{'='*74}")
    print("\nREAD (field -> consuming frames):")
    for f in sorted(read):
        who = ", ".join(f"{k}x{v}" for k, v in
                        sorted(reads[f].items(), key=lambda kv: -kv[1])[:4])
        print(f"  {f:<44} {who}")
    print(f"\nNEVER READ during {n_states} real law calls ({len(unread)}):")
    for f in unread:
        print(f"  {f}")

    print("\nNOTE: 'never read' is a NECESSARY condition for skipping a "
          "producer, not a sufficient one.\nA producer may still be "
          "load-bearing via cache mutation or a configuration guard.\n"
          "Side effects are audited separately.")


if __name__ == "__main__":
    main()
