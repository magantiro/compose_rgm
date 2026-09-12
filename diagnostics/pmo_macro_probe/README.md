# PMO macro-feedback development probe

Launched 2026-09-10. This is a bounded development experiment, not an official
PMO benchmark or evidence of general superiority.

- Retained JNK3 run: `7a4a09fc0268ad089692f6b70790d8254691112c030dc7496784fe078131f764`
  (four JNK3 cases only).
- Albuterol/perindopril retry: `e68821e364011d20d2fdb8aec23b0084e7fd9f74b9cb124d5be225ce4674fe32`
  (eight cases only). These twelve cases together are the active probe.
- Source: clean `pmo-macro-probe` worktree at `/private/tmp/compose-pmo-macro-probe`,
  commits `e9c22e89934b` (JNK3) and `bf89889c58f2` (other tasks). The later
  commit changes only image dependencies, task-subset launching/status and docs;
  scientific Python source and the contract are unchanged. Initial implementation:
  `ceb259429161`.
- Contract: `configs/pmo_macro_probe.json`; explanation: `docs/PMO_MACRO_PROBE.md`
  in that worktree. No branch has been pushed.
- Tasks: albuterol similarity, perindopril MPO, JNK3.
- Twelve cases: three tasks, guided/post-hoc arms, two paired seeds.
- Four shared roots, three completed-option decisions per trajectory, beam width
  two, two branches. At most 40 option attempts and 44 unique score queries per
  case, 528 score queries total. No docking, prescreen labels, or training.
- Twelve CPU containers; per-case 40-minute timeout, no automatic retries.
- Durable call IDs are in the current run's `spawn.json`. Remote artifacts live
  under `compose-v4-artifacts:pmo_macro_probe/<run>/<case>/`.

The initial run, `5d2900766c82d2735e70f8aa53aeb5a79f727f94c6a0ccbb48b1463536bdc7ca`,
failed in all twelve cases during package-version metadata collection, before
oracle initialization. Its downloaded `initialization_failures.json` and
per-case failures are retained. Zero oracle queries were spent. The correction
removed a request for standalone `guacamol` distribution metadata; all scientific
settings and oracle definitions remained unchanged.

The second run started JNK3 but failed on the other eight cases during PyTDC
import (`requests` missing), again before oracle initialization. Its
`partial_startup_receipt.json` and per-case failures are retained. The image
repair explicitly includes `requests` and `networkx`, matching the older PMO
harness, and passed a build-time construction check of both PyTDC oracles
without scoring molecules. Only those eight unscored cases were relaunched;
JNK3 was neither cancelled nor duplicated.

To inspect progress from the clean worktree:

```sh
PATH=/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH \
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
/Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_probe.py status \
  --tasks jnk3 \
  --output /Users/rmaganti/compose_rgm_git/diagnostics/pmo_macro_probe/7a4a09fc0268ad089692f6b70790d8254691112c030dc7496784fe078131f764/spawn.json

PATH=/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH \
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
/Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_probe.py status \
  --output /Users/rmaganti/compose_rgm_git/diagnostics/pmo_macro_probe/e68821e364011d20d2fdb8aec23b0084e7fd9f74b9cb124d5be225ce4674fe32/spawn.json
```

Heartbeats expose phase, root, elapsed seconds, query count and best score when
available. Final results expose short-prefix best/top-ten curves, initial versus
descendant values, realized structural changes, options, diversity and compute.
Do not infer a 10,000-query AUC from this probe or interpret a sparse JNK3 null
as a proof of unreachable activity.

Early JNK3 heartbeats showed about two completed options after 5-6 minutes
including initialization, slower than the historical T4 throughput estimate.
The 40-minute per-case timeout remains unchanged. Any timeout is incomplete,
not an optimization null or permission to extend the experiment automatically.

Verification: 12 focused tests passed before launch; the five PMO tests passed
again in 2.26 seconds after the metadata correction. Ruff and strict preflight
passed. No full repository suite was run; no controller milestone is declared
complete by this development launch.
