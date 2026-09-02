# Stale duplicates

Kept, not deleted, so nothing is lost. Each file here is a copy of a module that
lives authoritatively elsewhere, and each was verified to be imported by nothing.

| File | Authoritative version | Why archived |
|---|---|---|
| `modal_apps_region_rewrite.py` | `src/compose_v4/control/region_rewrite.py` | A copy of the control module that had been left in `modal_apps/`. It is 32 lines behind the real module (missing `structural_features_shared` and the later feature split) and `grep` found no import of `modal_apps.region_rewrite` anywhere in the tree. Having a stale second copy of the proposal kernel next to the apps is a correctness hazard: an edit could land in the wrong one. |
