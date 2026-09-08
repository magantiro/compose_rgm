# Exact-slot append-system contract repair

Decision recorded before implementation, 2026-09-08. The user authorized repairing
the verified closure rejection while preserving the general local/global search.

## Identity and scope

COMPOSE outputs executable molecular rewrite marks and their complete supported
molecular successors. This repair tests whether its pendant-ring option enforces
the already declared topology contract using persistent atom identity, rather
than unrelated atom numbers from separately canonicalized SMILES. It is not a new
generator, ring catalog, training run, or claim of improved docking performance.

The development inputs are the saved exact closure states and executor receipts
from the matched PARP1 seed0, d=0.4 audit at source revision `219c1cd`. The
comparison is the historical SMILES predicate versus the repaired exact-state
predicate and the independent graph-topology diagnosis. No winner structures,
new model samples, or docking labels are used to select this repair.

Keep Q(M), the applicability-aware balanced option prior, generic availability,
R_theta, the committor, kappa=1, budgets, and the executor unchanged. Supported
graphs remain charge-preserving, non-stereochemical, broad-organic molecules with
at most 40 active atoms. Optional ring channels do not define global support.

## Smallest repair and acceptance

- Add a graph-state contract adapter to the existing macro machinery. For
  append_system, identify ring systems in exact persistent slots. Keep the
  ring-system count increase and a new system of at least six atoms disjoint
  from every pre-closure ring atom. Do not require all those atoms to have been
  born during the entire program; that would change the existing contract.
- Route both T4 product applicability and ordinary-option execution, and the
  shared continuation kernel, through the exact-state adapter. Other macro
  predicates retain their existing semantics.
- Keep the historical SMILES-only API for legacy reproducibility, explicitly
  label its identity limitation, and prohibit its use in the exact-state paths.
  Do not attempt to recover training/replay states by SMILES rematching.
- Test pendant positives, fused/spiro/small-ring/non-closure negatives, joint
  slot permutations, sparse slots, and SCAR exclusion. Exercise the production
  executor and the actual controller paths with model-free test laws.
- Evaluate all relevant saved closure products, not just the two positive
  witnesses. Report coverage and precision against the independent saved
  diagnosis, exclusions, input hashes, code identity, software, and timing.
  These are counterfactual product checks, not sampled completed programs.
- Preserve the original candidate locks and negative audit. Publish new repair
  evidence separately. No additional T4 docking or training launch is authorized
  by this repair document.

Focused regression checks are the iteration gate. Repository-wide verification
remains required before a milestone-complete claim; existing unrelated failures
must remain visible. Commit the scoped repair after inspecting its diff. Do not
push without authorization.
