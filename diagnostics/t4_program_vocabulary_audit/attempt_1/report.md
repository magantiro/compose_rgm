# Frozen T4 Program-Vocabulary Audit

Generated from commit `4ec376b36a2020f2a1cf6e81c176fadb1e282b1a` with zero oracle calls.

## Outcome

The frozen bank is address-free and context-rebound, but it is not only an
abstract grammar. It is a shared, winner-informed executable program bank that
contains 77 compiled complete
winner-route programs and 69 other programs. Static
program payloads are fully specified primitive sequences; mutation and
recombination provide runtime variation.

- Literal target values: 0
- Literal seed SMILES: 0
- Literal winner endpoint SMILES: 0
- Programs with raw integer atom operands: 0
- Opaque source groups mapped to public T4 cells: 15/15
- Programs with exact execution on at least one cell: 143/146
- Programs with exact execution on another target: 65/146
- Existing direct retrieval recovered 26 new known
  public winners across 13/15 cells.

## Interpretation

The literal payload audit supports the implementation claim that physical source
slot addresses, explicit endpoint molecules, numeric scores and target-routing
fields are absent from program payloads. The source-group hashes are nevertheless
uniquely invertible by enumerating the public 15-cell registry, and complete
winner-derived primitive routes remain in the bank. Direct known-winner recovery
shows that some entries preserve enough route specificity to reconstruct public
answers. Cross-cell binding and exact execution measure transfer applicability,
not held-out task value.

The frozen official T4 controller uses one shared bank and disables direct
retrieval (`cold_start_retrieval_candidates=0`), while mutation and recombination
remain active. These facts reduce literal replay in that run but do not erase the
winner-informed prior. The scientifically accurate description is therefore
"shared, address-free, context-bound, winner-informed executable program bank,"
not a target-specific lookup table and not a purified task-independent grammar.
