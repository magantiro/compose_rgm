# Lipid reaction registry

This directory is the machine-readable gate between literature chemistry and
virtual lipid enumeration.  The schema is intentionally stricter than an RDKit
reaction file: a transform cannot become `qualified_for_enumeration` until it
has an atom-mapped reaction SMARTS, exact source locator, reactant-handle and
incompatibility rules, selectivity policy, known-positive product, known-negative
case, executable test manifest, and immutable implementation hash.

Do not populate reaction SMARTS by translating a reaction-family name from
memory.  Extract each transform from the cited scheme or supplement, reproduce
the reported products, and retain a `literature_spec` status until those checks
pass.  Enumerators must refuse any entry below `qualified_for_enumeration`.

This registry generates a structural training corpus.  It does not change
COMPOSE into a fragment generator: COMPOSE still learns stochastic rewrites on
complete, connected molecular graphs after the products have been constructed
and audited.
