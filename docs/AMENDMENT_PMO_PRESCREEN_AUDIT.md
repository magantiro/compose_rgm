# Amendment: GenMol's PMO initialization is a 249,455-call uncounted prescreen

Status: **verified against GenMol's released source**, 2026-08-22.
Repository: `NVIDIA-BioNeMo/genmol` (the `NVIDIA-Digital-Bio/genmol` path 301-redirects here).

## What the released code does

`scripts/exps/pmo/get_vocab.py`, verbatim in the essential lines:

```python
df = pd.read_csv('data/zinc250k.csv')
for prop in props:                                  # all 23 PMO tasks
    if prop not in df:
        df[prop] = Oracle(prop)(df['smiles'].tolist())   # <-- 249,455 calls
        df.to_csv('data/zinc250k.csv', index=False)
...
for frag in frags:
    frag2cnt[frag] += 1
    for prop in props:
        frag2score[prop][frag] += df[prop].iloc[i]
...
for k in frag2score[prop]:
    frag2score[prop][k] /= frag2cnt[k]              # mean over containing molecules
df = df.sort_values(by='score', ascending=False).iloc[:10000]
```

and `scripts/exps/pmo/main/genmol/run.py`:

```python
def set_initial_population(self):
    df = pd.read_csv(os.path.join(ROOT_DIR, f'vocab/{self.args.oracle_name}.csv'))
    df = df.iloc[:self.args.population_size]         # population_size: 100
    self.population = list(zip(df['score'], df['frag']))
```

So, per task: the oracle is called on **all 249,455 ZINC250k molecules**, every
molecule is cut into fragments, each fragment is scored by the **mean oracle
value of the molecules containing it**, the top 10,000 fragments are written to
`vocab/{task}.csv`, and the run takes the **top 100** as its initial population.
Those 100 arrive with their scores already known and never enter `mol_buffer`,
which is what `max_oracle_calls` counts.

**249,455 uncounted oracle calls per task, against a 10,000 counted budget.**
That is 25x the entire benchmark budget, spent before counted call 1, and
5,737,465 uncounted calls across the 23 tasks.

## Two further findings from the same source

**`warmup: 1000` is COUNTED.** It does not buy extra budget. In `generate()`:

```python
if self.iter > self.args.warmup:
    smiles = self.sampler.mask_modification(smiles, gamma=self.args.gamma)
```

Every iteration calls `self.oracle(smiles)` and is charged. `warmup` only gates
whether the diffusion sampler is used at all, so GenMol's first 1,000 counted
calls are **pure fragment recombination with the model switched off**.

**Hyperparameters are switched per task by name**, in `GenMolOpt.__init__`:

```python
if args.oracle_name in {'albuterol_similarity', 'isomers_c7h8n2o2',
                        'isomers_c9h10n2o2pf2cl', 'median1', 'qed',
                        'sitagliptin_mpo', 'zaleplon_mpo'}:
    args.min_mol_size, args.max_mol_size = 10, 30
elif args.oracle_name in {'gsk3b', 'jnk3'}:
    args.min_mol_size, args.max_mol_size = 30, 80
```

Three size regimes selected by task identity. GenMol's PMO numbers are therefore
not a single generic configuration applied uniformly across the 23 tasks.

## The shipped prescreen, measured

Fragment scores from `vocab/{task}.csv`. `top100` is the mean over exactly the
slice `set_initial_population` takes.

| task | n frags | top-1 | top-100 mean | median (all 10k) | enrichment |
|---|---|---|---|---|---|
| jnk3 | 10,000 | 0.6800 | **0.4435** | 0.1200 | **+0.3235** |
| osimertinib_mpo | 10,000 | 0.8288 | **0.8035** | 0.7351 | +0.0684 |
| scaffold_hop | 10,000 | 0.5261 | **0.5046** | 0.4485 | +0.0561 |

Against GenMol's published AUC-top-10 and our clean objective-blind pilot:

| task | our AUC | our top-10 @2K | GenMol AUC | prescreen top-100 |
|---|---|---|---|---|
| jnk3 | 0.236 | 0.3690 | 0.906 | 0.4435 |
| osimertinib_mpo | 0.749 | 0.8192 | 0.876 | 0.8035 |
| scaffold_hop | 0.453 | 0.4851 | 0.628 | 0.5046 |

## What this implies, per task

These three tasks turn out to sit in three different regimes, which is why the
matched-init experiment is worth running rather than assumed:

- **jnk3 — initialization-dominated.** The prescreen lifts the fragment pool from
  a 0.12 median to a 0.4435 top-100 mean. GenMol's *starting population* is
  already above where our clean run finished at 2,000 counted calls (0.369). A
  large part of this gap is information, not search.

- **osimertinib_mpo — nearly settled by initialization alone.** The prescreen
  top-100 mean is 0.8035; GenMol's published AUC is 0.802. Our clean run already
  reaches 0.8192 top-10 without any prescreen, i.e. above their starting point.

- **scaffold_hop — search-dominated. This is the important one.** The prescreen
  buys only +0.056 over the fragment-pool median, and its top-100 mean (0.5046)
  is barely above where our clean run plateaus (0.4851). **GenMol's 0.628 is
  therefore not explained by initialization.** Their search genuinely gets there
  from a weakly enriched start. Under our own decision rule, scaffold_hop is
  where search — not information — is the binding constraint, and so it is the
  task where a stronger controller has to earn its place.

## Caveat that the matched arm exists to remove

The prescreen numbers above are **fragment** scores: the mean oracle value over
molecules containing a fragment, not the score of any molecule. They are not
directly comparable to a molecule-level AUC. The matched-init arm resolves this
by scoring actual molecules.

## Naming: "oracle-prescreened", not "GenMol-matched"

This arm is **oracle-prescreened / matched-information COMPOSE**. It is not the
GenMol protocol and must never be labelled as such. GenMol scores ZINC250k,
decomposes those molecules into fragments, propagates the molecule scores into a
task-specific fragment vocabulary, and initializes from the top fragments. We
score the same 249,455 molecules with the same task oracle and initialize from
the top *molecules*.

What is matched is the **information**: 249,455 uncounted task-oracle evaluations
before search begins. What differs is the representation each method consumes it
through, and that difference is deliberate. The claim the paper can make is that
both methods receive the same kind and order of task-oracle prescreen
information, each through its native representation; the claim it cannot make is
that the protocols are identical.

## What our matched arm does

`modal_apps/pmo_matched_init_app.py`. Same information budget, COMPOSE's own
representation: score ZINC250k with the task oracle outside the counted budget,
take the top 100 **molecules** as the initial state bank, change nothing else.
Frozen R_theta, `N_LINEAGE`, `PER_ROUND`, `APPLY_CAP`, the 400-state cap, the
seeds, and `OracleMeter` PER_MOLECULE counting are all identical to the clean
pilot, so `clean@500` and `matched@500` differ in initialization alone.

We deliberately do **not** adopt GenMol's fragment-attachment representation.
Their unit is a fragment because their generator recombines fragments; ours is a
molecule because our kernel edits molecules. Matching the *information regime* is
the point; deforming COMPOSE into their action space would confound the
comparison rather than clean it.

Initial scores are seeded with `OracleMeter.prime()`, which records a value
without charging the budget, exactly as GenMol's vocab CSV supplies scores it
never re-derives. `prime()` carries a standing reporting obligation: **every use
must be reported together with the uncounted evaluation count that produced it**
(`uncounted_prescreen_calls` and `primed` are written into every run record).
It is not a budget escape hatch, and an arm that primes values it never paid for
anywhere is inflated rather than matched.

## What the three top-1 coincidences are, and are not

Scoring ZINC250k independently reproduces GenMol's top-1 vocabulary score exactly
on all three tasks: jnk3 0.6800, osimertinib_mpo 0.8288, scaffold_hop 0.5261.

That is a **corpus-scale sanity check, not an oracle-parity result.** Matching
three extrema is not logically stronger than exact molecule-by-molecule agreement
on a fixed panel, and it would remain consistent with disagreement anywhere below
the maximum. Establishing parity across the corpus would require verifying
identities and rankings throughout it, which has not been done.

The formal parity claim stays where it was earned: the Gate A and featurization
tests, at 0.000e+00 on their fixed panels. Cite those. The coincidences above are
worth one sentence as corroboration and no more.

## Counting hygiene

The corpus is called **ZINC250k**, but the number of molecules actually scored is
**249,455**, and every one of them scored uniquely -- no rows were lost to parse
failure or to canonicalization collisions. Tables and prose must carry the exact
figure for any *count* claim; "ZINC250k" is acceptable only as the dataset's name.
Every bank artifact records `uncounted_calls` exactly, and that field, not a
rounded restatement of it, is what any reported number must come from.

## Reporting consequence

Any table putting COMPOSE next to GenMol on PMO must state the initialization
convention for both arms. GenMol's published numbers are obtained with 249,455
uncounted task-oracle calls per task and per-task size hyperparameters. That is
a property of the baseline, not an accusation: it is simply not the same
information regime as an objective-blind start, and a comparison that omits it
is not measuring search.
