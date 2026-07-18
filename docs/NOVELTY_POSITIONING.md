# Novelty and positioning

## Central claim

**Rewrite Generator Matching (RGM) learns a continuous-time Markov generator
whose primitive events are executable, chemistry-constrained molecular graph
rewrites.** Endpoint-conditioned rewrite programs define tractable conditional
generators; Generator Matching marginalizes the endpoint and latent program to
learn a target-free generator. Because the process is supported only on a
chemistry-closed rewrite state graph, every committed intermediate state is a
complete valid, connected molecule.

The scoped novelty claim is:

> To our knowledge, RGM is the first generator-matching framework over
> executable chemistry-constrained molecular graph rewrite operators, yielding
> pathwise-valid, connected trajectories and a common process for de novo and
> source-conditional molecular generation.

This claim should remain qualified by “to our knowledge” until the final
camera-ready literature audit. It is intentionally stronger and more precise
than “we apply graph rewriting to molecules,” which would be false as a
historical claim.

## What is actually new

The contribution is the combination of four pieces:

1. **Generator object.** The model predicts CTMC firing rates for typed,
   guarded, executable chemical rewrites—not atoms, edges, denoised graphs, or
   a terminal edit script in isolation.
2. **Conditional-to-marginal construction.** Random source-to-target rewrite
   programs produce conditional generators whose endpoint/program mixture is
   learned with Generator Matching. The target and program are absent at
   inference.
3. **Pathwise constraint semantics.** Validity and connectivity are properties
   of the state graph and executor, so they hold at every accepted event rather
   than being terminal rewards, rejection filters, or soft penalties.
4. **Unified capability.** Changing the source state and legal action fiber
   turns the same generator into de novo generation, scaffold completion,
   fragment growing, or molecular optimization.

Insertions, deletions, bond changes, stochastic chemical rewriting, CTMCs, and
Generator Matching each have precedent. Their trainable synthesis above is the
claim.

## Nearest-neighbor distinctions

| Neighbor | Its principal generative object | Distinction of RGM |
|---|---|---|
| Generator Matching | A general framework for learning Markov generators from conditional probability paths | RGM supplies a chemistry-specific executable rewrite state graph, program compiler, and pathwise constraint semantics. |
| Morph | Unbalanced-optimal-transport morphing of flexible-size 3D geometric graphs | RGM is a discrete jump generator over guarded chemical graph operations; it targets executable edit trajectories and pathwise-valid complete 2D molecular states rather than a geometric OT morphing path. |
| Edit Flows / pCoMole | Flow-based insertion, deletion, or substitution processes for variable-length sequences; pCoMole adds terminal feasibility conditioning for biomolecular sequences | RGM operates on typed chemical graphs, aggregates rewrite aliases at molecular successors, and makes chemical feasibility an invariant of every event rather than chiefly a terminal constraint. |
| DDSBM | Discrete Schrödinger-bridge dynamics using node/edge state changes | RGM's events have explicit chemical operational semantics and guards, and its teacher is compiled from executable rewrite programs rather than an independent-coordinate graph corruption/interpolation process. |
| Molecular graph grammars / stochastic rewriting | Rule-based validity, compositional construction, or stochastic simulation | RGM learns a target-free time-inhomogeneous generator from endpoint-conditioned rewrite processes rather than using a fixed grammar policy or hand-specified rule propensities. |
| Fragment assembly | Select and attach members of a finite fragment vocabulary | The micro rewrite basis is universal over the supported chemistry and includes deletion, bond closure/deletion/reorder, and atom relabeling. Fragments may be sources or future macros, not the ontology of the generator. |

## Why the distinction matters in capability

The operator semantics allow hard constraints to act locally on the legal
fiber before rate normalization. This gives:

- zero probability for illegal transitions without post-hoc rejection;
- complete valid molecules at arbitrary stopping times;
- variable size and topology in one process;
- deletion and correction, so generation is not forced into monotone assembly;
- auditable trajectories whose events correspond to chemical graph operations;
- conditional generation by restricting the same action space, rather than
  defining a separate model for each task;
- a principled path to verified macro-operators that lower to the universal
  micro basis.

## Claims to avoid

- Do not claim the first use of graph rewriting, stochastic rewriting, CTMCs,
  or insert/delete edits for molecules.
- Do not claim Generator Matching itself guarantees chemical validity. The
  rewrite state space and executor provide the guarantee; GM learns rates on
  that space.
- Do not imply current 2D pathwise validity solves 3D geometric validity,
  synthesizability, or biological feasibility.
- Do not claim empirical superiority to Morph, DDSBM, or Edit Flows before
  matched benchmark comparisons.

## Primary references

- Holderrieth et al., [Generator Matching](https://proceedings.iclr.cc/paper_files/paper/2025/file/819aaee144cb40e887a4aa9e781b1547-Paper-Conference.pdf), ICLR 2025.
- Koziarski et al., [Generative Molecular Morphing for Flexible-Size Design via Unbalanced Optimal Transport](https://arxiv.org/abs/2606.07239), 2026.
- Kim et al., [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2438d634f0ed1640934d31376c110a92-Abstract-Conference.html), ICLR 2025.
- Karami et al., [pCoMole: Pre-trained Edit Flow for Biomolecular Sequence Design](https://openreview.net/forum?id=tTILzscPs4), 2026.
- Kajino, [Molecular Hypergraph Grammar](https://www.jstage.jst.go.jp/article/pjsai/JSAI2018/0/JSAI2018_3E104/_article/-char/en), 2018.
- Guo et al., [Data-Efficient Graph Grammar Learning for Molecular Generation](https://arxiv.org/abs/2203.08031), ICLR 2022.
