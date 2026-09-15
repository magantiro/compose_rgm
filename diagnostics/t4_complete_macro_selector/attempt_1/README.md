# T4 complete-macro selector, attempt 1

Status: frozen zero-oracle scientific gate passed.

The selector reranks exactly the immutable 128 learned candidates per held
source. It does not change generator support. All three grouped-source folds
fit with 50, 51 and 53 positive routes and 320 cross-fitted hard negatives per
fold.

At K=8, 16 and 32, source-balanced granular component coverage improved from
0.180126, 0.251341 and 0.346326 in the raw learned order to 0.320572, 0.401935
and 0.485553. Exact patch recall improved from zero to 0.004444 at K=8 and from
zero to 0.009206 at K=16; both arms recovered the same two held patches by
K=32. Exact endpoint and strict radius-two transformation recovery remained
zero at every cutoff.

Every selected endpoint and complete-macro signature is unique at each cutoff.
Exact realization precision is 1.0. The selector inherits learned-pool compile
coverage 0.935217. A realizability classifier abstains because rejected
compiler attempts are not candidate rows in the immutable lock. There were no
candidate-generation, oracle, docking or Modal calls.

Authoritative files:

- `candidate_lock.json.gz`: physical
  `aab18cc4fd737d3bf5953c79a7e3332459009813626aa6ee65519516c43d77c2`,
  payload `9ad8cea49864ceb286d0caee0cf714ac5034e78e4b9d7bd6557fb8721f78112f`;
- `result.json`: physical
  `2cc879e03ead3303ebbd5134a63e6e673f19d7fb0f9b90d173d5b5140ce53b80`,
  payload `03502a8661537635762d0369fd48fc250a6e440f0a3c93da59f50d028b6967b3`;
- `fit_summary.json`: physical
  `0c8ae2df564b3cb9709fc8b8a0a4f8c4147881a237235466ec6c91cd2e67d1ee`,
  payload `7903ee734df390669a9b49650f732a88bb0e9f6ed0851f7c7af9e86354576264`.

The entire directory was reproduced byte for byte in an independent full
rerun. See `docs/T4_COMPLETE_MACRO_SELECTOR.md` for methods, full interpretation
and limitations.
