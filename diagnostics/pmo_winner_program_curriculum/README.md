# Perindopril winner-program result

The frozen answer-known curriculum scored 15 charged molecules and obtained:

- official 10,000-query top-ten AUC: `0.8084043888960989`;
- final top-ten mean: `0.8090111472565413`;
- best score: `0.8106434833777776`;
- margin over IVG no-prescreen `0.645`: `+0.1634043888960989`;
- margin over IVG prescreen `0.753`: `+0.0554043888960989`.

Eleven of eleven complete programs replayed from four charged roots. All 15
queries are present in the immutable started/result receipt pairs. The primary
artifact is `result.json`, payload SHA-256
`cd01e59850dd7404bb162d9f51c64ddfc7dca14ebba99affae2f86d73d8ed19a`.

This is winner-informed Perindopril development evidence, not a held-out or
general PMO claim. Nine distinct endpoints share the same metric value and are
a plateau family, not independent objective improvements.

Reproduce the receipt reduction without rescoring:

```bash
PYTHONPATH=src /private/tmp/compose-pmo-tdc-env/bin/python \
  tools/pmo_winner_program_curriculum.py run \
  --output /Users/rmaganti/compose_rgm_git/diagnostics/pmo_winner_program_curriculum
```
