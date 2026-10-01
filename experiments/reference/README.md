# Frozen molecular reference

`model.json` identifies the NLL-trained molecular edit law used by the task
examples. Its selected tensor state is checked independently of the checkpoint
file hash. `local_assets/fragments/catalog.json` contains the ring-template data required to
reconstruct the model. The catalog fingerprint is checked after loading.

The model scores legal primitive edits. Canonical successor probabilities sum
the mass of edits with the same molecular product. Fragment generation uses
these preferences when sampling or selecting candidates. QED editing uses the
canonical successor kernel and a separate future-value model. PMO and T4 use
the same frozen reference to score replayed programs within their task
controllers. Their structured proposal laws and online feedback models remain
separate from the reference.

The canonical-successor NLL is implemented by
`factorized_successor_identity_loss` in
`src/compose_v4/experiments/factorized_successor_training.py`. The checkpoint
manifest fixes the selected trained state for inference. The original compiled
training corpus is not distributed here, so this checkout does not support
retraining that checkpoint from the raw molecular sources.

From the repository root, after fetching the LFS weights, run:

```bash
PYTHONPATH=src python tools/verify_shared_reference.py
```

This command loads the weights on CPU, checks the catalog, and verifies that
the fragment, PMO, and T4 example configurations name the same model. It makes
no oracle or docking call. QED's asset check is
`PYTHONPATH=src python tools/verify_qed_assets.py`.

The submitted benchmark numbers retain their original run identities. The
example configurations in this release define the shared-reference method and
do not reassign those numbers to different model weights.
