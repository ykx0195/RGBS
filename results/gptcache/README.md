# GPTCache experiment

Remote experiment directory: `/root/autodl-tmp/icde_gptcache_20261104`.

Read `实验报告.md` for scope, results, and limitations. Cache payloads are unique
provenance markers, not generated answers. These tests measure actual cache
retrieval/return under direct preload, not answer validity or query admission.

The remote directory retains the actual SQLite stores, FAISS indexes, embeddings,
inputs, isolated environment, and logs. No credentials are stored in these files.

To rerun on that server:

```sh
cd /root/autodl-tmp/icde_gptcache_20261104
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -u run_experiment.py
```

The script retains previous condition directories instead of overwriting their
databases. The encoder path and SHA-256 are fixed in the script. An independently
copied environment must also supply the model, `inputs/`, and
`frozen_embeddings/` with matching manifests.

Local preparation scripts use the original desktop result directories. They
verify Train/Test file hashes and preserve the mapping between texts and frozen
model embeddings. The local results archive excludes large SQLite, FAISS, and
embedding files, which remain available on the server.
