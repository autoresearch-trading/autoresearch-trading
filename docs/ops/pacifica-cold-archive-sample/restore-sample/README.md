# Pacifica Cold Archive Restore Sample

No R2 writes or deletes were executed. This is a local restore-sampling diagnostic.

Manifest: `/Users/diego/Dev/non-toxic/autoresearch-trading/docs/ops/pacifica-cold-archive-sample/manifest.csv`
Raw root: `/Users/diego/Dev/non-toxic/autoresearch-trading/data/pacifica_r2_rehydrate_sample`
Sampled sources: 3
Matched sources: 3
Mismatched sources: 0
OK: `True`

This checks that ordered `raw_json` rows in the cold parquet archive reconstruct the original source JSONL line sequence for sampled raw chunks.
