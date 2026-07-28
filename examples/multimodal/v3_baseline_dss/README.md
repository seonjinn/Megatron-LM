# DSS data recipes and migration artifacts

This directory contains the DSS-backed conversions requested in
`data_prep.md`, plus the provenance needed to repeat or extend the conversion.
Each entry point is self-contained apart from its DSS dataset references.

## Artifact index

- [`legacy_to_dss_mapping.csv`](legacy_to_dss_mapping.csv): editable source
  registry for historical actions and path mappings. It was seeded from the
  original
  2,373-row
  `v14_16k_49k_omni_262k_actions.csv`.
- [`legacy_to_dss_mapping_normalized.csv`](legacy_to_dss_mapping_normalized.csv):
  generated, one-candidate-per-row form of the mapping registry. Use this for
  filtering and review; regenerate it with `dss_mapping_tool.py`.
- [`recipe_pairs.csv`](recipe_pairs.csv): closest legacy and DSS entry-point
  pairs. Start here when converting a derivative recipe.
- [`path_prefix_aliases.csv`](path_prefix_aliases.csv): explicitly verified
  mount-prefix aliases used during lookup; the tool does not perform fuzzy
  path matching.
- [`dss_mapping_tool.py`](dss_mapping_tool.py): read-only mapping lookup,
  normalization, and recursive YAML audit utility.
- [`DSS_OPERATIONS_LOG.csv`](DSS_OPERATIONS_LOG.csv): append-only upload,
  cache, permission, and replacement history, including job IDs and observed
  status.
- [`DSS_MIGRATION_RUNBOOK.md`](DSS_MIGRATION_RUNBOOK.md): procedure for
  expanding a legacy graph, choosing and verifying mappings, uploading,
  caching, rewriting YAML, and validating.
- [`DSS_DATASET_MANIFEST.md`](DSS_DATASET_MANIFEST.md): exhaustive
  human-readable mapping and cache snapshot generated from the active graphs
  on 2026-07-23. Its cache columns are historical; consult the operations log
  for later changes.
- [`commit_missing_dss_on_nrt.sh`](commit_missing_dss_on_nrt.sh): historical
  helper used for the five post-mapping NRT uploads. Those uploads completed;
  do not rerun it blindly.

## Converting a derivative recipe

1. Find the closest legacy/DSS pair in `recipe_pairs.csv`.
2. Diff the derivative against that legacy baseline.
3. Copy the paired DSS recipe and preserve its unchanged DSS branches.
4. Run `dss_mapping_tool.py audit` on the derivative to inventory changed and
   new filesystem paths.
5. Review every `ambiguous_exact`, `ancestor_candidate`, and `unmapped` row.
   The tool intentionally does not rewrite YAML or choose among candidates.
6. Verify exact DSS snapshots and local caches, update the copied YAML, and run
   the validation checklist in `DSS_MIGRATION_RUNBOOK.md`.

Quick checks:

```bash
cd examples/multimodal/v3_baseline_dss

python dss_mapping_tool.py validate
python dss_mapping_tool.py normalize --check

python dss_mapping_tool.py lookup /exact/legacy/path \
  --recipe-key v14_vlm_16k \
  --role primary \
  --format json

python dss_mapping_tool.py audit /path/to/derivative.yaml \
  --recipe-key v14_vlm_16k \
  --output /tmp/derivative_dss_audit.csv
```

An audit exits with status 2 when human review is required. This is expected
for new data and for historical paths that were relocated before upload.
Run the tool in a Megatron-LM environment with PyYAML installed.

## Entry points

- `sft_combined_omni_16k_vlm_only_webbrowse_dss.yaml`: current V14 VLM-only
  16K web-browsing recipe, including the filtered/shuffled video updates.
- `sft_combined_omni_16k_vlm_only_webbrowse_no_ultra_text_dss.yaml`: the same
  recipe without `ultra_txt_materialized_0407_median_le_16384.yaml`.
- `../v3_baseline/sft_combined_omni_16k_vlm_only_webbrowse_dss_no_ultra.yaml`:
  companion no-ultra entry point using the current V1 DSS recipe snapshot and
  the standard `v3_baseline` directory layout.
- `sft_49k_video_dss.yaml`: V14 49K video additions.
- `mmlongbench_ultra_long_filtered_best_dss.yaml`: filtered MMLongBench
  ultra-long mix.

## Provenance

Leaf dataset and media mappings come from
`legacy_to_dss_mapping.csv`. The existing colleague-authored 16K DSS snapshot
was retained for unchanged dependencies, and the three updated 16K video
children were regenerated from the current local legacy snapshot.

The three NRT-only 49K child YAMLs were copied to the shared filesystem and
used to correct the exact active leaves and weights. The latest
`videomme_qa_0206.yaml` also revealed three leaves added after the mapping CSV
and original DSS upload were produced:

- `caprl_video_178k_dense_temporal_captions.jsonl`
- `internvid_dense_temporal_captions.jsonl`
- `hdvila_hopchain_qa_520k.jsonl`

All three primary leaves and the two new media roots were uploaded on
2026-07-23. The remaining media caches completed on 2026-07-24, and all three
leaves are now active in `sft_49k_video_dss/videomme_qa_0206.yaml`. Exact
references and job IDs are in `DSS_OPERATIONS_LOG.csv`.

## Final migration status as observed on 2026-07-24

- Of the 237 active references that were initially missing from the local
  cache, all 237 are now cached.
- The DSS-team restart successfully cached the original
  `nano_omni_262k_mmlongbench-cc-batch3-seed@v0`. The active recipe keeps that
  reference; the user-owned `-matthieul` upload was a fallback and is not used.
- All five post-mapping 49K datasets exist on DSS, are cached locally, and the
  three corresponding primary leaves are active.
- A post-migration media audit confirmed that the image-safety JSONL needs an
  explicit `filesystem+dss://mm_safety@v0` auxiliary source. The standard and
  no-ultra 16K leaves now include it, and all 22,560 referenced image basenames
  were found in the cache. The existing BenchFit and YT1B auxiliary mappings
  were also checked against 652,731 image and 968,697 video references
  respectively; no additional upload or cache job was required.
- The 16K recipes contain 138 inherited references that resolve from the
  prebuilt local cache but are not visible through exact live DSS lookup. They
  are locally usable but not portable to a fresh cluster.

## Final graph validation

All active YAML graphs parse and resolve locally. Current expanded-graph leaf
counts are 848 for 16K, 769 for 16K without ultra text, 74 for 49K, and 19 for
MMLongBench.
Every active DSS cache root and referenced subpath is present under
`/home/svc-dss/cache/nemotron`.
