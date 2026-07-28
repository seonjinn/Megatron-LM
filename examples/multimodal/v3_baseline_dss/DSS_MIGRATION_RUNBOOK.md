# Legacy recipe to DSS conversion runbook

This runbook captures the procedure and failure modes learned while converting
the V14 16K, V14 49K, and MMLongBench recipes in this directory. It is intended
to make a later conversion auditable and repeatable rather than dependent on
shell history or a pre-existing cache.

## Sources of truth

- `recipe_pairs.csv` identifies the closest legacy and converted DSS entry
  points. It is the first place to look for a derivative conversion.
- `legacy_to_dss_mapping.csv` is the historical action and path-mapping
  registry. It began as the 2,373-row
  `v14_16k_49k_omni_262k_actions.csv` supplied with the original migration.
  Add a row when a new legacy path has a stable DSS replacement.
- `legacy_to_dss_mapping_normalized.csv` is generated from the historical
  registry with one DSS candidate per row. Do not edit it directly.
- `path_prefix_aliases.csv` records the mount-prefix equivalences explicitly
  verified for this migration. Alias application is exact and prefix-based;
  no basename or fuzzy dataset matching is performed.
- `dss_mapping_tool.py` validates and normalizes the registry, looks up exact
  paths, and recursively audits a legacy YAML graph without modifying it.
- `DSS_OPERATIONS_LOG.csv` is append-only operational history. Record uploads,
  cache jobs, permission failures, replacement datasets, and their observed
  status. A later status observation should be a new row with a new timestamp.
- `DSS_DATASET_MANIFEST.md` is the human-readable expansion of the active YAML
  graphs as observed on 2026-07-23. Its embedded cache columns are a dated
  snapshot, not live state.
- The YAML files are the source of truth for what training actually consumes.
  A completed upload or cache job has no effect until its DSS reference is
  present in an active YAML branch.

Keep static mapping decisions separate from changing upload/cache status. This
prevents a successful local cache from being mistaken for a live DSS dataset,
or a historical `upload_new` action from being mistaken for a pending upload.

## Preferred workflow for a derivative

Do not rebuild a derivative from the mapping registry alone. Historical data
was sometimes relocated between a user workspace and a canonical upload root,
so an exact legacy path may intentionally have no CSV row.

Instead:

1. Select the closest row in `recipe_pairs.csv`.
2. Diff the derivative against the paired legacy baseline.
3. Copy the paired DSS entry point and its local include snapshot.
4. Preserve unchanged DSS branches from the paired conversion.
5. Audit the derivative and focus mapping work on added or changed paths.
6. Treat every ambiguous, ancestor-only, or unmapped result as a review item.
7. Validate the completed DSS graph against the derivative, not merely against
   the older baseline.

This uses the known-good paired conversion for unchanged content while still
making all derivative-specific decisions explicit.

## Status model

Classify every exact `DATASET_NAME@SNAPSHOT_NAME` reference along two independent
axes:

| Live DSS lookup | Local cache path | Meaning |
|---|---|---|
| present | present | Portable and immediately usable on this cluster |
| present | absent | Submit a cache job for this cluster |
| absent | present | Locally usable legacy cache; not portable to a fresh cluster |
| absent | absent | Find a different mapping or upload the source |

Use an exact lookup to test DSS:

```bash
NVDATASET_GROUPID=omni_vision \
nvdataset info DATASET_NAME \
  --snapshot-name SNAPSHOT_NAME \
  --output json
```

Do not infer that a dataset is absent merely because it did not appear in one
`nvdataset list` response. Conversely, a local directory only proves local
availability. During this migration, 138 older references from Tyler's recipe
were absent from exact live lookup but had working entries in the prebuilt
`/home/svc-dss/cache/nemotron` cache.

## Conversion workflow

### 1. Freeze and expand the legacy graph

Start from the exact requested entry point and recursively resolve every YAML
include. Preserve:

- split and blend structure;
- list order;
- repetitions and weights;
- `subflavors`, including the cook;
- primary `path` values;
- every auxiliary path, especially `aux.media_source`.

Record which leaf YAML contributed each primary and auxiliary reference. Do not
convert only the top-level YAML: most filesystem paths occur in included files.

### 2. Inventory filesystem references

Treat primary data and media roots as separate mapping records. A JSONL primary
may have a media tree owned and uploaded independently. Normalize obvious path
aliases only after verifying that they identify the same content.

Search the mapping registry by exact `source_path` first:

```bash
rg -n -F '/exact/legacy/path' legacy_to_dss_mapping.csv
```

Prefer the tool when the path comes from a YAML:

```bash
python dss_mapping_tool.py lookup /exact/legacy/path \
  --recipe-key v14_vlm_16k \
  --role primary \
  --format json
```

The tool applies only aliases declared in `path_prefix_aliases.csv`. If a new
cluster exposes another verified mount prefix, record it there with scope and
evidence before using it. Do not add aliases merely because two paths have the
same basename.

Interpret the historical `action` column as provenance:

- `reuse_dss_exact`: reuse was verified against an existing DSS dataset.
- `upload_new`: the source was uploaded when the original mapping was made; it
  does not mean that an upload is still pending.
- `upload_new_post_mapping`: the source was discovered and uploaded after the
  original CSV snapshot.
- `verified_previous_v14`: the reference was retained from the earlier V14 DSS
  conversion.

If no CSV row exists, check the colleague-authored DSS recipe and its exact
cache entry. Mark such a decision explicitly as inherited rather than claiming
an exact CSV match.

### 3. Verify the candidate mapping

For every candidate:

1. Verify the exact dataset and snapshot using `nvdataset info`.
2. Check the exact local path:
   `/home/svc-dss/cache/nemotron/DATASET_NAME/SNAPSHOT_NAME`.
3. Inspect the DSS subpath, if any. The dataset root matching is insufficient
   when the recipe needs a nested directory or JSONL.
4. Preserve whether the URL is a primary `dss://` path or an auxiliary
   `filesystem+dss://` media path.
5. Record uncertainty rather than selecting a fuzzy name match.

Generate a recursive audit report with:

```bash
cd examples/multimodal/v3_baseline_dss

python dss_mapping_tool.py audit /path/to/legacy_or_derivative.yaml \
  --recipe-key v14_vlm_16k \
  --output /tmp/legacy_to_dss_audit.csv
```

Use `--include-search-root` when an absolute include was copied locally and its
original path is unavailable. Exit status 2 means the CSV contains review
items; it is not a tool crash.

Interpret audit rows as follows:

| Status | Meaning |
|---|---|
| `include_resolved` | A YAML include was found and recursively inspected |
| `already_dss` | The input already contains a DSS reference |
| `mapped_exact` | One exact candidate remains after recipe, role, and alias filtering |
| `ambiguous_exact` | Multiple exact candidates remain; compare the paired baseline |
| `ancestor_candidate` | Only a parent path matched; DSS subpaths are not inferred |
| `unmapped` | No registry candidate exists; compare the paired baseline or upload/remap |
| `unresolved_include` | The graph is incomplete; copy the include or supply a search root |

The tool audits mappings only. It does not establish that a candidate is still
live on DSS or cached locally; perform the checks in the status model
separately.

As a regression check, the 2026-07-24 audit of the converted 49K graph produced
4 resolved includes and 142 `already_dss` rows with no blockers. Auditing the
exact legacy 16K graph produced 833 exact mappings plus 47 ambiguous, 12
ancestor-only, and 719 unmapped rows. Those review rows are expected because
many sources were relocated before Tyler's upload, and they are the reason the
paired baseline—not blind CSV substitution—is the preferred derivative
workflow.

### 4. Upload a genuinely missing source from NRT

For a directory:

```bash
export NVDATASET_GROUPID=omni_vision
export EDT_TEMP_UPLOAD=/lustre/fsw/portfolios/llmservice/users/$USER/edt_temp_upload

edt commit \
  /absolute/nrt/source/path \
  DATASET_NAME@v0 \
  --src-cluster-name oci-nrt-cs-001 \
  -y
```

For a standalone JSONL, ensure its Energon index exists first:

```bash
test -f /path/data.jsonl.idx || energon prepare /path/data.jsonl
```

Then use the same `edt commit` command. Directory uploads do not require
Energon merely because `edt` imports the Energon package.

Dataset ownership controls version creation. Read-only sharing permits lookup
and caching but not adding `v1`. If an owner is unavailable, commit the same
source under a new unique dataset name at `v0`, then update the mapping and
recipe after upload and cache validation. Do not launch both uploads: only use
the fallback when the existing-name request fails before a job is submitted.

List and inspect upload jobs with:

```bash
NVDATASET_GROUPID=omni_vision \
nvdataset job list DATASET_NAME \
  --type FILESYSTEM_UPLOAD \
  --output json

NVDATASET_GROUPID=omni_vision \
nvdataset job get DATASET_NAME JOB_ID \
  --output json
```

### 5. Cache on the destination cluster

Use explicit environment values. A value inherited from
`~/.config/shell/secrets.env` can silently direct the job to a different cache.

```bash
NVDATASET_GROUPID=omni_vision \
NVDATASET_CACHE_DIR=/home/svc-dss/cache/nemotron \
edt cache DATASET_NAME@SNAPSHOT_NAME \
  --dest-cluster-name oci-hsg-cs-001 \
  --no-wait
```

`edt cache` skips an existing cache directory and a non-terminal cache job.
Inspect the exact cache job type; a `FILESYSTEM_UPLOAD` job is not a cache job:

```bash
NVDATASET_GROUPID=omni_vision \
nvdataset job list DATASET_NAME \
  --type FILESYSTEM_CACHE \
  --output json
```

### 6. Rewrite the YAML

Only after exact mapping decisions are recorded:

- replace a primary path with `dss://NAME@VERSION[/SUBPATH]`;
- replace a media root with
  `filesystem+dss://NAME@VERSION[/SUBPATH]`;
- preserve all weights, ordering, cooks, and other metadata;
- keep newly uploaded leaves inactive until both primary and auxiliary caches
  exist on the training cluster.

When replacing a dataset because of ownership or a broken snapshot, update
every exact occurrence of the old reference and record the old and new mapping
in `DSS_OPERATIONS_LOG.csv`.

### 7. Validate before training

At minimum:

1. Parse every YAML and recursively resolve every include.
2. Compare legacy and DSS leaf counts, ordering, repetitions, and cooks.
3. Extract and deduplicate every DSS dataset name and snapshot.
4. Run exact DSS checks for references that are not already locally cached.
5. Verify every expected local cache path.
6. Confirm there are no active `filesystem://` paths in the converted graph.
7. Confirm intentionally inactive leaves are commented and documented.
8. Launch a data-loader smoke test before a full model job.

If the training process reports a different cache root than
`/home/svc-dss/cache/nemotron`, fix the environment before changing the recipe.

## Migration closure status

- The five post-mapping 49K uploads and their job IDs are recorded in
  `DSS_OPERATIONS_LOG.csv`. All required caches completed, and the three new
  primary leaves are active in
  `sft_49k_video_dss/videomme_qa_0206.yaml` at `repetitions: 1.0`.
- The original
  `nano_omni_262k_mmlongbench-cc-batch3-seed@v0` remains active. After two
  failed cache attempts and a rejected manifest repair, DSS job
  `0fcd205d-a21f-4b7e-91f7-528d190eaefd` completed successfully. The
  user-owned `-matthieul` fallback is not used.
- The 138 cache-only legacy references are acceptable for this cluster but
  remain a portability risk. A future fresh-cluster migration must restore
  their DSS visibility or upload/remap them.
- All four entry-point graphs parse and resolve, and every active DSS cache
  root and referenced subpath exists under `/home/svc-dss/cache/nemotron`.

## Recordkeeping checklist

For every new or changed mapping:

- [ ] Add the stable path mapping to `legacy_to_dss_mapping.csv`.
- [ ] Regenerate and check `legacy_to_dss_mapping_normalized.csv`.
- [ ] Add a new entry point to `recipe_pairs.csv` if it becomes a reusable
      baseline.
- [ ] Append upload and cache observations to `DSS_OPERATIONS_LOG.csv`.
- [ ] Update the relevant recipe only after exact DSS and cache validation.
- [ ] Update the current-status section of `README.md`.
- [ ] Regenerate or clearly date any exhaustive inventory.
- [ ] Record job IDs, clusters, snapshot names, ownership exceptions, and
      active or inactive recipe state.
