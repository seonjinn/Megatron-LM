# copy_to_s3.py

Copy MetadatasetV2 datasets + media to S3 and generate rewritten YAMLs for a
new cluster.

Handles the full lifecycle: analyzing YAML configs, resolving symlinks,
building a staged symlink tree mirroring the S3 layout, fixing datasets with
broken media references, and rewriting all paths for the target environment.

## Quick start (interactive)

The `interactive` subcommand runs the entire pipeline in one guided session:

```bash
python copy_to_s3.py interactive my_recipe.yaml \
    -o ./my_copy/
```

It will prompt for S3 destination, whether to fix broken datasets, whether to
rewrite YAMLs, etc.  For CI or scripting, pass `--yes` to accept all defaults:

```bash
python copy_to_s3.py interactive my_recipe.yaml \
    -o ./my_copy/ \
    --s3-dest "team-foo:bucket/prefix" \
    --new-root "/scratch/fsw/portfolios/.../my_project" \
    --yes
```

## Subcommands

### `analyze`

Parse input YAMLs (recursively following nested YAML references), resolve
symlinks, and generate:

- `datasets_mapping.txt` / `media_mapping.txt` — resolved path mappings
- `staging/` — symlink tree mirroring the S3 destination structure
- `dm_copy_staged.sh` — single datamover copy script
- `warnings.txt` — missing paths, root `filesystem:///` entries

```bash
python copy_to_s3.py analyze recipe.yaml \
    --s3-dest "team-foo:bucket/prefix" \
    -o ./my_copy/ \
    --dm-args "--slurm-nodes 1"
```

| Option | Default | Description |
|---|---|---|
| `--s3-dest` | *(required)* | S3 destination (e.g. `team-foo:bucket/prefix`) |
| `-o`, `--output-dir` | *(required)* | Output directory |
| `--dm-args` | `--slurm-nodes 1` | Extra args for `dm job copy` |
| `--s3-dataset-suffix` | `/datasets` | Subdir appended to `--s3-dest` for datasets |
| `--s3-media-suffix` | `/media_sources` | Subdir appended to `--s3-dest` for media |
| `--splits` | all | Limit to specific splits (e.g. `--splits train val`) |

### `rewrite`

Generate rewritten YAMLs where all `path:` and `media_source:` values point
to the new cluster.  Requires mapping files from a prior `analyze` run.

```bash
python copy_to_s3.py rewrite recipe.yaml \
    -o ./my_copy/ \
    --new-root "/scratch/fsw/portfolios/.../my_project"
```

| Option | Default | Description |
|---|---|---|
| `--new-root` | *(required)* | New root path on target cluster |
| `-o`, `--output-dir` | *(required)* | Must contain mappings from `analyze` |
| `--strip-prefix` | `/portfolios/` | Prefix to strip for mapping-file lookups |
| `--dataset-suffix` | `/datasets` | Subdir appended to `--new-root` for datasets |
| `--media-suffix` | `/media_sources` | Subdir appended to `--new-root` for media |
| `--fix-media-overrides` | — | Path to `fix_media_yaml_overrides.txt` from `fix-media` |

Rewritten YAMLs are placed in `<output-dir>/yamls/`.

### `fix-media`

Fix datasets whose tar metadata contains absolute image paths
(`media_source: filesystem:///`).  These datasets won't work on a new cluster
because the absolute paths won't resolve.

The fix:
1. Scan tar shards to find the common image-path prefix
2. Rewrite tars with relative paths (strip the prefix)
3. Copy `.nv-meta/` config files (`split.yaml`, `dataset.yaml`, `.info.json`)
   from the original dataset so `energon prepare` can run non-interactively
4. Generate helper files for integration with the rest of the pipeline

```bash
# Auto-discover broken datasets from YAMLs
python copy_to_s3.py fix-media \
    --from-yaml recipe.yaml \
    -o ./my_copy/

# Or specify datasets explicitly
python copy_to_s3.py fix-media \
    --datasets /path/to/dataset1 /path/to/dataset2 \
    -o ./my_copy/
```

| Option | Default | Description |
|---|---|---|
| `--datasets` | — | Dataset directory paths to fix |
| `--from-yaml` | — | Auto-discover affected datasets from YAML(s) |
| `-o`, `--output-dir` | *(required)* | Output directory |

**Generated files:**

| File | Purpose |
|---|---|
| `fixed_datasets/` | Rewritten tar shards (with `.nv-meta/` copied from originals) |
| `fix_media_report.txt` | Summary of what was fixed |
| `fix_media_mapping.txt` | Media dirs discovered from tar contents |
| `fix_media_yaml_overrides.txt` | Feed to `rewrite --fix-media-overrides` |
| `energon_prepare.sh` | Regenerate tar indices (`--tar-index-only`, non-interactive) |

### `interactive`

Guided end-to-end workflow that chains analyze, fix-media, and rewrite
together, prompting for decisions along the way.

```bash
python copy_to_s3.py interactive recipe.yaml other.yaml \
    -o ./my_copy/
```

**Phases:**

1. **Configuration** — Prompts for S3 destination, dm-args, etc. (skipped for
   values provided via CLI flags).
2. **Analyze** — Discovers all YAMLs, datasets, and media sources.
3. **Fix broken media** — If `filesystem:///` datasets are found, offers to
   fix them and merges the results into the main mappings. Skipped with
   `--no-fix-media`.
4. **Review mini-paths** — If the `claude` CLI is available, offers to run an
   LLM review to suggest better mini-path groupings (processed in batches of
   ~100 for reliability).  Then offers to open `$EDITOR` for manual review.
   Skipped with `--yes`.
5. **Write outputs** — Builds a `staging/` symlink tree mirroring the S3
   destination, writes mapping files and a single copy script. Also generates
   a `rewrite_yamls.sh` script (unless `--no-rewrite`).
6. **Rewrite YAMLs** — Prompts for the new cluster root path and generates
   rewritten YAMLs. Skipped with `--no-rewrite`.
7. **Summary** — Lists all generated files and numbered next steps.

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | *(required)* | Output directory |
| `--s3-dest` | *(prompted)* | S3 destination |
| `--new-root` | *(prompted)* | New root path on target cluster |
| `--dm-args` | *(none)* | Extra args for `dm job copy` |
| `--strip-prefix` | `/portfolios/` | Prefix to strip for rewrite lookups |
| `--s3-dataset-suffix` | `/datasets` | Subdir appended to S3 dest for datasets |
| `--s3-media-suffix` | `/media_sources` | Subdir appended to S3 dest for media |
| `--splits` | all | Limit to specific splits |
| `--yes`, `-y` | off | Accept all defaults, no prompts (requires `--s3-dest` and `--new-root`) |
| `--no-fix-media` | off | Skip the fix-media phase |
| `--no-rewrite` | off | Skip the rewrite phase |

### `review-mappings`

Re-review and edit mini-paths in existing mapping files from a prior
`interactive` or `analyze` run. After editing, regenerates `staging/` and
`dm_copy_staged.sh` to match.

```bash
# Default: LLM review then editor (both datasets and media)
python copy_to_s3.py review-mappings -o ./my_copy/

# Only review dataset mini-paths
python copy_to_s3.py review-mappings -o ./my_copy/ --datasets

# Editor only, media only
python copy_to_s3.py review-mappings -o ./my_copy/ --media --editor-only
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | *(required)* | Output directory from a prior run |
| `--datasets` | off | Review dataset mini-paths |
| `--media` | off | Review media mini-paths |
| `--llm-only` | off | Only run LLM review (skip editor) |
| `--editor-only` | off | Only open editor (skip LLM review) |
| `--dm-args` | *(from script)* | Override DM args for copy script |

If neither `--datasets` nor `--media` is given, both are reviewed.
If neither `--llm-only` nor `--editor-only` is given, LLM review runs first
then the editor opens for final adjustments.

### `add`

Add new YAML(s) to an existing output directory from a prior `interactive`
or `analyze` run. Datasets and media already in the mappings keep their
mini-paths unchanged; only truly new entries get computed and staged.

```bash
# Add a second YAML to an existing output
python copy_to_s3.py add extra.yaml -o ./my_copy/

# Non-interactive, skip fix-media
python copy_to_s3.py add extra.yaml -o ./my_copy/ --yes --no-fix-media
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | *(required)* | Existing output directory from a prior run |
| `--splits` | all | Limit to specific splits |
| `--yes`, `-y` | off | Accept all defaults without prompting |
| `--no-fix-media` | off | Skip fix-media step |
| `--no-rewrite` | off | Skip rewrite script update |
| `--strip-prefix` | *(from script)* | Override strip-prefix (default: read from existing `rewrite_yamls.sh`) |

S3 destination, DM args, and suffix settings are all extracted automatically
from the existing output files.

### `size`

Report disk sizes of all dataset and media directories referenced by the
input YAMLs.  Shows a per-entry breakdown sorted by size (largest first)
with subtotals for datasets and media, and a grand total.

```bash
python copy_to_s3.py size recipe.yaml

# Only train split
python copy_to_s3.py size recipe.yaml --splits train

# Only media sources
python copy_to_s3.py size recipe.yaml --media
```

| Option | Default | Description |
|---|---|---|
| `--splits` | all | Limit to specific splits |
| `--datasets` | off | Only report dataset sizes |
| `--media` | off | Only report media sizes |

If neither `--datasets` nor `--media` is given, both are reported.

## Typical end-to-end workflow

### Step 1: Run the interactive pipeline

```bash
python copy_to_s3.py interactive eagle_sft_v13.65.yaml \
    -o ./eagle_copy/
```

Answer the prompts.  At the end you'll see a summary like:

```
  Generated files:
    datasets_mapping.txt      472 dataset mappings
    media_mapping.txt         249 media mappings
    staging/                  Symlink tree mirroring S3 layout
    dm_copy_staged.sh         Single copy script (--follow-symlinks)
    warnings.txt              1 warning(s)
    fix_media_report.txt      24 fixed dataset(s)
    energon_prepare.sh        24 dataset(s) to prepare
    fixed_datasets/           Rewritten tar shards
    yamls/                    5 rewritten YAML(s)
    rewrite_yamls.sh          Rewrite YAMLs with a new root

  Next steps:
    1. Review warnings.txt
    2. Run: ./eagle_copy/energon_prepare.sh
    3. Run: ./eagle_copy/dm_copy_staged.sh --dry-run
    4. Remove --dry-run to copy data to S3
    5. Deploy rewritten YAMLs from ./eagle_copy/yamls/ dir
    6. To rewrite YAMLs later: ./eagle_copy/rewrite_yamls.sh <new-root>
```

### Step 2: Prepare fixed datasets (if any were fixed)

```bash
./eagle_copy/energon_prepare.sh
```

This runs `energon prepare --tar-index-only` on each fixed dataset to
regenerate the SQLite tar index.  It is fully non-interactive because
`split.yaml`, `dataset.yaml`, and `.info.json` are copied from the original
datasets.

### Step 3: Dry-run the copy

```bash
./eagle_copy/dm_copy_staged.sh --dry-run
```

Review the `dm job copy` command that will be run.

### Step 4: Copy to S3

```bash
./eagle_copy/dm_copy_staged.sh
```

This runs a single `dm job copy --follow-symlinks` that copies the entire
staging tree to S3. The staging tree contains symlinks to the real data, so
dm-copy follows them and uploads the actual files.

### Step 5: Deploy rewritten YAMLs

Copy the files from `./eagle_copy/yamls/` to the target cluster.

### Step 6 (optional): Rewrite YAMLs for a different root later

```bash
./eagle_copy/rewrite_yamls.sh /new/cluster/root
```

## Concepts

### Staged copy

Instead of generating a separate `dm job copy` command for each dataset and
media directory, the tool builds a **staging directory** that mirrors the exact
S3 destination structure using symlinks:

```
<output-dir>/
  staging/
    datasets/
      commercial_sft/dewiki_v5_0828   -> /lustre/.../dewiki_v5_0828
      avlm_sft_audio/MiraData         -> /lustre/.../MiraData
      fixed_datasets/some_dataset     -> <output-dir>/fixed_datasets/some_dataset
      ...
    media_sources/
      audioflamingo/VGGSound           -> /lustre/.../VGGSound
      grounding/coco                   -> /lustre/.../coco
      ...
  dm_copy_staged.sh
```

A single `dm job copy --follow-symlinks` then copies this entire tree to S3.
The symlinks are transparent to the copy — dm-copy follows them and uploads
the actual file contents.

This approach also handles **symlinks inside dataset directories** correctly.
Because `--follow-symlinks` is always enabled and works transitively, any
symlinks within the source data directories are resolved during the copy.

**Nested mini-paths:** When one mini-path is a prefix of another (e.g.
`image_data` and `image_data/data`), the tool detects the conflict. If the
child's real path is under the parent's real path, the child is redundant
(already reachable via the parent symlink) and is skipped. If they point to
unrelated real paths, the parent is created as a real directory with individual
child symlinks.

### Mini-paths

A "mini-path" is the shortest unique suffix of a resolved absolute path that
uniquely identifies it among all paths in the same namespace (datasets or
media).  Mini-paths determine the S3 object structure and the rewritten YAML
paths.

The algorithm works in two phases:

1. **Meaningful anchor** — Start from the basename and walk up past generic
   directory names (`images`, `train`, `data`, `video`, `audio`, etc.) to
   find a descriptive component.  This ensures the mini-path always contains
   a useful name, not just `images` or `train`.

2. **Uniqueness** — If any mini-paths collide, extend all conflicting paths
   by one parent component.  Repeat until unique.  The leading component of
   a mini-path is also checked — if it's generic, the path is extended
   further.

Examples:

```
/lustre/.../AVLM_SFT_AUDIO/MiraData             -> MiraData
/lustre/.../AVLM_SFT_AUDIO/Ego/action2sound/manifests -> action2sound/manifests
/lustre/.../audiocaps/audio/train                -> audiocaps/audio/train
/lustre/.../grounding_data/images/coco           -> grounding_data/images/coco
/lustre/.../sft_data/coco                        -> sft_data/coco
```

Dataset and media namespaces are independent — the same mini-path can appear
in both without conflict.

In `interactive` mode, you can review and edit mini-paths in `$EDITOR`
before output files are written. If the `claude` CLI is installed, the tool
can also run an LLM review pass to suggest better groupings (e.g. adding
common prefixes for related datasets). Large entry sets are processed in
batches of ~100 for reliable results, with cross-batch context to maintain
consistency.

### S3 directory structure

Datasets and media are placed under separate subdirectories of the S3
destination:

```
<s3-dest>/datasets/<mini-path>        # dataset files
<s3-dest>/media_sources/<mini-path>   # media source files
```

On the target cluster, rewritten YAML paths follow the same structure:

```
<new-root>/datasets/<mini-path>
<new-root>/media_sources/<mini-path>
```

The subdirectory names are configurable via `--s3-dataset-suffix` /
`--s3-media-suffix` (for `analyze` / `interactive`) and `--dataset-suffix` /
`--media-suffix` (for `rewrite`).

### MetadatasetV2 YAML format

The tool parses MetadatasetV2-style YAML files, which contain `splits:` with
`train:`/`val:` sections, each listing datasets with `path:` and optional
`media_source:` in `aux:`.  Nested YAML references (where `path:` points to
another `.yaml` file) are followed recursively.

### The `filesystem:///` problem

Some datasets have `media_source: filesystem:///` meaning their tar metadata
contains absolute image paths like `lustre/fs1/.../image.png`.  Combined with
the root media source `/`, this resolves to `/lustre/fs1/.../image.png` on
the original cluster.  On a new cluster, these paths won't exist.

The `fix-media` step rewrites tar shards to use relative paths and sets the
media source to the correct directory (e.g.
`filesystem:///lustre/fs1/.../images/`), making the datasets portable.
