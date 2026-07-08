# Nano v3.5 VLM recipes for OCI-NRT

This directory contains the OCI-NRT-specific Nano v3.5 VLM-only bootstrap and
training path. The scripts in the parent `nano_v35` directory remain unchanged.

1. Convert the Nano v3.5 HF LLM checkpoint to MCore TP=2, EP=32.
2. Combine it with the TP=2 RADIO-v4 checkpoint.
3. Pretrain the vision adaptor with ViT and LM frozen.
4. Run full VLM SFT with image and Conv3D video data.

Audio and omni stages are not part of this workflow.

## Destination defaults

`cluster_config.sh` defines the destination resource, workspace, and container
paths. Every value can be overridden through the environment. The important
defaults are:

- Standalone tokenizer: `resources/tokenizer/nemotron_3_nano_30b_a3b_tokenizer`
- HF LLM: `resources/checkpoints/nano_v35_checkpoint_hf`
- RADIO-v4: `resources/checkpoints/c-radio-v4-h-rc2-tp2`
- Container: `docker/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh`
- Output workspace: `/lustre/fsw/portfolios/llmservice/users/$USER/workspace`

Training intentionally uses the standalone tokenizer, not the tokenizer in the
HF checkpoint.

## Validate resources

Run this before submitting the bootstrap jobs:

```bash
examples/multimodal/nano_v35/oci-nrt/setup_vlm.sh
```

The training launchers intentionally use the same data YAMLs as the original
Nano v3.5 scripts:

- `examples/multimodal/v3_baseline/pretrain_vision_adaptor_recipe.yaml`
- `examples/multimodal/v3_baseline/1377_video_text.yaml`

`setup_vlm.sh` verifies that these top-level YAML files exist but deliberately
does not validate or filter their nested dataset references. This preserves
recipe parity even though some paths are unavailable on OCI-NRT.

## Bootstrap checkpoints

Convert the HF checkpoint:

```bash
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/oci-nrt/convert_hf_moe_to_mcore_nano_v35.sh \
  --name nano_v35_convert_hf_to_mcore_tp2_ep32 \
  --nodes 1 --duration-mins 30 \
  --partition batch_singlenode,batch_short,backfill
```

After conversion succeeds, combine the LLM and RADIO checkpoints:

```bash
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/oci-nrt/combine_nano_v35_with_radio.sh \
  --name nano_v35_combine_radio_v4 \
  --nodes 1 --duration-mins 30 \
  --partition batch_singlenode,batch_short,backfill
```

The merge defaults to iteration 1 for both inputs. Override `LM_ITERATION`,
`VISION_ITERATION`, or any checkpoint path when needed.

## Vision pretrain

Start with an 8-node, 20-iteration smoke run. Eight nodes is the minimum for
TP=2 and EP=32 with eight GPUs per node.

```bash
EARLY_EXIT_ITERS=20 SAVE_INTERVAL=20 \
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/oci-nrt/pretrain_nano_v35_radiov4_1377_svg_newcontainer.sh \
  --name nano_v35_vision_pretrain_radiov4_1377_smoke \
  --nodes 8 --duration-mins 30 \
  --partition batch_block1 \
  --overwrite-code-snapshot
```

For the full recipe, omit `EARLY_EXIT_ITERS` and use the script's 32-node
default, or override it explicitly with `--nodes 32`.

## VLM SFT

The default SFT input is the checkpoint from the full vision-pretrain run named
`nano_v35_vision_pretrain_radiov4_1377`. Set `CHECKPOINT_DIR` to use a smoke or
differently named run.

```bash
CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/$USER/workspace/output/nano_v35_vision_pretrain_radiov4_1377_smoke/checkpoints \
EARLY_EXIT_ITERS=20 SAVE_INTERVAL=20 \
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/oci-nrt/sft_nano_v35_conv3d_radiov4_1377_svg_newcontainer.sh \
  --name nano_v35_vlm_sft_conv3d_radiov4_1377_smoke \
  --nodes 8 --duration-mins 30 \
  --partition batch_block1 \
  --overwrite-code-snapshot
```

For the full recipe, omit the smoke overrides and use the script's 64-node
default. `DATA_TRAIN`, `CHECKPOINT_DIR`, `WORKSPACE`, `CONTAINER_IMAGE`, and all
parallelism and batch-size settings remain environment-overridable.

Use `--dry-run` with any `launch.sh` command to inspect the resolved training
command without submitting a job.
