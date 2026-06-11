# Nano v3.5 VLM recipes

This directory contains the nano v3.5 VLM bootstrap and staged training scripts.
They are derived from:

- `examples/multimodal/v3_omni_staged_conv3d_ga`
- `examples/multimodal/super/*_svg_newcontainer.sh`
- `nano-3.5-sft-alex-512k-hermes.sh` for the nano v3.5 LLM architecture

Default model parallelism is TP=2, EP=32 to match the existing nano VLM/RADIO
TP=2 checkpoint assembly pattern. Override `TP`, `EP`, `TOKENIZER_MODEL`,
`CHECKPOINT_DIR`, or `DATA_TRAIN` from the launch environment if needed.

## Checkpoint bootstrap

Convert the HF nano v3.5 LLM checkpoint to Megatron torch format:

```bash
HF_CKPT_DIR=/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano_v35_checkpoint_hf \
MCORE_CKPT_DIR=/lustre/fsw/portfolios/llmservice/users/$USER/workspace/checkpoints/nano_v35_llm_mcore_tp2_ep32_mtpfix \
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/convert_hf_moe_to_mcore_nano_v35.sh \
  --name nano_v35_convert_hf_to_mcore_tp2_ep32 \
  --nodes 1 \
  --duration-mins 30 \
  --partition cpu
```

Combine the converted LLM with standalone RADIO-v4 vision weights. The default
source is the RC2 TP=2 RADIO checkpoint.

```bash
LM_MCORE_DIR=/lustre/fsw/portfolios/llmservice/users/$USER/workspace/checkpoints/nano_v35_llm_mcore_tp2_ep32_mtpfix/torch \
VISION_CKPT_DIR=/lustre/fsw/portfolios/llmservice/users/tpoon/checkpoints/c-radio-v4-h-rc2-tp2 \
VISION_ITERATION=1 \
OUTPUT_CKPT_DIR=/lustre/fsw/portfolios/llmservice/users/$USER/workspace/checkpoints/nano_v35_vlm/nano_v35_moe_tp2_ep32_radio_v4_mtpfix \
examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/combine_nano_v35_with_radio.sh \
  --name nano_v35_combine_radio_v4 \
  --nodes 1 \
  --duration-mins 30 \
  --partition cpu
```

## Stage smoke runs

Use `EARLY_EXIT_ITERS=20` for first smoke tests. The stage scripts chain their
default checkpoint paths through the expected output names.

```bash
EARLY_EXIT_ITERS=20 examples/multimodal/launch.sh \
  --sbatch examples/multimodal/nano_v35/pretrain_nano_v35_radiov4_1377_svg_newcontainer.sh \
  --name nano_v35_pretrain_radiov4_1377_svg \
  --nodes 8 --duration-mins 30 --partition batch --overwrite-code-snapshot
```

Repeat with the SFT/audio/omni wrappers in this directory after each prior
stage has produced a usable checkpoint.
