# Model Conversion

1. Obtain/generate a VLM-compatible LLM checkpoint

2. Generate a VLM-compatible vision checkpoint
- See `examples/multimodal/model_converter/radio_converter.py`

3. Combine checkpoints for final VLM checkpoint

## LLM Checkpoints

- Option 1: obtain a pre-generated checkpoint
    - Nemotron v5.5 9B: `/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000`
    - Nemotron v6 30B-A3.5B: `/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500`
    - Nemotron Nano V3 30B-A3.5B: `/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001`

- Option 2: generate manually using `examples/multimodal/tools/prepare_vlm_checkpoint.sh`
    - May require updates to the bash script and/or python scripts called from within

## VLM: Nemotron Nano V3 30B-A3.5B (Latest LLM)

### Nemotron Nano V3 30B-A3.5B + C-RADIO-v3-H

```bash
./interactive.sh

# Conversion paths/vars for final Nemotron 3 LLM
CONVERTED_LLM=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v3/c-radio_v3-h_half.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v3-h-tp2
FINAL_VLM_TP=2
FINAL_VLM_EP=32
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v3-h-tp2-ep32/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v3-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP
```

### Nemotron Nano V3 30B-A3.5B + C-RADIO-v4-H RC2

```bash
./interactive.sh

# Conversion paths/vars for normal TP=2, EP=32 which works for 32-nodes with batch_size=128 (SFT)
CONVERTED_LLM=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001
FINAL_VLM_TP=2
FINAL_VLM_EP=32
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# RADIO-v4-H-RC2
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc2/c-radio_v4-h.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-rc2-tp2
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-h-rc2-tp2-ep32/iter_0000001
PREVIOUS_VLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-vlm-h-tp2-ep32/checkpoints/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP

# Verify with exiting pre-generated (because we had one available)
PYTHONPATH=. python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
--dir-a $PREVIOUS_VLM \
--dir-b $FINAL_VLM \
--model-only
```

### Nemotron Nano V3 30B-A3.5B + C-RADIO-v4-H RC3

```bash
./interactive.sh

# Conversion paths/vars for normal TP=2, EP=32 which works for 32-nodes with batch_size=128 (SFT)
CONVERTED_LLM=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001
FINAL_VLM_TP=2
FINAL_VLM_EP=32
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# RADIO-v4-H-RC3 (no PREVIOUS_VLM to compare to)
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc3/c-radio_v4-h.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-rc3-tp2
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-h-rc3-tp2-ep32/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP

# Vision model tester: need 4 nodes for EP=32 VLM
submit_job \
--nodes 4 \
--gpu 8 \
--duration 0.5 \
--partition backfill,batch_short \
--image "/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh" \
--setenv "PYTHONPATH=." \
--name vision_model_tester_radio_v4_h \
--workdir $(pwd) \
--wait \
-c "torchrun \
--nproc_per_node=\$SUBMIT_GPUS \
--master_addr=\$MASTER_ADDR \
--master_port=\$MASTER_PORT \
--nnodes=\$NUM_NODES \
--node_rank=\$NODE_RANK \
examples/multimodal/model_converter/vision_model_tester.py \
--use-te \
--mcore-model $(realpath $FINAL_VLM/..) \
--vision-resolution 512 \
--torchhub-version $PRIVATE_RADIO_REPO \
--mcore-model-type radio \
--language-model-type nemotron6-moe \
--torchhub-model-version $ORIGINAL_VISION_CHECKPOINT \
--tensor-parallel-size $FINAL_VLM_TP \
--expert-model-parallel-size $FINAL_VLM_EP"

# Tester output
# ==============================
# Correlation: 0.999946653842926 (relaxed thresholds: True)
# Mean diff: 0.00531005859375, current threshold: 0.2
# Max diff: 0.90625, current threshold: 100
# ==============================
# Test passed
# ==============================

# TODO: REMOVE
PYTHONPATH=. python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
--dir-a $PREVIOUS_VLM \
--dir-b $FINAL_VLM \
--model-only
```

### Nemotron Nano V3 30B-A3.5B + C-RADIO-v4-So400m RC3

```bash
./interactive.sh

# Conversion paths/vars for normal TP=2, EP=32 which works for 32-nodes with batch_size=128 (SFT)
CONVERTED_LLM=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001
FINAL_VLM_TP=2
FINAL_VLM_EP=32
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# RADIO-v4-So400m-RC3 (no PREVIOUS_VLM to compare to)
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc3/c-radio_v4-so400m.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-so400m-rc3-tp2
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-so400m-rc3-tp2-ep32/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio-so400m \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP

# Vision model tester: need 4 nodes for EP=32 VLM
submit_job \
--nodes 4 \
--gpu 8 \
--duration 0.5 \
--partition backfill,batch_short \
--image "/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/containers/pytorch25.06-moe-avlm-editable-energon.sqsh" \
--setenv "PYTHONPATH=." \
--name vision_model_tester_radio_v4_so400m \
--workdir $(pwd) \
--wait \
-c "torchrun \
--nproc_per_node=\$SUBMIT_GPUS \
--master_addr=\$MASTER_ADDR \
--master_port=\$MASTER_PORT \
--nnodes=\$NUM_NODES \
--node_rank=\$NODE_RANK \
examples/multimodal/model_converter/vision_model_tester.py \
--use-te \
--mcore-model $(realpath $FINAL_VLM/..) \
--vision-resolution 512 \
--torchhub-version $PRIVATE_RADIO_REPO \
--mcore-model-type radio-so400m \
--language-model-type nemotron6-moe \
--torchhub-model-version $ORIGINAL_VISION_CHECKPOINT \
--tensor-parallel-size $FINAL_VLM_TP \
--expert-model-parallel-size $FINAL_VLM_EP"

# Tester output:
# ==============================
# Correlation: 0.9999678134918213 (relaxed thresholds: True)
# Mean diff: 0.006683349609375, current threshold: 0.2
# Max diff: 4.0, current threshold: 100
# ==============================
# Test passed
# ==============================
```

### Nemotron Nano V3 30B-A3.5B + C-RADIO-v4-H-1D

```bash
./interactive.sh

# Conversion paths/vars for normal TP=2, EP=32 which works for 32-nodes with batch_size=128 (SFT)
CONVERTED_LLM=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/gheinrich/results/evfm/10-12-evfm-radio-h-1d-24-16nodes/checkpoints/checkpoint-170-export.pth.tar
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO
FINAL_VLM_TP=2
FINAL_VLM_EP=32

# Previous 1D, 170 epoch
# CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-1d-h-epoch170-mcore-tp2/
# FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-1d-h-ep170-tp2-ep32/iter_0000001

# Latest 1D, 300 epoch
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-1d-h-epoch300-mcore-tp2/
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-1d-h-ep300-tp2-ep32/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h-1d \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP
```

## VLM: Nemotron 6 30B-A3.5B (Older LLM)

### Nemotron 6 30B-A3.5B + C-RADIO-v2-VLM

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v2-vlm-tp2
FINAL_VLM_TP=2
FINAL_VLM_EP=32
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant-c-radio_v2-vlm-h-tp2/iter_0000001
PREVIOUS_VLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant-c-radio_v2-vlm-h-tp2/checkpoints/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--model-type c-radio_v2-vlm-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP

# Verify with exiting pre-generated (because we had one available)
PYTHONPATH=. python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
--dir-a $PREVIOUS_VLM \
--dir-b $FINAL_VLM \
--model-only
```

### Nemotron 6 30B-A3.5B + C-RADIO-v3-H

```bash
./interactive.sh

# Conversion paths/vars for initial Nemotron 3 LLM
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v3/c-radio_v3-h_half.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v3-h-tp2
FINAL_VLM_TP=2
FINAL_VLM_EP=32
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant-c-radio_v3-h-tp2/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v3-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP
```

### Nemotron 6 30B-A3.5B + C-RADIO-v4-H

```bash
./interactive.sh

# Conversion paths/vars for normal TP=2, EP=32 which works for 32-nodes with batch_size=128 (SFT)
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc2/c-radio_v4-h.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-rc2-tp2
FINAL_VLM_TP=2
FINAL_VLM_EP=32
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant-c-radio_v4-h-tp2/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP
```

### Nemotron 6 30B-A3.5B + SigLIP2-So400m-P16

- SigLIP2-So400m-P16-512, TP2 for Nemotron 6 MoE LLMs
    - Same weights as SigLIP from HF except:
        - vision_model.embeddings.patch_embedding.weight.shape: (1152, 3, 14, 14) -> (1152, 3, 16, 16)
        - vision_model.embeddings.position_embedding.weight.shape: (729, 1152) -> (1024, 1152)

```bash
./interactive.sh, but usually faster to allocate w/ `-g 1`

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/siglip2_checkpoints/google_siglip2_so400m_patch16_512_mcore_tp_2
FINAL_VLM_TP=2
FINAL_VLM_EP=32
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant_siglip2-so400m-p16-512-tp2/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/siglip_converter.py \
--tensor-parallel-size $FINAL_VLM_TP \
--model-id google/siglip2-so400m-patch16-512 \
--output $CONVERTED_VISION_PARENT_DIR \
--use-te

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_moe_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP \
--ep $FINAL_VLM_EP
```

## VLM: Nemotron 5.5 9B

### Convert 9B LLM: TP4 -> TP2

```bash
./interactive.sh

CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/checkpoint/convert.py \
--model-type hybrid \
--loader core \
--saver core \
--load-dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/llm_checkpoints/nemotron_5p5_9b_v2_tp4 \
--save-dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/llm_checkpoints/nemotron_5p5_9b_v2_tp2 \
--megatron-path . \
--max-queue-size 1 \
--target-tensor-parallel-size 2
```

### Nemotron 5.5 9B + C-RADIO-v2-VLM

```bash
./interactive.sh

# Conversion paths/vars (usually PREVIOUS_VLM doesn't exist, but here it does)
# - PREVIOUS_VLM from the previous VLM checkpoint, CHECKPOINT_DIR value
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v2-vlm-tp4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v2_vlm_tp4/iter_0000001
PREVIOUS_VLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/vlm/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
# Can also use `--version nvidia/C-RADIOv2-VLM-H` here
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version /lustre/fsw/portfolios/llmservice/users/mranzinger/output/evfm/commercial/v2/siglip2_only/huge/vit-h-16_anyres-v4_s8/checkpoints/last_release_half.pth.tar \
--model-type c-radio_v2-vlm-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_dense_vlm_checkpoints.py for dense VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP

# Verify with exiting pre-generated (because we had one available)
PYTHONPATH=. python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
--dir-a $PREVIOUS_VLM \
--dir-b $FINAL_VLM \
--model-only
```

### Nemotron 5.5 9B + C-RADIO-v4-H RC2

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc2/c-radio_v4-h.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-tp4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_h_tp4/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP
```

### Nemotron 5.5 9B + C-RADIO-v4-H RC3

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc3/c-radio_v4-h.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-rc3-tp4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_h_rc3_tp4/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP

# Vision model tester
# To debug: replace `torchrun --nproc_per_node=4`
#        -> `python -m debugpy --listen 0.0.0.0:3009 --wait-for-client -m torch.distributed.run --nproc_per_node=4`
PYTHONPATH=. torchrun --nproc_per_node=4 examples/multimodal/model_converter/vision_model_tester.py \
--use-te \
--mcore-model $FINAL_VLM/.. \
--vision-resolution 512 \
--torchhub-version $PRIVATE_RADIO_REPO \
--mcore-model-type radio \
--language-model-type nemotron5-hybrid-9b \
--torchhub-model-version $ORIGINAL_VISION_CHECKPOINT \
--tensor-parallel-size $FINAL_VLM_TP

# Test result
# ==============================
# Correlation: 0.9999356865882874 (relaxed thresholds: True)
# Mean diff: 0.00567626953125, current threshold: 0.2
# Max diff: 1.4375, current threshold: 100
# ==============================
# Test passed
# ==============================
```

### Nemotron 5.5 9B + C-RADIO-v4-So400m RC3

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/mranzinger/output/radio_releases/commercial/v4/rc3/c-radio_v4-so400m.pth.tar
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-so400m-rc3-tp4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_so400m_rc3_tp4/iter_0000001
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio-so400m \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP

# Vision model tester
# To debug: replace `torchrun --nproc_per_node=4`
#        -> `python -m debugpy --listen 0.0.0.0:3009 --wait-for-client -m torch.distributed.run --nproc_per_node=4`
PYTHONPATH=. torchrun --nproc_per_node=4 examples/multimodal/model_converter/vision_model_tester.py \
--use-te \
--mcore-model $FINAL_VLM/.. \
--vision-resolution 512 \
--torchhub-version $PRIVATE_RADIO_REPO \
--mcore-model-type radio-so400m \
--language-model-type nemotron5-hybrid-9b \
--torchhub-model-version $ORIGINAL_VISION_CHECKPOINT \
--tensor-parallel-size $FINAL_VLM_TP

# Tester output:
# ==============================
# Correlation: 0.9999687075614929 (relaxed thresholds: True)
# Mean diff: 0.006866455078125, current threshold: 0.2
# Max diff: 1.671875, current threshold: 100
# ==============================
# Test passed
# ==============================
```

### Nemotron 5.5 9B + C-RADIO-v4-H-1D

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
ORIGINAL_VISION_CHECKPOINT=/lustre/fsw/portfolios/llmservice/users/gheinrich/results/evfm/10-12-evfm-radio-h-1d-24-16nodes/checkpoints/checkpoint-170-export.pth.tar
PRIVATE_RADIO_REPO=/lustre/fsw/portfolios/llmservice/users/cmccarthy/RADIO
FINAL_VLM_TP=4

# Previous RADIO-1D, 170 epoch
# CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-1d-h-epoch170-mcore-tp4/
# FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_1d_h_ep170_tp4/iter_0000001
# PREVIOUS_VLM=/lustre/fsw/portfolios/llmservice/users/gheinrich/checkpoints/nemotron_5p5_9b_v2-c-radio-v4-1d-h-ep170/iter_0000001

# Latest RADIO-1D, 300 epoch
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-1d-h-epoch300-mcore-tp4/
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v4_1d_h_ep300_tp4/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/radio_converter.py \
--use-te \
--output $CONVERTED_VISION_PARENT_DIR \
--tensor-parallel-size $FINAL_VLM_TP \
--version $ORIGINAL_VISION_CHECKPOINT \
--model-type radio_v4-h-1d \
--torchhub-version $PRIVATE_RADIO_REPO

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM  \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP

# Verify with exiting pre-generated (because we had one available)
PYTHONPATH=. python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
--dir-a $PREVIOUS_VLM \
--dir-b $FINAL_VLM \
--model-only
```

### Nemotron 5.5 9B + SigLIP-So400m-P14

- Used for testing the SigLIP VLM checkpoint on a single node without an extra TP8 conversion
- SigLIP-So400m-P14-384, TP4 for Nemotron 9B/12B dense LLMs

```bash
./interactive.sh, but usually faster to allocate w/ `-g 1`

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/siglip_checkpoints/google_siglip_so400m_patch14_384_mcore_tp_4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_siglip-so400m-p14-384-tp4/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/siglip_converter.py \
--tensor-parallel-size $FINAL_VLM_TP \
--model-id google/siglip-so400m-patch14-384 \
--output $CONVERTED_VISION_PARENT_DIR \
--use-te

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP
```

### Nemotron 5.5 9B + SigLIP2-So400m-P16

- Used for testing the SigLIP2 VLM checkpoint on a single node without an extra TP8 conversion
- SigLIP2-So400m-P16-512, TP4 for Nemotron 9B/12B dense LLMs

```bash
./interactive.sh

# Conversion paths/vars
CONVERTED_LLM=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000
CONVERTED_VISION_PARENT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/siglip2_checkpoints/google_siglip2_so400m_patch16_512_mcore_tp_4
FINAL_VLM_TP=4
FINAL_VLM=/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_siglip2-so400m-p16-512-tp4/iter_0000001

# Generate VLM-compatible vision checkpoint
python examples/multimodal/model_converter/siglip_converter.py \
--tensor-parallel-size $FINAL_VLM_TP \
--model-id google/siglip2-so400m-patch16-512 \
--output $CONVERTED_VISION_PARENT_DIR \
--use-te

# Combine with LLM for final VLM checkpoint (uses combine_moe_vlm_checkpoints.py for MoE VLM)
PYTHONPATH=. python examples/multimodal/model_converter/combine_dense_vlm_checkpoints.py \
--llm-checkpoint $CONVERTED_LLM \
--vision-checkpoint $CONVERTED_VISION_PARENT_DIR/iter_0000001 \
--output-checkpoint $FINAL_VLM \
--tp $FINAL_VLM_TP
```
