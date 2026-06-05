# Super Training

## Cluster Setup

- Get access to NSC-SVG
    - [Docs: cluster info for NSC-SVG](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/3119929385/nsc-svg-slurm-1+Cluster-Specific+Information)
    - Generate SSH key, follow onboarding instructions / docs to upload SSH key (may take a few minutes to update)

- Run adlr utils setup to set $SHARE_OUTPUT and other env vars in ~/.bashrc
    - See [ADLR Utils: Setup instruction](https://nvidia.atlassian.net/wiki/spaces/ADLR/pages/2115111311/ADLR+Utils#Setup-instruction---adlr-utils-(once-per-cluster))

```bash
# Setup
/lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/latest/setup
```

- Add GitLab SSH key: https://gitlab-master.nvidia.com/-/user_settings/ssh_keys

- Clone / symlink / rebase

```bash
echo $SHARE_OUTPUT  # Should be set from above

# Setup megatron-lm, branch `tpoon/super-vlm2` (should be rebased onto `vlm2` already)
cd $SHARE_OUTPUT
git clone ssh://git@gitlab-master.nvidia.com:12051/ADLR/megatron-lm.git
cd megatron-lm
git checkout tpoon/super-vlm2
git pull

# Setup VLMEvalKitMcore, branch `super` (should be rebased onto `main` already)
cd $SHARE_OUTPUT
git clone ssh://git@gitlab-master.nvidia.com:12051/matthieul/VLMEvalKitMcore.git
cd VLMEvalKitMcore
git checkout super
```

- Copy any data / dirs from NRT -> SVG
    - Requires setting up SSH keys from NRT -> SVG
    - Requires adding GitLab access token for read_registry access
    - Requires install ADLR setup urils for NRT

```bash
# Required (most likely): Copy ~/nltk_data, used for audio/omni evals
slurm_copy \
--src="${HOME}/nltk_data/" \
--dest="nsc-svg-slurm-1-dc-02.nvidia.com:${HOME}/nltk_data/"

# Optional: Copy any trained models as follows
ITER_FOLDER=iter_0002200  # Best iter
SOURCE_OUTPUT_DIR=/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/sft_mmlongbench_txt_0403
DEST_OUTPUT_DIR=/lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/sft_mmlongbench_txt_0403

# Copy the checkpoints folder for the iter we care about, for both the original and the mcore/hf
slurm_copy \
--src="${SOURCE_OUTPUT_DIR}/checkpoints/${ITER_FOLDER}/" \
--dest="nsc-svg-slurm-1-dc-02.nvidia.com:${DEST_OUTPUT_DIR}/checkpoints/${ITER_FOLDER}/"

slurm_copy \
--src="${SOURCE_OUTPUT_DIR}/checkpoints/tp_1_hf/${ITER_FOLDER}/" \
--dest="nsc-svg-slurm-1-dc-02.nvidia.com:${DEST_OUTPUT_DIR}/checkpoints/tp_1_hf/${ITER_FOLDER}/"

slurm_copy \
--src="${SOURCE_OUTPUT_DIR}/checkpoints/tp_1/${ITER_FOLDER}/" \
--dest="nsc-svg-slurm-1-dc-02.nvidia.com:${DEST_OUTPUT_DIR}/checkpoints/tp_1/${ITER_FOLDER}/"

# Copy the rest, excluding checkpoints; rclone not working here, using rsync + rsync exclude syntax
slurm_copy \
--src="${SOURCE_OUTPUT_DIR}/" \
--dest="nsc-svg-slurm-1-dc-02.nvidia.com:${DEST_OUTPUT_DIR}/" \
--tool=rsync \
--tool-opts="--exclude=*.pt --exclude=*.safetensors"
```

- Build VLMEvalKit viewer venv for evals

```bash
cd $SHARE_OUTPUT/VLMEvalKitMcore
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r viewer/requirements.txt
```

- If modifying vLLM

```bash
echo $SHARE_OUTPUT  # Should be set from above

# Setup vllm, whatever current feature branch you're using
# Should be a personal fork, e.g. github.com/collinmccarthy/vllm
cd $SHARE_OUTPUT
git clone git@github.com:collinmccarthy/vllm.git
cd vllm
git checkout feature/nemotron-3-vl-super-registration

# Setup uv and install precommit hooks
# Following uv linux install from https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv venv
source .venv/bin/activate

uv pip install "pre-commit>=4.5.1"
pre-commit install
pre-commit install-hooks

# When committing to vllm, sign commits with `-s` e.g. `git commit -s -m <msg>`
# This will also run precommit hooks
# To run manually activate uv env and run `pre-commit run --all-files`
```

- Verify ~/.bashrc has appropriate env variables, e.g.

```bash
export WANDB_API_KEY="<my_wandb_api_key>"
export OPENAI_API_KEY="<my_nvidia_inference_api_key>"
export OPENAI_API_BASE=https://inference-api.nvidia.com/v1/chat/completions  # May not be used anymore
```

## Data Verification

- Common pipeline to verify data yamls for various stages

```bash
# - - - - - - - - - -
# Stage 1: Vision Pre-training
# - - - - - - - - - -
# This recipe is from:
#   https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/vlm2/examples/multimodal/v3_baseline/pretrain_vision_adaptor_packing_lower_bs.sh?ref_type=heads#L80
DATA_TRAIN=/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/yamls/pretrain_vision_adaptor_recipe.yaml
PREFIX=vision_pretrain
OUTPUT_DIR=$SHARE_OUTPUT/workspace/output/data_verification

# - - - - - - - - - -
# Stage 2: Vision 16K SFT
# - - - - - - - - - -
# This recipe is from:
#   https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/vlm2/examples/multimodal/v3_omni_staged_conv3d_ga/sft_13p77.sh?ref_type=heads#L47
# DATA_TRAIN=/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/super_yamls/1377_video_text.yaml
DATA_TRAIN=/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/yamls/1377_video_text.yaml
PREFIX=vision_sft_16k
OUTPUT_DIR=$SHARE_OUTPUT/workspace/output/data_verification

# - - - - - - - - - -
# Stage 5: Omni 16K SFT
# - - - - - - - - - -
# This recipe is from:
#   https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/vlm2/examples/multimodal/v3_omni_staged_conv3d_ga/sft_13p77.sh?ref_type=heads#L47
DATA_TRAIN=/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/super_yamls/sft_recipe_13p77.yaml
PREFIX=omni_sft_16k
OUTPUT_DIR=$SHARE_OUTPUT/workspace/output/data_verification

# - - - - - - - - - -
# Common
# - Choose one set of vars above and run the following
# - The first sbatch will take ~5-15 minutes
# - The second sbatch will take ~20-30 minutes
# - The third sbatch may take >4 hrs
# - - - - - - - - - -
cd $SHARE_OUTPUT/megatron-lm
mkdir -p $OUTPUT_DIR

# Permissions / paths check (checks permissions and the existance of directories)
# - Tail output with, `tail -f $SHARE_OUTPUT/workspace/output/data_verification/logs/${PREFIX}_check_read_access_yaml_permissions.log`
sbatch -p cpu -A llmservice_fm_vision -N 1 \
--job-name "${PREFIX}_check_read_access_yaml_permissions" \
--output $OUTPUT_DIR/logs/${PREFIX}_check_read_access_yaml_permissions.log \
--wrap "python3 $SHARE_OUTPUT/megatron-lm/examples/multimodal/tools/check_read_access_yaml.py \
$DATA_TRAIN \
--output $OUTPUT_DIR/${PREFIX}_check_read_access_yaml_permissions_report.txt"

# Shards check, up to 3 shards per folder (checks aux media paths and open tar files to check actual paths)
# - Tail output with, `tail -f $SHARE_OUTPUT/workspace/output/data_verification/logs/${PREFIX}_check_read_access_yaml_shards.log`
sbatch -p cpu -A llmservice_fm_vision -N 1 \
--job-name "${PREFIX}_check_read_access_yaml_shards" \
--output $OUTPUT_DIR/logs/${PREFIX}_check_read_access_yaml_shards.log \
--wrap "python3 $SHARE_OUTPUT/megatron-lm/examples/multimodal/tools/check_read_access_yaml.py \
$DATA_TRAIN \
--skip-permissions \
--verify-aux-media nonempty \
--verify-shard-paths \
--output $OUTPUT_DIR/${PREFIX}_check_read_access_yaml_shards_report.txt"

# \[Optional\] Shards check, exhaustive (much slower)
# - Tail output with, `tail -f $SHARE_OUTPUT/workspace/output/data_verification/logs/${PREFIX}_check_read_access_yaml_shards_full.log`
sbatch -p cpu -A llmservice_fm_vision -N 1 \
--job-name "${PREFIX}_check_read_access_yaml_shards_full" \
--output $OUTPUT_DIR/logs/${PREFIX}_check_read_access_yaml_shards_full.log \
--time 1-00:00:00 \
--wrap "python3 $SHARE_OUTPUT/megatron-lm/examples/multimodal/tools/check_read_access_yaml.py \
$DATA_TRAIN \
--skip-permissions \
--verify-aux-media nonempty \
--verify-shard-paths \
--exhaustive \
--workers 0 \
--output $OUTPUT_DIR/${PREFIX}_check_read_access_yaml_shards_full_report.txt"
```

## Build Eval Containers

### Build Mcore Container

- Mostly for testing one or two iterations compared to vLLM

```bash
cd $SHARE_OUTPUT/VLMEvalKitMcore

PARTITION=batch BASE_MCORE_CONTAINER=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.06-moe-avlm-editable-energon-super.sqsh/pytorch25.06-moe-avlm-editable-energon-super.sqsh \
./build_mcore_container.sh

# Test container
./interactive.sh --container $SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh

cd $SHARE_OUTPUT/VLMEvalKitMcore && python3 -c "
import torch; print(f'torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
import vlmeval; print(f'vlmeval: {vlmeval.__version__}')
import voicebench; print(f'voicebench: {voicebench.__version__}')
import av; print(f'av: {av.__version__}')
import decord; print(f'decord: {decord.__version__}')"
```

### Build vLLM Container

- We made small updates to vLLM registry to use latest naming conventions

```bash
cd $SHARE_OUTPUT/VLMEvalKitMcore

# Rebased onto vllm/main after commit 42c6bb4b7 (~1 week after 0.19.0 release)
VLLM_TAG=v0.19.0 VLLM_USE_PRECOMPILED=0 CONTAINER_SUFFIX="super-omni" EDITABLE=1 PARTITION=batch ./build_vllm_container.sh

# Test container; we expect 'Failed to load megatron' here, we're not installing megatron dependencies (e.g. transformer_engine)
./interactive.sh --container $SHARE_OUTPUT/vllm_containers/vllm-openai-v0.19.0-vlmeval-super-omni-editable.sqsh

cd $SHARE_OUTPUT/VLMEvalKitMcore && python3 -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
import vllm; print('vllm:', vllm.__version__)
import vlmeval; print('vlmeval: ok')
import voicebench; print('voicebench: ok')
import av; print('av:', av.__version__)"
```
## Vision Pre-training


### Vision Pre-training: Verify Script

- Generate command line from v3-nano PT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/v3_baseline/pretrain_vision_adaptor_packing_lower_bs.sh \
--dry-run
```

- Generate command line from LAX v3-super PT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/pretrain_super_final_ckpt_radiov4_1370_lax.sh \
--dry-run
```

- Generate command line from SVG v3-super PT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/pretrain_super_final_ckpt_radiov4_1377_svg.sh \
--dry-run
```

- Put each command line in a file, replace `\s+` with `\n` and diff v3-Nano -> v3-Super LAX -> v3-Super SVG

### Vision Pre-Training: Launch

- Run v3-Super SVG, Vision Pre-training
    - Batch size = 128 * TP = 2 = 256 max DP so max 32 nodes (256 GPUs)

```bash
# V1: Quick test
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/pretrain_super_final_ckpt_radiov4_1377_svg.sh \
--name pretrain_super_final_ckpt_radiov4_1377_0402 \
--num-jobs 1 \
--overwrite-code-snapshot \
--test-svg

# V2: Full test
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/pretrain_super_final_ckpt_radiov4_1377_svg.sh \
--name pretrain_super_final_ckpt_radiov4_1377_0402 \
--num-jobs 2
```

## Vision SFT

## Vision SFT: Verify Script

- Generate command line from v3-nano SFT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/v3_conv3d/sft_moe_rl_llm_eval_mode_radio_v4_two_epochs_bs_x2_videoaug.sh \
--dry-run
```

- Generate command line from LAX v3-super SFT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/sft_super_final_ckpt_conv3d_radiov4_1370_lax.sh \
--dry-run
```

- Generate command line from SVG v3-super SFT script

```bash
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/sft_super_final_ckpt_conv3d_radiov4_1377_svg.sh \
--dry-run
```

- Put each command line in a file, replace `\s+` with `\n` and diff v3-Nano -> v3-Super LAX -> v3-Super SVG

### Vision SFT: Launch

- Run v3-Super SVG, Vision SFT
    - Batch size = 128 * TP = 2 = 256 max DP so max 32 nodes (256 GPUs)
    - Using 64 nodes (default for this script) w/ 1 job
    - We know this is "bad" data, so using `_smokescreen` in name for now (will drop --name later)

```bash
# V1: Quick test; using `--overwrite-code-snapshot` by default
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/sft_super_final_ckpt_conv3d_radiov4_1377_svg.sh \
--name sft_super_final_ckpt_conv3d_radiov4_1377_0402 \
--num-jobs 1 \
--overwrite-code-snapshot \
--test-svg

# V2: Full run w/ bs=256 and lr=5e-5; add `--overwrite-code-snapshot` to update source
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/sft_super_final_ckpt_conv3d_radiov4_1377_svg.sh \
--name sft_super_final_ckpt_conv3d_radiov4_1377_0402 \
--num-jobs 3  # 13 total so far?

# V3: Full run; add `--overwrite-code-snapshot` to update source
examples/multimodal/launch.sh \
--sbatch examples/multimodal/super/sft_super_final_ckpt_conv3d_radiov4_2xbs_1377_svg.sh \
--name sft_super_final_ckpt_conv3d_radiov4_2xbs_1377_0402 \
--num-jobs 8
```

### Vision SFT: Evals

- For mcore (non-vLLM) testing: Create `config.yaml`

```bash
cd $SHARE_OUTPUT/megatron-lm
./interactive.sh

cd $SHARE_OUTPUT/megatron-lm

# Super needs TP=2,ETP=2 to avoid OOM
MODEL_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402
python examples/multimodal/tools/create_yaml_inference_config.py \
--model_name $MODEL_NAME \
--tensor-model-parallel-size 2 \
--expert-tensor-parallel-size 2
```

- Launch runs

```bash
cd $SHARE_OUTPUT/VLMEvalKitMcore

# Common: benchmarks
# TODO: Need to re-run OCRBenchV2
# BENCHMARKS=(
#     "InfoVQA_VAL" "TextVQA_VAL" "OCRBench"
#     "OCRBenchV2" "MMMU_DEV_VAL" "Video-MME"
#     "ScreenSpot_Pro_Development" "ScreenSpot_Pro_Creative" "ScreenSpot_Pro_CAD" "ScreenSpot_Pro_Scientific" "ScreenSpot_Pro_Office" "ScreenSpot_Pro_OS"
#     "MMLongBench_DOC"
# )

# Non-audio
# BENCHMARKS=("AI2D_TEST" "ChartQA_TEST" "DocVQA_VAL" "InfoVQA_VAL" "MathVista_MINI" "MMMU_DEV_VAL" "OCRBench" "OCRBenchV2" "TextVQA_VAL"
# "TreeBench" "CV-Bench-2D" "RefCOCO"
# "ScreenSpot_Mobile" "ScreenSpot_Desktop" "ScreenSpot_Web"
# "ScreenSpot_v2_Mobile" "ScreenSpot_v2_Desktop" "ScreenSpot_v2_Web"
# "ScreenSpot_Pro_Development" "ScreenSpot_Pro_Creative" "ScreenSpot_Pro_CAD" "ScreenSpot_Pro_Scientific" "ScreenSpot_Pro_Office" "ScreenSpot_Pro_OS"
# "MMLongBench_DOC" "OCR_Reasoning" "CharXiv_reasoning_val" "CharXiv_descriptive_val"
# "WeMath" "MathVerse_MINI_Vision_Only" "MathVision" "LogicVista"
# "Video-MME" "Video-MME-256" "MLVU" "MLVU-256" "LongVideoBench" "LongVideoBench-256")

# Non-audio rem
BENCHMARKS=("AI2D_TEST" "ChartQA_TEST" "DocVQA_VAL" "MathVista_MINI"  "OCRBenchV2"
"TreeBench" "CV-Bench-2D" "RefCOCO"
"ScreenSpot_Mobile" "ScreenSpot_Desktop" "ScreenSpot_Web"
"ScreenSpot_v2_Mobile" "ScreenSpot_v2_Desktop" "ScreenSpot_v2_Web"
"OCR_Reasoning" "CharXiv_reasoning_val" "CharXiv_descriptive_val"
"WeMath" "MathVerse_MINI_Vision_Only" "MathVision" "LogicVista"
"Video-MME-256" "MLVU" "MLVU-256" "LongVideoBench" "LongVideoBench-256")

# Full set
# BENCHMARKS=("AI2D_TEST" "ChartQA_TEST" "DocVQA_VAL" "InfoVQA_VAL" "MathVista_MINI" "MMMU_DEV_VAL" "OCRBench" "OCRBenchV2" "TextVQA_VAL"
# "TreeBench" "CV-Bench-2D" "RefCOCO"
# "ScreenSpot_Mobile" "ScreenSpot_Desktop" "ScreenSpot_Web"
# "ScreenSpot_v2_Mobile" "ScreenSpot_v2_Desktop" "ScreenSpot_v2_Web"
# "ScreenSpot_Pro_Development" "ScreenSpot_Pro_Creative" "ScreenSpot_Pro_CAD" "ScreenSpot_Pro_Scientific" "ScreenSpot_Pro_Office" "ScreenSpot_Pro_OS"
# "MMLongBench_DOC" "OCR_Reasoning" "CharXiv_reasoning_val" "CharXiv_descriptive_val"
# "WeMath" "MathVerse_MINI_Vision_Only" "MathVision" "LogicVista"
# "Video-MME" "Video-MME-256" "MLVU" "MLVU-256" "LongVideoBench" "LongVideoBench-256" "WorldSense-AVLM" "WorldSense-AVLM-256" "DailyOmni" "DailyOmni-256"
# "VoiceBench_ifeval" "VoiceBench_bbh" "VoiceBench_advbench" "VoiceBench_alpacaeval_full" "VoiceBench_commoneval" "VoiceBench_wildvoice" "VoiceBench_openbookqa" "VoiceBench_mmsu" "VoiceBench_sd-qa"
# "Earnings22_ASR_Test" "AMI_ASR_Test" "GigaSpeech_ASR_test" "LibriSpeech_test_clean" "LibriSpeech_test_other" "SPGISpeech_ASR_test" "TedLium_ASR_Test" "TedLium_Longform_ASR_Test" "VoxPopuli_ASR_test" "MMAU_test" "OmniBench")

# Option 1: Latest SFT; ran w/ mcore + vllm
MODEL_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402
EVAL_ITER_FLAGS="--eval-iters 10000,20000,30000,40000,50000,53520"
# MCORE_BENCHMARKS=("CONVERT_SUPER" "${BENCHMARKS[@]}")  # Yes conversion
MCORE_BENCHMARKS=("${BENCHMARKS[@]}")  # No conversion
EXTRA_ARGS=""  # Add --use-mtp after running w/o

# Option 2: Previous SFT; ran w/ mcore only, same tp_1 checkpoints + config.yaml
MODEL_NAME=sft_n3_super_20260202_1101_radiov4_1365_0211_dup_orig
EVAL_ITER_FLAGS="--eval-iters 10000,20000,30000,32916"
MCORE_BENCHMARKS=("${BENCHMARKS[@]}")  # No conversion

# Option 3: Previous SFT; ran w/ mcore + vllm, new tp_1/tp_1_hf checkpoints + config.yaml/config.json
MODEL_NAME=sft_n3_super_20260202_1101_radiov4_1365_0211_dup_new
# EVAL_ITER_FLAGS="--eval-iters 10000,20000,30000"
# EVAL_ITER_FLAGS="--eval-iters 10000,20000,30000,32916"
EVAL_ITER_FLAGS="--eval-iters 30000"
# EVAL_ITER_FLAGS="--eval-iters 20000,30000,32916"
# MCORE_BENCHMARKS=("CONVERT_SUPER" "${BENCHMARKS[@]}")
MCORE_BENCHMARKS=("${BENCHMARKS[@]}")

# Common: Mcore
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
shell/run_all_benchmark_auto.sh \
--model-name $MODEL_NAME \
--benchmarks "${MCORE_BENCHMARKS[*]}" \
--max-nodes 8 \
--megatron-src $SHARE_OUTPUT/megatron-lm \
$EVAL_ITER_FLAGS

# Common: vLLM; to run with MTP add `--use-mtp` (will output to a different dir)
INFERENCE_CONTAINER=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/cmccarthy/vllm_containers/vllm-openai-v0.19.0-vlmeval-super-omni-editable.sqsh \
shell/run_all_benchmark_vllm_auto.sh \
--model-name $MODEL_NAME \
--model-size super \
--benchmarks "${BENCHMARKS[*]}" \
--megatron-src $SHARE_OUTPUT/megatron-lm \
--use-defaults \
$EVAL_ITER_FLAGS \
$EXTRA_ARGS
```



- Export CSV for results

```bash
cd $SHARE_OUTPUT/VLMEvalKitMcore
source .venv/bin/activate

DIRS=(
    "$SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402/checkpoints/tp_1"
    "$SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402/checkpoints/tp_1_hf"
    # "$SHARE_OUTPUT/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211_dup_orig/checkpoints/tp_1"
    "$SHARE_OUTPUT/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211_dup_new/checkpoints/tp_1"
    "$SHARE_OUTPUT/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211_dup_new/checkpoints/tp_1_hf"
)
DATA_DIRS=$(IFS=' '; echo "${DIRS[*]}")

# Interactive
python3 viewer/app.py --data_dir $DATA_DIRS --megatron-path $SHARE_OUTPUT/megatron-lm --export-csv $SHARE_OUTPUT/workspace/output/benchmark_results_temp/sft_svg_vs_hsg.csv

# TODO COLLIN: Remove me before push
# Convert CSV to markdown comparison table (tp_1 vs tp_1_hf)
python3 ~/output/claude_scripts/csv_to_md.py $SHARE_OUTPUT/workspace/output/benchmark_results_temp/sft_svg_vs_hsg.csv
```

### Vision SFT: Evals on NRT

- Copy data to NRT

```bash
RUN_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402

# Copy the checkpoints folder for the iter(s) we want to run
ITER_FOLDER="iter_0053520"
rsync -avHP \
"nsc-svg-slurm-1-dc-02.nvidia.com:${SHARE_OUTPUT}/workspace/output/${RUN_NAME}/checkpoints/${ITER_FOLDER}/" \
"${SHARE_OUTPUT}/workspace/output/${RUN_NAME}/checkpoints/${ITER_FOLDER}/"

# Copy non-checkpoints
# Using same --exclude patterns as launch.sh, plus *.pt and *.safetensors
rsync -avHP \
"nsc-svg-slurm-1-dc-02.nvidia.com:${SHARE_OUTPUT}/workspace/output/${RUN_NAME}/" \
"${SHARE_OUTPUT}/workspace/output/${RUN_NAME}/" \
--exclude "*.pt" \
--exclude "*.safetensors" \
--exclude "__pycache__" \
--exclude "*.pyc" \
--exclude ".git/" \
--exclude "wandb/" \
--exclude ".venv/"
```

## Nano Test

```bash
# Run latest iter, TextVQA_VAL
cd $SHARE_OUTPUT/VLMEvalKitMcore

# Quick test, 30k iter only
MODEL_NAME=sft_mmlongbench_txt_0403
BENCHMARKS=("InfoVQA_VAL" "TextVQA_VAL")
MCORE_BENCHMARKS=("CONVERT_TO_TP_1" "${BENCHMARKS[@]}")
EVAL_ITER_FLAGS="--eval-iters 2200"

# Mcore
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
shell/run_all_benchmark_auto.sh \
--model-name $MODEL_NAME \
--benchmarks "${MCORE_BENCHMARKS[*]}" \
--max-nodes 8 \
--megatron-src $SHARE_OUTPUT/megatron-lm \
$EVAL_ITER_FLAGS

# vLLM; to run with MTP add `--use-mtp` (will output to a different dir)
# Need to use --model-size 30_3b and pass in MCORE_CONTAINER for SVG
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
INFERENCE_CONTAINER=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/cmccarthy/vllm_containers/vllm-openai-v0.19.0-vlmeval-super-omni-editable.sqsh \
shell/run_all_benchmark_vllm_auto.sh \
--model-name $MODEL_NAME \
--model-size 30_3b \
--benchmarks "CONVERT_TO_HF" \
--megatron-src $SHARE_OUTPUT/megatron-lm \
--use-defaults \
--no-reasoning \
--echo-only \
$EVAL_ITER_FLAGS
```

## Old Super Test

- Sanity check with tpoon checkpoints

```bash
# Symlink to a _dup dir for no-mtp checkpoint conversion
cd $SHARE_OUTPUT/megatron-lm

# Duplicate exactly, including all tp_1 checkpoints and top-level config.yaml
# We'll run these exactly, using mcore (only)
python3 examples/multimodal/tools/symlink_output_dir_files.py \
--src /scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211/ \
--dest $SHARE_OUTPUT/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211_dup_orig

# Duplicate again, w/o tp_1 and config.yaml; well run HF (which generates config.yaml) then mcore
python3 examples/multimodal/tools/symlink_output_dir_files.py \
--src /scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211/ \
--dest $SHARE_OUTPUT/workspace/output/sft_n3_super_20260202_1101_radiov4_1365_0211_dup_new \
--ignore-files config.yaml config.json \
--ignore-folders logs wandb dataloader tp_1 tp_1_hf

# Run latest iter, TextVQA_VAL
cd $SHARE_OUTPUT/VLMEvalKitMcore

MODEL_NAME=sft_n3_super_20260202_1101_radiov4_1365_0211_dup
BENCHMARKS=("InfoVQA_VAL" "TextVQA_VAL")
# EVAL_ITER_FLAGS="--eval-iters 32916"  # mcore, already exists
EVAL_ITER_FLAGS="--eval-iters 10000"  # vllm, needs to convert exists

# Mcore
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
shell/run_all_benchmark_auto.sh \
--model-name $MODEL_NAME \
--benchmarks "${BENCHMARKS[*]}" \
--max-nodes 8 \
--megatron-src $SHARE_OUTPUT/megatron-lm \
$EVAL_ITER_FLAGS

# vLLM; to run with MTP add `--use-mtp` (will output to a different dir)
INFERENCE_CONTAINER=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/cmccarthy/vllm_containers/vllm-openai-v0.19.0-vlmeval-super-omni-editable.sqsh \
shell/run_all_benchmark_vllm_auto.sh \
--model-name $MODEL_NAME \
--model-size super \
--benchmarks "${BENCHMARKS[*]}" \
--megatron-src $SHARE_OUTPUT/megatron-lm \
--use-defaults \
$EVAL_ITER_FLAGS
```