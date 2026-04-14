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
import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
import vlmeval; print('vlmeval: ok')
import voicebench; print('voicebench: ok')"
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
import av; print('av:', av.__version__)
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

# Quick test, 30k iter only
MODEL_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402
# BENCHMARKS=(
#     "OCRBench" "OCRBenchV2" "MMMU_DEV_VAL" "Video-MME"
#     "ScreenSpot_Pro_Development" "ScreenSpot_Pro_Creative" "ScreenSpot_Pro_CAD" "ScreenSpot_Pro_Scientific" "ScreenSpot_Pro_Office" "ScreenSpot_Pro_OS" "MMLongBench_DOC"
# )
BENCHMARKS=("TextVQA_VAL")
MCORE_BENCHMARKS=("${BENCHMARKS[@]}")  # Prepend "CONVERT_SUPER" to convert checkpoint
EVAL_ITER_FLAGS="--eval-iters 53520"

# Mcore
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
shell/run_all_benchmark_auto.sh \
--model-name $MODEL_NAME \
--benchmarks "${MCORE_BENCHMARKS[*]}" \
--max-nodes 8 \
--megatron-src $SHARE_OUTPUT/megatron-lm \
--disable-mtp \
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

- Alternative test with tp1 checkpoint w/o mtp

```bash
# Symlink to a _dup dir for no-mtp checkpoint conversion
cd $SHARE_OUTPUT/megatron-lm

python3 examples/multimodal/tools/symlink_output_dir_files.py \
--src $SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402 \
--dest $SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402_dup \
--ignore-files config.yaml config.json \
--ignore-folders wandb logs dataloader tp_1 tp_1_hf

# Create inference yaml w/o mtp flag
# Super needs TP=2,ETP=2 to avoid OOM
./interactive.sh

cd $SHARE_OUTPUT/megatron-lm

MODEL_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402_dup
python examples/multimodal/tools/create_yaml_inference_config.py \
--model_name $MODEL_NAME \
--tensor-model-parallel-size 2 \
--expert-tensor-parallel-size 2 \
--disable-mtp

# Run evals w/ _dup
cd $SHARE_OUTPUT/VLMEvalKitMcore

MODEL_NAME=sft_super_final_ckpt_conv3d_radiov4_1377_0402_dup
# MCORE_BENCHMARKS=("CONVERT_SUPER" "InfoVQA_VAL")
MCORE_BENCHMARKS=("TextVQA_VAL")
EVAL_ITER_FLAGS="--eval-iters 53520"

# Mcore
MCORE_CONTAINER=$SHARE_OUTPUT/mcore_containers/pytorch25.06-moe-avlm-editable-energon-super-vlmeval-voicebench.sqsh \
shell/run_all_benchmark_auto.sh \
--model-name $MODEL_NAME \
--benchmarks "${MCORE_BENCHMARKS[*]}" \
--max-nodes 8 \
--megatron-src $SHARE_OUTPUT/megatron-lm \
$EVAL_ITER_FLAGS
```

- Export CSV for results

```bash
cd $SHARE_OUTPUT/megatron-lm
./interactive_super_cpu.sh

cd $SHARE_OUTPUT/VLMEvalKitMcore

DIRS=(
    "$SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402/checkpoints/tp_1"
    "$SHARE_OUTPUT/workspace/output/sft_super_final_ckpt_conv3d_radiov4_1377_0402/checkpoints/tp_1_hf"
)
DATA_DIRS=$(IFS=' '; echo "${DIRS[*]}")

# Interactive
python viewer/app.py --data_dir $DATA_DIRS --megatron-path $SHARE_OUTPUT/megatron-lm
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