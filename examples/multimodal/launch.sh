#!/bin/bash

# --------------------
# Example usage:
# --------------------
# - - - - - - - - - -
# Normal launch
# - - - - - - - - - -
# ./examples/multimodal/launch.sh \
#   --name sft_moe_rl_llm_eval_mode_radio_multi_two_epochs_bs_x2_1212 \
#   --sbatch examples/multimodal/v3_experiments/sft_moe_rl_llm_eval_mode_radio_multi_two_epochs_bs_x2.sh \
# - - - - - - - - - -
# Important args
# - - - - - - - - - -
# - Prepend any env vars like `MY_VAR=my_val launch.sh <args>` to pass MY_VAR to sbatch script
# - Add `--overwrite-code-snapshot` to overwrite the code snapshot (default is to not override)
# - Add `--num-jobs N` to launch N jobs
# - Add `--dry-run` to only print the launch command (for testing any sbatch script changes)
# - Add `--debug` when attached to an interactive node to pause after DDP init and wait for debugger (prints host + port)
# - Add `--test-nrt` to launch 4-node test run for 15-minutes (appends `_test` to output dir as well)
# - Add `--source DIR` to use a different mcore source dir other than `pwd`
# --------------------

# Defaults
NUM_JOBS=1
SOURCE=`pwd`
OVERWRITE_CODE_SNAPSHOT=0
CODE_SNAPSHOT_FOLDER="code_snapshot"
DRY_RUN=0
DEPENDENCY=""
DURATION_HRS=""
DURATION_MINS=""
NODES=""
PARTITION=""
EXCLUDE=""
DEBUG=0
TEST_NRT=0
TEST_NRT_32N=0
TEST_SVG=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sbatch) SBATCH_FILE="$2"; shift ;;  # Required
        --name) MODEL_NAME="$2"; shift ;;  # Required
        --dependency) DEPENDENCY="$2"; shift ;;
        --num-jobs) NUM_JOBS="$2"; shift ;;
        --source) SOURCE="$2"; shift ;;
        --duration-hrs) DURATION_HRS="$2"; shift ;;
        --duration-mins) DURATION_MINS="$2"; shift ;;
        --nodes) NODES="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --exclude) EXCLUDE="$2"; shift ;;
        --overwrite-code-snapshot) OVERWRITE_CODE_SNAPSHOT=1 ;;
        --snapshot-folder) CODE_SNAPSHOT_FOLDER="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        --debug) DEBUG=1 ;;
        --test-nrt) TEST_NRT=1 ;;
        --test-nrt-32n) TEST_NRT_32N=1 ;;
        --test-svg) TEST_SVG=1 ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [args...]"
            exit 1
            ;;
    esac
    shift
done

if [[ $TEST_NRT -eq 1 || $TEST_NRT_32N -eq 1 || $TEST_SVG -eq 1 ]]; then
    NUM_JOBS=1
    if [[ -z "$DURATION_MINS" ]]; then
        DURATION_MINS=15
    fi
    if [[ $TEST_NRT_32N -eq 1 ]]; then
        NODES=32
        PARTITION="backfill,batch_block1"
    elif [[ $TEST_NRT -eq 1 ]]; then
        NODES=4
        PARTITION="backfill,batch_short,batch_block1"
    elif [[ $TEST_SVG -eq 1 ]]; then
        # EP=64 MoE needs at least 8 nodes with 8 GPUs/node for TP=2, PP=1.
        NODES=8
        PARTITION="batch"
    fi
    MODEL_NAME="${MODEL_NAME}_test"

    echo "Running in test mode with args:"
    echo "  model_name: ${MODEL_NAME}"
    echo "  num_jobs: ${NUM_JOBS}"
    echo "  duration_mins: ${DURATION_MINS}"
    echo "  nodes: ${NODES}"
    echo "  partition: ${PARTITION}"
fi

# Verify args
if [[ -z "$SBATCH_FILE" || -z "$MODEL_NAME" ]]; then
    echo -e "\nUsage: $0"
    echo "  --sbatch <my_sbatch.sh>"
    echo "  --name <job_name> [options...]"
    echo "Options:"
    echo "  --dependency <job_id> (default: none)"
    echo "  --num-jobs <num_jobs> (default: 1)"
    echo "  --source <source_dir> (default: current directory)"
    echo "  --exclude <nodelist> (default: none)"
    echo "  --overwrite-code-snapshot (default: no)"
    echo "  --snapshot-folder <snapshot_folder> (default: code_snapshot)"
    echo "  --update-code-only (default: no)"
    echo "  --dry-run (default: no)"
    echo -e "  --debug (default: no)\n"
    if [[ -z "$SBATCH_FILE" ]]; then echo "Error: Missing --sbatch (required arg)"; fi
    if [[ -z "$MODEL_NAME" ]]; then echo "Error: Missing --name (required arg)"; fi
    exit 1
fi

# Verify sbatch script is found
if [[ ! -f "$SBATCH_FILE" ]]; then
    echo "Error: Sbatch script $SBATCH_FILE not found"
    exit 1
fi

USER=${SLURM_JOB_USER:-${USER}}

OUTPUT_BASE="/lustre/fsw/portfolios/llmservice/users/${USER}/workspace/output/${MODEL_NAME}"
CODE_SNAPSHOT_DIR="${OUTPUT_BASE}/${CODE_SNAPSHOT_FOLDER}"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p $LOG_DIR

if [[ $OVERWRITE_CODE_SNAPSHOT -eq 1 ]]; then
    OVERWRITE_CODE_SNAPSHOT_STR="yes"
else
    OVERWRITE_CODE_SNAPSHOT_STR="no"
fi

echo "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
echo "Launching run(s) for ${MODEL_NAME}:"
echo "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
echo "  num jobs: ${NUM_JOBS}"
echo "  overwrite code snapshot: ${OVERWRITE_CODE_SNAPSHOT_STR}"
echo "  source: ${SOURCE}"
echo "  output base: ${OUTPUT_BASE}"

if [[ ! -z "$DEPENDENCY" ]]; then
    echo "  dependency: ${DEPENDENCY}"
fi

if [[ ! -z "$DURATION_HRS" ]]; then
    echo "  duration_hrs: ${DURATION_HRS}"
fi

if [[ ! -z "$DURATION_MINS" ]]; then
    echo "  duration_mins: ${DURATION_MINS}"
fi

if [[ ! -z "$NODES" ]]; then
    echo "  nodes: ${NODES}"
fi

if [[ ! -z "$PARTITION" ]]; then
    echo "  partition: ${PARTITION}"
fi

if [[ ! -z "$EXCLUDE" ]]; then
    echo "  exclude: ${EXCLUDE}"
fi

# Verify script supports MODEL_NAME by checking `MODEL_NAME=${MODEL_NAME:-`
if grep -q 'MODEL_NAME=\${MODEL_NAME:-' "$SBATCH_FILE"; then
    EXPORT_MODEL_NAME=1
else
    EXPORT_MODEL_NAME=0
    echo -e "\nWARNING: Script $SBATCH_FILE does not support MODEL_NAME environment variable, not setting MODEL_NAME explicitly"
    echo "To propagate MODEL_NAME to the script, add: MODEL_NAME=\${MODEL_NAME:-default_model_name}"
fi

# If using DRY_RUN, verify script supports DRY_RUN by checking `if [[ $DRY_RUN -eq 1 ]];`
if [[ $DRY_RUN -eq 1 ]] && ! grep -q 'DRY_RUN' "$SBATCH_FILE"; then
    echo -e "\nERROR: Sbatch script $SBATCH_FILE does not support DRY_RUN. Update script or remove --dry-run flag"
    exit 1
fi

# If using DEBUG, verify script supports DEBUG by checking `if [[ $DEBUG -eq 1 ]];`
if [[ $DEBUG -eq 1 ]] && ! grep -q 'DEBUG' "$SBATCH_FILE"; then
    echo -e "\nERROR: Sbatch script $SBATCH_FILE does not support DEBUG. Update script or remove --dry-run flag"
    exit 1
fi

echo "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "

# If we've SSH'd into the allocation from a new terminal, `srun` will be on path still,
#   so need to explicitly pass in `INTERACTIVE=1 <script>` (allow override here)
INTERACTIVE=${INTERACTIVE:-$(which srun >/dev/null 2>&1 && echo 0 || echo 1)}

if [[ $DRY_RUN -eq 0 && $DEBUG -eq 0 && $INTERACTIVE -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " RESPONSE
    if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Response: $RESPONSE. Launching runs."
    else
        echo "Response: $RESPONSE. Exiting."
        exit 1
    fi
fi

# Snapshot the source code into the OUTPUT directory on first run, and always run from the snapshot thereafter
# When debugging or dry-run, use SOURCE directly (no snapshot)
if [[ $DRY_RUN -eq 0 && $DEBUG -eq 0 && $INTERACTIVE -eq 0 && (! -d "${CODE_SNAPSHOT_DIR}" || $OVERWRITE_CODE_SNAPSHOT -eq 1) ]]; then
    if [[ -d "${CODE_SNAPSHOT_DIR}" ]]; then
        echo "Updating code snapshot directory: ${CODE_SNAPSHOT_DIR}"
    else
        echo "Creating code snapshot directory: ${CODE_SNAPSHOT_DIR}"
        mkdir -p ${CODE_SNAPSHOT_DIR}
    fi

    rsync -aH --stats --delete \
        --exclude "__pycache__" \
        --exclude "*.pyc" \
        --exclude ".git/" \
        --exclude "wandb/" \
        "${SOURCE}/" "${CODE_SNAPSHOT_DIR}/"
fi

# Use SOURCE directly when debugging or dry-run, otherwise use the code snapshot
if [[ $DEBUG -eq 1 || $DRY_RUN -eq 1 || $INTERACTIVE -eq 1 ]]; then
    CODE_DIR="${SOURCE}"
else
    CODE_DIR="${CODE_SNAPSHOT_DIR}"
fi

# Launch one or more jobs
cd $CODE_DIR

# Using --export=ALL is the default, just being explicit
# This allows us to pass in vars to our sbatch script like `MY_VAR=val ./launch.sh <args>`
# Manually adding MODEL_NAME and DRY_RUN here for convenience
if [[ $EXPORT_MODEL_NAME -eq 1 ]]; then
    export MODEL_NAME=$MODEL_NAME
fi

if [[ $DRY_RUN -eq 1 ]]; then
    # Don't actually launch, just set DRY_RUN and run the script, then exit
    DRY_RUN=1 $SBATCH_FILE
    exit 0
elif [[ $DEBUG -eq 1 ]]; then
    # Run the script directly with DEBUG=1 (no sbatch)
    DEBUG=1 $SBATCH_FILE
    exit 0
elif [[ $INTERACTIVE -eq 1 ]]; then
    # Interactive mode; run the script directly
    "$SBATCH_FILE"
    exit 0
else
    for ((i=1; i<=${NUM_JOBS}; i++)); do
        echo "Model $MODEL_NAME: launching run $i of $NUM_JOBS"

        EXTRA_SBATCH_ARGS=""
        if [[ ! -z "$DEPENDENCY" ]]; then
            EXTRA_SBATCH_ARGS+="--dependency=singleton,afterok:$DEPENDENCY "
        else
            EXTRA_SBATCH_ARGS+="--dependency=singleton "
        fi

        if [[ ! -z "$DURATION_HRS" ]]; then
            EXTRA_SBATCH_ARGS+="--time=${DURATION_HRS}:00:00 "
        fi

        if [[ ! -z "$DURATION_MINS" ]]; then
            EXTRA_SBATCH_ARGS+="--time=0:${DURATION_MINS}:00 "
        fi

        if [[ ! -z "$NODES" ]]; then
            EXTRA_SBATCH_ARGS+="--nodes=${NODES} "
        fi

        if [[ ! -z "$PARTITION" ]]; then
            EXTRA_SBATCH_ARGS+="--partition=${PARTITION} "
        fi

        if [[ ! -z "$EXCLUDE" ]]; then
            EXTRA_SBATCH_ARGS+="--exclude=${EXCLUDE} "
        fi

        sbatch \
        --export=ALL \
        --job-name=$MODEL_NAME \
        --output="$LOG_DIR/%x_%j_sbatch.log" \
        $EXTRA_SBATCH_ARGS \
        "$SBATCH_FILE"
    done
fi
