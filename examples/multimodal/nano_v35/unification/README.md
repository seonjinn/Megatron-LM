# Nano v3.5 unification VLM experiment

This directory contains the self-contained job scripts and checkpoint helpers
for the Nano v3.5 VLM experiment. The workflow converts the base Hugging Face
LLM checkpoint to MCore TP2/EP32, combines it with RADIO v4, pretrains the
vision adapter, and then runs multimodal SFT.

The experiment package has no runtime dependency on the sibling `oci-nrt`
directory. Cluster defaults live in `experiment_config.sh`; checkpoint
conversion and RADIO combination are implemented directly in their respective
jobs.

The two training stages are standalone Slurm scripts; they do not go through a
second launcher:

- `pretrain_vision_adapter.sh` uses
  `examples/multimodal/v3_baseline/pretrain_vision_adaptor_recipe_1377_dss.yaml`.
- `sft_vlm.sh` uses
  `examples/multimodal/v3_baseline/sft_combined_omni_16k_vlm_only_webbrowse_dss.yaml`.

Both recipes resolve `dss://` datasets through
`NVDATASET_CACHE_DIR=/home/svc-dss/cache/nemotron`. Both jobs use the
`nemotron_n4_post` account, write under
`/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/$USER/workspace/output`
by default, use the shared training image at
`/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/containers/pytorch25.11-moe-avlm-editable-energon-super-triton35.sqsh`,
and inherit the W&B API key from the submitted shell environment.

Run the existing checkpoint through the two training stages from the
Megatron-LM repository root:

```bash
experiment_dir=examples/multimodal/nano_v35/unification
pretrain_job=$(sbatch --parsable "${experiment_dir}/pretrain_vision_adapter.sh")
sbatch --dependency="afterok:${pretrain_job}" "${experiment_dir}/sft_vlm.sh"
```

If the converted/combined bootstrap checkpoint must be recreated first:

```bash
experiment_dir=examples/multimodal/nano_v35/unification
convert_job=$(sbatch --parsable "${experiment_dir}/convert_base_hf_to_mcore.sh")
combine_job=$(sbatch --parsable --dependency="afterok:${convert_job}" "${experiment_dir}/combine_base_with_radio.sh")
pretrain_job=$(sbatch --parsable --dependency="afterok:${combine_job}" "${experiment_dir}/pretrain_vision_adapter.sh")
sbatch --dependency="afterok:${pretrain_job}" "${experiment_dir}/sft_vlm.sh"
```

Paths and run names can be overridden with environment variables such as
`BASE_HF_CKPT_DIR`, `BASE_MCORE_CKPT_DIR`, `BASE_VLM_CKPT_DIR`,
`VISION_PRETRAIN_CKPT_DIR`, `TOKENIZER_MODEL`, `CONTAINER_IMAGE`,
`DATA_TRAIN`, and `OUTPUT_BASE`.

The collated local recipe and `data_yamls/` tree remain here as provenance; the
production SFT job uses the DSS recipe above.

`eval_sft_checkpoints.sh` submits the MCore, reasoning-disabled VLM evaluation
suite for SFT iterations 3284, 5000, and 6435 by default. Set
`BENCHMARKS_OVERRIDE` to a comma-separated subset when only selected datasets
should be submitted. For example:

```bash
BENCHMARKS_OVERRIDE=AI2D_TEST,OCRBench ADD_CONVERSION=false \
  ./examples/multimodal/nano_v35/unification/eval_sft_checkpoints.sh
```
