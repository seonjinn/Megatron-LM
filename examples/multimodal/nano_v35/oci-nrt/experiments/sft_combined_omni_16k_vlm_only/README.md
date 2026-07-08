# Nano v3.5 VLM-only experiment

This experiment starts from `resources/checkpoints/base_nano3p5_hf`, converts the
LLM to MCore TP2/EP32, combines it with the RADIO v4 vision checkpoint, pretrains
the vision adapter, and then runs VLM SFT with the local VLM-only recipe.

Run from the Megatron-LM repository root:

```bash
experiment_dir=examples/multimodal/nano_v35/oci-nrt/experiments/sft_combined_omni_16k_vlm_only
convert_job=$(sbatch --parsable "${experiment_dir}/convert_base_hf_to_mcore.sh")
combine_job=$(sbatch --parsable --dependency="afterok:${convert_job}" "${experiment_dir}/combine_base_with_radio.sh")
pretrain_job=$(sbatch --parsable --dependency="afterok:${combine_job}" "${experiment_dir}/pretrain_vision_adapter.sh")
sbatch --dependency="afterok:${pretrain_job}" "${experiment_dir}/sft_vlm.sh"
```

The top-level recipe uses a local Eagle v14 derivative that changes only its
CharXiv child to the judge-filtered, leakage-filtered Nano v3 mix. Every nested
recipe YAML is collated under `data_yamls/`; its README records the original
source of each file. Leaf dataset and media paths remain absolute. In
particular, the HopChain multipage JSONL indexes must remain readable by compute
jobs. The corrected recipe expands to 829 datasets. Compute-side validation
currently reports a missing ScaleCUA media file in the upstream Eagle v14 mix.

`eval_sft_checkpoints.sh` submits the MCore, reasoning-disabled VLM evaluation
suite for SFT iterations 3284, 5000, and 6435 by default. It generates one TP1
conversion per checkpoint and makes each benchmark depend on its conversion.
The launcher requires `OPENAI_API_KEY` for standard MathVista and MMLongBench
scoring, uses the `nemotron_omni_vision` Slurm account, and defaults to the
local 25.11 Mamba + VLMEval container.

Set `BENCHMARKS_OVERRIDE` to a comma-separated subset when only selected
datasets should be submitted. For example:

```bash
BENCHMARKS_OVERRIDE=AI2D_TEST,OCRBench ADD_CONVERSION=false \
  ./examples/multimodal/nano_v35/oci-nrt/experiments/sft_combined_omni_16k_vlm_only/eval_sft_checkpoints.sh
```
