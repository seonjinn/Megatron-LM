# AVLM + AU-Harness Evaluation

Evaluate AVLM models using [AU-Harness](https://github.com/NVIDIA/AU-Harness) audio benchmarks.

## Quick Start

```bash
bash run_avlm_au_harness.sh
```

## Configuration

Edit `run_avlm_au_harness.sh` to set:

| Variable | Description |
|----------|-------------|
| `MEGATRON_PATH` | Path to megatron-lm |
| `MODEL_PATH` | AVLM checkpoint (TP=1 format) |
| `MODEL_CONFIG_YAML` | Model config file |
| `AU_HARNESS_DIR` | Path to AU-Harness |

## AU-Harness Config

Edit `${AU_HARNESS_DIR}/run_configs/megatron.yaml` to configure:

- **Tasks & metrics**: Which benchmarks to run and which metrics to use for each
- **timeout**: Request timeout
- **generation_params**: Temperature, max tokens, etc.
- **prompt_overrides**: Custom prompts per task
- and more

Example config:

```yaml
task_metric:
  - ["librispeech_test_clean", "word_error_rate"]
  - ["librispeech_test_other", "word_error_rate"]

models:
  - name: "megatron-avlm"
    inference_type: "vllm"
    url: "http://localhost:8000"
    batch_size: 1
    chunk_size: 30
    timeout: 120
```

See [AU-Harness documentation](https://github.com/NVIDIA/AU-Harness) for all available options.
