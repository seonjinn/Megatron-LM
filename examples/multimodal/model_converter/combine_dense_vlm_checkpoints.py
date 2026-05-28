import subprocess
import argparse

def parse_args():
    description = """Combine dense LLM + vision checkpoints

    Example usage:
    ```bash
    cd ~/megatron-lm
    ./interactive_submit.sh -g 4  # Match GPUs == TP size

    python examples/multimodal/combine_dense_vlm_checkpoints.py \
      --llm-checkpoint /lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/torch_patched/iter_0000000 \
      --vision-checkpoint /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v2-vlm-tp4/iter_0000001 \
      --output-checkpoint /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/nemotron_5p5_9b_v2_c_radio_v2_vlm_tp4/iter_0000001 \
      --tp 4
    ```
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--llm-checkpoint", type=str, required=True)
    parser.add_argument("--vision-checkpoint", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    return parser.parse_args()

def main(args):
    lm = args.llm_checkpoint
    vision = args.vision_checkpoint
    output = args.output_checkpoint
    tp = args.tp

    command = "python examples/multimodal/combine_state_dicts.py --input \\ \n "

    for i in range(tp):
        command += f" {lm}/mp_rank_{i:02d}/model_optim_rng.pt {vision}/mp_rank_{i:02d}/model_optim_rng.pt \\ \n "

    prefixes = " ".join(["language_model", "vision_model"] * tp)
    command += f" --prefixes \\ \n {prefixes} \\ \n "

    command += " --output  \\ \n "
    for i in range(tp):
        command += f" {output}/mp_rank_{i:02d}/model_optim_rng.pt \\ \n "

    print(command)

    subprocess.run(command.replace("\n", "").replace("\\", ""), shell=True, check=True)

    last = output.split("/")[-1]
    latest = output.replace(last, "latest_checkpointed_iteration.txt")


    with open(f"{latest}", "w") as f:
        f.write("1")

if __name__ == "__main__":
    args = parse_args()
    main(args)