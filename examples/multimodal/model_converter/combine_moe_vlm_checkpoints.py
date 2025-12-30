import subprocess
import argparse

def parse_args():
    description = \
    """Combine MoE LLM + vision checkpoints

    Example usage:
    ```bash
    cd ~/megatron-lm
    ./interactive_submit.sh -g 4  # Match GPUs == TP size

    python examples/multimodal/combine_moe_vlm_checkpoints.py \
      --llm-checkpoint /lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant/checkpoints/iter_0015500 \
      --vision-checkpoint /lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/google_siglip2_so400m_patch16_512_mcore_tp_2/iter_0000001 \
      --output-checkpoint /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/vlm_checkpoints/v3-sft-16gbs-lurking-ringtail-lc-v2-1e-5-constant-siglip2-so400m-p16-512-tp2/iter_0000001 \
      --tp 2 \
      --ep 32
    ```
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--llm-checkpoint", type=str, required=True)
    parser.add_argument("--vision-checkpoint", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    return parser.parse_args()

def main(args):
    lm = args.llm_checkpoint
    vision = args.vision_checkpoint
    output = args.output_checkpoint
    tp = args.tp
    ep = args.ep

    command = "python examples/multimodal/combine_state_dicts.py --input \\ \n "

    for i in range(ep):
        j = i % tp
        command += f" {lm}/mp_rank_{j:02d}_{i:03d}/model_optim_rng.pt {vision}/mp_rank_{j:02d}/model_optim_rng.pt \\ \n "

    prefixes = " ".join(["language_model", "vision_model"] * ep)
    command += f" --prefixes \\ \n {prefixes} \\ \n "

    command += " --output  \\ \n "
    for i in range(ep):
        j = i % tp
        command += f" {output}/mp_rank_{j:02d}_{i:03d}/model_optim_rng.pt \\ \n "

    print(command)

    subprocess.run(command.replace("\n", "").replace("\\", ""), shell=True)

    last = output.split("/")[-1]
    latest = output.replace(last, "latest_checkpointed_iteration.txt")

    with open(f"{latest}", "w") as f:
        f.write("1")

if __name__ == "__main__":
    args = parse_args()
    main(args)