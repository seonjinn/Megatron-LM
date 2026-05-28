ep = 32
tp = 2

command = "python examples/multimodal/combine_state_dicts.py --input \ \n "
lm = "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/checkpoints/nano-v3-rc2-12-03-25-7-15-pm-mcore-tp2-ep32/iter_0000001"
# vision = "/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/c-radio_v2-vlm-h-tp2/iter_0000001"
vision = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/artifacts/mcore/radio_checkpoints/c-radio-v4-h-tp2/iter_0000001"
output = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nano-v3-rc2-12-03-25-7-15-pm-c-radio_v4-vlm-h-tp2-ep32"

for i in range(ep):
    j = i % tp
    command += f" {lm}/mp_rank_{j:02d}_{i:03d}/model_optim_rng.pt {vision}/mp_rank_{j:02d}/model_optim_rng.pt \ \n "

command += " --prefixes \ \n "
c = ["language_model", "vision_model"] * 32
command += " ".join(c)

command += " \ \n --output  \ \n "

for i in range(ep):
    j = i % tp
    command += f" {output}/mp_rank_{j:02d}_{i:03d}/model_optim_rng.pt \ \n "

print(command)

import subprocess

subprocess.run(command.replace("\n", "").replace("\\", ""), shell=True)

last = output.split("/")[-1]
latest = output.replace(last, "latest_checkpointed_iteration.txt")

with open(f"{latest}", "w") as f:
    f.write("1")