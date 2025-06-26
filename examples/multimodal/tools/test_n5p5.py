import torch
import sys

sys.path.append("/lustre/fsw/portfolios/llmservice/users/matthieul/repos_rebase/megatron-lm-vlm-hybrid-packing")

# ckpt_path = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_tp_8/torch/iter_1600000/mp_rank_00/model_optim_rng.pt"
ckpt_path = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_tp_4_dq/torch/iter_2560000/mp_rank_00/model_optim_rng.pt"

ckpt = torch.load(ckpt_path, weights_only=False)

print(ckpt['model']["decoder.layers.3.mlp.linear_fc2.weight"])
