import torch
import os

original_ckpt = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-g-v3-no-extra-state"
new_ckpt =      "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-g-v3-no-extra-state"
TP = 4

for i in range(TP):

    orig_path = os.path.join(original_ckpt, "iter_0000001", f"mp_rank_0{i}", "model_optim_rng.pt")
    new_dir_tp = os.path.join(new_ckpt, "iter_0000001", f"mp_rank_0{i}")
    os.makedirs(new_dir_tp, exist_ok=True)
    new_path = os.path.join(new_dir_tp, "model_optim_rng.pt")

    state_dict = torch.load(orig_path, weights_only=False)
    state_dict["model"] = {k: v for k, v in state_dict["model"].items() if "_extra_state" not in k}

    torch.save(state_dict, new_path)
