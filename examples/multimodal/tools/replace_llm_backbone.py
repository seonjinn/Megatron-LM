import torch
import os

# orig_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/N5p5_phase3_blend2_torch_cradio_vlm_v1_rc3_tp8_reinit_patched-no-extra-state"
# new_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/nemotron5p5_hybrid_12b_patch_vocab_cradio_vlm_v1_rc3_tp8"
# llm_dir = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_patch_vocab_tp_8/iter_1600000"
# prefix = "language_model."
# TP = 8

# orig_dir = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-vlm_v1_rc3-no-extra-state"
# new_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/nemotron5p5_hybrid_12b_patch_vocab_cradio_vlm_v1_rc3_tp4"
# llm_dir = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_patch_vocab_tp_4/iter_1600000"
# prefix = "language_model."
# TP = 4

# orig_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/N5p5_phase3_blend2_torch_cradio_vlm_v1_rc3_tp8_reinit_patched-no-extra-state"
# new_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/nemotron5p5_hybrid_12b_dq_patch_vocab_cradio_vlm_v1_rc3_tp8"
# llm_dir = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_tp_8_dq_patch_vocab/iter_2560000"
# prefix = "language_model."
# TP = 8

# orig_dir = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-vlm_v1_rc3-no-extra-state"
# new_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/nemotron5p5_hybrid_12b_dq_patch_vocab_cradio_vlm_v1_rc3_tp4"
# llm_dir = "/lustre/fsw/portfolios/llmservice/users/matthieul/workspace/output/nemotron5p5_hybrid_12b_tp_4_dq_patch_vocab/iter_2560000"
# prefix = "language_model."
# TP = 4

orig_dir = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/qwen2.5-7B-instruct-cradio-v3-g-mcore-tp4"
new_dir = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-g-v3-no-extra-state"
llm_dir = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-vlm_v1_rc3-no-extra-state/iter_0000001/"
prefix = ""
TP = 4

for i in range(TP):
    orig_path = os.path.join(orig_dir, "iter_0000001", f"mp_rank_0{i}", "model_optim_rng.pt")
    new_path = os.path.join(new_dir, "iter_0000001", f"mp_rank_0{i}", "model_optim_rng.pt")
    os.makedirs(os.path.join(new_dir, "iter_0000001", f"mp_rank_0{i}"), exist_ok=True)
    llm_path = os.path.join(llm_dir, f"mp_rank_0{i}", "model_optim_rng.pt")

    state_dict = torch.load(orig_path, weights_only=False)
    new_state_dict = state_dict.copy()
    new_state_dict["model"] = dict()

    for k, v in state_dict["model"].items():
        if "vision_model" in k:
            new_state_dict["model"][k] = v

    llm_state_dict = torch.load(llm_path, weights_only=False)
    for k, v in llm_state_dict["model"].items():
        if "vision_model" not in k:
            new_state_dict["model"][f"{prefix}{k}"] = v

    torch.save(new_state_dict, new_path)
    print("done", i)