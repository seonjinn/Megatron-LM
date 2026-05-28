import torch
import os
import argparse

orig_dir_TP_8 = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/N5p5_phase3_blend2_torch_cradio_vlm_v1_rc3_tp8_reinit_patched-no-extra-state"
orig_dir_TP_4 = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/mcore_mmodal_models/llama_3p1_8b_c-radio-vlm_v1_rc3-no-extra-state"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--iter", type=int, required=True)

    args = parser.parse_args()
    prefix = "language_model."
    if args.tp == 8:
        orig_dir = orig_dir_TP_8
    elif args.tp == 4:
        orig_dir = orig_dir_TP_4
    else:
        raise ValueError(f"Invalid tp: {args.tp}")

    iter_dir = f"iter_{args.iter:07d}"

    for i in range(args.tp):
        orig_path = os.path.join(orig_dir, iter_dir, f"mp_rank_0{i}", "model_optim_rng.pt")
        new_path = os.path.join(args.output_dir, iter_dir, f"mp_rank_0{i}", "model_optim_rng.pt")
        os.makedirs(os.path.join(args.output_dir, iter_dir, f"mp_rank_0{i}"), exist_ok=True)
        llm_path = os.path.join(args.input_dir, iter_dir, f"mp_rank_0{i}", "model_optim_rng.pt")

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