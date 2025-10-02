import argparse
import os
import torch

ep_size = 32
local_ep_size = 4
tp_size = 2

def convert_checkpoint(input_path, output_path):
    input_path = os.path.join(input_path, "checkpoints") if "checkpoints" not in input_path else input_path
    output_path = os.path.join(output_path, "checkpoints") if "checkpoints" not in output_path else output_path
    last_iter = int(open(os.path.join(input_path, "latest_checkpointed_iteration.txt")).read().strip())
    input_path = os.path.join(input_path, f"iter_{last_iter:07d}")

    new_sd = [dict() for _ in range(tp_size)]

    for ep in range(ep_size):
        tp = ep % tp_size
        path = os.path.join(input_path, f"mp_rank_{tp:02d}_{ep:03d}", "model_optim_rng.pt")
        sd = torch.load(path, weights_only=False)

        for k, v in sd["model"].items():
            v = v.cuda() if v is not None else v

            if ".experts." in k and "weight" in k:
                ks = k.split("weight")
                if len(ks) != 2:
                    print("mega failure")
                    breakpoint()
                new_k = ks[0]
                local_ep_rank = int(ks[1])
                global_ep_rank = ep * local_ep_size + local_ep_rank
                new_k += "weight" + str(global_ep_rank)
                for tp2 in range(tp_size):
                    new_sd[tp2][new_k] = v
            elif k in new_sd[tp]:
                if v is None:
                    assert new_sd[tp][k] is None
                else:
                    try:
                        assert torch.allclose(new_sd[tp][k], v)
                    except Exception as e:
                        breakpoint()
                        raise e
            else:
                new_sd[tp][k] = v

        print(f"converted {ep} / {ep_size}")


    for tp in range(tp_size):
        os.makedirs(os.path.join(output_path, f"iter_{last_iter:07d}", f"mp_rank_{tp:02d}"), exist_ok=True)
        sd["model"] = new_sd[tp]
        torch.save(sd, os.path.join(output_path, f"iter_{last_iter:07d}", f"mp_rank_{tp:02d}", "model_optim_rng.pt"))
        print(f"saved {tp} / {tp_size}")

    with open(os.path.join(output_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(last_iter))

if __name__ == "__main__":
    # Add args parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)

