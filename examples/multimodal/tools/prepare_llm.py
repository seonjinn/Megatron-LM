import torch
import os
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--iter", type=int, required=True)

    args = parser.parse_args()

    iter_dir = f"iter_{args.iter:07d}"

    # We're patching the embeddings weights from size 131072 to size 131584
    if args.tp == 8:
        num_pad = 132096 - 131072
    elif args.tp == 4:
        # num_pad = 131584 - 131072
        num_pad = 132096 - 131072

    # First, load all the embeddings and concatenate them.
    input_embeddings = []
    output_embeddings = []
    for i in range(args.tp):
        orig_path = os.path.join(args.input_dir, iter_dir, f"mp_rank_0{i}", "model_optim_rng.pt")
        state_dict = torch.load(orig_path, weights_only=False)
        input_embeddings.append(state_dict["model"]["embedding.word_embeddings.weight"].cpu())
        output_embeddings.append(state_dict["model"]["output_layer.weight"].cpu())

    input_embeddings = torch.concat(input_embeddings)
    output_embeddings = torch.concat(output_embeddings)

    plt.plot(input_embeddings.to(torch.float32).numpy().mean(axis=1)[:2000])
    plt.savefig("input_embeddings.png")
    plt.close()
    plt.plot(output_embeddings.to(torch.float32).numpy().mean(axis=1)[:2000])
    plt.savefig("output_embeddings.png")
    plt.close()

    plt.plot(input_embeddings.to(torch.float32).numpy().mean(axis=1)[:2000])
    plt.savefig("input_embeddings_patched.png")
    plt.close()
    plt.plot(output_embeddings.to(torch.float32).numpy().mean(axis=1)[:2000])
    plt.savefig("output_embeddings_patched.png")
    plt.close()

    # Padd the embeddings with new random embeddings at the end of the embedding table.
    input_embeddings_padded = torch.concat((
        input_embeddings, torch.randn(num_pad, input_embeddings.shape[1]) * input_embeddings.std() + input_embeddings.mean()
    )).to(input_embeddings.dtype)
    output_embeddings_padded = torch.concat((
        output_embeddings, torch.randn(num_pad, output_embeddings.shape[1]) * output_embeddings.std() + output_embeddings.mean()
    )).to(output_embeddings.dtype)

    # Save the chunked padded embeddings.
    for i in range(args.tp):
        orig_path = os.path.join(args.input_dir, iter_dir, f"mp_rank_0{i}", "model_optim_rng.pt")
        state_dict = torch.load(orig_path, weights_only=False)

        args.output_dir_tp = os.path.join(args.output_dir, iter_dir, f"mp_rank_0{i}")
        os.makedirs(args.output_dir_tp, exist_ok=True)
        new_path = os.path.join(args.output_dir_tp, "model_optim_rng.pt")

        start = int(i * input_embeddings_padded.shape[0]/args.tp)
        end = int((i+1) * input_embeddings_padded.shape[0]/args.tp)

        state_dict["model"]["embedding.word_embeddings.weight"] = input_embeddings_padded[start:end, :]
        state_dict["model"]["output_layer.weight"] = output_embeddings_padded[start:end, :]

        print(input_embeddings_padded[start:end, :].shape)
        print(output_embeddings_padded[start:end, :].shape)
        torch.save(state_dict, new_path)

    print(f"Saved patched embeddings to {args.output_dir}")
