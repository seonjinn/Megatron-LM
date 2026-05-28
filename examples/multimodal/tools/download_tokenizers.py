from huggingface_hub import snapshot_download

snapshot_download(
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        local_dir="nemotron_3_nano_30b_a3b_bf16_tokenizer",
        allow_patterns=[
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.*", "merges.*", "added_tokens.*", "chat_template.*", "config.json",],)
