import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image

import video_io


model_path = "."
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)


video_path = "images/demo.mp4"
video_fps = 1
video_nframe = 8
video_nframe_max = -1

# Get frames and metadata
image_urls, metadata = video_io.maybe_path_or_url_to_data_urls(
    video_path,
    fps=max(0, int(video_fps)),
    nframe=max(0, int(video_nframe)),
    nframe_max=int(video_nframe_max),
)
frames = [video_io.pil_image_from_base64(image_url) for image_url in image_urls]

print(f"Metadata: {metadata}")

# Build prompt with <image> tokens per frame (since <video> is not a valid token in this tokenizer)
prompt_parts = ["This is a video:\n"]
for j in range(len(frames)):
    if metadata and metadata.fps:
        timestamp = j / metadata.fps
        prompt_parts.append(f"Frame {j+1} sampled at {timestamp:.2f} seconds: <image>\n")
    else:
        prompt_parts.append(f"Frame {j+1}: <image>\n")
prompt_parts.append("Describe what you see.")
image_text = "".join(prompt_parts)

messages = [
    {
        "role": "system",
        "content": "/no_think"
    },
    {
        "role": "user",
        "content": image_text,
    }
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process frames as images
inputs = processor(
    text=[prompt],
    images=frames,
    return_tensors="pt",
)
inputs = inputs.to(device)

# Inference: Generation of the output
model.video_pruning_rate = 0.0
generated_ids = model.generate(
    pixel_values=inputs.pixel_values,
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=128,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Prompt: {prompt}\nOutput: {output_text[0]}\n\n\n")
