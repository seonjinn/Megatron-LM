import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor
from PIL import Image

import video_io


model_path = "/lustre/fsw/portfolios/llmservice/users/charlwang/vlm-hf-code/_ckpt/sft_nm_5p5_h_12b_6k_cradio_vlm_v1_rc3_video_13p41_lc_video_v2_div_2_49152_cp_2_0806"
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
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

messages = [
    {
        "role": "system",
        "content": "/no_think"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
            },
            {
                "type": "text",
                "text": "\nDescribe what you see.",
            },
        ],
    }
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process with FPS metadata
if metadata:
    inputs = processor(
        text=[prompt],
        videos=frames,
        videos_kwargs={'video_metadata': metadata},
        return_tensors="pt",
    )
else:
    inputs = processor(
        text=[prompt],
        videos=frames,
        return_tensors="pt",
    )
inputs = inputs.to(device)

# Inference: Generation of the output
model.video_pruning_rate = 0.0
# # TODO (charles): remove this line after debugging is done
# model._tokenizer = tokenizer
generated_ids = model.generate(
    pixel_values_videos=inputs.pixel_values_videos,
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
