import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor
from PIL import Image
from pathlib import Path


model_path = "."
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

generation_config = dict(max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)
img_lst = [
    "images/example1a.jpeg",
    "images/example1b.jpeg",
    "images/table.png",
    "images/tech.png",
]

print("="*50)
print("Text-only test")
print("="*50)
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": [{"type": "text", "text": "Write a short haiku about the moon."}]},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text_inputs = tokenizer([prompt], return_tensors="pt").to(device)
text_outputs = model.generate(
    input_ids=text_inputs.input_ids,
    attention_mask=text_inputs.attention_mask,
    max_new_tokens=64,
)
print(tokenizer.batch_decode(text_outputs[:, text_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])

print("="*50)
print("Test single image")
print("="*50)
for idx, img_path in enumerate(img_lst):
    images = [Image.open(img_lst[idx])]
    messages = [
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<image>\nDescribe the image.",
                },
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[Image.open(img_lst[idx])],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
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

print("="*50)
print("Test multi-images")
print("="*50)
multi_img_lst = [
    "images/example1a.jpeg",
    "images/example1b.jpeg",
]
images = [Image.open(p) for p in multi_img_lst]
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[prompt],
    images=images,
    return_tensors="pt",
)
inputs = inputs.to(device)

generated_ids = model.generate(
    pixel_values=inputs.pixel_values,
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1024,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Output: {output_text[0]}\n")