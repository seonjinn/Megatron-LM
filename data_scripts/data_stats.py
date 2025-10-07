import json
import os

root_path = '/lustre/fsw/portfolios/llmservice/users/amalasanjayd/eagle-next/'

with open("pretrain_eagle_sft_v13.52_no_text_reasoning.json") as f:
    data = json.load(f)

num_lines = 0

for k, v in data.items():
    print(k, v)
    ann_path = os.path.join(root_path, v['annotation'])
    with open(ann_path) as f:
        print('image path: ', os.path.join(root_path, v['root']))
        lines = f.readlines()
        num_lines += len(lines)

print(f'total lines: {num_lines}')