import webdataset as wds
import os
import json
import math
import random
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path
import functools

n_process = 16
root_path = '/lustre/fsw/portfolios/llmservice/users/amalasanjayd/eagle-next/'


def load_data(root_path, pretrain_path, subsample_rate):

    max_multi_images = 0

    all_data = []
    dat = json.load(open(pretrain_path))

    for k,v in dat.items():
        # v['root'], v['annotation'], v['data_augment'], v['repeat_time'], v['length']
        #if "nvpdftex" in k:
        #    ann_path = v['annotation']
        #else:
        ann_path = os.path.join(root_path, v['annotation'])
        with open(ann_path) as f:
            print('image path: ', os.path.join(root_path, v['root']))
            lines = f.readlines()
            print(f'current total: {len(lines)}; current length: {v["length"]}')

            if subsample_rate is not None:
                random.shuffle(lines)

                rate2 = subsample_rate * v["repeat_time"]
                if rate2 > 1:
                    raise ValueError(f"Subsample rate * repeat time is too high: {rate2}")

                lines = lines[:int(len(lines) * rate2)]
                print(f'subsampled to {len(lines)} samples')

            valid_lines = []

            for xid, content in enumerate(lines):
                sample = json.loads(lines[xid])

                sample["source"] = k
                if 'image' in sample and sample['image'] is not None:
                    if isinstance(sample['image'], list):
                        continue
                        max_multi_images = max(max_multi_images, len(sample['image']))
                        #import pdb; pdb.set_trace()
                        for idx, img in enumerate(sample['image']):
                            sample['image'][idx] = os.path.join(root_path, v['root'], sample['image'][idx])
                    else:
                        #if "nvpdftex" in k:
                        #    sample['image'] = os.path.join(v['root'], sample['image'])
                        #else:
                        sample['image'] = os.path.join(root_path, v['root'], sample['image'])
                        if '/' not in sample['image']:
                            #import pdb; pdb.set_trace()
                            continue

                if "image" in sample.keys() and sample['image'] is None:
                    print("none image: ", sample)
                    continue

                if 'image' in sample.keys() and '/' not in sample['image']:
                    continue
                    #print(sample['image'])
                    #import pdb; pdb.set_trace()
                valid_lines.append(json.dumps(sample))

            random.shuffle(valid_lines)
            all_data.extend(valid_lines)

    print('max multi image: ', max_multi_images)
    print('total length: ', len(all_data))

    return all_data


def build_wds(output_dir, items):
    cur_lines, pid = items
    # pid = pid + args.partid * n_process
    print('pid: ', pid)
    successful_samples, failed_samples = 0, 0
    with wds.ShardWriter(os.path.join(output_dir, f"pretrain-{pid:03d}-%06d.tar"), maxcount=10000) as shard_writer:
        for line in cur_lines:
            try:
                sample = json.loads(line)
                sample["source"] = sample["source"].replace(".", "-")
                json_sample = json.dumps(sample)
                wds_sample = {
                    # '__key__': str(i) + '_' + str(sample.get('id', i)).replace('.', '-'),
                    '__key__': sample["source"] + f'_{pid}_' + str(successful_samples),
                    'json': json_sample,
                }

                if 'image' in sample and sample['image'] is not None:
                    # image_path = sample['image'] # os.path.join(root_path, root, sample['image'])
                    if isinstance(sample['image'], list):
                        for idx, img in enumerate(sample['image']):
                            wds_sample[f'{idx+1}.img'] = Path(sample['image'][idx]).read_bytes()
                    else:
                        wds_sample['img'] = Path(sample['image']).read_bytes()
                    # with open(image_path, 'rb') as image_file:
                    #     wds_sample['img'] = image_file.read()

                shard_writer.write(wds_sample)
                successful_samples += 1
            except Exception as e:
                print('writing error. exception:', e, '. sample: ', json_sample)
                failed_samples += 1

    return successful_samples, failed_samples


def process_all(all_data, output_dir):

    random.shuffle(all_data)
    print('shuffling done.')
    print(f'total samples: {len(all_data)}')

    npart = len(all_data) // n_process
    all_parts = []
    for pid in range(n_process):
        start, end = pid * npart, (pid + 1) * npart
        if pid == n_process - 1:
            end = len(all_data)
        all_parts.append((all_data[start : end], pid))

    build_wds_with_output_dir = functools.partial(build_wds, output_dir)

    with Pool(processes=n_process) as pool:
        for result in pool.map(build_wds_with_output_dir, all_parts):
            success, fail = result
            print(f'success: {success}; failure: {fail}')


def main(input_json, output_dir, subsample_rate):
    os.makedirs(output_dir, exist_ok=False)
    all_data = load_data(root_path, input_json, subsample_rate)
    print(f'loaded item num: {len(all_data)}')
    process_all(all_data, output_dir)


if __name__ == '__main__':

    # write argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--subsample-rate', type=float, default=None)
    args = parser.parse_args()
    main(args.input_json, args.output_dir, args.subsample_rate)