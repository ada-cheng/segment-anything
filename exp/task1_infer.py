"""
In this script, we 
    1) Cut every 3D medical image into 2D slices, 
    2) Perform segmentation of each of the 13 organs on each slice,
    3) Combine the segmentation results into a 3D mask, and
    4) Calculate mDice score.
"""

import utils
import argparse
import sys
import torch
import numpy as np
import prompt
import random
import math
from tqdm import tqdm
sys.path.append('..')
import data_proc
import torch.nn.functional as F
    
def get_3d_segment(labels, range_start, used_targets, output):
    segment_result = np.zeros((14, *labels.shape), dtype=np.int8)

    for out, (z, targets) in zip(output, used_targets):
        batched_mask = out['masks'].cpu().numpy()
        segment_result[targets, :, :, z - range_start] = batched_mask[:, 0, :, :]
    return segment_result

def compute_mdice(dice_dict):
    count = 0
    sum = 0
    for val in dice_dict.values():
        if not math.isnan(val):
            count += 1
            sum += val
    return sum / count

def show_seg_result(image, mask, idx):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
 
    cmap = plt.get_cmap('rainbow')
    print(f"id: {idx}")
    print(f"mask shape: {mask.shape}")
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        # change mask_i to 0/1
        mask_i = mask_i > 0.5
        if mask_i.sum() == 0:
            print(f"mask {i} is empty")
            continue
        if i == 0:
            pass
        color = cmap(i / mask.shape[0])
        # set mask to color
     
        mask_i = mask_i.cpu().numpy()
        print(f"image type: {type(image)}")
        print(f"color type: {type(color)}")
        print(f"mask_i type: {type(mask_i)}")
        color = torch.tensor(color[:3]).to(image.device)
        image = image.double()
        image = (image - image.min()) / (image.max() - image.min())
        
        image[mask_i] = color
    ax.imshow(image)
    plt.savefig(f"result_finetuned/seg_result_{idx}.png")
    plt.cla()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command-line arguments for '
                                     'segmentation on BTCV using SAM')
    parser.add_argument('--target', type=int, default=0,
                        help='target to segment, if 0, segment all targets')
    parser.add_argument('--data_dir', type=str, default='dataset/RawData',
                        help='directory of the raw data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to load model and data to')
    parser.add_argument('--json_path', type=str, default='dataset/RawData/dataset_0.json',
                        help='path of the json file that describes data')
    parser.add_argument('--point_prompt', type=str, default = "center")
    parser.add_argument('--bounding_box_prompt', action='store_true',
                        help='whether to use bounding box prompt')
    parser.add_argument('--box_margin', type=int, default=0,
                        help='margin of the bounding box')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='how many 2D slice to copy to GPU at a time')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

#    training_data, training_labels = data_proc.load_dataset('training', args.json_path, args.data_dir)
    validation_data, validation_labels = data_proc.load_dataset('validation', args.json_path, args.data_dir)
    
    sam = utils.get_sam(device=args.device, sam_checkpoint= '../fine_tuned_sam_vit_h.pth')

    if args.point_prompt is None:
        args.point_prompt = []

    prompt.print_prompt(args.point_prompt, args.bounding_box_prompt, args.box_margin)

    training_dices = []
    validation_dices = []
    '''
    print("Segmenting on training dataset...", file=sys.stderr)
    for i_sample, (data, label) in enumerate(zip(training_data, training_labels)):
        print(f"Sample {i_sample}:")
        split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
        segment_results = []
        for idx in tqdm(range(len(split_ranges) - 1)):
            next_range = range(split_ranges[idx], split_ranges[idx + 1])
            used_targets, batched_input = data_proc.prepare_input(
                sam, data, label, next_range, 
                args.point_prompt, args.bounding_box_prompt, args.box_margin
            )
            if len(batched_input) == 0:
                segment_results.append(np.zeros((14, *label[..., next_range].shape), dtype=np.int8))
                torch.cuda.empty_cache()
                print("Empty batch, skipping...", file=sys.stderr)
                continue
            output = sam(batched_input, multimask_output=False)
            segment_result = get_3d_segment(label[..., next_range], 
                                            split_ranges[idx],
                                            used_targets, output)
            segment_results.append(segment_result)
            torch.cuda.empty_cache()
        segment_result = np.concatenate(segment_results, axis=-1)

        dice_dict = {}
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            if truth.sum() == 0:
                dice_dict[target] = float('nan')
                continue
            pred = segment_result[target]
            dice_dict[target] = utils.dice_score(pred, truth)

        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {compute_mdice(dice_dict)}")
        torch.cuda.empty_cache()
    '''
    print("Segmenting on validation dataset...", file=sys.stderr)
    
    visualize_cnt = 0
    
    visualize_dict = [] 
    
    for i_sample, (data, label) in enumerate(zip(validation_data, validation_labels)):
        print(f"Sample {i_sample}:")
        split_ranges = list(range(0, data.shape[-1], args.batch_size)) + [data.shape[-1]]
        segment_results = []
        for idx in tqdm(range(len(split_ranges) - 1)):
            next_range = range(split_ranges[idx], split_ranges[idx + 1])
            
            used_targets, batched_input = data_proc.prepare_input(
                sam, data, label, next_range, 
                args.point_prompt, args.bounding_box_prompt, args.box_margin
            )
            
            if len(batched_input) == 0:
                segment_results.append(np.zeros((14, *label[..., next_range].shape), dtype=np.int8))
                torch.cuda.empty_cache()
                continue
            output = sam(batched_input, multimask_output=False)
            print(f"output shape: {output[0]['masks'].shape}")
            print(f"batched input shape: {batched_input[0]}")
            if visualize_cnt < 140:
                visualize_dict.append((batched_input[0]['image'], output[0]['masks']))
                visualize_cnt += 1
            segment_result = get_3d_segment(label[..., next_range], 
                                            split_ranges[idx],
                                            used_targets, output)
            print(f"segment_result {segment_result.shape}")
            segment_results.append(segment_result)

            torch.cuda.empty_cache()
        segment_result = np.concatenate(segment_results, axis=-1)
        print(f'seg shape after concat:{segment_result.shape}')
        dice_dict = {}
        print(f"label shape: {label.shape}")
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            
            if truth.sum() == 0:
                print(f"target {target} is empty")
                dice_dict[target] = float('nan')
                continue
            pred = segment_result[target]
        
            dice_dict[target] = utils.dice_score(pred, truth)
  
        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {compute_mdice(dice_dict)}")
        torch.cuda.empty_cache()
        
        if i_sample == 0:
            break
        
    
    for i in range(len(visualize_dict)):
        image, mask = visualize_dict[i]
        image = image.cpu()
       
    
        image = image.permute(1, 2, 0)
        mask = mask.float()
        mask = F.interpolate(mask, size=(image.shape[0], image.shape[1]), mode='nearest')
        mask = mask.squeeze(1)
    
      
        show_seg_result(image, mask, i)
     
    
    