import argparse
from tqdm import tqdm
import json
import os
import sys
import numpy as np
from pycocotools import mask as mask_utils
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch.utils.data import RandomSampler

from datasets.dataloader_sailvos import SailVos_diffusion_vas
from eval_utils import set_seed

def erode_video_sequence(video_sequence, kernel_size=(5, 5), iterations=1):

    if video_sequence.dtype != np.uint8:
        video_sequence = np.clip(video_sequence, 0, 1)  # Ensure the data is within the range [0, 1]
        video_sequence = (video_sequence * 255).astype(np.uint8)

    frames = video_sequence # This changes shape to (num_frames, height, width)
    kernel = np.ones(kernel_size, np.uint8)
    eroded_frames = np.empty_like(frames)
    for i in range(frames.shape[0]):
        eroded_frames[i] = cv2.erode(frames[i], kernel, iterations=iterations)

    eroded_frames = (eroded_frames / 255).astype(np.int64)

    return eroded_frames


def convert_pixels_to_masks(pixels, threshold=0):
    
    pixels = pixels.permute(0, 1, 3, 4, 2)
    summed_channels = pixels.sum(dim=4)
    binary_masks = (summed_channels > threshold).float()
    binary_masks = binary_masks.unsqueeze(1)

    return binary_masks


def resize_masks_and_encode_union_modal(pred_rles, input_shape, new_shape, modal_pixels):
    # Decode RLEs to masks
    masks = np.array([mask_utils.decode(rle) for rle in pred_rles]).astype(np.uint8).reshape(1, 1, len(pred_rles),
                                                                                             input_shape[0],
                                                                                             input_shape[1])
    resized_masks = np.zeros((1, 1, masks.shape[2], 800, 1280))

    # Loop through each frame and resize
    for i in range(masks.shape[2]):  # iterating over the third dimension which has size 25
        resized_masks[0, 0, i] = cv2.resize(masks[0, 0, i], (1280, 800), interpolation=cv2.INTER_LINEAR)

    modal_pixels_np = modal_pixels.numpy().astype(np.uint8)
    resized_masks = np.logical_or(resized_masks, modal_pixels_np).astype(np.uint8)
    new_rles = [convert_masks_to_rle(resized_masks[:, :, t, :, :]) for t in range(masks.shape[2])]

    return new_rles


def convert_masks_to_rle(masks):
    return [mask_utils.encode(np.asfortranarray(mask[0].astype(np.uint8))) for mask in masks]




def get_metrics_for_diffusion_vas(val_loader, pred_annot_path, input_shape=(128, 256), lo_thresh=0, hi_thresh=1):

    # num_timesteps = 25

    num_samples = 0
    skip_samples = 0

    pred_annot = json.load(open(pred_annot_path, 'r'))

    loader_iter = iter(val_loader)


    ious = []
    ious_occ = []

    for batch_idx, batch_data in enumerate(tqdm(loader_iter, ncols=0)):
        modal_pixels, amodal_pixels = batch_data["modal_res"], batch_data["amodal_res"]
        num_timesteps = modal_pixels.shape[1]
        image_ids, obj_id, cat_id = batch_data["image_ids"], batch_data["obj_id"], batch_data["cat_id"]

        modal_pixels, label_data = convert_pixels_to_masks(modal_pixels), convert_pixels_to_masks(amodal_pixels)

        modal_masks = modal_pixels.numpy().astype(int).squeeze(0).squeeze(0)
        amodal_masks = label_data.numpy().astype(int).squeeze(0).squeeze(0)

        intersection = (amodal_masks & modal_masks).sum(axis=(1, 2))  # Sum over spatial dimensions (800, 1280)
        union = (amodal_masks | modal_masks).sum(axis=(1, 2))
        check_result = np.mean(intersection / (union + 1e-6))

        if check_result < lo_thresh or check_result > hi_thresh:
            skip_samples += 1
            continue

        pred_rles = []
        for i in range(image_ids[0].shape[0]):
            key = (int(image_ids[0][i]), int(obj_id[0]))
            key = str(key)
            pred_rles.append(pred_annot[key]['diffvas_res_rle']) # svd_res_600


        pred_rles = resize_masks_and_encode_union_modal(pred_rles, input_shape, (800, 1280), modal_pixels)

        pred_rles_masks = np.array([mask_utils.decode(rle) for rle in pred_rles]).astype(int).reshape(25, 800, 1280)

        intersection = (amodal_masks & pred_rles_masks).sum(axis=(1, 2))  # Sum over spatial dimensions (800, 1280)
        union = (amodal_masks | pred_rles_masks).sum(axis=(1, 2))
        tmp_ious = intersection / (union + 1e-6)
        ious += tmp_ious.tolist()

        amodal_masks_occ = amodal_masks & (~modal_masks)
        amodal_masks_occ = erode_video_sequence(amodal_masks_occ) # Erode the amodal masks, since the modal masks are slightly smaller than amodal masks even without occlusion
        pred_rles_masks_occ = pred_rles_masks & (~modal_masks)
        pred_rles_masks_occ = erode_video_sequence(pred_rles_masks_occ)

        intersection = (amodal_masks_occ & pred_rles_masks_occ).sum(axis=(1, 2))  # Sum over spatial dimensions (800, 1280)
        union = (amodal_masks_occ | pred_rles_masks_occ).sum(axis=(1, 2))
        tmp_ious_occ = intersection / (union + 1e-6)
        ious_occ += [val for val in tmp_ious_occ.tolist() if val >= 1e-6]

    return {
        'miou': np.mean(ious),
        'miou_occ': np.mean(ious_occ)
    }




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video amodal segmentation and content completion using Diffusion-VAS."
    )

    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to input RGB data.", # /data/SAILVOS_2D/
    )

    parser.add_argument(
        "--eval_annot_path",
        type=str,
        required=True,
        help="Path to input annotation data.", # diffusion_vas_sailvos_val.json
    )

    parser.add_argument(
        "--pred_annot_path",
        type=str,
        required=True,
        help="Diffusion_vas eval result path.", # eval_outputs/diffusion_vas_sailvos_eval_results.json
    )

    args = parser.parse_args()

    set_seed(0)

    val_dataset = SailVos_diffusion_vas(
        path=args.eval_annot_path,
        rgb_base_path=args.eval_data_path,
        total_num=-1,
        channel_num=3,
        width=1280,
        height=800,
        read_rgb=False
    )
    sampler = RandomSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        sampler=sampler,
    )

    diffusion_results = get_metrics_for_diffusion_vas(val_dataloader, args.pred_annot_path, input_shape=(256, 512))

    print("diffusion_results:") 
    print(diffusion_results)




