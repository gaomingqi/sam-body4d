import argparse
from tqdm import tqdm
import json
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch.utils.data import RandomSampler

from datasets.dataloader_tao_amodal import TAO_amodal_diffusion_vas
from eval_utils import get_bbox_from_mask, compute_iou, set_seed

def get_metrics_for_modal(val_loader, lo_thresh=0, hi_thresh=1):

    print(f"thresh: [{lo_thresh}, {hi_thresh}]")

    num_ap25 = 0
    num_ap50 = 0
    num_ap75 = 0
    ious = []

    loader_iter = iter(val_loader)
    for batch_idx, batch_data in enumerate(tqdm(loader_iter, ncols=0)):

        gt_amodal_bboxes = batch_data['amodal_bboxes'][0].detach().cpu().numpy()
        modal_pixels = (batch_data['modal_res'][0,:,0,:,:].detach().cpu().numpy() + 1) // 2
        gt_modal_bboxes = [get_bbox_from_mask(modal_pixels[i]) for i in range(len(modal_pixels))]


        tmp_ious = [compute_iou(gt_amodal_bboxes[i], gt_modal_bboxes[i]) for i in range(len(gt_amodal_bboxes))]

        check_result = np.mean(tmp_ious)
        if check_result < lo_thresh or check_result > hi_thresh:
            continue


        ious += tmp_ious
        num_ap25 += (np.array(tmp_ious) > 0.25).sum()
        num_ap50 += (np.array(tmp_ious) > 0.5).sum()
        num_ap75 += (np.array(tmp_ious) > 0.75).sum()

    result = {}
    result['miou'] = np.mean(ious)
    result['ap25'] = num_ap25 / len(ious)
    result['ap50'] = num_ap50 / len(ious)
    result['ap75'] = num_ap75 / len(ious)

    return result


def get_metrics_for_diffusion_vas(val_loader, pred_annot_path, track_ids, lo_thresh=0, hi_thresh=1):

    print(f"thresh: [{lo_thresh}, {hi_thresh}]")

    num_ap25 = 0
    num_ap50 = 0
    num_ap75 = 0
    ious = []

    pred_annot = json.load(open(pred_annot_path, 'r'))


    loader_iter = iter(val_loader)
    for batch_idx, batch_data in enumerate(tqdm(loader_iter, ncols=0)):
        gt_amodal_bboxes = batch_data['amodal_bboxes'][0].detach().cpu().numpy()
        vid_id = batch_data['vid_id']
        rel_track_id = int(batch_data['track_id']) - track_ids[str(int(vid_id))] + 1
        file_names = batch_data['image_file_names']

        modal_pixels = (batch_data['modal_res'][0, :, 0, :, :].detach().cpu().numpy() + 1) // 2
        gt_modal_bboxes = [get_bbox_from_mask(modal_pixels[i]) for i in range(len(modal_pixels))]
        check_result = np.mean([compute_iou(gt_amodal_bboxes[i], gt_modal_bboxes[i]) for i in range(len(gt_amodal_bboxes))])
        if check_result < lo_thresh or check_result > hi_thresh:
            continue

        pred_amodal_bboxes = []
        for i in range(len(file_names)):
            index = str((int(vid_id), file_names[i][0], str(rel_track_id)))
            pred_amodal_bboxes.append(pred_annot[index]["diffvas_res_bbox"])
        pred_amodal_bboxes = np.array(pred_amodal_bboxes)


        tmp_ious = [compute_iou(gt_amodal_bboxes[i], pred_amodal_bboxes[i]) for i in range(len(gt_amodal_bboxes))]
        ious += tmp_ious
        num_ap25 += (np.array(tmp_ious) > 0.25).sum()
        num_ap50 += (np.array(tmp_ious) > 0.5).sum()
        num_ap75 += (np.array(tmp_ious) > 0.75).sum()


    result = {}
    result['miou'] = np.mean(ious)
    result['ap25'] = num_ap25 / len(ious)
    result['ap50'] = num_ap50 / len(ious)
    result['ap75'] = num_ap75 / len(ious)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video amodal segmentation and content completion using Diffusion-VAS."
    )

    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to input RGB data.", # /data/TAO/frames/
    )

    parser.add_argument(
        "--eval_annot_path",
        type=str,
        required=True,
        help="Path to input annotation data.", # diffusion_vas_tao_amodal_val.json
    )

    parser.add_argument(
        "--pred_annot_path",
        type=str,
        required=True,
        help="Diffusion_vas eval result path.", # eval_outputs/diffusion_vas_tao_amodal_eval_results.json
    )

    parser.add_argument(
        "--track_ids_path",
        type=str,
        required=True,
        help="Path to track ids.", # tao_amodal_track_ids_abs2rel_val.json
    )

    args = parser.parse_args()

    set_seed(0)

    val_dataset = TAO_amodal_diffusion_vas(
        path=args.eval_annot_path,
        rgb_base_path=args.eval_data_path,
        total_num=-1,
        channel_num=3,
    )
    sampler = RandomSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        sampler=sampler,
    )

    track_ids = json.load(open(args.track_ids_path, 'r'))

    diffusion_results = get_metrics_for_diffusion_vas(val_dataloader, args.pred_annot_path, track_ids)
    print("diffusion_results:")
    print(diffusion_results)

