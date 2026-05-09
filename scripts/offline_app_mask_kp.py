import os
ROOT = os.path.dirname(os.path.dirname(__file__))

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)
sys.path.append(os.path.join(top_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(top_dir, 'models', 'diffusion_vas'))

import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

import argparse
import time
import cv2
import glob
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from typing import List, Sequence

def cap_consecutive_ones_by_iou(
    flag: Sequence[int],
    iou: Sequence[float],
    max_keep: int = 3,
) -> List[int]:
    """
    Output rule:
      - If flag[i] == 0 -> output[i] = 1
      - If flag[i] == 1 -> for each consecutive run of 1s:
          - If run_length <= max_keep: keep all as 1
          - If run_length >  max_keep: keep only the indices of the top `max_keep`
            IoU values within the run as 1, set the rest to 0.
            (Tie-breaking: if IoU is the same, prefer the smaller index for stability.)

    Args:
        flag: A 0/1 sequence indicating positions to be processed. Runs of consecutive 1s
              are handled together.
        iou:  A float sequence (same length as `flag`), used to rank elements inside each
              run of consecutive 1s when the run is longer than `max_keep`.
        max_keep: Maximum number of 1s to keep within any consecutive-ones run.

    Returns:
        out: A list of 0/1 integers with the same length as `flag`, following the rules above.

    Raises:
        ValueError: If `flag` and `iou` have different lengths.
    """
    n = len(flag)
    if len(iou) != n:
        raise ValueError(f"len(flag)={n} != len(iou)={len(iou)}")

    # Initialize:
    # - positions where flag==0 are forced to 1
    # - positions where flag==1 are set to 0 first, and will be selected back to 1 per-run
    out = [1 if flag[i] == 0 else 0 for i in range(n)]

    i = 0
    while i < n:
        if flag[i] != 1:
            i += 1
            continue

        # Find a consecutive run of 1s: [i, j)
        j = i
        while j < n and flag[j] == 1:
            j += 1

        run_idx = list(range(i, j))
        if len(run_idx) <= max_keep:
            # Short run: keep all
            for k in run_idx:
                out[k] = 1
        else:
            # Long run: keep top `max_keep` by IoU within the run.
            # Sort by (-IoU, index) to ensure stable tie-breaking.
            top = sorted(run_idx, key=lambda k: (-float(iou[k]), k))[:max_keep]
            for k in top:
                out[k] = 1

        i = j

    return out


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_predictor
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bpe_path = f"{script_dir}/models/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    predictor = build_sam3_video_predictor(bpe_path=bpe_path, checkpoint_path=cfg.sam3['ckpt_path'])
    return predictor


def read_frame_at(path: str, idx: int):
    """Read a specific frame (by index) from a video file."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def build_sam3_3d_body_config(cfg,human_detector=None):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    detector_path = cfg.sam_3d_body['detector_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)
    # from models.sam_3d_body.tools.build_detector import HumanDetector
    # human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator

def mask_completion_and_iou_init(pred_amodal_masks, pred_res, obj_id, batch_masks, i, W, H, OUTPUT_DIR):
    obj_ratio_dict_obj_id = None
    iou_dict_obj_id = None
    occ_dict_obj_id = None
    idx_dict_obj_id = None
    idx_path_obj_id = None
    
    # for completion
    pred_amodal_masks_com = [np.array(img.resize((pred_res[1], pred_res[0]))) for img in pred_amodal_masks]
    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
    # for iou
    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
    # compute iou
    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
    ious = []
    masks_margin_shrink = [bm.copy() for bm in masks]
    mask_H, mask_W = masks_margin_shrink[0].shape
    occlusion_threshold = 0.4
    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
        # mute objects near margin
        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
        mask_binary_cp = zero_mask_cp.astype(np.uint8)
        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
        if mask_binary_cp.max() == 0:   # margin objects
            ious.append(occlusion_threshold)
            continue
        area_a = (a > 0).sum()
        area_b = (b > 0).sum()
        if area_a == 0 and area_b == 0:
            ious.append(occlusion_threshold)
        elif area_a > area_b:
            ious.append(occlusion_threshold)
        else:
            inter = np.logical_and(a > 0, b > 0).sum()
            uni = np.logical_or(a > 0, b > 0).sum()
            obj_iou = inter / (uni + 1e-6)
            ious.append(obj_iou)

        if i == 0 and bi == 0:
            if ious[0] < occlusion_threshold:
                obj_ratio_dict_obj_id = bbox_from_mask(b)
            else:
                obj_ratio_dict_obj_id = bbox_from_mask(a)

    # remove fake completions (empty or from MARGINs)
    for pi, pamc in enumerate(pred_amodal_masks_com):
        # zero predictions, back to original masks
        if masks[pi].sum() > pred_amodal_masks[pi].sum():
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        elif is_skinny_mask(pred_amodal_masks[pi]):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

    # confirm occlusions & save masks (for HMR)
    iou_dict_obj_id = [float(iou_) for iou_ in ious]
    arr = iou_dict_obj_id[:]
    for isb in range(1, len(arr) - 1):
        if arr[isb] == occlusion_threshold and arr[isb-1] < occlusion_threshold and arr[isb+1] < occlusion_threshold:
            arr[isb] = 0.0

    iou_dict_obj_id = arr
    occ_dict_obj_id = [1 if ix >= occlusion_threshold else 0 for ix in iou_dict_obj_id]
    start, end = (idxs := [ix for ix,x in enumerate(iou_dict_obj_id) if x < occlusion_threshold]) and (idxs[0], idxs[-1]) or (None, None)

    if start is not None and end is not None:
        start = max(0, start-2)
        end = min(len(pred_amodal_masks), end+2)
        idx_dict_obj_id = (start, end)
        completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
        completion_image_path = f'{OUTPUT_DIR}/completion/{completion_path}/images'
        completion_masks_path = f'{OUTPUT_DIR}/completion/{completion_path}/masks'
        os.makedirs(completion_image_path, exist_ok=True)
        os.makedirs(completion_masks_path, exist_ok=True)
        idx_path_obj_id = {'images': completion_image_path, 'masks': completion_masks_path}
    return obj_ratio_dict_obj_id, iou_dict_obj_id, occ_dict_obj_id, idx_dict_obj_id, idx_path_obj_id

def mask_completion_and_iou_final(pred_amodal_masks, pred_res, obj_id, batch_masks, W, H, iou_dict_obj_id, occ_dict_obj_id, idx_path_obj_id, keep_idx):    
    keep_id = [io for io, vo in enumerate(keep_idx) if vo == 1]
    batch_masks_ = [batch_masks[io] for io in keep_id]

    # for completion
    zero_com = np.zeros_like(np.array(pred_amodal_masks[0].resize((pred_res[1], pred_res[0])))[:,:,0])
    pred_amodal_masks_com = [np.array(img.resize((pred_res[1], pred_res[0]))) for img in pred_amodal_masks]
    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
    # for iou
    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
    # compute iou
    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks_]
    ious = []
    masks_margin_shrink = [bm.copy() for bm in masks]
    mask_H, mask_W = masks_margin_shrink[0].shape
    occlusion_threshold = 0.5
    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
        # mute objects near margin
        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
        mask_binary_cp = zero_mask_cp.astype(np.uint8)
        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
        if mask_binary_cp.max() == 0:   # margin objects
            ious.append(occlusion_threshold)
            continue
        area_a = (a > 0).sum()
        area_b = (b > 0).sum()
        if area_a == 0 and area_b == 0:
            ious.append(occlusion_threshold)
        elif area_a > area_b:
            ious.append(occlusion_threshold)
        else:
            inter = np.logical_and(a > 0, b > 0).sum()
            uni = np.logical_or(a > 0, b > 0).sum()
            obj_iou = inter / (uni + 1e-6)
            ious.append(obj_iou)

    # remove fake completions (empty or from MARGINs)
    for pi, pamc in enumerate(pred_amodal_masks_com):
        # zero predictions, back to original masks
        if masks[pi].sum() > pred_amodal_masks[pi].sum():
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        elif is_skinny_mask(pred_amodal_masks[pi]):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

    # confirm occlusions & save masks (for HMR)
    iou_dict_obj_id_ = [float(iou_) for iou_ in ious]
    arr = iou_dict_obj_id_[:]
    for isb in range(1, len(arr) - 1):
        if arr[isb] == occlusion_threshold and arr[isb-1] < occlusion_threshold and arr[isb+1] < occlusion_threshold:
            arr[isb] = 0.0

    iou_dict_obj_id_ = arr
    occ_dict_obj_id_ = [1 if ix >= occlusion_threshold else 0 for ix in iou_dict_obj_id_]
    
    completion_masks_path = idx_path_obj_id['masks']
    current_id = 0  # within start & end
    final_pred_amodal_masks_com = []
    for ki, keep_id in enumerate(keep_idx): # all within batch_size
        if keep_id == 0:
            final_pred_amodal_masks_com.append(zero_com)
            continue
        
        occ_dict_obj_id[ki] = occ_dict_obj_id_[current_id]
        iou_dict_obj_id[ki] = iou_dict_obj_id_[current_id]
        
        if occ_dict_obj_id_[current_id] == 1: # only save heavy occluded results
            current_id += 1
            final_pred_amodal_masks_com.append(zero_com)
            continue
        final_pred_amodal_masks_com.append(pred_amodal_masks_com[current_id])
        mask_idx_ = pred_amodal_masks[current_id].copy()
        mask_idx_[mask_idx_ > 0] = obj_id
        mask_idx_ = Image.fromarray(mask_idx_).convert('P')
        mask_idx_.putpalette(DAVIS_PALETTE)
        mask_idx_.save(os.path.join(completion_masks_path, f"{ki:08d}.png"))
        current_id += 1
    
    return iou_dict_obj_id, occ_dict_obj_id, final_pred_amodal_masks_com

def propagate_in_video(predictor, session_id, max_num_objects=None):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            max_num_objects=max_num_objects,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

class OfflineApp:
    def __init__(self, config_path: str = os.path.join(ROOT, "configs", "body4d.yaml"), use_detector=False):
        """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
        self.CONFIG = OmegaConf.load(config_path)
        if use_detector:
            from models.sam_3d_body.tools.build_detector import HumanDetector
            human_detector = HumanDetector(name="vitdet", device=device, path="")
        else:
            human_detector = None
        self.predictor = build_sam3_from_config(self.CONFIG)
        self.sam3_3d_body_model = build_sam3_3d_body_config(self.CONFIG, human_detector=human_detector)

        if self.CONFIG.completion.get('enable', False):
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = build_diffusion_vas_config(self.CONFIG)
        else:
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = None, None, None, None, None

        self.RUNTIME = {}  # clear any old state
        self.OUTPUT_DIR = os.path.join(self.CONFIG.runtime['output_dir'], gen_id())
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.RUNTIME['batch_size'] = self.CONFIG.sam_3d_body.get('batch_size', 1)
        self.RUNTIME['detection_resolution'] = self.CONFIG.completion.get('detection_resolution', [256, 512])
        self.RUNTIME['completion_resolution'] = self.CONFIG.completion.get('completion_resolution', [512, 1024])
        self.RUNTIME['smpl_export'] = self.CONFIG.runtime.get('smpl_export', False)
        self.RUNTIME['bboxes'] = None
        self.RUNTIME['session_id'] = None

    def save_masks(self, 
        start_frame_idx,
        outputs_per_frame, 
        obj_dict, 
        resized_batch_frames,
        original_size,
        frame_list=None,
    ):
        """
        Mask generation across the video.
        Currently runs SAM-3 propagation and renders a mask video.
        """
        if frame_list is not None:
            frame_name_list = [os.path.basename(fl).split('.')[0] for fl in frame_list]
        else:
            frame_name_list = None

        # print("[DEBUG] Save Masks.")
        out_w = original_size[0]
        out_h = original_size[1]
        MASKS_PATH = os.path.join(self.OUTPUT_DIR, 'masks')  # for sam3-3d-body
        os.makedirs(MASKS_PATH, exist_ok=True)
        IMAGE_PATH = os.path.join(self.OUTPUT_DIR, 'painted_images') # for sam3-3d-body
        os.makedirs(IMAGE_PATH, exist_ok=True)
        # FEATS_PATH = os.path.join(self.OUTPUT_DIR, 'feats')  # for confidence
        # os.makedirs(FEATS_PATH, exist_ok=True)

        for out_frame_idx, resized_frame in enumerate(resized_batch_frames):
            output = outputs_per_frame.get(out_frame_idx+start_frame_idx, None)
            img = np.array(resized_frame).astype("uint8")
            msk = np.zeros_like(img[:, :, 0])
            if output is not None:
                for out_obj_id, sam_obj_id in obj_dict.items():
                    # which mask belongs to out_obj_id
                    idx = np.where(output['out_obj_ids'] == sam_obj_id)[0]
                    if len(idx) == 0:
                        continue
                    # mask_hw: (H,W) np, bool or 0/1 or 0/255. True/1/255 = foreground
                    mask_u8 = (output['out_binary_masks'][idx.item()].astype(np.uint8) > 0).astype(np.uint8)
                    fg_area = mask_u8.sum()          # True 总面积
                    min_area = fg_area / 100.0       # 阈值
                    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
                    keep = np.zeros_like(mask_u8)
                    for i in range(1, num):          # 0 是背景
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            keep[labels == i] = 1
                    mask_hw_clean = keep.astype(bool)   # (H,W)
                    msk[mask_hw_clean] = out_obj_id
                    mask = mask_hw_clean.astype(np.uint8) * 255
                    img = mask_painter(img, mask, mask_color=4 + out_obj_id)
            
            msk_pil = cv2.resize(msk, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            msk_pil = Image.fromarray(msk_pil).convert('P')
            msk_pil.putpalette(DAVIS_PALETTE)
            if frame_name_list is None:
                msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx+start_frame_idx:08d}.png"))
                # np.savez_compressed(os.path.join(FEATS_PATH, f"{out_frame_idx+start_frame_idx:08d}.npz"), x=output['feature_cache'])
                # Image.fromarray(img).save(os.path.join(IMAGE_PATH, f"{out_frame_idx+start_frame_idx:08d}.jpg"))
            else:
                msk_pil.save(os.path.join(MASKS_PATH, f"{frame_name_list[out_frame_idx]}.png"))
                # Image.fromarray(img).save(os.path.join(IMAGE_PATH, f"{frame_name_list[out_frame_idx]}.jpg"))
                # np.savez_compressed(os.path.join(FEATS_PATH, f"{frame_name_list[out_frame_idx]}.npz"), x=output['feature_cache'])

    def on_4d_generation(self, images_list: str=None, seq_path=None, kps_list=None, box_list=None, render=True):
        """
        Placeholder for 4D generation.
        Later:
        - run sam3_3d_body_model on per-frame images + masks
        - render 4D visualization video
        For now, just log and return None.
        """
        # print("[DEBUG] 4D Generation button clicked.")

        MASKS_PATH = os.path.join(self.OUTPUT_DIR, 'masks')  # for sam3-3d-body
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]
        masks_list = sorted(
            [
                image
                for ext in image_extensions
                for image in glob.glob(os.path.join(MASKS_PATH, ext))
            ]
        )

        os.makedirs(f"{self.OUTPUT_DIR}/mhr_params", exist_ok=True)
        if render:
            os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames", exist_ok=True)
            for obj_id in self.RUNTIME['out_obj_ids']:
                os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames_individual/{obj_id+1}", exist_ok=True)

        batch_size = self.RUNTIME['batch_size']
        n = len(images_list)
        
        # Optional, detect occlusions
        w, h = Image.open(images_list[0]).size
        pred_res = self.RUNTIME['detection_resolution']
        pred_res_hi = self.RUNTIME['completion_resolution']
        pred_res = pred_res if h < w else pred_res[::-1]
        pred_res_hi = pred_res_hi if h < w else pred_res_hi[::-1]

        modal_pixels_list = []
        if self.pipeline_mask is not None:
            for obj_id in self.RUNTIME['out_obj_ids']:
                modal_pixels, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
                modal_pixels_list.append(modal_pixels)
            rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(seq_path, resolution=pred_res)
            depth_pixels = rgb_to_depth(rgb_pixels, self.depth_model)

        mhr_shape_scale_dict = {}   # each element is a list storing input parameters for mhr_forward
        obj_ratio_dict = {}         # avoid fake completion by obj ratio on the first frame

        # same cam_int across ALL frames
        input_image = np.array(Image.open(images_list[0])).astype('uint8')
        cam_int = self.sam3_3d_body_model.fov_estimator.get_cam_intrinsics(input_image)

        for i in tqdm(range(0, n, batch_size)):
            batch_images = images_list[i:i + batch_size]
            batch_masks  = masks_list[i:i + batch_size]

            W, H = Image.open(batch_masks[0]).size

            # Optional, detect occlusions
            idx_dict = {}
            idx_path = {}
            occ_dict = {}
            iou_dict = {}
            if len(modal_pixels_list) > 0:
                print("detect occlusions ...")
                pred_amodal_masks_dict = {}
                for (modal_pixels, obj_id) in zip(modal_pixels_list, self.RUNTIME['out_obj_ids']):
                    # detect occlusions for each object
                    # predict amodal masks (amodal segmentation)
                    pred_amodal_masks = self.pipeline_mask(
                        modal_pixels[:, i:i + batch_size, :, :, :],
                        depth_pixels[:, i:i + batch_size, :, :, :],
                        height=pred_res[0],
                        width=pred_res[1],
                        num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    obj_ratio_dict_obj_id, iou_dict_obj_id, occ_dict_obj_id, idx_dict_obj_id, idx_path_obj_id = mask_completion_and_iou_init(pred_amodal_masks, pred_res, obj_id, batch_masks, i, W, H, self.OUTPUT_DIR)
                    if obj_ratio_dict_obj_id is not None:
                        obj_ratio_dict[obj_id] = obj_ratio_dict_obj_id
                    if iou_dict_obj_id is not None:                     # list [batch_size], iou
                        iou_dict[obj_id] = iou_dict_obj_id
                    if occ_dict_obj_id is not None:                     # list [batch_size], 1: non_occ, 0: occ
                        occ_dict[obj_id] = occ_dict_obj_id
                    if idx_dict_obj_id is not None:                     # cell, (start, end)
                        idx_dict[obj_id] = idx_dict_obj_id
                    if idx_path_obj_id is not None:                     # dict, {'images': 'image_root_path', 'masks': 'mask_root_path'}
                        idx_path[obj_id] = idx_path_obj_id

                # completion
                for obj_id, (start, end) in idx_dict.items(): 

                    completion_image_path = idx_path[obj_id]['images']
                    completion_mask_path = idx_path[obj_id]['masks']
                    # # prepare inputs
                    modal_pixels_current, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res_hi, obj_id=obj_id)
                    rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(seq_path, resolution=pred_res_hi)
                    depth_pixels = rgb_to_depth(rgb_pixels_current, self.depth_model)
                    modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
                    modal_pixels_current = modal_pixels_current[:, start:end]

                    rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                    modal_obj_mask = (modal_pixels_current > 0).float()
                    modal_background = 1 - modal_obj_mask
                    rgb_pixels_current = (rgb_pixels_current + 1) / 2
                    modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                    keep_idx = cap_consecutive_ones_by_iou(occ_dict[obj_id][start:end], iou_dict[obj_id][start:end])
                    mask_idx = torch.tensor(keep_idx, device=modal_rgb_pixels.device).bool()

                    pred_amodal_masks_ = self.pipeline_mask(
                        modal_pixels_current[:, mask_idx],
                        depth_pixels[:, i:i + batch_size, :, :, :][:, start:end][:, mask_idx],
                        height=pred_res_hi[0],
                        width=pred_res_hi[1],
                        num_frames=sum(keep_idx),
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    iou_dict_obj_id, occ_dict_obj_id, pred_amodal_masks_com = mask_completion_and_iou_final(
                        pred_amodal_masks_, 
                        pred_res_hi, 
                        obj_id, 
                        batch_masks, 
                        W, 
                        H, 
                        iou_dict[obj_id],
                        occ_dict[obj_id],
                        idx_path[obj_id],
                        [0]*start + keep_idx + [0]*(len(occ_dict_obj_id)-end),  # keep idx full
                    )
                    if iou_dict_obj_id is not None:
                        iou_dict[obj_id] = iou_dict_obj_id
                    if occ_dict_obj_id is not None:
                        occ_dict[obj_id] = occ_dict_obj_id

                    print("content completion by diffusion-vas ...")
                    keep_idx = cap_consecutive_ones_by_iou(occ_dict[obj_id][start:end], iou_dict[obj_id][start:end])
                    mask_idx = torch.tensor(keep_idx, device=modal_rgb_pixels.device).bool()
                    pred_amodal_masks_current = pred_amodal_masks_com[start:end]
                    pred_amodal_masks_current = [xxx for xxx, mmm in zip(pred_amodal_masks_current, keep_idx) if mmm == 1]
                    modal_mask_union = (modal_pixels_current[:, mask_idx][0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
                    pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
                        2).repeat(1, 1, 3, 1, 1)

                    # predict amodal rgb (content completion)
                    pred_amodal_rgb = self.pipeline_rgb(
                        modal_rgb_pixels[:, mask_idx],
                        pred_amodal_masks_tensor, 
                        height=pred_res_hi[0], # my_res[0]
                        width=pred_res_hi[1],  # my_res[1]
                        num_frames=sum(keep_idx),
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    pred_i = 0
                    save_i = start-1
                    for keep_i, occ_i in zip(keep_idx, occ_dict[obj_id][start:end]):
                        save_i += 1
                        if occ_i == 1:
                            if keep_i == 1:
                                pred_i += 1
                            continue
                        if keep_i == 1:
                            rgb_i = np.array(pred_amodal_rgb[pred_i]).astype('uint8')
                            rgb_i = cv2.resize(rgb_i, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                            cv2.imwrite(os.path.join(completion_image_path, f"{save_i:08d}.jpg"), cv2.cvtColor(rgb_i, cv2.COLOR_RGB2BGR))
                            pred_i += 1
                            continue
            else:
            
                idx_dict = idx_dict or None
                idx_path = idx_path or None
                iou_dict = iou_dict or None
                for obj_id in self.RUNTIME['out_obj_ids']:
                    occ_dict[obj_id] = [1] * len(batch_masks)

            # batch_boxes = [bboxes[i:i + batch_size] for bboxes in box_list]
            batch_kps = None if kps_list is None else [kps[i:i + batch_size] for kps in kps_list]

            # Process with external mask
            mask_outputs, id_batch, empty_frame_list = process_image_with_mask(self.sam3_3d_body_model, batch_images, batch_masks, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, cam_int=cam_int, batch_kps=batch_kps, iou_dict=iou_dict, predictor=self.predictor)
            
            num_empth_ids = 0
            for frame_id in range(len(batch_images)):
                image_path = batch_images[frame_id]
                if frame_id in empty_frame_list:
                    mask_output = None
                    id_current = None
                    num_empth_ids += 1
                else:
                    # mask_output = mask_outputs[frame_id-num_empth_ids]
                    # id_current = id_batch[frame_id-num_empth_ids]
                    mask_output = mask_outputs[frame_id]
                    id_current = id_batch[frame_id]
                
                # if render:
                #     img = cv2.imread(image_path)
                #     rend_img = visualize_sample_together(img, mask_output, self.sam3_3d_body_model.faces, id_current)
                #     cv2.imwrite(
                #         f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                #         rend_img.astype(np.uint8),
                #     )

                np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_data.npz", data=mask_output)
                # np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_id.npz", data=id_current)


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()
    if args.output_dir is not None:
        predictor.OUTPUT_DIR = args.output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

    # human detection on the frame where human FIRST appear
    if os.path.isfile(args.input_video) and args.input_video.endswith(".mp4"):
        input_type = "video"
        image = read_frame_at(args.input_video, 0)
        width, height = image.size
        for starting_frame_idx in range(10, 100):
            image = np.array(read_frame_at(args.input_video, starting_frame_idx))
            outputs = predictor.sam3_3d_body_model.process_one_image(image, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
        
        inference_state = predictor.predictor.init_state(video_path=args.input_video)
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    elif os.path.isdir(args.input_video):
        input_type = "images"
        image_list = glob.glob(os.path.join(args.input_video, '*.jpg'))
        image_list.sort()
        image = Image.open(image_list[0]).convert('RGB')
        width, height = image.size
        starting_frame_idx = 0
        for image_path in image_list:
            outputs = predictor.sam3_3d_body_model.process_one_image(image_path, bbox_thr=0.6,)
            if len(outputs) > 0:
                break
            starting_frame_idx += 1

        inference_state = predictor.predictor.init_state(video_path=image_list)
        predictor.predictor.clear_all_points_in_video(inference_state)
        predictor.RUNTIME['inference_state'] = inference_state
        predictor.RUNTIME['out_obj_ids'] = []

        # 1. load bbox (first frame)
        for obj_id, output in enumerate(outputs):
            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
            xmin, ymin, xmax, ymax = output['bbox']
            rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height]]
            rel_box = np.array(rel_box, dtype=np.float32)
            _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                inference_state=predictor.RUNTIME['inference_state'],
                frame_idx=starting_frame_idx,
                obj_id=obj_id+1,
                box=rel_box,
            )

    # 2. tracking
    predictor.on_mask_generation(start_frame_idx=0)
    # 3. hmr upon masks
    with torch.autocast("cuda", enabled=False):
        predictor.on_4d_generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline 4D Body Generation from Videos")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video (either *.mp4 or a directory containing image sequences)")
    args = parser.parse_args()

    input_path = args.input_video
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input_video does not exist: {input_path}")
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".mp4"):
            raise ValueError(
                f"--input_video must be an .mp4 file or a directory, got file: {input_path}"
            )
    elif os.path.isdir(input_path):
        # Optional: check whether the directory contains images
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        images = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(valid_ext)
        ]
        if len(images) == 0:
            raise ValueError(
                f"--input_video directory contains no image files: {input_path}"
            )
    else:
        raise ValueError(
            f"--input_video must be an .mp4 file or a directory: {input_path}"
        )

    inference(args)


from typing import List
def resize_images_longest_side(
    images: List[Image.Image],
    max_side: int = 1280
) -> List[Image.Image]:
    resized = []
    for img in images:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale >= 1.0:
            resized.append(img)
            continue
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized.append(
            img.resize((new_w, new_h), Image.BICUBIC)
        )
    return resized


import numpy as np

def majority_keypoints_in_mask(keypoints: np.ndarray, mask: np.ndarray) -> bool:
    """
    Return True if more than half of valid keypoints lie inside the mask.
    """
    H, W = mask.shape
    pts = keypoints[:, :2]

    x = np.rint(pts[:, 0]).astype(int)  # column (x)
    y = np.rint(pts[:, 1]).astype(int)  # row (y)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    num_valid = np.sum(valid)
    if num_valid == 0:
        return False

    num_inside = np.sum(mask[y[valid], x[valid]])
    return num_inside > (num_valid / 2)

# from __future__ import annotations

# import numpy as np
# import cv2


def same_object_by_bbox_mask_iou(
    bbox_xyxy: np.ndarray,
    mask: np.ndarray,
    *,
    shrink_px: int = 10,
    iou_thresh: float = 0.15,
    min_intersection_px: int = 50,
    min_box_area_px: int = 200,
) -> bool:
    """
    Decide whether bbox (xyxy) and mask belong to the same object using IoU,
    with a *strict* bias (easy to say NOT same):
      - erode (shrink) mask before IoU
      - require BOTH IoU >= iou_thresh AND intersection >= min_intersection_px
      - reject tiny boxes by min_box_area_px

    Args:
        bbox_xyxy: (4,) np, [x1,y1,x2,y2]
        mask: (H,W) np, 0/255 or 0/1
        shrink_px: erode mask by pixels
        iou_thresh: IoU threshold
        min_intersection_px: minimal overlap pixels
        min_box_area_px: minimal bbox area, otherwise False

    Returns:
        True if same object else False
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    H, W = mask.shape

    b = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"bbox_xyxy must have shape (4,), got {np.asarray(bbox_xyxy).shape}")

    x1, y1, x2, y2 = b.tolist()
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    ix1 = int(np.floor(x1))
    iy1 = int(np.floor(y1))
    ix2 = int(np.ceil(x2))
    iy2 = int(np.ceil(y2))

    ix1 = max(0, min(W, ix1))
    ix2 = max(0, min(W, ix2))
    iy1 = max(0, min(H, iy1))
    iy2 = max(0, min(H, iy2))

    bw = ix2 - ix1
    bh = iy2 - iy1
    box_area = bw * bh
    if box_area <= 0 or box_area < int(min_box_area_px):
        return False

    mask_bool = mask > 0
    if shrink_px > 0:
        k = 2 * int(shrink_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_bool = cv2.erode(mask_bool.astype(np.uint8), kernel) > 0

    # intersection pixels inside bbox
    inter = int(mask_bool[iy1:iy2, ix1:ix2].sum())
    if inter <= 0 or inter < int(min_intersection_px):
        return False

    mask_area = int(mask_bool.sum())
    union = box_area + mask_area - inter
    if union <= 0:
        return False

    iou = float(inter / union)
    return iou >= float(iou_thresh)

def bbox_similar_to_mask_bbox(
    bbox: np.ndarray,
    mask: np.ndarray,
    iou_thr: float = 0.5,
    center_thr: float = 0.25,
    size_thr: float = 0.5,
) -> bool:
    """
    Return True if input bbox [x1,y1,x2,y2] is similar to the bbox derived from mask.

    Similarity checks:
      1) IoU(bbox, mask_bbox) >= iou_thr
      2) center distance (normalized by mask bbox size) <= center_thr
      3) size ratio similarity (min(w)/max(w) and min(h)/max(h)) >= size_thr

    Args:
        bbox: np.ndarray/list [x1,y1,x2,y2]
        mask: HxW mask (bool or 0/1 or 0..255)
        iou_thr: IoU threshold
        center_thr: normalized center distance threshold
        size_thr: minimum ratio similarity for width and height

    Returns:
        bool
    """
    if bbox is None or mask is None:
        return False

    m = mask.astype(bool)
    if m.sum() == 0:
        return False

    ys, xs = np.nonzero(m)
    x1m, x2m = int(xs.min()), int(xs.max()) + 1
    y1m, y2m = int(ys.min()), int(ys.max()) + 1

    x1, y1, x2, y2 = np.rint(np.asarray(bbox)).astype(int)

    # sanitize
    if x2 <= x1 or y2 <= y1 or x2m <= x1m or y2m <= y1m:
        return False

    # IoU
    ix1, iy1 = max(x1, x1m), max(y1, y1m)
    ix2, iy2 = min(x2, x2m), min(y2, y2m)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_b = (x2 - x1) * (y2 - y1)
    area_m = (x2m - x1m) * (y2m - y1m)
    union = area_b + area_m - inter
    iou = inter / union if union > 0 else 0.0

    # center distance normalized by mask bbox size
    cbx, cby = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cmx, cmy = (x1m + x2m) / 2.0, (y1m + y2m) / 2.0
    mw, mh = (x2m - x1m), (y2m - y1m)
    norm = max(np.hypot(mw, mh), 1e-6)
    center_dist = np.hypot(cbx - cmx, cby - cmy) / norm

    # size similarity (ratio close to 1)
    bw, bh = (x2 - x1), (y2 - y1)
    w_sim = min(bw, mw) / max(bw, mw)
    h_sim = min(bh, mh) / max(bh, mh)
    size_sim = min(w_sim, h_sim)

    return (iou >= iou_thr) and (center_dist <= center_thr) and (size_sim >= size_thr)

import numpy as np

def iou_over_threshold(bbox1: np.ndarray, bbox2: np.ndarray, threshold=0.7) -> bool:
    bbox1 = np.asarray(bbox1, dtype=np.float64).reshape(-1)
    bbox2 = np.asarray(bbox2, dtype=np.float64).reshape(-1)

    # intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # areas
    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])

    small_area = min(area1, area2)
    if small_area <= 0:
        return False

    # 小框未被覆盖比例
    uncovered_ratio = 1.0 - inter_area / small_area

    return uncovered_ratio

def masks_low_overlap(mask_path1, mask_path2, threshold=0.2):
    # 两个文件必须都存在
    if not (os.path.exists(mask_path1) and os.path.exists(mask_path2)):
        return False

    mask1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    if mask1 is None or mask2 is None:
        return False

    m1 = mask1 > 0
    m2 = mask2 > 0

    area1 = m1.sum()
    area2 = m2.sum()
    if area1 == 0 or area2 == 0:
        return False

    intersection = (m1 & m2).sum()
    overlap_ratio = intersection / min(area1, area2)

    # 不重合或重合率低，都返回 True
    return overlap_ratio < threshold
