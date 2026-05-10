import argparse
import gc
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
DIFFUSION_VAS_DIR = os.path.join(REPO_DIR, "models", "diffusion_vas")
SAM3D_DIR = os.path.join(REPO_DIR, "models", "sam_3d_body")
for path in (SCRIPTS_DIR, REPO_DIR, DIFFUSION_VAS_DIR, SAM3D_DIR):
    if path not in sys.path:
        sys.path.append(path)

from utils import DAVIS_PALETTE, keep_largest_component  # noqa: E402


DEFAULT_DATA_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D"
DEFAULT_RESULT_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask"
DEFAULT_SAVE_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask_completion_opt"
DEFAULT_CONFIG_PATH = os.path.join(REPO_DIR, "configs", "body4d.yaml")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass
class SequenceRecord:
    name: str
    frame_dir: str
    result_dir: str
    output_dir: str


@dataclass
class Segment:
    start: int
    end: int
    clip_start: int
    clip_end: int
    score: float
    mean_area_ratio: float
    mean_kp_jump: float


def configure_library_logging(level):
    level = str(level).upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level}")
    os.environ["LOG_LEVEL"] = level
    numeric_level = getattr(logging, level)
    logging.getLogger().setLevel(numeric_level)
    for name in (
        "sam3",
        "sam3.model",
        "sam3.model.sam3_video_predictor",
        "sam3.model.sam3_video_inference",
        "sam3.model.sam3_video_base",
        "diffusers",
        "transformers",
        "mhr_smpl_conversion",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


def image_paths(folder):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def parse_sequence_filter(value, sequence_file=None):
    if sequence_file:
        wanted = set()
        with open(sequence_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                wanted.add(line)
        return wanted
    if value is None or value == "" or value == "all":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def resolve_frame_dir(data_root, seq_name):
    parts = seq_name.split(os.sep)
    candidates = []
    if len(parts) >= 2:
        candidates.append(os.path.join(data_root, parts[0], "images", *parts[1:]))
    candidates.extend(
        [
            os.path.join(data_root, "images", seq_name),
            os.path.join(data_root, seq_name),
        ]
    )
    for path in candidates:
        if os.path.isdir(path) and image_paths(path):
            return path
    return candidates[0]


def list_sequences(data_root, result_root, save_root, sequence="all", sequence_file=None):
    wanted = parse_sequence_filter(sequence, sequence_file=sequence_file)
    records = []
    for masks_dir in sorted(glob.glob(os.path.join(result_root, "*", "*", "masks"))):
        result_dir = os.path.dirname(masks_dir)
        if not os.path.isdir(os.path.join(result_dir, "mhr_params")):
            continue
        seq_name = os.path.relpath(result_dir, result_root)
        basename = os.path.basename(seq_name)
        scene = seq_name.split(os.sep)[0]
        if wanted is not None and seq_name not in wanted and basename not in wanted and scene not in wanted:
            continue
        records.append(
            SequenceRecord(
                name=seq_name,
                frame_dir=resolve_frame_dir(data_root, seq_name),
                result_dir=result_dir,
                output_dir=os.path.join(save_root, seq_name),
            )
        )
    if not records:
        raise RuntimeError(f"No VOccl3D mask result sequences found under {result_root}")
    return records


def sequence_frame_count(record):
    return len(image_paths(record.frame_dir))


def split_sequences_balanced(records, num_shards):
    shards = [[] for _ in range(num_shards)]
    shard_costs = [0 for _ in range(num_shards)]
    weighted = [(sequence_frame_count(record), record) for record in records]
    for cost, record in sorted(weighted, reverse=True, key=lambda x: x[0]):
        idx = int(np.argmin(shard_costs))
        shards[idx].append(record)
        shard_costs[idx] += cost
    return shards, shard_costs


def visible_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() and visible.strip() != "-1":
        return [x.strip() for x in visible.split(",") if x.strip()]
    if not torch.cuda.is_available():
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def parse_gpus(gpus):
    if gpus == "auto":
        return visible_gpu_ids()
    return [x.strip() for x in gpus.split(",") if x.strip()]


def link_or_copy_file(src, dst, overwrite=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        if not overwrite:
            return
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_dir(src_dir, dst_dir, overwrite=False):
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(src_dir, "*"))):
        if os.path.isfile(src):
            link_or_copy_file(src, os.path.join(dst_dir, os.path.basename(src)), overwrite=overwrite)


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def stem_from_frame(path):
    return os.path.splitext(os.path.basename(path))[0]


def mhr_path_for_frame(mhr_dir, frame_path):
    return os.path.join(mhr_dir, f"{stem_from_frame(frame_path)}_data.npz")


def load_mhr_item(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)["data"]
    except Exception:
        return None
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        if data.shape == ():
            data = data.item()
        elif data.size > 0:
            data = data[0]
        else:
            return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None
    return data


def load_keypoints3d(mhr_dir, frame_paths):
    kps = np.full((len(frame_paths), 70, 3), np.nan, dtype=np.float32)
    for idx, frame_path in enumerate(frame_paths):
        item = load_mhr_item(mhr_path_for_frame(mhr_dir, frame_path))
        if item is None or "pred_keypoints_3d" not in item:
            continue
        pts = np.asarray(item["pred_keypoints_3d"], dtype=np.float32)
        if pts.shape != (70, 3):
            continue
        if "pred_cam_t" in item:
            cam_t = np.asarray(item["pred_cam_t"], dtype=np.float32).reshape(1, 3)
            pts = pts + cam_t
        kps[idx] = pts
    return kps


def load_mask_areas(mask_paths, obj_id=1):
    areas = np.zeros(len(mask_paths), dtype=np.float32)
    for idx, path in enumerate(mask_paths):
        mask = np.array(Image.open(path).convert("P"))
        areas[idx] = float((mask == obj_id).sum())
    return areas


def rolling_median(values, start, end):
    start = max(0, start)
    end = min(len(values), end)
    if start >= end:
        return np.nan
    window = values[start:end]
    window = window[np.isfinite(window)]
    if len(window) == 0:
        return np.nan
    return float(np.median(window))


def area_drop_flags(areas, ref_window, drop_ratio, min_area):
    n = len(areas)
    ratios = np.ones(n, dtype=np.float32)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        prev = rolling_median(areas, i - ref_window, i)
        nxt = rolling_median(areas, i + 1, i + 1 + ref_window)
        refs = [x for x in (prev, nxt) if np.isfinite(x)]
        if not refs:
            continue
        baseline = max(refs)
        if baseline < min_area:
            continue
        ratios[i] = float(areas[i] / (baseline + 1e-6))
        flags[i] = ratios[i] < drop_ratio
    return flags, ratios


def keypoint_jump_scores(kps):
    n = kps.shape[0]
    centered = kps.copy()
    hips = np.full((n, 1, 3), np.nan, dtype=np.float32)
    for idx in range(n):
        hip_pair = centered[idx, [9, 10], :]
        valid = np.isfinite(hip_pair).all(axis=1)
        if valid.any():
            hips[idx, 0] = hip_pair[valid].mean(axis=0)
    centered = centered - hips

    pair_steps = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        a = centered[i - 1]
        b = centered[i]
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if valid.sum() == 0:
            pair_steps[i] = np.nan
            continue
        pair_steps[i] = float(np.median(np.linalg.norm(b[valid] - a[valid], axis=1)))

    frame_scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vals = []
        if i > 0 and np.isfinite(pair_steps[i]):
            vals.append(pair_steps[i])
        if i + 1 < n and np.isfinite(pair_steps[i + 1]):
            vals.append(pair_steps[i + 1])
        frame_scores[i] = max(vals) if vals else 0.0
    return frame_scores


def keypoint_jump_flags(scores, ref_window, jump_ratio, jump_min):
    n = len(scores)
    flags = np.zeros(n, dtype=bool)
    thresholds = np.full(n, jump_min, dtype=np.float32)
    for i in range(n):
        prev = rolling_median(scores, i - ref_window, i)
        nxt = rolling_median(scores, i + 1, i + 1 + ref_window)
        refs = [x for x in (prev, nxt) if np.isfinite(x) and x > 0]
        baseline = float(np.median(refs)) if refs else 0.0
        threshold = max(float(jump_min), float(jump_ratio) * baseline)
        thresholds[i] = threshold
        flags[i] = scores[i] > threshold
    return flags, thresholds


def fill_short_gaps(flags, max_gap):
    if max_gap <= 0:
        return flags
    flags = flags.copy()
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            flags[i:j] = True
        i = j
    return flags


def bool_runs(flags):
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def detect_segments(frame_paths, mask_paths, mhr_dir, args):
    areas = load_mask_areas(mask_paths, obj_id=args.obj_id)
    kps = load_keypoints3d(mhr_dir, frame_paths)
    area_flags, area_ratios = area_drop_flags(
        areas,
        ref_window=args.ref_window,
        drop_ratio=args.area_drop_ratio,
        min_area=args.min_mask_area,
    )
    kp_scores = keypoint_jump_scores(kps)
    kp_flags, kp_thresholds = keypoint_jump_flags(
        kp_scores,
        ref_window=args.ref_window,
        jump_ratio=args.kp_jump_ratio,
        jump_min=args.kp_jump_min,
    )
    bad = fill_short_gaps(area_flags & kp_flags, args.merge_gap)

    segments = []
    for start, end in bool_runs(bad):
        length = end - start + 1
        if length > args.max_occ_len:
            continue
        clip_start = start - args.context
        clip_end = end + args.context
        if clip_start < 0 or clip_end >= len(frame_paths):
            continue
        if bad[clip_start:start].any() or bad[end + 1 : clip_end + 1].any():
            continue
        area_severity = float(np.mean(1.0 - area_ratios[start : end + 1]))
        jump_mean = float(np.mean(kp_scores[start : end + 1]))
        jump_threshold = float(np.mean(kp_thresholds[start : end + 1]) + 1e-6)
        score = area_severity + min(3.0, jump_mean / jump_threshold)
        segments.append(
            Segment(
                start=start,
                end=end,
                clip_start=clip_start,
                clip_end=clip_end,
                score=score,
                mean_area_ratio=float(np.mean(area_ratios[start : end + 1])),
                mean_kp_jump=jump_mean,
            )
        )

    selected = []
    occupied = np.zeros(len(frame_paths), dtype=bool)
    for seg in sorted(segments, key=lambda x: x.score, reverse=True):
        if len(selected) >= args.max_segments_per_video:
            break
        if occupied[seg.clip_start : seg.clip_end + 1].any():
            continue
        selected.append(seg)
        occupied[seg.clip_start : seg.clip_end + 1] = True
    selected = sorted(selected, key=lambda x: x.start)

    debug = {
        "areas": areas,
        "area_ratios": area_ratios,
        "area_flags": area_flags.astype(np.uint8),
        "kp_scores": kp_scores,
        "kp_thresholds": kp_thresholds,
        "kp_flags": kp_flags.astype(np.uint8),
        "bad_flags": bad.astype(np.uint8),
    }
    return selected, debug


def oriented_resolution(base_resolution, first_image_path):
    width, height = Image.open(first_image_path).size
    base = tuple(int(x) for x in base_resolution)
    return base if height < width else base[::-1]


def indexed_link_folder(paths, dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    out_paths = []
    for idx, src in enumerate(paths):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(dst_dir, f"{idx:08d}{ext}")
        link_or_copy_file(src, dst, overwrite=True)
        out_paths.append(dst)
    return out_paths


def save_palette_mask(mask, path, obj_id=1):
    mask_idx = np.zeros(mask.shape, dtype=np.uint8)
    mask_idx[mask > 0] = obj_id
    img = Image.fromarray(mask_idx).convert("P")
    img.putpalette(DAVIS_PALETTE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    img.save(tmp)
    os.replace(tmp, path)


def binarize_pipeline_masks(mask_frames, target_hw):
    masks = []
    target_h, target_w = target_hw
    for img in mask_frames:
        arr = np.asarray(img.resize((target_w, target_h))).astype(np.uint8)
        if arr.ndim == 3:
            arr = arr.sum(axis=-1)
        mask = (arr > 600).astype(np.uint8)
        masks.append(keep_largest_component(mask))
    return np.asarray(masks, dtype=np.uint8)


def run_completion_for_segment(models, frame_paths, mask_paths, segment, out_seq_dir, args):
    (
        cfg,
        estimator,
        pipeline_mask,
        pipeline_rgb,
        depth_model,
        generator,
        load_and_transform_masks_fn,
        load_and_transform_rgbs_fn,
        rgb_to_depth_fn,
    ) = models
    clip_indices = list(range(segment.clip_start, segment.clip_end + 1))
    clip_frames = [frame_paths[i] for i in clip_indices]
    clip_masks = [mask_paths[i] for i in clip_indices]
    occ_global = set(range(segment.start, segment.end + 1))
    occ_local = [i for i, global_i in enumerate(clip_indices) if global_i in occ_global]

    seg_name = f"segment_{segment.start:04d}_{segment.end:04d}"
    seg_dir = os.path.join(out_seq_dir, "completion", seg_name)
    input_image_dir = os.path.join(seg_dir, "input_images")
    input_mask_dir = os.path.join(seg_dir, "input_masks")
    completed_image_dir = os.path.join(seg_dir, "images")
    completed_mask_dir = os.path.join(seg_dir, "masks")
    os.makedirs(completed_image_dir, exist_ok=True)
    os.makedirs(completed_mask_dir, exist_ok=True)

    indexed_link_folder(clip_frames, input_image_dir)
    indexed_link_folder(clip_masks, input_mask_dir)

    pred_res = oriented_resolution(cfg.completion.completion_resolution, clip_frames[0])
    modal_pixels, ori_shape = load_and_transform_masks_fn(
        input_mask_dir,
        resolution=pred_res,
        obj_id=args.obj_id,
    )
    rgb_pixels, _, _ = load_and_transform_rgbs_fn(input_image_dir, resolution=pred_res)
    depth_pixels = rgb_to_depth_fn(rgb_pixels, depth_model)

    with torch.inference_mode():
        pred_mask_frames = pipeline_mask(
            modal_pixels,
            depth_pixels,
            height=pred_res[0],
            width=pred_res[1],
            num_frames=len(clip_indices),
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]

        pred_masks = binarize_pipeline_masks(pred_mask_frames, pred_res)
        modal_union = (modal_pixels[0, :, 0, :, :].cpu().numpy() > 0).astype(np.uint8)
        pred_masks = np.logical_or(pred_masks, modal_union).astype(np.uint8)

        pred_mask_tensor = (
            torch.from_numpy(np.where(pred_masks == 0, -1, 1))
            .float()
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
        )
        modal_obj_mask = (modal_pixels > 0).float()
        modal_background = 1 - modal_obj_mask
        modal_rgb_pixels = ((rgb_pixels + 1) / 2) * modal_obj_mask + modal_background
        modal_rgb_pixels = modal_rgb_pixels * 2 - 1

        pred_rgb_frames = pipeline_rgb(
            modal_rgb_pixels,
            pred_mask_tensor,
            height=pred_res[0],
            width=pred_res[1],
            num_frames=len(clip_indices),
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]

    full_h, full_w = ori_shape
    completed_frame_paths = []
    completed_mask_paths = []
    for local_i in occ_local:
        global_i = clip_indices[local_i]
        frame_stem = stem_from_frame(frame_paths[global_i])

        mask = cv2.resize(
            pred_masks[local_i].astype(np.uint8),
            (full_w, full_h),
            interpolation=cv2.INTER_NEAREST,
        )
        original_mask = np.array(Image.open(mask_paths[global_i]).convert("P"))
        mask = np.logical_or(mask > 0, original_mask == args.obj_id).astype(np.uint8)

        local_mask_path = os.path.join(completed_mask_dir, f"{local_i:08d}.png")
        output_mask_path = os.path.join(out_seq_dir, "masks", f"{frame_stem}.png")
        save_palette_mask(mask, local_mask_path, obj_id=args.obj_id)
        save_palette_mask(mask, output_mask_path, obj_id=args.obj_id)
        completed_mask_paths.append(output_mask_path)

        rgb = np.asarray(pred_rgb_frames[local_i]).astype(np.uint8)
        rgb = cv2.resize(rgb, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        local_rgb_path = os.path.join(completed_image_dir, f"{local_i:08d}.jpg")
        output_rgb_path = os.path.join(out_seq_dir, "completed_images", f"{frame_stem}.jpg")
        os.makedirs(os.path.dirname(output_rgb_path), exist_ok=True)
        cv2.imwrite(local_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        completed_frame_paths.append(output_rgb_path)

    del modal_pixels, rgb_pixels, depth_pixels, pred_mask_tensor, modal_rgb_pixels
    torch.cuda.empty_cache()
    gc.collect()

    if completed_frame_paths:
        cam_int = estimator.fov_estimator.get_cam_intrinsics(
            np.array(Image.open(frame_paths[0]).convert("RGB")).astype("uint8")
        )
        reconstruct_completed_frames(
            estimator,
            completed_frame_paths,
            completed_mask_paths,
            [frame_paths[i] for i in range(segment.start, segment.end + 1)],
            out_seq_dir,
            cam_int=cam_int,
            batch_size=args.batch_size,
            obj_id=args.obj_id,
        )

    return {
        "segment": asdict(segment),
        "clip_indices": clip_indices,
        "completed_global_indices": list(range(segment.start, segment.end + 1)),
        "completion_dir": seg_dir,
        "num_completed": len(completed_frame_paths),
    }


def reconstruct_completed_frames(
    estimator,
    completed_frame_paths,
    completed_mask_paths,
    original_frame_paths,
    out_seq_dir,
    cam_int,
    batch_size,
    obj_id=1,
):
    from models.sam_3d_body.notebook.utils import process_image_with_mask  # noqa: WPS433

    mhr_shape_scale_dict = {}
    mhr_dir = os.path.join(out_seq_dir, "mhr_params")
    os.makedirs(mhr_dir, exist_ok=True)
    batch_size = max(1, int(batch_size))

    for start in range(0, len(completed_frame_paths), batch_size):
        end = min(len(completed_frame_paths), start + batch_size)
        images = completed_frame_paths[start:end]
        masks = completed_mask_paths[start:end]
        originals = original_frame_paths[start:end]
        occ_dict = {obj_id: [1] * len(images)}
        with torch.autocast("cuda", enabled=False):
            outputs, _, empty_frames = process_image_with_mask(
                estimator,
                images,
                masks,
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict=mhr_shape_scale_dict,
                occ_dict=occ_dict,
                batch_kps=None,
                cam_int=cam_int,
                iou_dict=None,
                predictor=None,
            )
        empty_frames = set(empty_frames)
        for local_idx, original_path in enumerate(originals):
            data = None if local_idx in empty_frames else outputs[local_idx]
            out_path = os.path.join(mhr_dir, f"{stem_from_frame(original_path)}_data.npz")
            tmp = out_path + ".tmp.npz"
            np.savez_compressed(tmp, data=data)
            os.replace(tmp, out_path)
        torch.cuda.empty_cache()
        gc.collect()


def apply_config_overrides(args):
    cfg = OmegaConf.load(args.config_path)
    if args.ckpt_root is not None:
        cfg.paths.ckpt_root = args.ckpt_root
    if args.sam3_ckpt_path is not None:
        cfg.sam3.ckpt_path = args.sam3_ckpt_path
    if args.sam_3d_body_ckpt_path is not None:
        cfg.sam_3d_body.ckpt_path = args.sam_3d_body_ckpt_path
    if args.mhr_path is not None:
        cfg.sam_3d_body.mhr_path = args.mhr_path
    if args.fov_path is not None:
        cfg.sam_3d_body.fov_path = args.fov_path
    if args.detector_path is not None:
        cfg.sam_3d_body.detector_path = args.detector_path
    cfg.completion.enable = True
    cfg.completion.max_occ_len = int(args.max_occ_len)
    if args.batch_size is not None:
        cfg.sam_3d_body.batch_size = int(args.batch_size)
    if args.completion_resolution is not None:
        cfg.completion.completion_resolution = [int(x) for x in args.completion_resolution.split(",")]
    cfg.runtime.output_dir = args.save_root

    cfg_dir = os.path.join(args.save_root, ".runtime_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(
        cfg_dir,
        f"body4d_completion_opt_worker{args.worker_id}_{os.getpid()}.yaml",
    )
    OmegaConf.save(config=cfg, f=cfg_path, resolve=True)
    return cfg, cfg_path


def build_models(args):
    from models.diffusion_vas.demo import (  # noqa: WPS433
        load_and_transform_masks,
        load_and_transform_rgbs,
        rgb_to_depth,
    )
    from offline_app_mask_kp import (  # noqa: WPS433
        build_diffusion_vas_config,
        build_sam3_3d_body_config,
    )

    cfg, cfg_path = apply_config_overrides(args)
    if not args.quiet:
        print(f"[CONFIG] completion-enabled runtime config -> {cfg_path}")
    estimator = build_sam3_3d_body_config(cfg, human_detector=None)
    pipeline_mask, pipeline_rgb, depth_model, _, generator = build_diffusion_vas_config(cfg)
    return (
        cfg,
        estimator,
        pipeline_mask,
        pipeline_rgb,
        depth_model,
        generator,
        load_and_transform_masks,
        load_and_transform_rgbs,
        rgb_to_depth,
    )


def prepare_output_sequence(record, overwrite=False):
    os.makedirs(record.output_dir, exist_ok=True)
    mirror_dir(os.path.join(record.result_dir, "masks"), os.path.join(record.output_dir, "masks"), overwrite=overwrite)
    mirror_dir(
        os.path.join(record.result_dir, "mhr_params"),
        os.path.join(record.output_dir, "mhr_params"),
        overwrite=overwrite,
    )
    meta_src = os.path.join(record.result_dir, "inference_meta.json")
    if os.path.exists(meta_src):
        link_or_copy_file(meta_src, os.path.join(record.output_dir, "inference_meta_source.json"), overwrite=overwrite)


def process_sequence(record, args, models=None):
    frame_paths = image_paths(record.frame_dir)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames found for {record.name}: {record.frame_dir}")

    mask_paths = [os.path.join(record.result_dir, "masks", f"{stem_from_frame(p)}.png") for p in frame_paths]
    missing_masks = [p for p in mask_paths if not os.path.exists(p)]
    if missing_masks:
        raise RuntimeError(f"{record.name}: missing {len(missing_masks)} masks, first={missing_masks[0]}")
    mhr_dir = os.path.join(record.result_dir, "mhr_params")

    prepare_output_sequence(record, overwrite=args.overwrite_base)
    segments, debug = detect_segments(frame_paths, mask_paths, mhr_dir, args)

    debug_path = os.path.join(record.output_dir, "completion_detection_debug.npz")
    np.savez_compressed(debug_path, **debug)

    if args.dry_run or not segments:
        save_json(
            os.path.join(record.output_dir, "completion_optimization_meta.json"),
            {
                "sequence": record.name,
                "frame_dir": record.frame_dir,
                "source_result_dir": record.result_dir,
                "output_dir": record.output_dir,
                "num_frames": len(frame_paths),
                "selected_segments": [asdict(seg) for seg in segments],
                "processed_segments": [],
                "dry_run": bool(args.dry_run),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return {"sequence": record.name, "frames": len(frame_paths), "segments": len(segments), "processed": 0}

    if models is None:
        raise RuntimeError("Models are required when segments are selected and --dry_run is not used.")

    processed = []
    for segment in segments:
        processed.append(run_completion_for_segment(models, frame_paths, mask_paths, segment, record.output_dir, args))

    save_json(
        os.path.join(record.output_dir, "completion_optimization_meta.json"),
        {
            "sequence": record.name,
            "frame_dir": record.frame_dir,
            "source_result_dir": record.result_dir,
            "output_dir": record.output_dir,
            "num_frames": len(frame_paths),
            "selected_segments": [asdict(seg) for seg in segments],
            "processed_segments": processed,
            "dry_run": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "thresholds": {
                "area_drop_ratio": args.area_drop_ratio,
                "kp_jump_ratio": args.kp_jump_ratio,
                "kp_jump_min": args.kp_jump_min,
                "max_occ_len": args.max_occ_len,
                "context": args.context,
                "max_segments_per_video": args.max_segments_per_video,
            },
        },
    )
    return {
        "sequence": record.name,
        "frames": len(frame_paths),
        "segments": len(segments),
        "processed": len(processed),
    }


def run_single_process(args):
    records = list_sequences(
        args.data_root,
        args.result_root,
        args.save_root,
        sequence=args.sequence,
        sequence_file=args.sequence_file,
    )
    if args.max_sequences is not None:
        records = records[: args.max_sequences]
    if args.expected_sequences is not None and len(records) != args.expected_sequences:
        raise RuntimeError(f"Expected {args.expected_sequences} sequences, found {len(records)}.")

    os.makedirs(args.save_root, exist_ok=True)
    todo = []
    for record in records:
        meta_path = os.path.join(record.output_dir, "completion_optimization_meta.json")
        if os.path.exists(meta_path) and not args.overwrite:
            continue
        todo.append(record)

    if not args.quiet:
        print(f"[VOCCL3D] selected={len(records)} todo={len(todo)} result_root={args.result_root}")
    if not todo:
        return

    models = None
    summaries = []
    iterator = tqdm(todo, desc="Optimizing VOccl3D masks", disable=not args.show_progress)
    for record in iterator:
        if not args.dry_run and models is None:
            models = build_models(args)
        summary = process_sequence(record, args, models=models)
        summaries.append(summary)
        if args.show_progress:
            iterator.set_postfix_str(f"{record.name} seg={summary['segments']}")

    save_json(
        os.path.join(args.save_root, f"completion_optimization_worker{args.worker_id}_summary.json"),
        {
            "worker_id": args.worker_id,
            "num_sequences": len(summaries),
            "summaries": summaries,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def append_optional_arg(cmd, name, value):
    if value is not None:
        cmd.extend([name, str(value)])


def build_worker_command(args, seq_names, worker_id, show_progress):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--data_root",
        args.data_root,
        "--result_root",
        args.result_root,
        "--save_root",
        args.save_root,
        "--config_path",
        args.config_path,
        "--sequence",
        ",".join(seq_names),
        "--obj_id",
        str(args.obj_id),
        "--ref_window",
        str(args.ref_window),
        "--area_drop_ratio",
        str(args.area_drop_ratio),
        "--kp_jump_ratio",
        str(args.kp_jump_ratio),
        "--kp_jump_min",
        str(args.kp_jump_min),
        "--min_mask_area",
        str(args.min_mask_area),
        "--merge_gap",
        str(args.merge_gap),
        "--context",
        str(args.context),
        "--max_occ_len",
        str(args.max_occ_len),
        "--max_segments_per_video",
        str(args.max_segments_per_video),
        "--library_log_level",
        args.library_log_level,
        "--worker_id",
        str(worker_id),
    ]
    append_optional_arg(cmd, "--ckpt_root", args.ckpt_root)
    append_optional_arg(cmd, "--sam3_ckpt_path", args.sam3_ckpt_path)
    append_optional_arg(cmd, "--sam_3d_body_ckpt_path", args.sam_3d_body_ckpt_path)
    append_optional_arg(cmd, "--mhr_path", args.mhr_path)
    append_optional_arg(cmd, "--fov_path", args.fov_path)
    append_optional_arg(cmd, "--detector_path", args.detector_path)
    append_optional_arg(cmd, "--batch_size", args.batch_size)
    append_optional_arg(cmd, "--completion_resolution", args.completion_resolution)
    append_optional_arg(cmd, "--max_frames", args.max_frames)
    if args.overwrite:
        cmd.append("--overwrite")
    if args.overwrite_base:
        cmd.append("--overwrite_base")
    if args.dry_run:
        cmd.append("--dry_run")
    if show_progress:
        cmd.append("--show_progress")
    else:
        cmd.append("--quiet")
    return cmd


def tail_file(path, num_lines=40):
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[-num_lines:])


def launch_multi_gpu(args):
    gpus = parse_gpus(args.gpus)
    if not gpus:
        print("[MULTI-GPU] No CUDA GPUs detected; falling back to single process.")
        return run_single_process(args)

    records = list_sequences(
        args.data_root,
        args.result_root,
        args.save_root,
        sequence=args.sequence,
        sequence_file=args.sequence_file,
    )
    if args.max_sequences is not None:
        records = records[: args.max_sequences]
    if args.expected_sequences is not None and len(records) != args.expected_sequences:
        raise RuntimeError(f"Expected {args.expected_sequences} sequences, found {len(records)}.")
    if not args.overwrite:
        records = [
            r
            for r in records
            if not os.path.exists(os.path.join(r.output_dir, "completion_optimization_meta.json"))
        ]
    if not records:
        print("[MULTI-GPU] all selected sequences are already complete.")
        return

    num_workers = min(len(gpus), len(records))
    shards, shard_costs = split_sequences_balanced(records, num_workers)
    log_dir = os.path.join(args.save_root, ".logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_gpu = str(args.progress_gpu).strip()
    progress_enabled = progress_gpu.lower() != "none"
    active_gpus = gpus[:num_workers]
    fallback_progress_worker = 0 if progress_enabled and progress_gpu not in active_gpus else None

    manifest = {
        "timestamp": timestamp,
        "data_root": args.data_root,
        "result_root": args.result_root,
        "save_root": args.save_root,
        "gpus": gpus,
        "shards": [],
    }
    procs = []
    for worker_id, (gpu_id, shard, cost) in enumerate(zip(gpus, shards, shard_costs)):
        if not shard:
            continue
        seq_names = [record.name for record in shard]
        show_progress = progress_enabled and (gpu_id == progress_gpu or worker_id == fallback_progress_worker)
        cmd = build_worker_command(args, seq_names, worker_id, show_progress=show_progress)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        env["LOG_LEVEL"] = args.library_log_level
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_path = None
        log_f = None
        if not show_progress:
            log_path = os.path.join(log_dir, f"completion_opt_gpu{gpu_id}_worker{worker_id}_{timestamp}.log")
            log_f = open(log_path, "w")
        print(
            f"[MULTI-GPU] worker {worker_id} -> GPU {gpu_id}, "
            f"{len(seq_names)} seqs/{cost} frames, "
            f"{'terminal progress' if show_progress else 'log ' + log_path}"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((worker_id, gpu_id, log_path, log_f, proc))
        manifest["shards"].append(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "num_sequences": len(seq_names),
                "num_frames": int(cost),
                "sequences": seq_names,
                "log_path": log_path,
            }
        )

    manifest_path = os.path.join(log_dir, f"completion_opt_{timestamp}_manifest.json")
    save_json(manifest_path, manifest)
    print(f"[MULTI-GPU] manifest -> {manifest_path}")

    failures = []
    for worker_id, gpu_id, log_path, log_f, proc in procs:
        ret = proc.wait()
        if log_f is not None:
            log_f.close()
        print(f"[MULTI-GPU] worker {worker_id} on GPU {gpu_id} exited with {ret}")
        if ret != 0:
            failures.append((worker_id, gpu_id, ret, log_path))
    if failures:
        for worker_id, gpu_id, ret, log_path in failures:
            print(f"\n[ERROR] worker {worker_id} GPU {gpu_id} failed with code {ret}")
            if log_path:
                print(tail_file(log_path))
        raise RuntimeError(f"{len(failures)} worker(s) failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize existing VOccl3D mask results with Diffusion-VAS local mask completion and MHR reconstruction."
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result_root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--save_root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence", default="all")
    parser.add_argument(
        "--sequence_file",
        default=None,
        help="Optional newline-separated sequence list. Each line can be scene/name, name only, or scene only.",
    )
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--ckpt_root", default=None)
    parser.add_argument("--sam3_ckpt_path", default=None)
    parser.add_argument("--sam_3d_body_ckpt_path", "--body_ckpt_path", "--ckpt_path", dest="sam_3d_body_ckpt_path", default=None)
    parser.add_argument("--mhr_path", "--mhr_model_path", dest="mhr_path", default=None)
    parser.add_argument("--fov_path", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--completion_resolution", default=None, help="Override config completion_resolution, e.g. 256,512")
    parser.add_argument("--ref_window", type=int, default=8)
    parser.add_argument("--area_drop_ratio", type=float, default=0.65)
    parser.add_argument("--kp_jump_ratio", type=float, default=2.5)
    parser.add_argument("--kp_jump_min", type=float, default=0.08)
    parser.add_argument("--min_mask_area", type=float, default=512.0)
    parser.add_argument("--merge_gap", type=int, default=1)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--max_occ_len", type=int, default=32)
    parser.add_argument("--max_segments_per_video", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--expected_sequences", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Re-process sequences even when optimization meta exists.")
    parser.add_argument("--overwrite_base", action="store_true", help="Overwrite mirrored source masks/MHR before optimization.")
    parser.add_argument("--dry_run", action="store_true", help="Only detect local segments and mirror source outputs; do not load GPU models.")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--progress_gpu", default="0")
    parser.add_argument("--show_progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--library_log_level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="WARNING",
        help="Suppress noisy library logs by default while keeping warnings/errors.",
    )
    args = parser.parse_args()
    configure_library_logging(args.library_log_level)

    if args.multi_gpu:
        launch_multi_gpu(args)
    else:
        if not args.quiet:
            args.show_progress = True
        run_single_process(args)


if __name__ == "__main__":
    main()
