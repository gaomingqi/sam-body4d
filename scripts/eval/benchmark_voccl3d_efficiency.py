import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
DIFFUSION_VAS_DIR = os.path.join(REPO_DIR, "models", "diffusion_vas")
SAM3D_DIR = os.path.join(REPO_DIR, "models", "sam_3d_body")
for path in (CURRENT_DIR, SCRIPTS_DIR, REPO_DIR, DIFFUSION_VAS_DIR, SAM3D_DIR):
    if path not in sys.path:
        sys.path.append(path)

from run_voccl3d import (  # noqa: E402
    apply_config_overrides,
    detect_person_boxes_for_sequence,
    load_frame_paths,
    segment_frame_person,
    segment_sequence_person_per_frame,
)
from optimize_voccl3d_mask_completion import (  # noqa: E402
    binarize_pipeline_masks,
    detect_segments,
    indexed_link_folder,
    oriented_resolution,
)
from offline_app_box_kp import build_diffusion_vas_config, offline_app as BoxOfflineApp  # noqa: E402
from offline_app_mask_kp import OfflineApp as MaskOfflineApp  # noqa: E402
from models.diffusion_vas.demo import (  # noqa: E402
    load_and_transform_masks,
    load_and_transform_rgbs,
    rgb_to_depth,
)
from models.sam_3d_body.notebook.utils import process_image_with_bbox, process_image_with_mask  # noqa: E402
from models.sam_3d_body.tools.build_detector import HumanDetector  # noqa: E402


DEFAULT_DATA_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D"
DEFAULT_MASK_RESULT_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask"
DEFAULT_CONFIG = os.path.join(REPO_DIR, "configs", "body4d.yaml")
DEFAULT_SEQUENCE = "scene20_view1/SMPLX-male_kneel_down_with_stool_left_hand10_stageii"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    cuda_sync()
    start = time.perf_counter()
    result = fn()
    cuda_sync()
    return time.perf_counter() - start, result


def profiler_flops(fn, enabled=True):
    if not enabled:
        return float("nan")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    cuda_sync()
    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            fn()
            cuda_sync()
    total = 0
    for evt in prof.key_averages():
        total += int(getattr(evt, "flops", 0) or 0)
    return float(total)


def fps(num_frames, seconds):
    return float(num_frames) / float(seconds) if seconds > 0 else float("nan")


def gflops(value):
    return float(value) / 1e9 if np.isfinite(value) else float("nan")


def make_runtime_config(args, output_dir, prompt_mode):
    cfg_args = SimpleNamespace(
        config_path=args.config_path,
        ckpt_root=args.ckpt_root,
        sam3_ckpt_path=args.sam3_ckpt_path,
        sam_3d_body_ckpt_path=args.sam_3d_body_ckpt_path,
        mhr_path=args.mhr_path,
        fov_path=args.fov_path,
        detector_path=args.detector_path,
        batch_size=args.batch_size,
        output_dir=output_dir,
        prompt_mode=prompt_mode,
        worker_id=0,
    )
    return apply_config_overrides(cfg_args)


def resolve_seq_dir(data_root, sequence):
    seq_dir = os.path.join(data_root, sequence.split(os.sep)[0], "images", *sequence.split(os.sep)[1:])
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence images not found: {seq_dir}")
    return seq_dir


def release_cuda(*items):
    for item in items:
        del item
    cuda_sync()
    torch.cuda.empty_cache()


def run_mhr_box_no_save(app, frame_paths, boxes, batch_size):
    cam_int = app.sam3_3d_body_model.fov_estimator.get_cam_intrinsics(
        np.array(Image.open(frame_paths[0]).convert("RGB")).astype("uint8")
    )
    mhr_shape_scale_dict = {}
    for start in range(0, len(frame_paths), batch_size):
        end = min(len(frame_paths), start + batch_size)
        batch_images = frame_paths[start:end]
        batch_boxes = [torch.from_numpy(boxes[start:end, obj_idx, :]).float() for obj_idx in range(boxes.shape[1])]
        occ_dict = {obj_idx + 1: [1] * len(batch_images) for obj_idx in range(boxes.shape[1])}
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            process_image_with_bbox(
                app.sam3_3d_body_model,
                batch_images,
                batch_boxes,
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict=mhr_shape_scale_dict,
                occ_dict=occ_dict,
                batch_kps=None,
                flip=False,
                cam_int=cam_int,
            )


def run_mhr_mask_no_save(app, frame_paths, mask_paths, batch_size):
    cam_int = app.sam3_3d_body_model.fov_estimator.get_cam_intrinsics(
        np.array(Image.open(frame_paths[0]).convert("RGB")).astype("uint8")
    )
    mhr_shape_scale_dict = {}
    for start in range(0, len(frame_paths), batch_size):
        end = min(len(frame_paths), start + batch_size)
        batch_images = frame_paths[start:end]
        batch_masks = mask_paths[start:end]
        occ_dict = {1: [1] * len(batch_images)}
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            process_image_with_mask(
                app.sam3_3d_body_model,
                batch_images,
                batch_masks,
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict=mhr_shape_scale_dict,
                occ_dict=occ_dict,
                batch_kps=None,
                cam_int=cam_int,
                iou_dict=None,
                predictor=None,
            )


def benchmark_box(args, frame_paths, output_dir):
    config_path = make_runtime_config(args, output_dir, "box")
    app = BoxOfflineApp(config_path=config_path, load_sam3=False)
    app.RUNTIME["batch_size"] = args.batch_size
    detector = HumanDetector(
        name="vitdet",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        path=args.detector_path or "",
    )

    det_seconds, (boxes, box_debug) = timed(
        lambda: detect_person_boxes_for_sequence(
            detector,
            frame_paths,
            max_people=args.max_people,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            fallback_mode=args.box_fallback,
        )
    )
    mhr_seconds, _ = timed(lambda: run_mhr_box_no_save(app, frame_paths, boxes, args.batch_size))

    image0 = cv2.imread(frame_paths[0])
    box0 = boxes[:1]
    det_flops = profiler_flops(
        lambda: detector.run_human_detection(
            image0,
            det_cat_id=0,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            default_to_full_image=False,
        ),
        enabled=args.profile_flops,
    )
    mhr_flops = profiler_flops(
        lambda: run_mhr_box_no_save(app, frame_paths[:1], box0, 1),
        enabled=args.profile_flops,
    )

    release_cuda(app, detector)
    return {
        "frames": len(frame_paths),
        "detector_seconds": det_seconds,
        "mhr_seconds": mhr_seconds,
        "total_seconds": det_seconds + mhr_seconds,
        "detector_fps": fps(len(frame_paths), det_seconds),
        "mhr_fps": fps(len(frame_paths), mhr_seconds),
        "total_fps": fps(len(frame_paths), det_seconds + mhr_seconds),
        "fallback_frames": int((box_debug[:, 3] > 0).sum()) if len(box_debug) else 0,
        "flops_single_frame": {
            "detector_gflops": gflops(det_flops),
            "mhr_box_gflops": gflops(mhr_flops),
            "total_gflops": gflops(det_flops + mhr_flops),
            "note": "PyTorch profiler lower bound; only supported torch ops are counted.",
        },
    }


def benchmark_mask(args, frame_paths, output_dir):
    config_path = make_runtime_config(args, output_dir, "mask")
    app = MaskOfflineApp(config_path=config_path)
    app.OUTPUT_DIR = output_dir
    app.RUNTIME["batch_size"] = args.batch_size
    app.RUNTIME["out_obj_ids"] = [1]
    os.makedirs(output_dir, exist_ok=True)

    sam_seconds, _ = timed(lambda: segment_sequence_person_per_frame(app, frame_paths, save_feature_cache=False))
    mask_paths = [
        os.path.join(output_dir, "masks", f"{os.path.splitext(os.path.basename(path))[0]}.png")
        for path in frame_paths
    ]
    mhr_seconds, _ = timed(lambda: run_mhr_mask_no_save(app, frame_paths, mask_paths, args.batch_size))

    sam_flops = profiler_flops(lambda: segment_frame_person(app, frame_paths[0]), enabled=args.profile_flops)
    mhr_flops = profiler_flops(
        lambda: run_mhr_mask_no_save(app, frame_paths[:1], mask_paths[:1], 1),
        enabled=args.profile_flops,
    )

    release_cuda(app)
    return {
        "frames": len(frame_paths),
        "sam_mask_seconds": sam_seconds,
        "mhr_seconds": mhr_seconds,
        "total_seconds": sam_seconds + mhr_seconds,
        "sam_mask_fps": fps(len(frame_paths), sam_seconds),
        "mhr_fps": fps(len(frame_paths), mhr_seconds),
        "total_fps": fps(len(frame_paths), sam_seconds + mhr_seconds),
        "flops_single_frame": {
            "sam_mask_gflops": gflops(sam_flops),
            "mhr_mask_gflops": gflops(mhr_flops),
            "total_gflops": gflops(sam_flops + mhr_flops),
            "note": "PyTorch profiler lower bound; only supported torch ops are counted.",
        },
    }


def apply_diffusion_config_overrides(args):
    cfg = OmegaConf.load(args.config_path)
    if args.ckpt_root is not None:
        cfg.paths.ckpt_root = args.ckpt_root
    if args.completion_resolution is not None:
        cfg.completion.completion_resolution = [int(x) for x in args.completion_resolution.split(",")]
    if args.batch_size is not None:
        cfg.sam_3d_body.batch_size = int(args.batch_size)
    cfg.completion.max_occ_len = int(args.max_occ_len)
    return cfg


def pick_completion_clip(args, frame_paths):
    result_dir = os.path.join(args.mask_result_root, args.sequence)
    mask_paths = [
        os.path.join(result_dir, "masks", f"{os.path.splitext(os.path.basename(path))[0]}.png")
        for path in frame_paths
    ]
    mhr_dir = os.path.join(result_dir, "mhr_params")
    detect_args = SimpleNamespace(
        obj_id=1,
        ref_window=args.ref_window,
        area_drop_ratio=args.area_drop_ratio,
        min_mask_area=args.min_mask_area,
        kp_jump_ratio=args.kp_jump_ratio,
        kp_jump_min=args.kp_jump_min,
        segment_signal="mask",
        merge_gap=args.merge_gap,
        max_occ_len=args.max_occ_len,
        context=args.context,
        max_segments_per_video=1,
        replace_normal_frames=False,
    )
    segments, _ = detect_segments(frame_paths, mask_paths, mhr_dir, detect_args)
    if segments:
        seg = segments[0]
        indices = list(range(seg.clip_start, seg.clip_end + 1))
        replace_indices = list(seg.replace_indices)
        source = "detected_segment"
    else:
        count = min(args.diffusion_fallback_frames, len(frame_paths))
        indices = list(range(count))
        replace_indices = indices
        source = "fallback_first_frames"
    return result_dir, indices, replace_indices, source


def diffusion_forward_once(
    pipeline_mask,
    pipeline_rgb,
    depth_model,
    generator,
    image_dir,
    mask_dir,
    pred_res,
    obj_id=1,
    num_inference_steps=25,
):
    modal_pixels, ori_shape = load_and_transform_masks(mask_dir, resolution=pred_res, obj_id=obj_id)
    rgb_pixels, _, _ = load_and_transform_rgbs(image_dir, resolution=pred_res)
    depth_pixels = rgb_to_depth(rgb_pixels, depth_model)
    with torch.inference_mode():
        pred_mask_frames = pipeline_mask(
            modal_pixels,
            depth_pixels,
            height=pred_res[0],
            width=pred_res[1],
            num_frames=modal_pixels.shape[1],
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            num_inference_steps=num_inference_steps,
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
            num_frames=modal_pixels.shape[1],
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]
    return pred_masks, pred_rgb_frames, ori_shape


def benchmark_diffusion(args, frame_paths, output_dir):
    result_dir, indices, replace_indices, source = pick_completion_clip(args, frame_paths)
    clip_frames = [frame_paths[i] for i in indices]
    clip_masks = [
        os.path.join(result_dir, "masks", f"{os.path.splitext(os.path.basename(frame_paths[i]))[0]}.png")
        for i in indices
    ]
    clip_dir = os.path.join(output_dir, "diffusion_clip")
    image_dir = os.path.join(clip_dir, "images")
    mask_dir = os.path.join(clip_dir, "masks")
    indexed_link_folder(clip_frames, image_dir)
    indexed_link_folder(clip_masks, mask_dir)

    cfg = apply_diffusion_config_overrides(args)
    pipeline_mask, pipeline_rgb, depth_model, _, generator = build_diffusion_vas_config(cfg)
    pred_res = oriented_resolution(cfg.completion.completion_resolution, clip_frames[0])

    diffusion_seconds, _ = timed(
        lambda: diffusion_forward_once(
            pipeline_mask,
            pipeline_rgb,
            depth_model,
            generator,
            image_dir,
            mask_dir,
            pred_res,
            obj_id=1,
            num_inference_steps=args.diffusion_num_inference_steps,
        )
    )
    diffusion_flops = profiler_flops(
        lambda: diffusion_forward_once(
            pipeline_mask,
            pipeline_rgb,
            depth_model,
            generator,
            image_dir,
            mask_dir,
            pred_res,
            obj_id=1,
            num_inference_steps=args.diffusion_num_inference_steps,
        ),
        enabled=args.profile_flops,
    )
    flops_per_clip_frame = diffusion_flops / max(len(clip_frames), 1)
    if args.diffusion_num_inference_steps != 25 and np.isfinite(flops_per_clip_frame):
        flops_per_clip_frame_25step = flops_per_clip_frame * (25.0 / float(args.diffusion_num_inference_steps))
        clip_total_gflops_25step = diffusion_flops * (25.0 / float(args.diffusion_num_inference_steps))
    else:
        flops_per_clip_frame_25step = flops_per_clip_frame
        clip_total_gflops_25step = diffusion_flops

    release_cuda(pipeline_mask, pipeline_rgb, depth_model)
    return {
        "source": source,
        "clip_frames": len(clip_frames),
        "replace_frames": len(replace_indices),
        "clip_indices": indices,
        "replace_indices": replace_indices,
        "resolution": list(pred_res),
        "diffusion_num_inference_steps": int(args.diffusion_num_inference_steps),
        "total_seconds": diffusion_seconds,
        "clip_fps": fps(len(clip_frames), diffusion_seconds),
        "effective_replace_fps": fps(len(replace_indices), diffusion_seconds),
        "flops_single_clip_frame": {
            "diffusion_vas_gflops": gflops(flops_per_clip_frame),
            "diffusion_vas_gflops_est_25step": gflops(flops_per_clip_frame_25step),
            "clip_total_gflops": gflops(diffusion_flops),
            "clip_total_gflops_est_25step": gflops(clip_total_gflops_25step),
            "note": "Profiler lower bound for depth + mask diffusion + RGB diffusion, divided by clip frames. est_25step linearly scales denoising steps.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark VOccl3D box/mask/Diffusion-VAS efficiency without saving MHR npz.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--mask_result_root", default=DEFAULT_MASK_RESULT_ROOT)
    parser.add_argument("--output_dir", default="/home/mingqi/data/results/hmr/bench_runtime_voccl3d/nosave")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt_root", default="/home/mingqi/data/checkpoints/hmr")
    parser.add_argument("--sam3_ckpt_path", default=None)
    parser.add_argument("--sam_3d_body_ckpt_path", "--body_ckpt_path", "--ckpt_path", dest="sam_3d_body_ckpt_path", default=None)
    parser.add_argument("--mhr_path", "--mhr_model_path", dest="mhr_path", default=None)
    parser.add_argument("--fov_path", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--modes", default="box,mask,diffusion")
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--max_people", type=int, default=1)
    parser.add_argument("--bbox_thr", type=float, default=0.5)
    parser.add_argument("--nms_thr", type=float, default=0.3)
    parser.add_argument("--box_fallback", choices=("previous", "full_image"), default="previous")
    parser.add_argument("--completion_resolution", default=None)
    parser.add_argument("--ref_window", type=int, default=8)
    parser.add_argument("--area_drop_ratio", type=float, default=0.65)
    parser.add_argument("--kp_jump_ratio", type=float, default=2.5)
    parser.add_argument("--kp_jump_min", type=float, default=0.08)
    parser.add_argument("--min_mask_area", type=float, default=512.0)
    parser.add_argument("--merge_gap", type=int, default=1)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--max_occ_len", type=int, default=32)
    parser.add_argument("--diffusion_fallback_frames", type=int, default=8)
    parser.add_argument("--diffusion_num_inference_steps", type=int, default=25)
    args = parser.parse_args()

    seq_dir = resolve_seq_dir(args.data_root, args.sequence)
    frame_paths = load_frame_paths(seq_dir, max_frames=args.max_frames)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "sequence": args.sequence,
        "frames": len(frame_paths),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "profile_flops": bool(args.profile_flops),
        "modes": {},
    }

    for mode in modes:
        mode_out = os.path.join(args.output_dir, mode)
        if os.path.exists(mode_out):
            shutil.rmtree(mode_out)
        os.makedirs(mode_out, exist_ok=True)
        if mode == "box":
            results["modes"]["box"] = benchmark_box(args, frame_paths, mode_out)
        elif mode == "mask":
            results["modes"]["mask"] = benchmark_mask(args, frame_paths, mode_out)
        elif mode == "diffusion":
            results["modes"]["diffusion"] = benchmark_diffusion(args, frame_paths, mode_out)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    output_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(output_path + ".tmp", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    os.replace(output_path + ".tmp", output_path)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[OK] saved benchmark summary -> {output_path}")


if __name__ == "__main__":
    main()
