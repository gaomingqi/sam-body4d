import argparse
import glob
import json
import logging
import os
import subprocess
import sys
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
for path in (SCRIPTS_DIR, REPO_DIR):
    if path not in sys.path:
        sys.path.append(path)

from offline_app_box_kp import offline_app as BoxOfflineApp
from offline_app_mask_kp import (
    OfflineApp as MaskOfflineApp,
    propagate_in_video,
    resize_images_longest_side,
)
from models.sam_3d_body.tools.build_detector import HumanDetector


DEFAULT_VOCCL3D_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D/scene9_view1"
DEFAULT_OUTPUT_DIR = "/home/mingqi/data/predictions/hmr/VOccl3D/scene9_view1"
DEFAULT_CONFIG_PATH = os.path.join(REPO_DIR, "configs", "body4d.yaml")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


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
        "mhr_smpl_conversion",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


def list_sequences(voccl3d_root):
    images_root = os.path.join(voccl3d_root, "images")
    if os.path.isdir(images_root):
        seq_dirs = [
            p for p in sorted(glob.glob(os.path.join(images_root, "*")))
            if os.path.isdir(p)
        ]
        if seq_dirs:
            return seq_dirs

    seq_dirs = []
    for scene_images_root in sorted(glob.glob(os.path.join(voccl3d_root, "*", "images"))):
        seq_dirs.extend(
            p for p in sorted(glob.glob(os.path.join(scene_images_root, "*")))
            if os.path.isdir(p)
        )
    return seq_dirs


def sequence_name(seq_dir, voccl3d_root):
    rel_path = os.path.relpath(os.path.abspath(seq_dir), os.path.abspath(voccl3d_root))
    parts = rel_path.split(os.sep)
    if "images" not in parts:
        return os.path.basename(seq_dir.rstrip(os.sep))

    image_idx = parts.index("images")
    seq_parts = parts[image_idx + 1 :]
    if image_idx == 0:
        return os.path.join(*seq_parts)
    return os.path.join(*(parts[:image_idx] + seq_parts))


def sequence_out_dir(seq_dir, output_root, voccl3d_root):
    return os.path.join(output_root, sequence_name(seq_dir, voccl3d_root))


def parse_sequence_filter(value):
    if value is None or value == "" or value == "all":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def selected_sequence_dirs(args):
    wanted = parse_sequence_filter(args.sequence)
    seq_dirs = list_sequences(args.voccl3d_root)
    if wanted is not None:
        seq_dirs = [
            p for p in seq_dirs
            if os.path.basename(p) in wanted or sequence_name(p, args.voccl3d_root) in wanted
        ]
    if not seq_dirs:
        raise RuntimeError("No VOccl3D sequences selected.")
    return seq_dirs


def load_frame_paths(seq_dir, max_frames=None):
    frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {seq_dir}")
    return frame_paths


def count_masks(out_dir):
    masks_dir = os.path.join(out_dir, "masks")
    if not os.path.isdir(masks_dir):
        return 0
    return len(glob.glob(os.path.join(masks_dir, "*.png")))


def count_boxes(out_dir):
    box_path = os.path.join(out_dir, "boxes.npz")
    if not os.path.exists(box_path):
        return 0
    try:
        data = np.load(box_path, allow_pickle=True)
        return int(data["bboxes"].shape[0])
    except Exception:
        return 0


def count_mhr_outputs(out_dir):
    mhr_dir = os.path.join(out_dir, "mhr_params")
    if not os.path.isdir(mhr_dir):
        return 0
    return len(glob.glob(os.path.join(mhr_dir, "*_data.npz")))


def close_session(predictor, session_id):
    if session_id is not None:
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )


def _output_mask_by_id(outputs, out_obj_id):
    out_obj_ids = np.asarray(outputs["out_obj_ids"])
    obj_id = int(out_obj_id.item() if hasattr(out_obj_id, "item") else out_obj_id)
    idx = np.where(out_obj_ids == obj_id)[0]
    if len(idx) == 0:
        raise KeyError(f"Object id {obj_id} not found in SAM outputs {out_obj_ids.tolist()}")
    return outputs["out_binary_masks"][idx[0]]


def _output_mask_area(outputs, out_obj_id):
    return int(np.asarray(_output_mask_by_id(outputs, out_obj_id)).astype(bool).sum())


def select_person_sam_id(outputs, preferred_sam_id=1):
    out_obj_ids = outputs["out_obj_ids"]
    ids = [int(x.item() if hasattr(x, "item") else x) for x in out_obj_ids]
    if not ids:
        raise RuntimeError("SAM3 did not return any object for prompt 'person'.")
    if preferred_sam_id is not None and preferred_sam_id in ids:
        return preferred_sam_id, "preferred"

    best_id = max(
        ids,
        key=lambda sid: _output_mask_area(outputs, sid),
    )
    return best_id, "largest_mask"


def propagate_sequence_person_first(predictor_app, frame_paths, preferred_sam_id=1):
    batch_frames = [Image.open(frame).convert("RGB") for frame in frame_paths]
    resized_batch_frames = resize_images_longest_side(batch_frames)

    response = predictor_app.predictor.handle_request(
        request=dict(type="start_session", resource_path=resized_batch_frames)
    )
    session_id = response["session_id"]
    predictor_app.RUNTIME["session_id"] = session_id

    try:
        response = predictor_app.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="person",
            )
        )
        outputs = response["outputs"]
        target_sam_id, select_mode = select_person_sam_id(
            outputs, preferred_sam_id=preferred_sam_id
        )

        for out_obj_id in outputs["out_obj_ids"]:
            sam_id = int(out_obj_id.item() if hasattr(out_obj_id, "item") else out_obj_id)
            if sam_id == target_sam_id:
                continue
            predictor_app.predictor.handle_request(
                request=dict(type="remove_object", session_id=session_id, obj_id=sam_id)
            )

        outputs_per_frame = propagate_in_video(
            predictor_app.predictor,
            session_id,
            max_num_objects=1,
        )
        features = [outputs_per_frame[i]["feature_cache"] for i in sorted(outputs_per_frame)]
        return outputs_per_frame, resized_batch_frames, features, target_sam_id, select_mode
    finally:
        close_session(predictor_app.predictor, session_id)
        predictor_app.RUNTIME["session_id"] = None


def segment_frame_person(predictor_app, frame_path):
    frame = Image.open(frame_path).convert("RGB")
    resized_frame = resize_images_longest_side([frame])

    response = predictor_app.predictor.handle_request(
        request=dict(type="start_session", resource_path=resized_frame)
    )
    session_id = response["session_id"]
    predictor_app.RUNTIME["session_id"] = session_id

    try:
        response = predictor_app.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="person",
            )
        )
        outputs = response["outputs"]
        out_obj_ids = outputs["out_obj_ids"]
        ids = [int(x.item() if hasattr(x, "item") else x) for x in out_obj_ids]
        if ids:
            target_sam_id, select_mode = select_person_sam_id(
                outputs, preferred_sam_id=None
            )
            target_area = _output_mask_area(outputs, target_sam_id)
            obj_dict = {1: target_sam_id}
        else:
            target_sam_id = -1
            select_mode = "none"
            target_area = 0
            obj_dict = {}

        return (
            {0: outputs},
            resized_frame,
            outputs.get("feature_cache"),
            target_sam_id,
            select_mode,
            target_area,
            len(ids),
            obj_dict,
        )
    finally:
        close_session(predictor_app.predictor, session_id)
        predictor_app.RUNTIME["session_id"] = None


def normalize_feature(feature):
    if feature is None:
        return None
    feature = np.asarray(feature)
    if feature.ndim == 3:
        feature = feature[None]
    return feature


def stack_feature_cache(features):
    valid = [normalize_feature(feature) for feature in features if feature is not None]
    if not valid:
        raise RuntimeError("No SAM feature_cache was produced.")
    zero = np.zeros_like(valid[0])
    filled = [normalize_feature(feature) if feature is not None else zero for feature in features]
    return np.concatenate(filled, axis=0)


def segment_sequence_person_per_frame(predictor_app, frame_paths, save_feature_cache=False):
    features = []
    debug = []
    for frame_idx, frame_path in enumerate(
        tqdm(frame_paths, desc="Per-frame SAM person masks", leave=False)
    ):
        (
            outputs_per_frame,
            resized_frames,
            feature,
            sam_id,
            select_mode,
            target_area,
            num_candidates,
            obj_dict,
        ) = segment_frame_person(predictor_app, frame_path)
        predictor_app.save_masks(
            start_frame_idx=0,
            outputs_per_frame=outputs_per_frame,
            obj_dict=obj_dict,
            resized_batch_frames=resized_frames,
            original_size=Image.open(frame_path).size,
            frame_list=[frame_path],
        )
        if save_feature_cache:
            features.append(feature)
        debug.append(
            (
                frame_idx,
                sam_id,
                target_area,
                num_candidates,
                1 if select_mode == "largest_mask" else 0,
            )
        )
    return features, np.asarray(debug, dtype=np.float32)


def full_image_box(frame_path):
    width, height = Image.open(frame_path).size
    return np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)


def sort_boxes_by_area(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid]
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[np.argsort(-areas)]


def detect_person_boxes_for_sequence(
    detector,
    frame_paths,
    max_people=1,
    bbox_thr=0.5,
    nms_thr=0.3,
    fallback_mode="previous",
):
    if max_people < 1:
        raise ValueError("--max_people must be >= 1")

    all_boxes = []
    debug = []
    previous = None
    for frame_idx, frame_path in enumerate(
        tqdm(frame_paths, desc="Per-frame person boxes", leave=False)
    ):
        image = cv2.imread(frame_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {frame_path}")
        detected = detector.run_human_detection(
            image,
            det_cat_id=0,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            default_to_full_image=False,
        )
        detected = sort_boxes_by_area(detected)
        selected = detected[:max_people]
        detected_count = int(len(detected))
        selected_count = int(len(selected))
        fallback_count = max_people - selected_count

        if fallback_count > 0:
            if fallback_mode == "previous":
                if previous is None:
                    fill = np.repeat(full_image_box(frame_path)[None], fallback_count, axis=0)
                else:
                    fill = previous[selected_count:max_people]
                    if len(fill) < fallback_count:
                        first_box = previous[:1] if len(previous) else full_image_box(frame_path)[None]
                        fill = np.concatenate(
                            [fill, np.repeat(first_box, fallback_count - len(fill), axis=0)],
                            axis=0,
                        )
            elif fallback_mode == "full_image":
                fill = np.repeat(full_image_box(frame_path)[None], fallback_count, axis=0)
            else:
                raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")
            selected = np.concatenate([selected, fill], axis=0) if len(selected) else fill

        selected = selected.astype(np.float32)
        previous = selected
        all_boxes.append(selected)
        debug.append([frame_idx, detected_count, selected_count, fallback_count])

    return np.stack(all_boxes, axis=0), np.asarray(debug, dtype=np.float32)


def load_or_detect_boxes(args, detector, out_dir, frame_paths, frame_names):
    box_path = os.path.join(out_dir, "boxes.npz")
    if os.path.exists(box_path) and not args.overwrite:
        data = np.load(box_path, allow_pickle=True)
        boxes = data["bboxes"].astype(np.float32)
        if boxes.shape[0] >= len(frame_paths) and boxes.shape[1] == args.max_people:
            print(f"[BOXES] reusing {box_path}")
            debug = data["debug"] if "debug" in data.files else np.zeros((len(frame_paths), 4), dtype=np.float32)
            return boxes[: len(frame_paths)], debug[: len(frame_paths)]

    boxes, debug = detect_person_boxes_for_sequence(
        detector,
        frame_paths,
        max_people=args.max_people,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        fallback_mode=args.box_fallback,
    )
    tmp_path = box_path + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        bboxes=boxes,
        frame_names=np.asarray(frame_names),
        debug=debug,
        max_people=args.max_people,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        box_fallback=args.box_fallback,
    )
    os.replace(tmp_path, box_path)
    return boxes, debug


def save_inference_metadata(out_dir, metadata):
    metadata_path = os.path.join(out_dir, "inference_meta.json")
    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp_path, metadata_path)


def infer_sequence_mask(
    predictor,
    seq_dir,
    output_root,
    args,
):
    seq_name = sequence_name(seq_dir, args.voccl3d_root)
    out_dir = sequence_out_dir(seq_dir, output_root, args.voccl3d_root)
    out_path = os.path.join(out_dir, "feature_cache.npz")
    os.makedirs(out_dir, exist_ok=True)
    predictor.OUTPUT_DIR = out_dir
    if args.batch_size is not None:
        predictor.RUNTIME["batch_size"] = int(args.batch_size)

    frame_paths = load_frame_paths(seq_dir, max_frames=args.max_frames)
    frame_names = [os.path.basename(p) for p in frame_paths]
    num_frames = len(frame_paths)

    existing_mhr = count_mhr_outputs(out_dir)
    if existing_mhr == num_frames and not args.overwrite:
        print(f"[SKIP] {seq_name}: mhr_params already has {existing_mhr}/{num_frames} frames")
        return

    predictor.RUNTIME["out_obj_ids"] = [1]
    masks_count = count_masks(out_dir)
    if args.reuse_masks and masks_count == num_frames:
        print(f"[MASKS] {seq_name}: reusing {masks_count}/{num_frames} existing masks")
    else:
        if args.mask_mode == "propagate":
            outputs_per_frame, resized_frames, features, sam_id, select_mode = propagate_sequence_person_first(
                predictor,
                frame_paths,
                preferred_sam_id=args.preferred_sam_id,
            )

            first_size = Image.open(frame_paths[0]).size
            predictor.save_masks(
                start_frame_idx=0,
                outputs_per_frame=outputs_per_frame,
                obj_dict={1: sam_id},
                resized_batch_frames=resized_frames,
                original_size=first_size,
                frame_list=frame_paths,
            )
            feature_data = np.concatenate(features, axis=0) if args.save_feature_cache else None
            mask_debug = np.asarray([[0, sam_id, -1, -1, -1]], dtype=np.float32)
        elif args.mask_mode == "per_frame":
            features, mask_debug = segment_sequence_person_per_frame(
                predictor,
                frame_paths,
                save_feature_cache=args.save_feature_cache,
            )
            feature_data = stack_feature_cache(features) if args.save_feature_cache else None
        else:
            raise ValueError(f"Unsupported mask_mode: {args.mask_mode}")

        if args.save_feature_cache:
            tmp_out_path = out_path + ".tmp.npz"
            np.savez_compressed(
                tmp_out_path,
                data=feature_data,
                frame_names=np.asarray(frame_names),
                sequence_name=seq_name,
                prompt_mode="mask",
                mask_mode=args.mask_mode,
                mask_debug=mask_debug,
                output_obj_id=1,
            )
            os.replace(tmp_out_path, out_path)

    with torch.autocast("cuda", enabled=False):
        predictor.on_4d_generation(
            frame_paths,
            seq_path=seq_dir,
            kps_list=None,
            render=False,
        )

    mhr_count = count_mhr_outputs(out_dir)
    save_inference_metadata(
        out_dir,
        dict(
            sequence=seq_name,
            prompt_mode="mask",
            mask_mode=args.mask_mode,
            frames=num_frames,
            masks=count_masks(out_dir),
            mhr=mhr_count,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        ),
    )
    print(
        f"[OK] {seq_name}: {mhr_count}/{num_frames} MHR frames, "
        f"output {os.path.join(out_dir, 'mhr_params')}"
    )


def infer_sequence_box(
    predictor,
    detector,
    seq_dir,
    output_root,
    args,
):
    seq_name = sequence_name(seq_dir, args.voccl3d_root)
    out_dir = sequence_out_dir(seq_dir, output_root, args.voccl3d_root)
    os.makedirs(out_dir, exist_ok=True)
    predictor.OUTPUT_DIR = out_dir
    if args.batch_size is not None:
        predictor.RUNTIME["batch_size"] = int(args.batch_size)

    frame_paths = load_frame_paths(seq_dir, max_frames=args.max_frames)
    frame_names = [os.path.basename(p) for p in frame_paths]
    num_frames = len(frame_paths)

    existing_mhr = count_mhr_outputs(out_dir)
    if existing_mhr == num_frames and not args.overwrite:
        print(f"[SKIP] {seq_name}: mhr_params already has {existing_mhr}/{num_frames} frames")
        return

    boxes, box_debug = load_or_detect_boxes(args, detector, out_dir, frame_paths, frame_names)
    box_list = [
        torch.from_numpy(boxes[:, obj_idx, :]).float()
        for obj_idx in range(args.max_people)
    ]
    predictor.RUNTIME["bboxes"] = box_list
    predictor.RUNTIME["out_obj_ids"] = list(range(1, args.max_people + 1))

    with torch.autocast("cuda", enabled=False):
        predictor.on_4d_generation(
            frame_paths,
            box_list,
            kps_list=None,
            flip=False,
            render=args.render_box,
        )

    mhr_count = count_mhr_outputs(out_dir)
    fallback_frames = int((box_debug[:, 3] > 0).sum()) if len(box_debug) else 0
    save_inference_metadata(
        out_dir,
        dict(
            sequence=seq_name,
            prompt_mode="box",
            frames=num_frames,
            boxes=count_boxes(out_dir),
            max_people=args.max_people,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            box_fallback=args.box_fallback,
            fallback_frames=fallback_frames,
            mhr=mhr_count,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        ),
    )
    print(
        f"[OK] {seq_name}: {mhr_count}/{num_frames} MHR frames, "
        f"{fallback_frames} frames used box fallback, "
        f"output {os.path.join(out_dir, 'mhr_params')}"
    )


def has_complete_mhr(seq_dir, output_root, voccl3d_root, max_frames=None):
    out_dir = sequence_out_dir(seq_dir, output_root, voccl3d_root)
    try:
        num_frames = len(load_frame_paths(seq_dir, max_frames=max_frames))
    except RuntimeError:
        return False
    return count_mhr_outputs(out_dir) == num_frames


def sequence_frame_count(seq_dir, max_frames=None):
    return len(load_frame_paths(seq_dir, max_frames=max_frames))


def sequence_scene_name(seq_name):
    parts = seq_name.split(os.sep)
    return parts[0] if len(parts) > 1 else "<single_scene>"


def summarize_sequence_names(seq_names):
    summary = {}
    for seq_name in seq_names:
        scene = sequence_scene_name(seq_name)
        summary[scene] = summary.get(scene, 0) + 1
    return dict(sorted(summary.items()))


def print_sequence_summary(label, seq_names):
    scene_counts = summarize_sequence_names(seq_names)
    detail = ", ".join(f"{scene}:{count}" for scene, count in scene_counts.items())
    print(f"[VOCCL3D] {label}: {len(seq_names)} sequences")
    if detail:
        print(f"[VOCCL3D] {label} by scene: {detail}")


def validate_expected_sequences(args, seq_names):
    if args.expected_sequences is None:
        return
    expected = int(args.expected_sequences)
    if len(seq_names) != expected:
        scene_counts = summarize_sequence_names(seq_names)
        raise RuntimeError(
            f"Expected {expected} VOccl3D sequences, but found {len(seq_names)}. "
            f"Scene counts: {scene_counts}. Check --data_root/--sequence before running."
        )


def split_sequences_balanced(seq_dirs, num_shards, max_frames=None):
    shards = [[] for _ in range(num_shards)]
    shard_costs = [0 for _ in range(num_shards)]
    weighted = [(sequence_frame_count(seq_dir, max_frames=max_frames), seq_dir) for seq_dir in seq_dirs]
    for cost, seq_dir in sorted(weighted, reverse=True):
        idx = int(np.argmin(shard_costs))
        shards[idx].append(seq_dir)
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


def normalize_progress_gpu(progress_gpu):
    if progress_gpu is None:
        return "0"
    progress_gpu = str(progress_gpu).strip()
    return progress_gpu if progress_gpu else "0"


def append_optional_arg(cmd, name, value):
    if value is None:
        return
    cmd.extend([name, str(value)])


def build_worker_command(args, seq_names, worker_id):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--voccl3d_root",
        args.voccl3d_root,
        "--output_dir",
        args.output_dir,
        "--sequence",
        ",".join(seq_names),
        "--prompt_mode",
        args.prompt_mode,
        "--mask_mode",
        args.mask_mode,
        "--config_path",
        args.config_path,
        "--max_people",
        str(args.max_people),
        "--bbox_thr",
        str(args.bbox_thr),
        "--nms_thr",
        str(args.nms_thr),
        "--box_fallback",
        args.box_fallback,
        "--preferred_sam_id",
        str(args.preferred_sam_id),
        "--library_log_level",
        args.library_log_level,
        "--worker_id",
        str(worker_id),
    ]
    append_optional_arg(cmd, "--max_frames", args.max_frames)
    append_optional_arg(cmd, "--batch_size", args.batch_size)
    append_optional_arg(cmd, "--ckpt_root", args.ckpt_root)
    append_optional_arg(cmd, "--sam3_ckpt_path", args.sam3_ckpt_path)
    append_optional_arg(cmd, "--sam_3d_body_ckpt_path", args.sam_3d_body_ckpt_path)
    append_optional_arg(cmd, "--mhr_path", args.mhr_path)
    append_optional_arg(cmd, "--fov_path", args.fov_path)
    append_optional_arg(cmd, "--detector_path", args.detector_path)
    if args.overwrite:
        cmd.append("--overwrite")
    if args.save_feature_cache:
        cmd.append("--save_feature_cache")
    if args.reuse_masks:
        cmd.append("--reuse_masks")
    if args.render_box:
        cmd.append("--render_box")
    return cmd


def tail_file(path, num_lines=40):
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[-num_lines:])


def launch_multi_gpu(args):
    gpus = parse_gpus(args.gpus)
    if not gpus:
        print("[MULTI-GPU] No CUDA GPUs detected; falling back to single process.")
        return run_single_process(args)

    seq_dirs = selected_sequence_dirs(args)
    selected_names = [sequence_name(p, args.voccl3d_root) for p in seq_dirs]
    print_sequence_summary("selected", selected_names)
    validate_expected_sequences(args, selected_names)
    if not args.overwrite:
        skipped = [
            sequence_name(p, args.voccl3d_root)
            for p in seq_dirs
            if has_complete_mhr(p, args.output_dir, args.voccl3d_root, max_frames=args.max_frames)
        ]
        seq_dirs = [
            p for p in seq_dirs
            if not has_complete_mhr(p, args.output_dir, args.voccl3d_root, max_frames=args.max_frames)
        ]
        if skipped:
            print_sequence_summary("skipped complete", skipped)
    if not seq_dirs:
        print("[MULTI-GPU] all selected sequences are already complete.")
        return

    to_run_names = [sequence_name(p, args.voccl3d_root) for p in seq_dirs]
    print_sequence_summary("to run", to_run_names)
    num_workers = min(len(gpus), len(seq_dirs))
    shards, shard_costs = split_sequences_balanced(seq_dirs, num_workers, max_frames=args.max_frames)
    log_dir = os.path.join(args.output_dir, ".logs")
    os.makedirs(log_dir, exist_ok=True)
    progress_gpu = normalize_progress_gpu(args.progress_gpu)
    progress_gpu_enabled = progress_gpu.lower() != "none"
    active_gpus = gpus[:num_workers]
    stream_fallback_worker = None
    if progress_gpu_enabled and progress_gpu not in active_gpus:
        stream_fallback_worker = 0
        print(
            f"[MULTI-GPU] progress GPU {progress_gpu} is not in selected GPUs "
            f"({', '.join(active_gpus)}); streaming worker 0 instead."
        )

    procs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "timestamp": timestamp,
        "data_root": args.voccl3d_root,
        "output_dir": args.output_dir,
        "prompt_mode": args.prompt_mode,
        "mask_mode": args.mask_mode,
        "gpus": gpus,
        "selected_sequences": selected_names,
        "skipped_sequences": skipped if not args.overwrite else [],
        "to_run_sequences": to_run_names,
        "shards": [],
    }
    for worker_id, (gpu_id, shard, cost) in enumerate(zip(gpus, shards, shard_costs)):
        if not shard:
            continue
        seq_names = [sequence_name(p, args.voccl3d_root) for p in shard]
        manifest["shards"].append(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "num_sequences": len(seq_names),
                "num_frames": int(cost),
                "sequences": seq_names,
            }
        )
        stream_to_terminal = progress_gpu_enabled and (
            gpu_id == progress_gpu or worker_id == stream_fallback_worker
        )
        log_path = None
        if not stream_to_terminal:
            log_path = os.path.join(
                log_dir,
                f"run_voccl3d_{args.prompt_mode}_gpu{gpu_id}_worker{worker_id}_{timestamp}.log",
            )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        env["LOG_LEVEL"] = args.library_log_level
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = build_worker_command(args, seq_names, worker_id)
        log_f = None if stream_to_terminal else open(log_path, "w")
        output_target = "terminal progress" if stream_to_terminal else f"log {log_path}"
        print(
            f"[MULTI-GPU] worker {worker_id} -> GPU {gpu_id}, "
            f"{len(seq_names)} seqs/{cost} frames, {output_target}"
        )
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((worker_id, gpu_id, log_path, log_f, proc))

    manifest_path = os.path.join(log_dir, f"run_voccl3d_{args.prompt_mode}_{timestamp}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[MULTI-GPU] manifest -> {manifest_path}")

    failures = []
    for worker_id, gpu_id, log_path, log_f, proc in procs:
        ret = proc.wait()
        if log_f is not None:
            log_f.close()
        if ret != 0:
            failures.append((worker_id, gpu_id, ret, log_path))
        print(f"[MULTI-GPU] worker {worker_id} on GPU {gpu_id} exited with {ret}")

    if failures:
        for worker_id, gpu_id, ret, log_path in failures:
            if log_path is None:
                print(
                    f"\n[ERROR] worker {worker_id} GPU {gpu_id} failed with code {ret}; "
                    "output was streamed to terminal."
                )
            else:
                print(f"\n[ERROR] worker {worker_id} GPU {gpu_id} failed with code {ret}; tail of {log_path}:")
                print(tail_file(log_path))
        raise RuntimeError(f"{len(failures)} worker(s) failed.")


def apply_config_overrides(args):
    overrides = [
        args.ckpt_root,
        args.sam3_ckpt_path,
        args.sam_3d_body_ckpt_path,
        args.mhr_path,
        args.fov_path,
        args.detector_path,
        args.batch_size,
    ]
    if not any(value is not None for value in overrides):
        return args.config_path

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
    if args.batch_size is not None:
        cfg.sam_3d_body.batch_size = int(args.batch_size)

    cfg_dir = os.path.join(args.output_dir, ".runtime_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(
        cfg_dir,
        f"body4d_{args.prompt_mode}_worker{args.worker_id}_{os.getpid()}.yaml",
    )
    OmegaConf.save(config=cfg, f=cfg_path, resolve=True)
    return cfg_path


def run_single_process(args):
    seq_dirs = selected_sequence_dirs(args)
    selected_names = [sequence_name(p, args.voccl3d_root) for p in seq_dirs]
    print_sequence_summary("selected", selected_names)
    validate_expected_sequences(args, selected_names)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = apply_config_overrides(args)

    if args.prompt_mode == "mask":
        predictor = MaskOfflineApp(config_path=config_path)
        for seq_dir in tqdm(seq_dirs, desc="Processing VOccl3D mask"):
            infer_sequence_mask(
                predictor,
                seq_dir=seq_dir,
                output_root=args.output_dir,
                args=args,
            )
    elif args.prompt_mode == "box":
        if args.save_feature_cache:
            print("[WARN] --save_feature_cache is ignored in box mode.")
        predictor = BoxOfflineApp(config_path=config_path, load_sam3=False)
        detector = HumanDetector(
            name="vitdet",
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            path=args.detector_path or "",
        )
        for seq_dir in tqdm(seq_dirs, desc="Processing VOccl3D box"):
            infer_sequence_box(
                predictor,
                detector,
                seq_dir=seq_dir,
                output_root=args.output_dir,
                args=args,
            )
    else:
        raise ValueError(f"Unsupported prompt_mode: {args.prompt_mode}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM-Body4D MHR inference on VOccl3D image sequences"
    )
    parser.add_argument("--voccl3d_root", "--data_root", dest="voccl3d_root", default=DEFAULT_VOCCL3D_ROOT)
    parser.add_argument("--output_dir", "--save_root", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence", default="all", help="Comma-separated sequence names, or all")
    parser.add_argument(
        "--prompt_mode",
        choices=("mask", "box"),
        default="mask",
        help="mask: per-frame SAM person mask prompt; box: per-frame person detector boxes only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_feature_cache", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional smoke-test frame cap")
    parser.add_argument("--batch_size", type=int, default=None, help="Override SAM-Body4D MHR batch size")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ckpt_root", default=None, help="Override configs/body4d.yaml paths.ckpt_root")
    parser.add_argument("--sam3_ckpt_path", default=None)
    parser.add_argument(
        "--sam_3d_body_ckpt_path",
        "--body_ckpt_path",
        "--ckpt_path",
        dest="sam_3d_body_ckpt_path",
        default=None,
    )
    parser.add_argument("--mhr_path", "--mhr_model_path", dest="mhr_path", default=None)
    parser.add_argument("--fov_path", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument(
        "--reuse_masks",
        action="store_true",
        help="Mask mode only: skip SAM mask generation when masks already exist for every selected frame.",
    )
    parser.add_argument(
        "--preferred_sam_id",
        type=int,
        default=1,
        help="Mask propagate mode only: prefer this SAM id from the first-frame 'person' prompt.",
    )
    parser.add_argument(
        "--mask_mode",
        choices=("per_frame", "propagate"),
        default="per_frame",
        help="Mask mode: per_frame independently prompts every frame; propagate tracks from first frame.",
    )
    parser.add_argument("--max_people", type=int, default=1, help="Box mode: keep this many largest person boxes per frame.")
    parser.add_argument("--bbox_thr", type=float, default=0.5, help="Box mode: person detector score threshold.")
    parser.add_argument("--nms_thr", type=float, default=0.3, help="Box mode: person detector NMS threshold.")
    parser.add_argument(
        "--box_fallback",
        choices=("previous", "full_image"),
        default="previous",
        help="Box mode: fallback when a frame has fewer detections than --max_people.",
    )
    parser.add_argument("--render_box", action="store_true", help="Box mode: also save rendered_frames with boxes.")
    parser.add_argument("--multi_gpu", action="store_true", help="Launch one worker process per selected GPU.")
    parser.add_argument("--gpus", default="auto", help="auto or comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument(
        "--expected_sequences",
        type=int,
        default=None,
        help="Abort if selected VOccl3D sequence count does not match this value.",
    )
    parser.add_argument(
        "--progress_gpu",
        default="0",
        help="Multi-GPU: stream this GPU worker to terminal for tqdm progress; use none to log every worker.",
    )
    parser.add_argument(
        "--library_log_level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="WARNING",
        help="Suppress noisy library logs by default; WARNING keeps tqdm clean while preserving warnings/errors.",
    )
    parser.add_argument("--worker_id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_library_logging(args.library_log_level)

    if args.multi_gpu:
        launch_multi_gpu(args)
    else:
        run_single_process(args)


if __name__ == "__main__":
    main()
