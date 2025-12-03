import os

ROOT = os.path.dirname(__file__)
os.environ["GRADIO_TEMP_DIR"] = os.path.join(ROOT, "gradio_tmp")

import cv2
import gradio as gr
from PIL import Image
import yaml
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F

import random
import glob
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(current_dir, 'models', 'diffusion_vas'))

# import sam3
from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.sam_3d_body.tools.vis_utils import visualize_sample_together
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
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
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


import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 到毫秒
    u = uuid.uuid4().hex[:8]  # 取 8 位即可
    return f"{t}_{u}"


# ===============================
# Global runtime objects
# ===============================
CONFIG = None           # loaded config
sam3_model = None       # your SAM-3 model instance
sam3_image_model = None
image_predictor = None
predictor = None        # sam-3 predictor
inference_state = None  # sam-3 inference_state
RUNTIME = {}            # global dict to store runtime data per video/run


def load_config(config_path: str):
    """Load YAML config into a SimpleNamespace for convenient attribute access."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    return sam3_model, predictor


def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    detector_path = cfg.sam_3d_body['detector_path']
    segmentor_path = cfg.sam_3d_body['segmentor_path']
    fov_path = cfg.sam_3d_body['fov_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_detector import HumanDetector
    # human_detector = HumanDetector(
    #     name='vitdet', device=device, path=detector_path
    # )
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)

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
    model_path_depth = model_path_depth + f"/depth_anything_v2_{depth_encoder}.pth"
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator


from concurrent.futures import ThreadPoolExecutor

def build_all_models_parallel(CONFIG):
    enable = CONFIG.completion.get("enable", False)

    # -------- parallel part: SAM3 / SAM3D / Depth --------
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_sam3 = ex.submit(build_sam3_from_config, CONFIG)
        fut_body = ex.submit(build_sam3_3d_body_config, CONFIG)

        if enable:
            cfg = CONFIG.completion
            depth_encoder = cfg["depth_encoder"]
            depth_ckpt = cfg["model_path_depth"] + f"/depth_anything_v2_{depth_encoder}.pth"
            fut_depth = ex.submit(init_depth_model, depth_ckpt, depth_encoder, device)

    sam3_model, predictor = fut_sam3.result()
    sam3_3d_body_model = fut_body.result()

    if not enable:
        return sam3_model, predictor, sam3_3d_body_model, None, None, None, None, None

    depth_model = fut_depth.result()

    # -------- sequential part: HF pipelines --------
    cfg = CONFIG.completion
    pipeline_mask = init_amodal_segmentation_model(cfg["model_path_mask"], device)
    pipeline_rgb  = init_rgb_model(cfg["model_path_rgb"], device)

    max_occ_len = cfg["max_occ_len"]
    generator = torch.manual_seed(23)

    return (
        sam3_model,
        predictor,
        sam3_3d_body_model,
        pipeline_mask,
        pipeline_rgb,
        depth_model,
        max_occ_len,
        generator,
    )


def init_runtime(config_path: str = os.path.join(ROOT, "configs", "4db.yaml")):
    """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
    global CONFIG, sam3_model, predictor, inference_state, sam3_3d_body_model, RUNTIME, OUTPUT_DIR, pipeline_mask \
        , pipeline_rgb, depth_model, max_occ_len, generator
    CONFIG = load_config(config_path)
    sam3_model, predictor = build_sam3_from_config(CONFIG)
    sam3_3d_body_model = build_sam3_3d_body_config(CONFIG)

    if CONFIG.completion.get('enable', False):
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = build_diffusion_vas_config(CONFIG)
    else:
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = None, None, None, None, None

    # sam3_model, predictor, sam3_3d_body_model, pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = build_all_models_parallel(CONFIG)

    OUTPUT_DIR = os.path.join(CONFIG.runtime['output_dir'], gen_id())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RUNTIME = {}  # clear any old state
    RUNTIME['batch_size'] = CONFIG.sam_3d_body.get('batch_size', 1)

# ===============================
# Paths & supported formats
# ===============================

EXAMPLE_1 = os.path.join(ROOT, "examples", "example1.mp4")
EXAMPLE_2 = os.path.join(ROOT, "examples", "example2.mp4")
EXAMPLE_3 = os.path.join(ROOT, "examples", "example3.mp4")

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


# ===============================
# Video utilities
# ===============================

def read_video_metadata(path: str):
    """Return FPS and total frame count."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    RUNTIME['video_fps'] = fps
    return fps, total


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


def get_thumb(path: str):
    """Return the first frame of a video as thumbnail, or None if missing."""
    if not os.path.exists(path):
        return None
    return read_frame_at(path, 0)


EX1_THUMB = get_thumb(EXAMPLE_1)
EX2_THUMB = get_thumb(EXAMPLE_2)
EX3_THUMB = get_thumb(EXAMPLE_3)


# ===============================
# Prepare video
# ===============================

def prepare_video(path: str):
    """
    Common helper:
    - validate path & extension
    - read fps & total frames
    - read first frame
    - init slider & time text
    """
    if path is None:
        return (
            None,  # video_state
            1.0,   # fps_state
            None,  # current_frame
            gr.update(minimum=0, maximum=0, value=0),  # frame_slider
            "00:00 / 00:00",  # time_text
        )

    if not os.path.exists(path):
        raise gr.Error(f"Video not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise gr.Error(
            f"Unsupported video format {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    fps, total = read_video_metadata(path)
    if fps <= 0 or total <= 0:
        raise gr.Error("Invalid video metadata (fps or frame count).")

    first_frame = read_frame_at(path, 0)
    if first_frame is None:
        raise gr.Error("Failed to read first frame.")

    slider_cfg = gr.update(minimum=0, maximum=total - 1, value=0)

    dur = total / fps
    total_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
    time_text = f"00:00 / {total_text}"

    inference_state = predictor.init_state(video_path=path)
    predictor.clear_all_points_in_video(inference_state)
    RUNTIME['inference_state'] = inference_state
    RUNTIME['clicks'] = {}
    RUNTIME['id'] = 1
    RUNTIME['objects'] = {} # points
    RUNTIME['masks'] = {}   # masks
    RUNTIME['out_obj_ids'] = []

    return path, fps, first_frame, slider_cfg, time_text


# ===============================
# Handlers
# ===============================

def on_upload(file_obj):
    """Handle uploaded video file."""
    if file_obj is None:
        return prepare_video(None)
    return prepare_video(file_obj.name)


def on_example_select(evt: gr.SelectData):
    """
    Handle click in the examples gallery.
    evt.index is the clicked item index (0, 1, 2, ...).
    """
    idx = evt.index
    if isinstance(idx, (list, tuple)):  # gallery 有时给 (row, col)
        idx = idx[0]

    if idx == 0:
        path = EXAMPLE_1
    elif idx == 1:
        path = EXAMPLE_2
    elif idx == 2:
        path = EXAMPLE_3
    else:
        raise gr.Error("Unknown example index.")

    return prepare_video(path)


def update_frame(idx, path, fps):
    """Update current frame + time text when slider moves."""
    if path is None:
        return None, "00:00 / 00:00"

    idx = int(idx)
    frame = read_frame_at(path, idx)
    if frame is None:
        raise gr.Error(f"Failed to read frame {idx}.")

    cur_sec = idx / fps if fps > 0 else 0.0
    cur_text = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = total / fps if fps > 0 else 0.0
    end_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"

    return frame, f"{cur_text} / {end_text}"


def on_click(evt: gr.SelectData, point_type: str, video_path: str, frame_idx: int):
    """
    Handle click on the current frame:
    1. Read the corresponding frame from video (by frame_idx).
    2. Compute width/height (for sanity / later use).
    3. Draw a + / - marker at the clicked position.
    4. Return updated image.
    """
    if video_path is None:
        # No video loaded, nothing to do
        return None

    # 1) Read the frame corresponding to current slider value
    frame = read_frame_at(video_path, int(frame_idx))
    if frame is None:
        raise gr.Error(f"Failed to read frame {frame_idx} from video.")

    # 2) Get image size
    width, height = frame.size
    # (If you want to log it)
    print(f"[FRAME] size = {width} x {height}")

    # 3) Get click coordinates and point type
    x, y = evt.index  # evt.index = (x, y) in pixel coords
    print(f"[CLICK] ({x}, {y}), type={point_type}")

    # 4) Draw marker and return updated image
    point_type_norm = point_type.lower()
    if point_type_norm not in ("positive", "negative"):
        point_type_norm = "positive"

    try:
        clicks = RUNTIME['clicks'][frame_idx]
        clicks.append((x, y, point_type))
    except:
        clicks = [(x, y, point_type)]

    pts = []
    lbs = []
    for (x, y, t) in clicks:
        pts.append([int(x), int(y)])
        lbs.append(1 if t.lower() == "positive" else 0)
    input_point = np.array(pts, dtype=np.int32)
    input_label = np.array(lbs, dtype=np.int32)
    
    try:
        RUNTIME['clicks'][frame_idx] = clicks
    except:
        RUNTIME['clicks'][frame_idx] = {}
        RUNTIME['clicks'][frame_idx] = clicks

    prompts = {}
    prompts[RUNTIME['id']] = input_point, input_label

    rel_points = [[x / width, y / height] for x, y in input_point]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    points_labels_tensor = torch.tensor(input_label, dtype=torch.int32)

    _, RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.add_new_points_or_box(
        inference_state=RUNTIME['inference_state'],
        frame_idx=frame_idx,
        obj_id=RUNTIME['id'],
        points=points_tensor,
        labels=points_labels_tensor,
    )
    mask_np = (video_res_masks[-1, 0].detach().cpu().numpy() > 0)
    mask = (mask_np > 0).astype(np.uint8) * 255
    
    painted_image = mask_painter(np.array(frame, dtype=np.uint8), mask, mask_color=4+RUNTIME['id'])
    RUNTIME['masks'][RUNTIME['id']] = {frame_idx: mask}

    for k, v in RUNTIME['masks'].items():
        if k == RUNTIME['id']:
            continue
        if frame_idx in v:
            mask = v[frame_idx]
            painted_image = mask_painter(painted_image, mask, mask_color=4+k)

    frame = Image.fromarray(painted_image)

    updated = draw_point_marker(frame, x, y, point_type_norm)
    return updated


def add_target(targets, selected):
    """Add new target and select it by default."""
    name = f"Target {len(targets) + 1}"
    
    if RUNTIME['clicks'] == {}:
        return targets, selected, gr.update(choices=targets, value=selected)

    targets = targets + [name]
    selected = selected + [name]

    RUNTIME['objects'][RUNTIME['id']] = RUNTIME['clicks']
    RUNTIME['id'] += 1
    RUNTIME['clicks'] = {}

    return targets, selected, gr.update(choices=targets, value=selected)


def toggle_upload(open_state: bool):
    """Toggle upload panel visibility and button label."""
    new_state = not open_state
    label = (
        "Upload Video (click to close)"
        if new_state
        else "Upload Video (click to open)"
    )
    return new_state, gr.update(visible=new_state), gr.update(value=label)


def on_mask_generation(video_path: str):
    """
    Mask generation across the video.
    Currently runs SAM-3 propagation and renders a mask video.
    """
    print("[DEBUG] Mask Generation button clicked.")
    if video_path is None:
        raise gr.Error("No video loaded.")

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in predictor.propagate_in_video(
        RUNTIME['inference_state'],
        start_frame_idx=0,
        max_frame_num_to_track=1800,
        reverse=False,
        propagate_preflight=True,
    ):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(RUNTIME['out_obj_ids'])
        } 

    # render the segmentation results every few frames
    vis_frame_stride = 1
    out_h = RUNTIME['inference_state']['video_height']
    out_w = RUNTIME['inference_state']['video_width']
    img_to_video = []

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images') # for sam3-3d-body
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')  # for sam3-3d-body
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)

    for out_frame_idx in range(0, len(video_segments), vis_frame_stride):
        img = RUNTIME['inference_state']['images'][out_frame_idx].detach().float().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img).convert('RGB')
        msk = np.zeros_like(img[:, :, 0])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            img = mask_painter(img, mask, mask_color=4 + out_obj_id)
            msk[mask == 255] = out_obj_id
        img_to_video.append(img)

        msk_pil = Image.fromarray(msk).convert('P')
        msk_pil.putpalette(DAVIS_PALETTE)
        img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx:08d}.jpg"))
        msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx:08d}.png"))

    out_video_path = os.path.join(OUTPUT_DIR, 'video_mask.mp4')
    images_to_mp4(img_to_video, out_video_path, fps=RUNTIME['video_fps'])

    return out_video_path


def on_4d_generation(video_path: str):
    """
    Placeholder for 4D generation.
    Later:
      - run sam3_3d_body_model on per-frame images + masks
      - render 4D visualization video
    For now, just log and return None.
    """
    print("[DEBUG] 4D Generation button clicked.")
    # TODO: implement 4D body generation, write to a video and return its path

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images') # for sam3-3d-body
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')  # for sam3-3d-body
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob.glob(os.path.join(IMAGE_PATH, ext))
        ]
    )
    masks_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob.glob(os.path.join(MASKS_PATH, ext))
        ]
    )

    os.makedirs(f"{OUTPUT_DIR}/mask_4d", exist_ok=True)
    batch_size = RUNTIME['batch_size']
    n = len(images_list)
    
    # Optional, detect occlusions
    pred_res = [256, 512]
    pred_res_hi = [512, 1024]
    modal_pixels_list = []
    if pipeline_mask is not None:
        for obj_id in RUNTIME['out_obj_ids']:
            modal_pixels, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
            modal_pixels_list.append(modal_pixels)
        rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res)
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    for i in tqdm(range(0, n, batch_size)):
        batch_images = images_list[i:i + batch_size]
        batch_masks  = masks_list[i:i + batch_size]

        W, H = Image.open(batch_masks[0]).size

        # Optional, detect occlusions
        idx_dict = {}
        idx_path = {}
        if len(modal_pixels_list) > 0:
            print("detect occlusions ...")
            # rgb_pixels_batch = rgb_pixels[:, i:i + batch_size, :, :, :]
            # raw_rgb_pixels_batch = raw_rgb_pixels[i:i + batch_size, :, :, :]
            # modal_pixels_batch = torch.cat([mp[:, i:i + batch_size, :, :, :] for mp in modal_pixels_list], dim=0)
            pred_amodal_masks_dict = {}
            for (modal_pixels, obj_id) in zip(modal_pixels_list, RUNTIME['out_obj_ids']):
                # predict amodal masks (amodal segmentation)
                pred_amodal_masks = pipeline_mask(
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
                    generator=generator,
                ).frames[0]

                # for completion
                pred_amodal_masks_com = [np.array(img.resize((pred_res_hi[1], pred_res_hi[0]))) for img in pred_amodal_masks]
                pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
                pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
                pred_amodal_masks_dict[obj_id] = pred_amodal_masks_com
                # for iou
                pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
                pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
                pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                # compute iou
                masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
                ious = [
                    (
                        1.0 if ((a > 0).sum() == 0 and (b > 0).sum() == 0)
                        else np.logical_and(a > 0, b > 0)[(a > 0) | (b > 0)].sum() /
                            (np.logical_or(a > 0, b > 0)[(a > 0) | (b > 0)].sum() + 1e-6)
                    )
                    for a, b in zip(masks, pred_amodal_masks)
                ]
                
                # confirm occlusions & save masks (for HMR)
                start, end = (idxs := [ix for ix,x in enumerate(ious) if x < 0.7]) and (idxs[0], idxs[-1]) or (None, None)
                if start is not None and end is not None:
                    start = max(0, start-2)
                    end = min(modal_pixels[:, i:i + batch_size, :, :, :].shape[1]-1, end+2)
                    idx_dict[obj_id] = (start, end)
                    completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                    completion_image_path = f'{OUTPUT_DIR}/completion/{completion_path}/images'
                    completion_masks_path = f'{OUTPUT_DIR}/completion/{completion_path}/masks'
                    os.makedirs(completion_image_path, exist_ok=True)
                    os.makedirs(completion_masks_path, exist_ok=True)
                    idx_path[obj_id] = {'images': completion_image_path, 'masks': completion_masks_path}
                    # save completion masks
                    for idx_ in range(start, end):
                        mask_idx_ = pred_amodal_masks[idx_].copy()
                        mask_idx_[mask_idx_ > 0] = obj_id
                        mask_idx_ = Image.fromarray(mask_idx_).convert('P')
                        mask_idx_.putpalette(DAVIS_PALETTE)
                        mask_idx_.save(os.path.join(completion_masks_path, f"{idx_:08d}.png"))

            # completion
            for obj_id, (start, end) in idx_dict.items(): 
                completion_image_path = idx_path[obj_id]['images']
                # prepare inputs
                modal_pixels_current, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res_hi, obj_id=obj_id)
                rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res_hi)
                modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
                modal_pixels_current = modal_pixels_current[:, start:end]
                pred_amodal_masks_current = pred_amodal_masks_dict[obj_id][start:end]
                modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
                pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
                    2).repeat(1, 1, 3, 1, 1)
                
                rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                modal_obj_mask = (modal_pixels_current > 0).float()
                modal_background = 1 - modal_obj_mask
                rgb_pixels_current = (rgb_pixels_current + 1) / 2
                modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                print("content completion by diffusion-vas ...")
                # predict amodal rgb (content completion)
                pred_amodal_rgb = pipeline_rgb(
                    modal_rgb_pixels,
                    pred_amodal_masks_tensor,
                    height=pred_res_hi[0], # my_res[0]
                    width=pred_res_hi[1],  # my_res[1]
                    num_frames=end-start,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=8,
                    noise_aug_strength=0.02,
                    min_guidance_scale=1.5,
                    max_guidance_scale=1.5,
                    generator=generator,
                ).frames[0]

                pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

                # save pred_amodal_rgb
                pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
                pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                                for frame in pred_amodal_rgb])
                idx_ = start
                for img in pred_amodal_rgb_save:
                    cv2.imwrite(os.path.join(completion_image_path, f"{idx_:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    idx_ += 1

        # Process with external mask
        mask_outputs, id_batch = process_image_with_mask(sam3_3d_body_model, batch_images, batch_masks, idx_path, idx_dict)
        
        for image_path, mask_output, id_current in zip(batch_images, mask_outputs, id_batch):
            img = cv2.imread(image_path)
            rend_img = visualize_sample_together(img, mask_output, sam3_3d_body_model.faces, id_current)
            cv2.imwrite(
                f"{OUTPUT_DIR}/mask_4d/{os.path.basename(image_path)[:-4]}.jpg",
                rend_img.astype(np.uint8),
            )

    jpg_folder_to_mp4(f"{OUTPUT_DIR}/mask_4d", f"{OUTPUT_DIR}/4d.mp4", fps=RUNTIME['video_fps'])

    return f"{OUTPUT_DIR}/4d.mp4"


# ===============================
# UI Layout
# ===============================

with gr.Blocks(title="SAM3-4D-Body") as demo:
    # States
    video_state = gr.State(None)
    fps_state = gr.State(1.0)
    point_type_state = gr.State("positive")   # "positive" or "negative"
    targets_state = gr.State([])
    selected_targets_state = gr.State([])
    upload_open_state = gr.State(False)

    with gr.Row():
        # -------- Left column: examples + frame + controls --------
        with gr.Column(scale=1):
            gr.Markdown("### Example Videos")

            examples_gallery = gr.Gallery(
                value=[
                    (EX1_THUMB, "Example 1"),
                    (EX2_THUMB, "Example 2"),
                    (EX3_THUMB, "Example 3"),
                ],
                show_label=False,
                columns=3,
                height=160,
            )

            current_frame = gr.Image(
                label="Current Frame (click to annotate)",
                interactive=True,
                sources=[],
            )

            toggle_upload_btn = gr.Button(
                "Upload Video (click to open)",
                size="sm",
                variant="secondary",
            )

            upload_panel = gr.Row(visible=False)
            with upload_panel:
                upload = gr.File(
                    label="Video File",
                    file_count="single",
                )

            frame_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Frame Index",
            )

            time_text = gr.Text("00:00 / 00:00", label="Time")

            point_radio = gr.Radio(
                choices=["Positive", "Negative"],
                value="Positive",
                label="Point Type",
                interactive=True,
            )

            targets_box = gr.CheckboxGroup(
                label="Targets",
                choices=[],
                value=[],
            )
            add_target_btn = gr.Button("Add Target")

        # -------- Right column: result image + buttons + 4D video --------
        with gr.Column(scale=1):
            result_display = gr.Video(label="Segmentation Result")
            with gr.Row():
                mask_gen_btn = gr.Button("Mask Generation")
                gen4d_btn = gr.Button("4D Generation")
            fourd_display = gr.Video(label="4D Result")

    # ===============================
    # Event bindings
    # ===============================

    # Toggle upload panel
    toggle_upload_btn.click(
        fn=toggle_upload,
        inputs=[upload_open_state],
        outputs=[upload_open_state, upload_panel, toggle_upload_btn],
    )

    # Upload → load video
    upload.change(
        fn=on_upload,
        inputs=[upload],
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text],
    )

    # Click example thumbnail in gallery → load that example
    examples_gallery.select(
        fn=on_example_select,
        inputs=None,
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text],
    )

    # Slider → update frame + time
    frame_slider.change(
        fn=update_frame,
        inputs=[frame_slider, video_state, fps_state],
        outputs=[current_frame, time_text],
    )

    point_radio.change(
        fn=lambda v: v.lower(),   # "Positive" / "Negative" → "positive" / "negative"
        inputs=[point_radio],
        outputs=[point_type_state],
    )

    # Click on current frame
    current_frame.select(
        fn=on_click,
        inputs=[point_type_state, video_state, frame_slider],
        outputs=[current_frame],
    )

    # Add target
    add_target_btn.click(
        fn=add_target,
        inputs=[targets_state, selected_targets_state],
        outputs=[targets_state, selected_targets_state, targets_box],
    )

    mask_gen_btn.click(
        fn=on_mask_generation,
        inputs=[video_state], 
        outputs=[result_display],
    )

    gen4d_btn.click(
        fn=on_4d_generation,
        inputs=[video_state],      
        outputs=[fourd_display],  
    )


if __name__ == "__main__":

    init_runtime()
    demo.launch()
