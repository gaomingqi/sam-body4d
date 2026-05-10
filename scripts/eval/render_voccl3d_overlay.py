import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
for path in (REPO_DIR, SCRIPTS_DIR, os.path.join(REPO_DIR, "models", "sam_3d_body")):
    if path not in sys.path:
        sys.path.append(path)


DEFAULT_MHR_MODEL_PATH = (
    "/home/mingqi/data/checkpoints/hmr/sam-3d-body-dinov3/assets/mhr_model.pt"
)


def load_faces(mhr_model_path):
    state = torch.jit.load(mhr_model_path, map_location="cpu").state_dict()
    return state["character_torch.mesh.faces"].cpu().numpy().astype(np.int64)


def load_mhr_outputs(path):
    data = np.load(path, allow_pickle=True)["data"]
    if data.shape == () or data.item() is None or len(data) == 0:
        return None
    return list(data)


def image_paths(image_dir):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Render saved SAM-Body4D MHR predictions as mesh overlays."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mhr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mhr_model_path", default=DEFAULT_MHR_MODEL_PATH)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    from models.sam_3d_body.tools.vis_utils import visualize_sample_together

    os.makedirs(args.output_dir, exist_ok=True)
    faces = load_faces(args.mhr_model_path)
    frames = image_paths(args.image_dir)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    rendered = 0
    missing = 0
    for image_path in tqdm(frames, desc="Rendering mesh overlay"):
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        mhr_path = os.path.join(args.mhr_dir, f"{frame_name}_data.npz")
        if not os.path.exists(mhr_path):
            missing += 1
            continue

        img = cv2.imread(image_path)
        outputs = load_mhr_outputs(mhr_path)
        id_current = list(range(len(outputs))) if outputs is not None else []
        overlay = visualize_sample_together(img, outputs, faces, id_current)
        out_path = os.path.join(args.output_dir, f"{frame_name}_overlay.jpg")
        cv2.imwrite(out_path, overlay.astype(np.uint8))
        rendered += 1

    print(f"[OK] rendered {rendered} frames -> {args.output_dir}")
    if missing:
        print(f"[WARN] missing MHR for {missing} frames")


if __name__ == "__main__":
    main()
