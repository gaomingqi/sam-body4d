import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_DATA_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D"
DEFAULT_MASK_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask"
DEFAULT_COMPLETION_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask_completion_opt"
DEFAULT_OUTPUT_DIR = "/home/mingqi/projects/hmr/visualisation"
DEFAULT_SMPL_MODEL = "/home/mingqi/data/checkpoints/hmr/body_models/smpl/SMPL_NEUTRAL.pkl"
MASK_EVAL = "eval_voccl3d_3dpw_metrics_post.pt"
COMPLETION_EVAL = "eval_voccl3d_3dpw_metrics_completion_opt_post.pt"


def load_eval(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def metric_at(record, key, idx):
    arr = np.asarray(record["metrics"][key])
    if idx >= arr.shape[0]:
        return float("nan")
    return float(arr[idx])


def load_faces(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return np.asarray(data["f"], dtype=np.int32)


def load_camera_intrinsics(data_root, scene):
    path = Path(data_root) / scene / "images" / "camera_parameters.txt"
    lines = path.read_text().splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            values = [float(x.strip()) for x in line.strip("[]").split(",")]
            rows.append(values)
            if len(rows) == 3:
                break
    if len(rows) != 3:
        raise RuntimeError(f"Could not parse camera intrinsics from {path}")
    return np.asarray(rows, dtype=np.float32)


def project_vertices(vertices, K):
    z = np.maximum(vertices[:, 2], 1e-4)
    x = K[0, 0] * vertices[:, 0] / z + K[0, 2]
    y = K[1, 1] * vertices[:, 1] / z + K[1, 2]
    return np.stack([x, y], axis=1), z


def render_overlay(image_bgr, vertices, faces, K, color, alpha=0.55):
    h, w = image_bgr.shape[:2]
    points, z = project_vertices(vertices, K)
    face_pts = points[faces]
    face_z = z[faces].mean(axis=1)
    valid = (
        np.isfinite(face_pts).all(axis=(1, 2))
        & (z[faces] > 1e-4).all(axis=1)
        & (face_pts[:, :, 0].max(axis=1) >= 0)
        & (face_pts[:, :, 0].min(axis=1) < w)
        & (face_pts[:, :, 1].max(axis=1) >= 0)
        & (face_pts[:, :, 1].min(axis=1) < h)
    )
    order = np.argsort(face_z[valid])[::-1]
    valid_faces = face_pts[valid][order]

    mesh = image_bgr.copy()
    for tri in valid_faces:
        tri_i = np.round(tri).astype(np.int32)
        cv2.fillConvexPoly(mesh, tri_i, color)
    out = cv2.addWeighted(mesh, alpha, image_bgr, 1.0 - alpha, 0)

    # A light contour from sampled triangles helps the overlay read as a mesh.
    for tri in valid_faces[::8]:
        tri_i = np.round(tri).astype(np.int32)
        cv2.polylines(out, [tri_i], isClosed=True, color=tuple(int(c * 0.55) for c in color), thickness=1)
    return out


def add_header(image, title, lines, color):
    out = image.copy()
    pad_h = 86
    canvas = np.full((out.shape[0] + pad_h, out.shape[1], 3), 245, dtype=np.uint8)
    canvas[pad_h:] = out
    cv2.rectangle(canvas, (0, 0), (out.shape[1] - 1, pad_h - 1), color, 3)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    y = 56
    for line in lines:
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22
    return canvas


def image_path(data_root, sequence, frame_name):
    return Path(data_root) / sequence.split(os.sep)[0] / "images" / sequence.split(os.sep)[1] / frame_name


def selected_completed_indices(meta_path):
    if not meta_path.exists():
        return set()
    meta = json.load(open(meta_path))
    indices = set()
    for seg in meta.get("processed_segments", []) or []:
        indices.update(int(x) for x in seg.get("completed_global_indices", []) or [])
    return indices


def gather_candidates(args):
    rows = list(csv.DictReader(open(Path(args.completion_root) / args.case_csv_name)))
    candidates = []
    for row in rows:
        if row["source"] != "completion_recomputed":
            continue
        seq = row["case"]
        mask = load_eval(Path(args.mask_root) / seq / MASK_EVAL)
        comp = load_eval(Path(args.completion_root) / seq / COMPLETION_EVAL)
        completed = selected_completed_indices(Path(args.completion_root) / seq / "completion_optimization_meta.json")

        for idx, frame_name in enumerate(comp["frame_names"]):
            d_pve = metric_at(mask, "pve", idx) - metric_at(comp, "pve", idx)
            d_mpjpe = metric_at(mask, "mpjpe", idx) - metric_at(comp, "mpjpe", idx)
            d_pa = metric_at(mask, "pa_mpjpe", idx) - metric_at(comp, "pa_mpjpe", idx)
            if not np.isfinite([d_pve, d_mpjpe, d_pa]).all():
                continue
            score = d_pve + d_mpjpe + 0.5 * d_pa
            if score <= args.min_score:
                continue
            candidates.append(
                {
                    "score": score,
                    "sequence": seq,
                    "idx": idx,
                    "frame_name": str(frame_name),
                    "completed": idx in completed,
                    "d_pve": d_pve,
                    "d_mpjpe": d_mpjpe,
                    "d_pa": d_pa,
                    "mask_pve": metric_at(mask, "pve", idx),
                    "comp_pve": metric_at(comp, "pve", idx),
                    "mask_mpjpe": metric_at(mask, "mpjpe", idx),
                    "comp_mpjpe": metric_at(comp, "mpjpe", idx),
                    "mask_pa": metric_at(mask, "pa_mpjpe", idx),
                    "comp_pa": metric_at(comp, "pa_mpjpe", idx),
                }
            )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[: args.num_frames]


def make_visualization(args, item, faces):
    seq = item["sequence"]
    idx = item["idx"]
    scene = seq.split(os.sep)[0]
    K = load_camera_intrinsics(args.data_root, scene)
    img_path = image_path(args.data_root, seq, item["frame_name"])
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(img_path)

    mask = load_eval(Path(args.mask_root) / seq / MASK_EVAL)
    comp = load_eval(Path(args.completion_root) / seq / COMPLETION_EVAL)

    gt_verts = np.asarray(comp["gt_smpl_verts_cam"][idx], dtype=np.float32)
    ori_verts = np.asarray(mask["pred_smpl_verts_cam"][idx], dtype=np.float32)
    comp_verts = np.asarray(comp["pred_smpl_verts_cam"][idx], dtype=np.float32)

    gt_panel = render_overlay(image, gt_verts, faces, K, color=(80, 190, 80))
    ori_panel = render_overlay(image, ori_verts, faces, K, color=(65, 120, 245))
    comp_panel = render_overlay(image, comp_verts, faces, K, color=(245, 130, 60))

    metric_line_ori = f"PVE {item['mask_pve']:.1f}  MPJPE {item['mask_mpjpe']:.1f}  PA {item['mask_pa']:.1f}"
    metric_line_comp = f"PVE {item['comp_pve']:.1f}  MPJPE {item['comp_mpjpe']:.1f}  PA {item['comp_pa']:.1f}"
    delta_line = f"dPVE {item['d_pve']:+.1f}  dMPJPE {item['d_mpjpe']:+.1f}  dPA {item['d_pa']:+.1f}"
    tag = "completed-frame" if item["completed"] else "postprocess-spillover"

    gt_panel = add_header(gt_panel, "GT", [seq, item["frame_name"]], (80, 190, 80))
    ori_panel = add_header(ori_panel, "Original Mask", [metric_line_ori, tag], (65, 120, 245))
    comp_panel = add_header(comp_panel, "Completion", [metric_line_comp, delta_line], (245, 130, 60))
    combined = np.concatenate([gt_panel, ori_panel, comp_panel], axis=1)

    out_dir = Path(args.output_dir) / "voccl3d_completion_better_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_seq = seq.replace(os.sep, "_").replace("(", "").replace(")", "")
    out_path = out_dir / f"{item['rank']:02d}_{safe_seq}_{Path(item['frame_name']).stem}_score_{item['score']:.1f}.jpg"
    cv2.imwrite(str(out_path), combined)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize frames where VOccl3D mask completion improves over original mask results.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mask_root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--completion_root", default=DEFAULT_COMPLETION_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smpl_model", default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--case_csv_name", default="eval_case_metrics_completion_opt_post.csv")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--min_score", type=float, default=0.0)
    args = parser.parse_args()

    faces = load_faces(args.smpl_model)
    candidates = gather_candidates(args)
    if not candidates:
        raise RuntimeError("No improved completion frames found.")

    outputs = []
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        outputs.append(make_visualization(args, item, faces))
        print(
            f"[{rank:02d}] {item['sequence']} {item['frame_name']} "
            f"score={item['score']:.2f} dPVE={item['d_pve']:.2f} "
            f"dMPJPE={item['d_mpjpe']:.2f} dPA={item['d_pa']:.2f} "
            f"completed={item['completed']} -> {outputs[-1]}",
            flush=True,
        )

    manifest = Path(args.output_dir) / "voccl3d_completion_better_frames" / "selected_frames.json"
    with open(manifest, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"[OK] wrote {len(outputs)} visualizations -> {manifest.parent}")
    print(f"[OK] manifest -> {manifest}")


if __name__ == "__main__":
    main()
