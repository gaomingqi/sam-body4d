import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import smplx


DEFAULT_VOCCL3D_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D/scene9_view1"
DEFAULT_AMASS_ROOT = "/home/mingqi/data/datasets/hmr/aamas"
DEFAULT_PAIR_FILE = (
    "/home/mingqi/projects/hmr/VOccl3D-dataset/prepare_and_download_dataset/"
    "voccl3d_dataset_release_amass_file_pairs.npy"
)
DEFAULT_BODY_MODEL_PATH = "/home/mingqi/data/checkpoints/hmr/body_models"
DEFAULT_SMPLX2SMPL = (
    "/home/mingqi/projects/hmr/sam-body4d/scripts/eval/eval_utils/body_model/"
    "smplx2smpl_sparse.pt"
)
DEFAULT_J_REGRESSOR = (
    "/home/mingqi/projects/hmr/sam-body4d/scripts/eval/eval_utils/body_model/"
    "smpl_3dpw14_J_regressor_sparse.pt"
)


def parse_matrix(lines, start_index, rows):
    matrix = []
    for i in range(rows):
        row = lines[start_index + i].strip("[]\n").split(",")
        matrix.append([float(x) for x in row])
    return np.asarray(matrix, dtype=np.float32)


def load_camera_params(camera_txt_path):
    with open(camera_txt_path) as f:
        lines = f.readlines()
    intrinsic_start = lines.index("Camera Intrinsic Matrix:\n") + 1
    extrinsic_start = lines.index("Camera Extrinsic Matrix:\n") + 1
    return (
        parse_matrix(lines, intrinsic_start, rows=3),
        parse_matrix(lines, extrinsic_start, rows=4),
    )


def get_project2d(joints_3d, translation, camera_intrinsic):
    joints_3d = joints_3d + translation[:, None, :]
    projected = joints_3d / joints_3d[..., -1:]
    projected = np.einsum("ij,bkj->bki", camera_intrinsic, projected)
    return projected.astype(np.float32)


def load_joint_regressor(path, device):
    if path.endswith(".pt") or path.endswith(".pth"):
        regressor = torch.load(path, map_location=device)
        if regressor.is_sparse:
            regressor = regressor.to_dense()
        return regressor.float().to(device)
    return torch.from_numpy(np.load(path)).float().to(device)


def get_bbox_valid(joints, img_height, img_width, rescale):
    valid = []
    for j in joints:
        if 0 <= j[0] <= img_width and 0 <= j[1] <= img_height:
            valid.append(j)

    if len(valid) < 1:
        return [-1, -1], -1, len(valid), [-1, -1, -1, -1]

    valid = np.asarray(valid)
    bbox = [
        float(valid[:, 0].min()),
        float(valid[:, 1].min()),
        float(valid[:, 0].max()),
        float(valid[:, 1].max()),
    ]
    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    return center, float(scale * rescale), len(valid), bbox


def infer_gender(sequence_name):
    match = re.search(r"(male|female|neutral)", sequence_name)
    if not match:
        raise ValueError(f"Could not infer gender from sequence name: {sequence_name}")
    return match.group(0)


def select_sequences(pair_file, voccl3d_root, amass_root, requested):
    pairs = np.load(pair_file, allow_pickle=True).item()
    voccl3d_root = Path(voccl3d_root)
    amass_root = Path(amass_root)
    requested_set = None if requested in (None, "", "all") else {
        x.strip() for x in requested.split(",") if x.strip()
    }

    selected = []
    skipped = []
    for pair_key, amass_rel in sorted(pairs.items()):
        if voccl3d_root.name not in pair_key:
            continue

        seq_name = Path(pair_key).stem
        if requested_set is not None and seq_name not in requested_set:
            continue

        seq_dir = voccl3d_root / "images" / seq_name
        trans_path = voccl3d_root / "transformation_files" / f"{seq_name}.npy"
        amass_path = amass_root / Path(*Path(amass_rel).parts[1:])

        if not seq_dir.is_dir() or not trans_path.exists() or not amass_path.exists():
            skipped.append((seq_name, seq_dir, trans_path, amass_path))
            continue
        selected.append((seq_name, seq_dir, trans_path, amass_path))

    return selected, skipped


def build_models(body_model_path, device):
    models = {}
    for gender in ("male", "female", "neutral"):
        models[gender] = smplx.create(
            body_model_path,
            model_type="smplx",
            gender=gender,
            ext="npz",
            num_betas=11,
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        models[gender].eval()
    return models


def process_sequence(
    seq_name,
    seq_dir,
    trans_path,
    amass_path,
    smplx_model,
    smplx2smpl,
    j_regressor,
    camera_intrinsic,
    camera_extrinsic,
    batch_size,
    bbox_img_size,
    device,
):
    transform_anim = np.load(trans_path, allow_pickle=True).item()
    first_index = next(iter(transform_anim))
    frame_names = sorted([p.name for p in seq_dir.glob("*.png")])

    orig_amass = np.load(amass_path, allow_pickle=True)
    amass_poses = orig_amass["poses"].astype(np.float32)
    betas = orig_amass["betas"].astype(np.float32)

    gender = infer_gender(seq_name)
    gt_single = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(frame_names), batch_size), desc=seq_name, leave=False):
            names = frame_names[start:start + batch_size]
            pose_list = []
            global_orient_list = []
            centroid_list = []
            corrected_pose_list = []

            for offset, img_name in enumerate(names, start=start):
                frame_key = first_index + offset
                transform_data = transform_anim[frame_key]
                global_orient = np.asarray(transform_data["global_orient_vert"], dtype=np.float32).reshape(3)
                centroid = np.asarray(transform_data["camera_verts_centroid"], dtype=np.float32).reshape(3)
                pose = amass_poses[(frame_key - 1) * 4].copy()
                corrected_pose = pose.copy()
                corrected_pose[:3] = global_orient

                pose_list.append(pose)
                corrected_pose_list.append(corrected_pose)
                global_orient_list.append(global_orient)
                centroid_list.append(centroid)

            pose_arr = np.stack(pose_list, axis=0).astype(np.float32)
            global_orient_arr = np.stack(global_orient_list, axis=0).astype(np.float32)
            centroid_arr = np.stack(centroid_list, axis=0).astype(np.float32)

            out = smplx_model(
                betas=torch.from_numpy(betas[:11]).to(device).float().unsqueeze(0).repeat(len(names), 1),
                global_orient=torch.from_numpy(global_orient_arr).to(device).float(),
                body_pose=torch.from_numpy(pose_arr[:, 3:66]).to(device).float(),
                left_hand_pose=torch.zeros((len(names), 45), device=device),
                right_hand_pose=torch.zeros((len(names), 45), device=device),
                jaw_pose=torch.zeros((len(names), 3), device=device),
                leye_pose=torch.zeros((len(names), 3), device=device),
                reye_pose=torch.zeros((len(names), 3), device=device),
                expression=torch.zeros((len(names), 10), device=device),
            )
            vertices = out.vertices.detach()
            joints = out.joints.detach()

            mean_vertices = vertices.mean(dim=1)
            cam_translation = torch.from_numpy(centroid_arr).to(device).float() - mean_vertices
            cam_vertices = vertices + cam_translation[:, None, :]
            cam_joints = joints + cam_translation[:, None, :]
            smpl_vertices = torch.matmul(smplx2smpl, cam_vertices)
            joints3d = torch.matmul(j_regressor, smpl_vertices)

            cam_vertices_np = cam_vertices.cpu().numpy().astype(np.float32)
            cam_joints_np = cam_joints.cpu().numpy().astype(np.float32)
            cam_translation_np = cam_translation.cpu().numpy().astype(np.float32)
            joints3d_np = joints3d.cpu().numpy().astype(np.float32)
            gtkps_batch = get_project2d(
                cam_joints_np,
                np.zeros_like(cam_translation_np),
                camera_intrinsic,
            )

            for i, img_name in enumerate(names):
                center, scale, _, bbox = get_bbox_valid(
                    gtkps_batch[i, :22],
                    img_height=bbox_img_size,
                    img_width=bbox_img_size,
                    rescale=1.2,
                )
                gt_single[img_name] = {
                    "center": center,
                    "scale": scale,
                    "global_orient_vert": global_orient_arr[i:i + 1],
                    "pose_cam": corrected_pose_list[i].astype(np.float32),
                    "shape": betas[:11].astype(np.float32),
                    "cam_translation": cam_translation_np[i:i + 1],
                    "gender": gender,
                    "gtkps": gtkps_batch[i],
                    "camera_extrinsic": camera_extrinsic,
                    "camera_intrinsic": camera_intrinsic,
                    "bbox": bbox,
                    "cam_vertices_raw": cam_vertices_np[i],
                    "joints3D": joints3d_np[i:i + 1],
                }

    return gt_single


def main():
    parser = argparse.ArgumentParser(description="Prepare VOccl3D GT labels, skipping missing AMASS files")
    parser.add_argument("--voccl3d_root", default=DEFAULT_VOCCL3D_ROOT)
    parser.add_argument("--amass_root", default=DEFAULT_AMASS_ROOT)
    parser.add_argument("--pair_file", default=DEFAULT_PAIR_FILE)
    parser.add_argument("--body_model_path", default=DEFAULT_BODY_MODEL_PATH)
    parser.add_argument("--smplx2smpl", default=DEFAULT_SMPLX2SMPL)
    parser.add_argument("--j_regressor", default=DEFAULT_J_REGRESSOR)
    parser.add_argument("--sequence", default="all", help="Comma-separated sequence names, or all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bbox_img_size", type=int, default=720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    voccl3d_root = Path(args.voccl3d_root)
    output = Path(args.output) if args.output else voccl3d_root / f"{voccl3d_root.name}_ground_truth_labels.npy"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {output}")

    selected, skipped = select_sequences(args.pair_file, voccl3d_root, args.amass_root, args.sequence)
    if not selected:
        raise RuntimeError("No sequences selected with available image, transform, and AMASS files.")

    camera_intrinsic, camera_extrinsic = load_camera_params(voccl3d_root / "images" / "camera_parameters.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_models(args.body_model_path, device)
    smplx2smpl = torch.load(args.smplx2smpl, map_location=device).to(device).to_dense()
    j_regressor = load_joint_regressor(args.j_regressor, device)

    save_dict = {}
    for seq_name, seq_dir, trans_path, amass_path in tqdm(selected, desc="Preparing VOccl3D GT"):
        gender = infer_gender(seq_name)
        gt_single = process_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            trans_path=trans_path,
            amass_path=amass_path,
            smplx_model=models[gender],
            smplx2smpl=smplx2smpl,
            j_regressor=j_regressor,
            camera_intrinsic=camera_intrinsic,
            camera_extrinsic=camera_extrinsic,
            batch_size=args.batch_size,
            bbox_img_size=args.bbox_img_size,
            device=device,
        )
        save_dict[os.path.join("images", seq_name)] = gt_single

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, save_dict)
    print(f"[OK] saved {len(save_dict)} sequences -> {output}")
    if skipped:
        print(f"[SKIP] {len(skipped)} sequences missing required files:")
        for seq_name, _, _, amass_path in skipped:
            print(f"  {seq_name}: missing AMASS {amass_path}")


if __name__ == "__main__":
    main()
