import argparse
import glob
import os
import sys

import numpy as np
import smplx
import torch
from einops import einsum
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(repo_dir, "scripts"))
sys.path.append(os.path.join(repo_dir, "scripts", "mhr_smpl_conversion"))
sys.path.append(os.path.join(current_dir, "eval_utils"))

from mhr_smpl_conversion.conversion import Conversion
from eval_utils.eval_tools import as_np_array, compute_camcoord_metrics
from eval_utils.smooth import postprocess_smpl_params
from eval_utils.std import suppress_stdout_stderr


DEFAULT_VOCCL3D_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D/scene9_view1"
DEFAULT_PRED_ROOT = "/home/mingqi/data/predictions/hmr/VOccl3D/scene9_view1"
DEFAULT_SEQUENCE = "SMPLX-female_gesture_etc-04-set_4-kawaguchi_stageii"
DEFAULT_BODY_MODEL_PATH = "/home/mingqi/data/checkpoints/hmr/body_models"
DEFAULT_MHR_MODEL_PATH = "/home/mingqi/data/checkpoints/hmr/sam-3d-body-dinov3/assets/mhr_model.pt"
EVAL_BODY_MODEL_DIR = os.path.join(current_dir, "eval_utils", "body_model")


def parse_pelvis_idxs(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_eval_body_model_file(filename):
    path = os.path.join(EVAL_BODY_MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def output_file_name(args):
    if args.output_file:
        return args.output_file
    return "eval_voccl3d_3dpw_metrics.pt"


def load_mhr_frame_data(path, obj_id):
    data_arr = np.load(path, allow_pickle=True)["data"]
    if data_arr.shape == () or data_arr.item() is None or len(data_arr) == 0:
        raise RuntimeError("empty MHR frame")
    if len(data_arr) == 1:
        return data_arr[0]
    return data_arr[int(obj_id)]


class _AttrObject:
    pass


class JitMHRWrapper:
    def __init__(self, model_path, device):
        self.jit_model = torch.jit.load(model_path, map_location=device)
        state = self.jit_model.state_dict()

        mesh = _AttrObject()
        mesh.faces = state["character_torch.mesh.faces"].cpu().numpy().astype(np.int64)
        mesh.vertices = state["character_torch.mesh.rest_vertices"].cpu().numpy()

        character = _AttrObject()
        character.mesh = mesh
        self.character = character

    def to(self, device):
        self.jit_model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        return self.jit_model(*args, **kwargs)


def load_mhr_vertices_3dpw_style(mhr_dir, obj_id=0):
    mhr_files = sorted(glob.glob(os.path.join(mhr_dir, "*_data.npz")))
    if not mhr_files:
        raise FileNotFoundError(f"No MHR data npz found under {mhr_dir}")

    initial_vertices = None
    for path in mhr_files:
        try:
            data = load_mhr_frame_data(path, obj_id)
            initial_vertices = (data["pred_vertices"] + data["pred_cam_t"][None]) * 100.0
            break
        except Exception:
            pass
    if initial_vertices is None:
        raise RuntimeError(f"No valid MHR frames found under {mhr_dir}")

    vertices = []
    frame_names = []
    failures = []
    last_vertices = initial_vertices

    for path in tqdm(mhr_files, desc="Loading MHR"):
        try:
            data = load_mhr_frame_data(path, obj_id)
            last_vertices = (data["pred_vertices"] + data["pred_cam_t"][None]) * 100.0
        except Exception as exc:
            failures.append((os.path.basename(path), str(exc)))
        vertices.append(last_vertices.copy())
        frame_names.append(os.path.basename(path).replace("_data.npz", ".png"))

    return np.stack(vertices, axis=0), np.asarray(frame_names), failures


def convert_mhr_to_smplx_params(args, converter, mhr_vertices):
    with suppress_stdout_stderr():
        conversion_results = converter.convert_mhr2smpl(
            mhr_vertices=mhr_vertices,
            single_identity=args.single_identity,
            is_tracking=args.is_tracking,
            return_smpl_meshes=False,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            return_fitting_errors=True,
            batch_size=args.batch_size,
        )
    smplx_params = conversion_results.result_parameters
    for key in ["left_hand_pose", "right_hand_pose", "expression"]:
        smplx_params.pop(key, None)
    if not args.no_postprocess:
        smplx_params = postprocess_smpl_params(smplx_params)
    return smplx_params, conversion_results.result_errors


def forward_smplx_params_to_smpl_vertices(smplx_params, body_model_path, batch_size, device):
    smplx_model = smplx.SMPLX(
        model_path=f"{body_model_path}/smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=True,
    ).to(device)
    smplx2smpl = torch.load(resolve_eval_body_model_file("smplx2smpl_sparse.pt")).to(device)

    num_frames = next(v.shape[0] for v in smplx_params.values() if torch.is_tensor(v))
    smpl_vertices = []
    smplx_vertices = []
    for start in tqdm(range(0, num_frames, batch_size), desc="SMPL-X forward"):
        end = min(start + batch_size, num_frames)
        batch_params = {
            key: value[start:end].to(device).contiguous()
            for key, value in smplx_params.items()
            if torch.is_tensor(value) and value.shape[0] == num_frames
        }
        current_batch = end - start
        batch_params.setdefault("left_hand_pose", torch.zeros(current_batch, 12, device=device))
        batch_params.setdefault("right_hand_pose", torch.zeros(current_batch, 12, device=device))
        batch_params.setdefault("expression", torch.zeros(current_batch, 10, device=device))
        batch_params.setdefault("jaw_pose", torch.zeros(current_batch, 3, device=device))
        batch_params.setdefault("leye_pose", torch.zeros(current_batch, 3, device=device))
        batch_params.setdefault("reye_pose", torch.zeros(current_batch, 3, device=device))
        out = smplx_model(**batch_params)
        smplx_verts = out.vertices
        smpl_verts = torch.stack([torch.matmul(smplx2smpl, v) for v in smplx_verts])
        smplx_vertices.append(smplx_verts.detach())
        smpl_vertices.append(smpl_verts.detach())
    return torch.cat(smpl_vertices, dim=0), torch.cat(smplx_vertices, dim=0)


def load_voccl3d_gt_smpl(voccl3d_root, gt_file, sequence, frame_names, batch_size, device):
    gt_path = gt_file or os.path.join(
        voccl3d_root,
        f"{os.path.basename(voccl3d_root.rstrip(os.sep))}_ground_truth_labels.npy",
    )
    gt_all = np.load(gt_path, allow_pickle=True).item()
    gt_key = os.path.join("images", sequence)
    if gt_key not in gt_all:
        raise KeyError(f"Sequence {gt_key} not found in GT file {gt_path}")

    gt_seq = gt_all[gt_key]
    missing = [name for name in frame_names if name not in gt_seq]
    if missing:
        raise KeyError(f"GT missing {len(missing)} frames, first: {missing[0]}")

    gt_smplx_verts = np.stack(
        [gt_seq[name]["cam_vertices_raw"] for name in frame_names],
        axis=0,
    )
    gt_smplx_verts = torch.from_numpy(gt_smplx_verts).float().to(device)
    smplx2smpl = torch.load(resolve_eval_body_model_file("smplx2smpl_sparse.pt")).to(device)

    gt_smpl_verts = []
    for start in tqdm(range(0, gt_smplx_verts.shape[0], batch_size), desc="GT SMPL-X -> SMPL"):
        end = min(start + batch_size, gt_smplx_verts.shape[0])
        gt_smpl_verts.append(
            torch.stack([torch.matmul(smplx2smpl, v) for v in gt_smplx_verts[start:end]])
        )
    return torch.cat(gt_smpl_verts, dim=0), gt_path


def compute_voccl3d_metrics(args, pred_smpl_verts, gt_smpl_verts, device):
    j_regressor = torch.load(
        resolve_eval_body_model_file("smpl_3dpw14_J_regressor_sparse.pt")
    ).to_dense().to(device)
    pred_j3d = einsum(j_regressor, pred_smpl_verts, "j v, l v i -> l j i")
    gt_j3d = einsum(j_regressor, gt_smpl_verts, "j v, l v i -> l j i")
    batch_eval = {
        "pred_j3d": pred_j3d,
        "target_j3d": gt_j3d,
        "pred_verts": pred_smpl_verts,
        "target_verts": gt_smpl_verts,
    }
    metrics = compute_camcoord_metrics(
        batch_eval,
        pelvis_idxs=args.pelvis_idxs,
        fps=args.fps,
    )
    return {key: as_np_array(value) for key, value in metrics.items()}, pred_j3d, gt_j3d


def summarize_metric(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float("nan")


def evaluate_sequence(args, sequence):
    if not torch.cuda.is_available():
        raise RuntimeError("eval_voccl3d needs CUDA for MHR -> SMPL conversion.")
    device = torch.device("cuda")
    pred_dir = os.path.join(args.pred_root, sequence)
    mhr_dir = os.path.join(pred_dir, "mhr_params")

    mhr_model = JitMHRWrapper(args.mhr_model_path, device)
    smplx_model = smplx.SMPLX(
        model_path=f"{args.body_model_path}/smplx",
        gender="neutral",
    ).to(device)
    converter = Conversion(mhr_model=mhr_model, smpl_model=smplx_model, method="pytorch")

    mhr_vertices, frame_names, failures = load_mhr_vertices_3dpw_style(
        mhr_dir,
        obj_id=args.obj_id,
    )
    smplx_params, fitting_errors = convert_mhr_to_smplx_params(args, converter, mhr_vertices)
    pred_smpl_verts, pred_smplx_verts = forward_smplx_params_to_smpl_vertices(
        smplx_params,
        args.body_model_path,
        args.batch_size,
        device,
    )
    gt_smpl_verts, gt_path = load_voccl3d_gt_smpl(
        args.voccl3d_root,
        args.gt_file,
        sequence,
        frame_names,
        args.batch_size,
        device,
    )
    metrics, pred_j3d, gt_j3d = compute_voccl3d_metrics(
        args,
        pred_smpl_verts,
        gt_smpl_verts,
        device,
    )

    print("\n --------------- VOccl3D 3DPW-style metrics -------------")
    print(f"Sequence: {sequence}")
    print(f"Frames: {len(frame_names)}")
    print(f"GT: {gt_path}")
    print(f"PA-MPJPE (mm): {summarize_metric(metrics['pa_mpjpe']):.2f}")
    print(f"MPJPE (mm): {summarize_metric(metrics['mpjpe']):.2f}")
    print(f"PVE (mm): {summarize_metric(metrics['pve']):.2f}")
    print(f"ACCEL (m/s^2): {summarize_metric(metrics['accel']):.2f}")
    if failures:
        print(f"[WARN] Reused previous MHR vertices for {len(failures)} empty/failed frames.")

    out_path = os.path.join(pred_dir, output_file_name(args))
    torch.save(
        {
            "sequence": sequence,
            "frame_names": frame_names,
            "pred_smpl_verts_cam": pred_smpl_verts.detach().cpu(),
            "pred_smplx_verts_cam": pred_smplx_verts.detach().cpu(),
            "pred_joints_cam": pred_j3d.detach().cpu(),
            "gt_smpl_verts_cam": gt_smpl_verts.detach().cpu(),
            "gt_joints_cam": gt_j3d.detach().cpu(),
            "smplx_params": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in smplx_params.items()
            },
            "fitting_errors": fitting_errors,
            "mhr_failures": failures,
            "metrics": metrics,
            "pelvis_idxs": args.pelvis_idxs,
            "metric_style": "3dpw_camcoord",
            "postprocess": not args.no_postprocess,
            "single_identity": args.single_identity,
            "is_tracking": args.is_tracking,
        },
        out_path,
    )
    print(f"[OK] saved eval -> {out_path}")
    return metrics


def selected_sequences(args):
    if args.sequence != "all":
        return [x.strip() for x in args.sequence.split(",") if x.strip()]

    gt_file = args.gt_file or os.path.join(
        args.voccl3d_root,
        f"{os.path.basename(args.voccl3d_root.rstrip(os.sep))}_ground_truth_labels.npy",
    )
    gt_all = np.load(gt_file, allow_pickle=True).item()
    return [os.path.basename(key) for key in sorted(gt_all)]


def print_aggregate(all_metrics):
    if len(all_metrics) <= 1:
        return
    print("\n --------------- Aggregate -------------")
    for key, label in [
        ("pa_mpjpe", "PA-MPJPE (mm)"),
        ("mpjpe", "MPJPE (mm)"),
        ("pve", "PVE (mm)"),
        ("accel", "ACCEL (m/s^2)"),
    ]:
        values = np.concatenate([metrics[key] for metrics in all_metrics])
        print(f"{label}: {summarize_metric(values):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM-Body4D MHR outputs on VOccl3D with 3DPW-style metrics"
    )
    parser.add_argument("--pred_root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--voccl3d_root", default=DEFAULT_VOCCL3D_ROOT)
    parser.add_argument("--gt_file", default=None)
    parser.add_argument("--body_model_path", default=DEFAULT_BODY_MODEL_PATH)
    parser.add_argument("--mhr_model_path", default=DEFAULT_MHR_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="Comma-separated sequence names, or all")
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pelvis_idxs", type=parse_pelvis_idxs, default=parse_pelvis_idxs("2,3"))
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--single_identity", action="store_true")
    parser.add_argument("--is_tracking", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    args = parser.parse_args()

    all_metrics = []
    for sequence in selected_sequences(args):
        all_metrics.append(evaluate_sequence(args, sequence))
    print_aggregate(all_metrics)


if __name__ == "__main__":
    main()
