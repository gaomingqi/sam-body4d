import argparse
import os
import re
import sys

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)
sys.path.append(os.path.join(repo_dir, "scripts"))

from eval_voccl3d import (
    DEFAULT_BODY_MODEL_PATH,
    DEFAULT_VOCCL3D_ROOT,
    compute_voccl3d_metrics,
    forward_smplx_params_to_smpl_vertices,
    load_voccl3d_gt_smpl,
    parse_pelvis_idxs,
    summarize_metric,
)


DEFAULT_GT_FILE = "/home/mingqi/data/datasets/hmr/VOccl3D/VOccl3D_ground_truth_labels.npy"
DEFAULT_INPUT_FILE = "eval_voccl3d_3dpw_metrics.pt"
DEFAULT_OUTPUT_FILE = "eval_voccl3d_from_smplx_params.pt"


def resolve_params_file(args):
    if args.params_file:
        return args.params_file
    if not args.pred_root or not args.sequence:
        raise ValueError("Use --params_file, or provide both --pred_root and --sequence.")
    return os.path.join(args.pred_root, args.sequence, args.input_file)


def to_tensor_dict(smplx_params):
    tensor_params = {}
    for key, value in smplx_params.items():
        if torch.is_tensor(value):
            tensor_params[key] = value.detach().cpu().float()
        elif isinstance(value, np.ndarray):
            tensor_params[key] = torch.from_numpy(value).float()
        else:
            tensor_params[key] = value
    return tensor_params


def num_param_frames(smplx_params):
    for value in smplx_params.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("smplx_params does not contain any tensor-like batched value.")


def infer_scene_from_path(path):
    for part in reversed(os.path.abspath(path).split(os.sep)):
        match = re.match(r"(scene\d+_view\d+)", part)
        if match:
            return match.group(1)
    return None


def resolve_sequence(args, record, params_file):
    if args.sequence:
        return args.sequence

    sequence = record.get("sequence")
    if sequence is None:
        raise ValueError("No --sequence provided and source file has no 'sequence' field.")
    sequence = str(sequence)
    if os.sep in sequence:
        return sequence

    scene = infer_scene_from_path(params_file)
    if scene is not None:
        return os.path.join(scene, sequence)
    return sequence


def load_source_record(params_file):
    record = torch.load(params_file, map_location="cpu", weights_only=False)
    if "smplx_params" not in record:
        raise KeyError(f"{params_file} does not contain 'smplx_params'.")
    if "frame_names" not in record:
        raise KeyError(f"{params_file} does not contain 'frame_names'.")
    return record


def resolve_output_path(args, params_file):
    if os.path.isabs(args.output_file):
        return args.output_file
    return os.path.join(os.path.dirname(params_file), args.output_file)


def evaluate_from_smplx_params(args):
    if not torch.cuda.is_available():
        raise RuntimeError("eval_voccl3d_smplx_params needs CUDA for SMPL-X forward.")

    params_file = resolve_params_file(args)
    record = load_source_record(params_file)
    sequence = resolve_sequence(args, record, params_file)
    frame_names = np.asarray(record["frame_names"])
    smplx_params = to_tensor_dict(record["smplx_params"])

    num_frames = num_param_frames(smplx_params)
    if len(frame_names) != num_frames:
        raise ValueError(
            f"frame_names has {len(frame_names)} frames, but smplx_params has {num_frames}."
        )

    device = torch.device("cuda")
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

    print("\n --------------- VOccl3D SMPL-X-param metrics -------------")
    print(f"Source: {params_file}")
    print(f"Sequence: {sequence}")
    print(f"Frames: {len(frame_names)}")
    print(f"GT: {gt_path}")
    print(f"PA-MPJPE (mm): {summarize_metric(metrics['pa_mpjpe']):.2f}")
    print(f"MPJPE (mm): {summarize_metric(metrics['mpjpe']):.2f}")
    print(f"PVE (mm): {summarize_metric(metrics['pve']):.2f}")
    print(f"ACCEL (m/s^2): {summarize_metric(metrics['accel']):.2f}")

    out_path = resolve_output_path(args, params_file)
    save_data = {
        "sequence": sequence,
        "frame_names": frame_names,
        "source_params_file": params_file,
        "smplx_params": {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in smplx_params.items()
        },
        "metrics": metrics,
        "pelvis_idxs": args.pelvis_idxs,
        "metric_style": "3dpw_camcoord_from_smplx_params",
        "body_model_path": args.body_model_path,
        "gt_file": gt_path,
    }
    if not args.metrics_only:
        save_data.update(
            {
                "pred_smpl_verts_cam": pred_smpl_verts.detach().cpu(),
                "pred_smplx_verts_cam": pred_smplx_verts.detach().cpu(),
                "pred_joints_cam": pred_j3d.detach().cpu(),
                "gt_smpl_verts_cam": gt_smpl_verts.detach().cpu(),
                "gt_joints_cam": gt_j3d.detach().cpu(),
            }
        )
    torch.save(save_data, out_path)
    print(f"[OK] saved eval -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VOccl3D using saved SMPL-X params and GT only."
    )
    parser.add_argument("--params_file", default=None, help="Existing .pt containing smplx_params and frame_names.")
    parser.add_argument("--pred_root", default=None, help="Alternative to --params_file.")
    parser.add_argument("--sequence", default=None, help="Sequence name, or scene/sequence when using parent VOccl3D GT.")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--voccl3d_root", default=DEFAULT_VOCCL3D_ROOT)
    parser.add_argument("--gt_file", default=DEFAULT_GT_FILE)
    parser.add_argument("--body_model_path", default=DEFAULT_BODY_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pelvis_idxs", type=parse_pelvis_idxs, default=parse_pelvis_idxs("2,3"))
    parser.add_argument("--metrics_only", action="store_true", help="Do not save pred/GT vertices in output .pt.")
    args = parser.parse_args()
    evaluate_from_smplx_params(args)


if __name__ == "__main__":
    main()
