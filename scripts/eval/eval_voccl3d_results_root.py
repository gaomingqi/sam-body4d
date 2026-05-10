import argparse
import csv
import gc
import glob
import json
import os
from argparse import Namespace

import numpy as np
import torch

from eval_voccl3d import evaluate_sequence, parse_pelvis_idxs, summarize_metric


DEFAULT_DATA_ROOT = "/home/mingqi/data/datasets/hmr/VOccl3D"
DEFAULT_GT_FILE = "/home/mingqi/data/datasets/hmr/VOccl3D/VOccl3D_ground_truth_labels.npy"
DEFAULT_BODY_MODEL_PATH = "/home/mingqi/data/checkpoints/hmr/body_models"
DEFAULT_MHR_MODEL_PATH = "/home/mingqi/data/checkpoints/hmr/sam-3d-body-dinov3/assets/mhr_model.pt"
METRIC_KEYS = ("pa_mpjpe", "mpjpe", "pve", "accel")


def find_sequences(result_root):
    mhr_dirs = sorted(glob.glob(os.path.join(result_root, "*", "*", "mhr_params")))
    sequences = []
    for mhr_dir in mhr_dirs:
        seq_dir = os.path.dirname(mhr_dir)
        sequence = os.path.relpath(seq_dir, result_root)
        sequences.append(sequence)
    return sequences


def gt_key_candidates(data_root, sequence):
    scene_name = os.path.basename(data_root.rstrip(os.sep))
    seq_parts = sequence.split(os.sep)
    candidates = [
        os.path.join("images", sequence),
        sequence,
    ]
    if scene_name.startswith("scene"):
        candidates.append(os.path.join(scene_name, "images", sequence))
    if len(seq_parts) >= 2 and seq_parts[0].startswith("scene"):
        candidates.append(os.path.join(seq_parts[0], "images", *seq_parts[1:]))
    return candidates


def resolve_gt_key(gt_all, data_root, sequence):
    for key in gt_key_candidates(data_root, sequence):
        if key in gt_all:
            return key
    return None


def output_eval_path(result_root, sequence, output_file):
    return os.path.join(result_root, sequence, output_file)


def load_eval_record(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def metric_mean(metrics, key):
    return summarize_metric(metrics[key]) if key in metrics else float("nan")


def make_case_row(result_root, sequence, eval_path, status="ok", error=""):
    row = {
        "status": status,
        "scene": sequence.split(os.sep)[0],
        "sequence": os.path.basename(sequence),
        "case": sequence,
        "frames": "",
        "mhr_failures": "",
        "pa_mpjpe": "",
        "mpjpe": "",
        "pve": "",
        "accel": "",
        "eval_path": eval_path,
        "error": error,
    }
    if status != "ok":
        return row

    record = load_eval_record(eval_path)
    metrics = record["metrics"]
    row["frames"] = int(np.asarray(metrics["pve"]).size)
    row["mhr_failures"] = len(record.get("mhr_failures", []))
    for key in METRIC_KEYS:
        row[key] = metric_mean(metrics, key)
    return row


def write_case_csv(path, rows):
    fieldnames = [
        "status",
        "scene",
        "sequence",
        "case",
        "frames",
        "mhr_failures",
        "pa_mpjpe",
        "mpjpe",
        "pve",
        "accel",
        "eval_path",
        "error",
    ]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def summarize_rows(rows):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    summary = {
        "num_cases": len(rows),
        "num_ok": len(ok_rows),
        "num_no_gt": sum(row["status"] == "no_gt" for row in rows),
        "num_failed": sum(row["status"] == "failed" for row in rows),
        "total_frames": int(sum(int(row["frames"]) for row in ok_rows)),
    }
    for key in METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in ok_rows], dtype=np.float64)
        summary[f"case_avg_{key}"] = summarize_metric(values)

    per_frame = {key: [] for key in METRIC_KEYS}
    for row in ok_rows:
        record = load_eval_record(row["eval_path"])
        for key in METRIC_KEYS:
            per_frame[key].append(np.asarray(record["metrics"][key]))
    for key in METRIC_KEYS:
        values = np.concatenate(per_frame[key], axis=0) if per_frame[key] else np.asarray([])
        summary[f"frame_avg_{key}"] = summarize_metric(values)
    return summary


def write_summary(path, summary):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def cleanup_after_sequence():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_root(args):
    sequences = find_sequences(args.result_root)
    if not sequences:
        raise RuntimeError(f"No mhr_params found under {args.result_root}")

    case_csv = args.case_csv or os.path.join(args.result_root, "eval_case_metrics.csv")
    summary_json = args.summary_json or os.path.join(args.result_root, "eval_summary.json")
    rows = []
    print(f"[GT] loading {args.gt_file}")
    gt_all = np.load(args.gt_file, allow_pickle=True).item()
    print(f"[GT] loaded {len(gt_all)} sequences")

    for index, sequence in enumerate(sequences, start=1):
        eval_path = output_eval_path(args.result_root, sequence, args.output_file)
        print(f"\n[{index}/{len(sequences)}] {sequence}")
        gt_key = resolve_gt_key(gt_all, args.data_root, sequence)
        if gt_key is None:
            row = make_case_row(
                args.result_root,
                sequence,
                eval_path,
                status="no_gt",
                error=f"GT missing; tried {gt_key_candidates(args.data_root, sequence)}",
            )
            print(f"[NO_GT] {sequence}")
        elif os.path.exists(eval_path) and not args.overwrite:
            print(f"[SKIP] existing eval: {eval_path}")
            row = make_case_row(args.result_root, sequence, eval_path)
        else:
            eval_args = Namespace(
                pred_root=args.result_root,
                voccl3d_root=args.data_root,
                gt_file=args.gt_file,
                body_model_path=args.body_model_path,
                mhr_model_path=args.mhr_model_path,
                batch_size=args.batch_size,
                sequence=sequence,
                obj_id=args.obj_id,
                fps=args.fps,
                pelvis_idxs=args.pelvis_idxs,
                output_file=args.output_file,
                single_identity=args.single_identity,
                is_tracking=args.is_tracking,
                no_postprocess=args.no_postprocess,
                gt_all=gt_all,
            )
            try:
                cleanup_after_sequence()
                evaluate_sequence(eval_args, sequence)
                row = make_case_row(args.result_root, sequence, eval_path)
            except Exception as exc:
                row = make_case_row(
                    args.result_root,
                    sequence,
                    eval_path,
                    status="failed",
                    error=repr(exc),
                )
                print(f"[ERROR] {sequence}: {exc}")
            finally:
                cleanup_after_sequence()
        rows.append(row)
        write_case_csv(case_csv, rows)
        cleanup_after_sequence()

    summary = summarize_rows(rows)
    write_summary(summary_json, summary)
    print("\n--------------- Bulk Summary -------------")
    print(f"Result root: {args.result_root}")
    print(f"Cases: {summary['num_ok']}/{summary['num_cases']} ok")
    print(f"Total frames: {summary['total_frames']}")
    for prefix, label in [("case_avg", "case-average"), ("frame_avg", "frame-weighted")]:
        print(
            f"{label}: "
            f"PA-MPJPE {summary[f'{prefix}_pa_mpjpe']:.2f}, "
            f"MPJPE {summary[f'{prefix}_mpjpe']:.2f}, "
            f"PVE {summary[f'{prefix}_pve']:.2f}, "
            f"ACCEL {summary[f'{prefix}_accel']:.2f}"
        )
    print(f"[OK] case csv -> {case_csv}")
    print(f"[OK] summary -> {summary_json}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate a VOccl3D result root.")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gt_file", default=DEFAULT_GT_FILE)
    parser.add_argument("--body_model_path", default=DEFAULT_BODY_MODEL_PATH)
    parser.add_argument("--mhr_model_path", default=DEFAULT_MHR_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pelvis_idxs", type=parse_pelvis_idxs, default=parse_pelvis_idxs("2,3"))
    parser.add_argument("--output_file", default="eval_voccl3d_3dpw_metrics.pt")
    parser.add_argument("--case_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--single_identity", action="store_true")
    parser.add_argument("--is_tracking", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate_root(args)


if __name__ == "__main__":
    main()
