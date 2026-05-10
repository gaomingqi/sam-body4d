import argparse
import csv
import gc
import glob
import json
import os
import shutil
from argparse import Namespace

import numpy as np
import torch

from eval_voccl3d import evaluate_sequence, parse_pelvis_idxs, summarize_metric
from eval_voccl3d_results_root import (
    DEFAULT_BODY_MODEL_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_GT_FILE,
    DEFAULT_MHR_MODEL_PATH,
    cleanup_after_sequence,
    resolve_gt_key,
)


METRIC_KEYS = ("pa_mpjpe", "mpjpe", "pve", "accel")
DEFAULT_MASK_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask"
DEFAULT_COMPLETION_ROOT = "/home/mingqi/data/results/hmr/VOccl3D_mask_completion_opt"
DEFAULT_SOURCE_EVAL_FILE = "eval_voccl3d_3dpw_metrics_post.pt"
DEFAULT_OUTPUT_FILE = "eval_voccl3d_3dpw_metrics_completion_opt_post.pt"


def find_sequences(root):
    return [
        os.path.relpath(os.path.dirname(path), root)
        for path in sorted(glob.glob(os.path.join(root, "*", "*", "mhr_params")))
    ]


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_eval(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_eval(path, record):
    tmp_path = path + ".tmp"
    torch.save(record, tmp_path)
    os.replace(tmp_path, path)


def annotate_eval(path, **extra):
    record = load_eval(path)
    record.update(extra)
    save_eval(path, record)


def copy_reused_eval(source_path, output_path, sequence, meta, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    record = load_eval(source_path)
    record.update(
        {
            "sequence": sequence,
            "combined_source": "mask_reused_no_completion_segments",
            "source_eval_path": source_path,
            "completion_processed_segments": [],
            "completion_selected_segments": meta.get("selected_segments", []) if meta else [],
            "completion_meta_path": meta.get("_path") if meta else None,
        }
    )
    save_eval(output_path, record)


def metric_mean(record, key):
    return summarize_metric(record["metrics"][key])


def row_from_eval(sequence, eval_path, source, meta, status="ok", error=""):
    row = {
        "status": status,
        "source": source,
        "scene": sequence.split(os.sep)[0],
        "sequence": os.path.basename(sequence),
        "case": sequence,
        "frames": "",
        "processed_segments": "",
        "completed_frames": "",
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

    record = load_eval(eval_path)
    processed = meta.get("processed_segments", []) if meta else []
    row["frames"] = int(np.asarray(record["metrics"]["pve"]).size)
    row["processed_segments"] = len(processed)
    row["completed_frames"] = sum(int(seg.get("num_completed", 0)) for seg in processed)
    row["mhr_failures"] = len(record.get("mhr_failures", []))
    for key in METRIC_KEYS:
        row[key] = metric_mean(record, key)
    return row


def write_case_csv(path, rows):
    fieldnames = [
        "status",
        "source",
        "scene",
        "sequence",
        "case",
        "frames",
        "processed_segments",
        "completed_frames",
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
        "num_failed": sum(row["status"] == "failed" for row in rows),
        "num_no_gt": sum(row["status"] == "no_gt" for row in rows),
        "num_reused_mask": sum(row["source"] == "mask_reused" for row in ok_rows),
        "num_completion_recomputed": sum(row["source"] == "completion_recomputed" for row in ok_rows),
        "total_processed_segments": int(sum(int(row["processed_segments"]) for row in ok_rows)),
        "total_completed_frames": int(sum(int(row["completed_frames"]) for row in ok_rows)),
        "total_frames": int(sum(int(row["frames"]) for row in ok_rows)),
    }
    for key in METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in ok_rows], dtype=np.float64)
        summary[f"case_avg_{key}"] = summarize_metric(values)

    per_frame = {key: [] for key in METRIC_KEYS}
    for row in ok_rows:
        record = load_eval(row["eval_path"])
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


def evaluate_combined(args):
    sequences = find_sequences(args.completion_root)
    if not sequences:
        raise RuntimeError(f"No mhr_params found under {args.completion_root}")

    case_csv = args.case_csv or os.path.join(args.completion_root, "eval_case_metrics_completion_opt_post.csv")
    summary_json = args.summary_json or os.path.join(args.completion_root, "eval_summary_completion_opt_post.json")

    print(f"[GT] loading {args.gt_file}", flush=True)
    gt_all = np.load(args.gt_file, allow_pickle=True).item()
    print(f"[GT] loaded {len(gt_all)} sequences", flush=True)

    rows = []
    for index, sequence in enumerate(sequences, start=1):
        seq_dir = os.path.join(args.completion_root, sequence)
        meta_path = os.path.join(seq_dir, "completion_optimization_meta.json")
        meta = load_json(meta_path, default={})
        meta["_path"] = meta_path if os.path.exists(meta_path) else None
        processed = meta.get("processed_segments", []) or []

        output_path = os.path.join(seq_dir, args.output_file)
        source_path = os.path.join(args.mask_root, sequence, args.source_eval_file)

        print(f"\n[{index}/{len(sequences)}] {sequence}", flush=True)
        gt_key = resolve_gt_key(gt_all, args.data_root, sequence)
        if gt_key is None:
            row = row_from_eval(sequence, output_path, "no_gt", meta, status="no_gt", error="GT missing")
        elif not processed:
            if not os.path.exists(source_path):
                row = row_from_eval(
                    sequence,
                    output_path,
                    "mask_reused",
                    meta,
                    status="failed",
                    error=f"source eval missing: {source_path}",
                )
            else:
                copy_reused_eval(source_path, output_path, sequence, meta, overwrite=args.overwrite)
                row = row_from_eval(sequence, output_path, "mask_reused", meta)
                print(f"[REUSE] {source_path}", flush=True)
        elif os.path.exists(output_path) and not args.overwrite:
            row = row_from_eval(sequence, output_path, "completion_recomputed", meta)
            print(f"[SKIP] existing completion eval: {output_path}", flush=True)
        else:
            eval_args = Namespace(
                pred_root=args.completion_root,
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
                annotate_eval(
                    output_path,
                    combined_source="completion_recomputed",
                    source_eval_path=source_path,
                    completion_meta_path=meta_path,
                    completion_processed_segments=processed,
                    completion_selected_segments=meta.get("selected_segments", []),
                )
                row = row_from_eval(sequence, output_path, "completion_recomputed", meta)
            except Exception as exc:
                row = row_from_eval(
                    sequence,
                    output_path,
                    "completion_recomputed",
                    meta,
                    status="failed",
                    error=repr(exc),
                )
                print(f"[ERROR] {sequence}: {exc}", flush=True)
            finally:
                cleanup_after_sequence()

        rows.append(row)
        write_case_csv(case_csv, rows)

    summary = summarize_rows(rows)
    write_summary(summary_json, summary)

    print("\n--------------- Completion Combined Summary -------------")
    print(f"Completion root: {args.completion_root}")
    print(f"Cases: {summary['num_ok']}/{summary['num_cases']} ok")
    print(f"Reused mask / recomputed completion: {summary['num_reused_mask']} / {summary['num_completion_recomputed']}")
    print(f"Processed segments / completed frames: {summary['total_processed_segments']} / {summary['total_completed_frames']}")
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
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate VOccl3D mask completion results combined with original mask results.")
    parser.add_argument("--mask_root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--completion_root", default=DEFAULT_COMPLETION_ROOT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gt_file", default=DEFAULT_GT_FILE)
    parser.add_argument("--body_model_path", default=DEFAULT_BODY_MODEL_PATH)
    parser.add_argument("--mhr_model_path", default=DEFAULT_MHR_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--obj_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pelvis_idxs", type=parse_pelvis_idxs, default=parse_pelvis_idxs("2,3"))
    parser.add_argument("--source_eval_file", default=DEFAULT_SOURCE_EVAL_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--case_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--single_identity", action="store_true")
    parser.add_argument("--is_tracking", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate_combined(args)


if __name__ == "__main__":
    main()
