# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import cv2
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from utils.painter import color_list

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces, id_current):
	from sam_3d_body.visualization.renderer import Renderer

	img_mesh = img_cv2.copy()
	img_mesh = np.ones_like(img_mesh) * 255

	if outputs is None:
		return img_mesh

	rend_img = []
	for pid, person_output in enumerate(outputs):
		renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
		img2 = (
			renderer(
				person_output["pred_vertices"],
				person_output["pred_cam_t"],
				img_mesh.copy(),
				mesh_base_color=color_list[id_current[pid]+4],
				scene_bg_color=(1, 1, 1),
			)
			* 255
		)

		# cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
		rend_img.append(img2)

	return rend_img

def draw_keypoints_with_index(
    image: np.ndarray,
    keypoints,  # np.ndarray or torch.Tensor, (N,2)
    radius: int = 3,
    point_color=(0, 255, 0),
    text_color=(255, 255, 255),
    text_scale: float = 0.4,
    text_thickness: int = 1,
    offset=(5, -5),
) -> np.ndarray:
    out = image.copy()
    H, W = out.shape[:2]

    # accept numpy or torch, normalize to numpy float32
    if isinstance(keypoints, torch.Tensor):
        kps = keypoints.detach().cpu().float().numpy()
    else:
        kps = np.asarray(keypoints, dtype=np.float32)

    if kps.ndim != 2 or kps.shape[1] != 2:
        raise ValueError(f"keypoints must be (N,2), got {kps.shape}")

    for i, (x, y) in enumerate(kps):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(out, (xi, yi), radius, point_color, -1, cv2.LINE_AA)
            cv2.putText(
                out, str(i),
                (xi + offset[0], yi + offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness,
                cv2.LINE_AA,
            )

    return out

def visualize_sample_together(img_cv2, outputs, faces, id_current):
	from sam_3d_body.visualization.renderer import Renderer

	# Render everything together
	img_mesh = img_cv2.copy()
	if outputs is None:
		return img_mesh
	# img_mesh = np.ones_like(img_mesh) * 255

	# return draw_keypoints_with_index(img_mesh, outputs[0]['pred_keypoints_2d'])
	
	if outputs is None:
		return img_mesh

	# First, sort by depth, furthest to closest
	try:
		all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
	except:
		return img_mesh
	outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

	id_sorted = np.argsort(-all_depths)   # by id not depth for consistent coloring

	# Then, put all meshes together as one super mesh
	all_pred_vertices = []
	all_faces = []
	all_color = []
	for pid, person_output in enumerate(outputs_sorted):
		all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
		all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
		all_color.append(color_list[id_current[id_sorted[pid]]+4])
	all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
	all_faces = np.concatenate(all_faces, axis=0)

	# Pull out a fake translation; take the closest two
	fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
	all_pred_vertices = all_pred_vertices - fake_pred_cam_t
	
	# Render front view
	renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
	img_mesh = (
		renderer(
			all_pred_vertices,
			fake_pred_cam_t,
			img_mesh,
			# mesh_base_color=LIGHT_BLUE,
			mesh_base_color=all_color,
			scene_bg_color=(1, 1, 1),
		)
		* 255
	)

	return img_mesh
