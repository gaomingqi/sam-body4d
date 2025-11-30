# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union, List

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_frames(
        self,
        img_list: List[str], # Union[str, np.ndarray],
        bboxes: Optional[List[np.ndarray]] = None, # Optional[np.ndarray] = None,
        masks: Optional[List[np.ndarray]] = None, # Optional[np.ndarray] = None,
        cam_int: Optional[List[np.ndarray]] = None, # Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        batch_list = []
        image_list =[]
        num_objs_per_image = []
        for i, img in enumerate(img_list):
            if type(img) == str:
                img = load_image(img, backend="cv2", image_format="bgr")
                image_format = "bgr"
            else:
                print("####### Please make sure the input image is in RGB format")
                image_format = "rgb"
            height, width = img.shape[:2]
            image_list.append(img)

            if bboxes[i] is not None:
                boxes = bboxes[i].reshape(-1, 4)
                self.is_crop = True
            elif self.detector is not None:
                if image_format == "rgb":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    image_format = "bgr"
                print("Running object detector...")
                boxes = self.detector.run_human_detection(
                    img,
                    det_cat_id=det_cat_id,
                    bbox_thr=bbox_thr,
                    nms_thr=nms_thr,
                    default_to_full_image=False,
                )
                print("Found boxes:", boxes)
                self.is_crop = True
            else:
                boxes = np.array([0, 0, width, height]).reshape(1, 4)
                self.is_crop = False

            num_objs_per_image.append(len(boxes))
            # If there are no detected humans, don't run prediction
            if len(boxes) == 0:
                # return []
                continue

            # The following models expect RGB images instead of BGR
            if image_format == "bgr":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Handle masks - either provided externally or generated via SAM2
            masks_score = None
            if masks[i] is not None:
                # Use provided masks - ensure they match the number of detected boxes
                print(f"Using provided masks: {masks[i].shape}")
                assert (
                    bboxes[i] is not None
                ), "Mask-conditioned inference requires bboxes input!"
                masks_binary = masks[i].reshape(-1, height, width, 1).astype(np.uint8)
                masks_score = np.ones(
                    len(masks), dtype=np.float32
                )  # Set high confidence for provided masks
                use_mask = True
            elif use_mask and self.sam is not None:
                print("Running SAM to get mask from bbox...")
                # Generate masks using SAM2
                masks, masks_score = self.sam.run_sam(img, boxes)
            else:
                masks, masks_score = None, None

        #################### Construct batch data samples ####################
            batch = prepare_batch(img, self.transform, boxes, masks_binary, masks_score)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
            if cam_int is not None:
                print("Using provided camera intrinsics...")
                cam_int = cam_int.to(batch["img"])
                batch["cam_int"] = cam_int.clone()
            elif self.fov_estimator is not None:
                print("Running FOV estimator ...")
                input_image = batch["img_ori"][0].data
                cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                    batch["img"]
                )
                batch["cam_int"] = cam_int.clone()
            else:
                cam_int = batch["cam_int"].clone()

            batch_list.append(batch)

        batch_dict = {}
        for bi in batch_list:
            for k, v in bi.items():
                if k == 'bbox_format':
                    continue
                if k == 'img_ori':
                    v = v[0]
                if k not in batch_dict:
                    batch_dict[k] = [v]
                else:
                    batch_dict[k].append(v)

        for k, v in batch_dict.items():
            if k == 'img_ori':
                continue
            batch_dict[k] = torch.concat(v, dim=0)

        batch_dict['bbox_format'] = batch_list[0]['bbox_format']

        #################### Run model inference on an image ####################
        batch_dict = recursive_to(batch_dict, "cuda")
        self.model._initialize_batch(batch_dict)

        outputs = self.model.run_inference_batch(
            image_list,
            batch_dict,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")

        all_out_batch = []
        
        batch_size = batch_dict["img"].shape[0]
        num_objects = batch_dict["img"].shape[1]
        for b_idx in range(batch_dict["img"].shape[0]):    # batch 
            all_out = []
            for idx in range(batch_dict["img"].shape[1]):    # person
                all_out.append(
                    {
                        "bbox": batch_dict["bbox"][b_idx, idx].cpu().numpy(),
                        "focal_length": out["focal_length"][b_idx * num_objects + idx],
                        "pred_keypoints_3d": out["pred_keypoints_3d"][b_idx * num_objects + idx],
                        "pred_keypoints_2d": out["pred_keypoints_2d"][b_idx * num_objects + idx],
                        "pred_vertices": out["pred_vertices"][b_idx * num_objects + idx],
                        "pred_cam_t": out["pred_cam_t"][b_idx * num_objects + idx],
                        "pred_pose_raw": out["pred_pose_raw"][b_idx * num_objects + idx],
                        "global_rot": out["global_rot"][b_idx * num_objects + idx],
                        "body_pose_params": out["body_pose"][b_idx * num_objects + idx],
                        "hand_pose_params": out["hand"][b_idx * num_objects + idx],
                        "scale_params": out["scale"][b_idx * num_objects + idx],
                        "shape_params": out["shape"][b_idx * num_objects + idx],
                        "expr_params": out["face"][b_idx * num_objects + idx],
                        "mask": masks[b_idx] if masks is not None else None,
                        "pred_joint_coords": out["pred_joint_coords"][b_idx * num_objects + idx],
                        "pred_global_rots": out["joint_global_rots"][b_idx * num_objects + idx],
                    }
                )

                if inference_type == "full":
                    all_out[-1]["lhand_bbox"] = np.array(
                        [
                            (
                                batch_lhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][0]
                                - batch_lhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][0] / 2
                            ).item(),
                            (
                                batch_lhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][1]
                                - batch_lhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][1] / 2
                            ).item(),
                            (
                                batch_lhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][0]
                                + batch_lhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][0] / 2
                            ).item(),
                            (
                                batch_lhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][1]
                                + batch_lhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][1] / 2
                            ).item(),
                        ]
                    )
                    all_out[-1]["rhand_bbox"] = np.array(
                        [
                            (
                                batch_rhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][0]
                                - batch_rhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][0] / 2
                            ).item(),
                            (
                                batch_rhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][1]
                                - batch_rhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][1] / 2
                            ).item(),
                            (
                                batch_rhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][0]
                                + batch_rhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][0] / 2
                            ).item(),
                            (
                                batch_rhand["bbox_center"].flatten(0, 1)[b_idx * num_objects + idx][1]
                                + batch_rhand["bbox_scale"].flatten(0, 1)[b_idx * num_objects + idx][1] / 2
                            ).item(),
                        ]
                    )
            all_out_batch.append(all_out)

        return all_out_batch
