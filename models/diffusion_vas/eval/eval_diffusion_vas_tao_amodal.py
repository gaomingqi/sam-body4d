import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import json
from pycocotools import mask as cocomask
import argparse
from datetime import datetime

import torch.utils.checkpoint
from torch.utils.data import RandomSampler

from datasets.dataloader_tao_amodal import TAO_amodal_diffusion_vas
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from eval_utils import set_seed

import warnings
warnings.filterwarnings("ignore")

set_seed(0)


def get_bbox_from_mask(mask):
    # Find the coordinates of the non-zero values in the mask
    y_coords, x_coords = np.nonzero(mask)

    # If there are no non-zero values, return an empty bbox
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None

    # Get the bounding box coordinates
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Calculate width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Return the bounding box as [x_min, y_min, width, height]
    return [int(x_min), int(y_min), int(width), int(height)]

def image_to_bbox(image, threshold=387.5):

    # Convert to binary mask
    binary_mask = np.sum(image, axis=-1) > threshold
    binary_mask = binary_mask.astype(np.uint8)

    pred_amodal_bbox = get_bbox_from_mask(binary_mask)

    return pred_amodal_bbox

def union_bboxes(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def image_to_coco_rle_mu(image, modal_mask, threshold=387.5):
    # Convert to binary mask
    binary_mask = np.sum(image, axis=-1) > threshold
    binary_mask = binary_mask.astype(np.uint8)

    binary_mask = np.logical_or(binary_mask, modal_mask)

    # Encode to RLE using COCO tool

    rle_encoded = cocomask.encode(np.asfortranarray(binary_mask))
    rle_encoded = {
        'counts': rle_encoded['counts'].decode('utf-8'),  # Convert bytes to string
        'size': rle_encoded['size']
    }

    rle_encoded2 = cocomask.encode(np.asfortranarray(modal_mask.astype(np.uint8)))
    rle_encoded2 = {
        'counts': rle_encoded2['counts'].decode('utf-8'),  # Convert bytes to string
        'size': rle_encoded2['size']
    }

    return rle_encoded, rle_encoded2


def post_process_pred_bbox(pred_bbox, pad_width, pad_height, modal_bbox):
    if pred_bbox is not None:
        pred_bbox = [pred_bbox[0] - pad_width, pred_bbox[1] - pad_height, pred_bbox[2], pred_bbox[3]]
    else:
        pred_bbox = modal_bbox

    return pred_bbox


def convert_rgb_to_depth2(rgb_images, depth_model):


    # Convert the RGB images to depth maps
    depth_maps = [depth_model.infer_image(rgb_image.cpu().numpy()[0]) for rgb_image in rgb_images]

    depth_maps = np.array(depth_maps)
    # Normalize the depth maps to the range [0, 1]
    depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())

    return depth_maps



def init_depth_model(model_path_depth, depth_encoder):

    from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

    depth_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_model = DepthAnythingV2(**depth_model_configs[depth_encoder]).to('cuda')
    depth_model.load_state_dict(
        torch.load(model_path_depth))
    depth_model.eval()

    return depth_model



def eval_diffusion_vas_on_tao_amodal(args):

    val_dataset = TAO_amodal_diffusion_vas(
        path=args.eval_annot_path,
        rgb_base_path=args.eval_data_path,
        total_num=-1,
        channel_num=3,
        read_rgb=True
    )
    sampler = RandomSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        sampler=sampler,
    )

    track_ids = json.load(open(args.track_ids_path, 'r'))

    pipeline = DiffusionVASPipeline.from_pretrained(
        args.model_path_mask, dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    seed_num = 23 # different seeds would generate various results, due to multimodal generation
    generator = torch.manual_seed(seed_num)
    pipeline.set_progress_bar_config(disable=True)


    model_path_depth = args.model_path_depth + f"/depth_anything_v2_{args.depth_encoder}.pth"
    depth_model = init_depth_model(model_path_depth, args.depth_encoder)

    samples = {}
    for step, batch_data in enumerate(tqdm(val_dataloader, ncols=0)):
        modal_pixels = batch_data["modal_res"]
        rgb_imgs = batch_data['rgb_res']
        amodal_bboxes = batch_data["amodal_bboxes"][0].detach().cpu()
        modal_pixels2 = (batch_data['modal_res'][0, :, 0, :, :].detach().cpu().numpy() + 1) // 2
        modal_bboxes = [get_bbox_from_mask(modal_pixels2[i]) for i in range(len(modal_pixels2))]
        vid_id = batch_data['vid_id']
        rel_track_id = int(batch_data['track_id']) - track_ids[str(int(vid_id))] + 1
        file_names = batch_data['image_file_names']
        ori_h, ori_w = int(batch_data['height']), int(batch_data['width'])


        # preprocess the modal pixels
        modal_pixels_np = modal_pixels.detach().cpu().numpy()
        modal_pixels_np = (modal_pixels_np + 1) / 2
        pad_height, pad_width = ori_h // 4, ori_w // 4
        padded_h, padded_w = ori_h + 2 * pad_height, ori_w + 2 * pad_width

        # preprocess the depth images
        depth_imgs_rgb = []
        depth_imgs = convert_rgb_to_depth2(rgb_imgs, depth_model)
        for i in range(depth_imgs.shape[0]):
            rgb_frame = cv2.cvtColor((depth_imgs[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            depth_imgs_rgb.append(rgb_frame)
        depth_imgs_rgb = np.stack(depth_imgs_rgb)
        depth_imgs_rgb = np.transpose(depth_imgs_rgb, (0, 3, 1, 2))
        depth_imgs_rgb = depth_imgs_rgb / 255.0
        depth_imgs_np = np.expand_dims(depth_imgs_rgb, axis=0)



        padded_and_resized_frames = []
        padded_and_resized_depth_frames = []
        for frame in range(modal_pixels_np.shape[1]):

            padded_frame = np.pad(modal_pixels_np[0, frame], ((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                                  mode='constant', constant_values=0)
            resized_frame = cv2.resize(padded_frame.transpose(1, 2, 0), (args.width, args.height)).transpose(2, 0, 1)
            padded_and_resized_frames.append(resized_frame)

            padded_depth_frame = np.pad(depth_imgs_np[0, frame], ((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                                  mode='constant', constant_values=1)
            resized_depth_frame = cv2.resize(padded_depth_frame.transpose(1, 2, 0), (args.width, args.height)).transpose(2, 0, 1)
            padded_and_resized_depth_frames.append(resized_depth_frame)


        padded_and_resized_np = np.stack(padded_and_resized_frames, axis=0)
        padded_and_resized_np = (padded_and_resized_np * 2) - 1
        modal_pixels = torch.tensor(padded_and_resized_np).unsqueeze(0).float()

        padded_and_resized_depth_np = np.stack(padded_and_resized_depth_frames, axis=0)
        padded_and_resized_depth_np = (padded_and_resized_depth_np * 2) - 1
        depth_imgs = torch.tensor(padded_and_resized_depth_np).unsqueeze(0).float()


        video_frames = pipeline(
            modal_pixels,
            depth_imgs,
            height=args.height,
            width=args.width,
            num_frames=25,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=2,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]

        video_frames = np.array(video_frames).astype('uint8')

        for i in range(len(file_names)):
            key = str((int(vid_id), file_names[i][0], str(rel_track_id)))
            video_frame = video_frames[i]
            video_frame = cv2.resize(video_frame, (padded_w, padded_h))

            diffvas_rle = image_to_bbox(video_frame, threshold=600)
            diffvas_rle = post_process_pred_bbox(diffvas_rle, pad_width, pad_height, modal_bboxes[i])
            diffvas_rle_union_box = union_bboxes(diffvas_rle, modal_bboxes[i])
            
            modal_frame = modal_pixels2[i]
            modal_frame = np.pad(modal_frame, ((pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant', constant_values=0)

            diffvas_rle_union_mask, modal_mask_padded = image_to_coco_rle_mu(video_frame, modal_frame, threshold=600)

            samples[key] = {
                            "diffvas_res_bbox": diffvas_rle_union_box,
                            "diffvas_res_rle": diffvas_rle_union_mask,
                            "modal_res_rle": modal_mask_padded,
                        }
            
        
    timestamp = datetime.now().strftime('%b%d_%H%M')
    os.makedirs(args.eval_output_path, exist_ok=True)
    with open(args.eval_output_path + f'diffusion_vas_tao_amodal_eval_results_seed{seed_num}.json', 'w') as file:
        json.dump(samples, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video amodal segmentation and content completion using Diffusion-VAS."
    )

    parser.add_argument(
        "--model_path_mask",
        type=str,
        default="checkpoints/diffusion-vas-amodal-segmentation",
        help="Path to diffusion-vas amodal segmentation checkpoint.",
    )

    parser.add_argument(
        "--depth_encoder",
        type=str,
        default="vitl",  # or 'vits', vitl, 'vitg'
        help="Path to diffusion-vas content completion checkpoint.",
    )

    parser.add_argument(
        "--model_path_depth",
        type=str,
        default="checkpoints/",
        help="Path to diffusion-vas content completion checkpoint.",
    )

    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to input RGB data.", # /data/TAO/frames/
    )

    parser.add_argument(
        "--eval_annot_path",
        type=str,
        required=True,
        help="Path to input annotation data.", # diffusion_vas_tao_amodal_val.json
    )


    parser.add_argument(
        "--eval_output_path",
        type=str,
        required=True,
        help="Output path.",
    )

    parser.add_argument(
        "--track_ids_path",
        type=str,
        required=True,
        help="Path to track ids.", # tao_amodal_track_ids_abs2rel_val.json
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target prediction's height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Target prediction's width.",
    )

    args = parser.parse_args()

    print(args)
    eval_diffusion_vas_on_tao_amodal(args)
