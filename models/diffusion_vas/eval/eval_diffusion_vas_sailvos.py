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
import matplotlib.pyplot as plt

import torch.utils.checkpoint
from torch.utils.data import RandomSampler

from datasets.dataloader_sailvos import SailVos_diffusion_vas
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from eval_utils import set_seed

import warnings
warnings.filterwarnings("ignore")

set_seed(0)



def image_to_coco_rle(image, threshold=387.5):
    """
    Convert an RGB image to a binary mask and encode it using COCO RLE.

    Args:
    image (np.array): The input image array of shape (H, W, 3) with dtype uint8.
    threshold (float): The threshold to convert image to binary mask. Default is 387.5.

    Returns:
    dict: The RLE-encoded binary mask.
    """
    # Convert to binary mask
    binary_mask = np.sum(image, axis=-1) > threshold
    binary_mask = binary_mask.astype(np.uint8)

    # Encode to RLE using COCO tool
    rle_encoded = cocomask.encode(np.asfortranarray(binary_mask))
    rle_encoded = {
        'counts': rle_encoded['counts'].decode('utf-8'),  # Convert bytes to string
        'size': rle_encoded['size']
    }

    return rle_encoded




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



def eval_diffusion_vas_on_sailvos(args):

    val_dataset = SailVos_diffusion_vas(
            path=args.eval_annot_path,
            rgb_base_path=args.eval_data_path,
            total_num=-1,
            channel_num=3,
            width=args.width,
            height=args.height,
            read_rgb=True
    )
    sampler = RandomSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        sampler=sampler,
    )


    pipeline = DiffusionVASPipeline.from_pretrained(
        args.model_path_mask, dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipeline.enable_model_cpu_offload()
    seed_num = 23 # different seeds would generate various results, due to multimodal generation
    generator = torch.manual_seed(seed_num)
    pipeline.set_progress_bar_config(disable=True)

    samples = {}
    for step, batch_data in enumerate(tqdm(val_dataloader, ncols=0)):

        modal_pixels = batch_data["modal_res"]
        rgb_imgs = batch_data['rgb_res']

        model_path_depth = args.model_path_depth + f"/depth_anything_v2_{args.depth_encoder}.pth"
        depth_model = init_depth_model(model_path_depth, args.depth_encoder)

        depth_imgs = convert_rgb_to_depth2(rgb_imgs, depth_model)
        depth_imgs = torch.from_numpy(depth_imgs).float() * 2.0 - 1.0
        depth_imgs = depth_imgs.unsqueeze(1).repeat(1, 3, 1, 1).unsqueeze(0)

        # print("depth_imgs:", depth_imgs.shape, depth_imgs.min(), depth_imgs.max())
        # plt.imsave('depth_frame0_ch0.png', ((depth_imgs[0, 0, 0] + 1) / 2).cpu().numpy(), cmap='gray')

        image_ids, obj_id, cat_id = batch_data["image_ids"], batch_data["obj_id"], batch_data["cat_id"]

        video_frames = pipeline(
            modal_pixels,
            depth_imgs,
            height=args.height,
            width=args.width,
            num_frames=25,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        ).frames[0]

        video_frames = [np.array(img) for img in video_frames]
        video_frames = np.array(video_frames).astype('uint8')

        # print("video_frames:", video_frames.shape, video_frames.min(), video_frames.max())
        # plt.imsave('video_frame0.png', video_frames[0].astype(np.uint8))


        for i in range(video_frames.shape[0]):
            key = (int(image_ids[0][i]), int(obj_id[0]))
            video_frame = video_frames[i]
            diffvas_rle_mask = image_to_coco_rle(video_frame, threshold=600)

            samples[key] = {
                "diffvas_res_rle": diffvas_rle_mask
            }

    output_dict = {str(key): value for key, value in samples.items()}

    with open(args.eval_output_path + f'diffusion_vas_sailvos_eval_results_seed{seed_num}.json', 'w') as file:
        json.dump(output_dict, file, indent=4)



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
        help="Path to input RGB data.", # /data/SAILVOS_2D/
    )

    parser.add_argument(
        "--eval_annot_path",
        type=str,
        required=True,
        help="Path to input annotation data.", # diffusion_vas_sailvos_val.json
    )

    parser.add_argument(
        "--eval_output_path",
        type=str,
        required=True,
        help="Output path.",
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


    args.eval_data_path = "/data3/kaihuac/SAILVOS_2D/"
    args.eval_annot_path = "/data3/kaihuac/my_svd_datasets/sailvos/sailvos_seqs_val_len25_gap25_svd.json"
    args.eval_output_path = "/data3/kaihuac/diffusion_vas/eval_outputs/"

    print(args)
    eval_diffusion_vas_on_sailvos(args)

