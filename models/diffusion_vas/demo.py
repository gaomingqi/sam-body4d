import os
import cv2
import numpy as np
import argparse
import imageio
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from PIL import Image


import torch.utils.checkpoint
from torchvision import transforms

from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline

import warnings
warnings.filterwarnings("ignore")


def init_amodal_segmentation_model(model_path_mask):
    pipeline_mask = DiffusionVASPipeline.from_pretrained(
        model_path_mask, dtype=torch.float16
    ).to("cuda")
    pipeline_mask.enable_model_cpu_offload()
    pipeline_mask.set_progress_bar_config(disable=True)

    return pipeline_mask


def init_rgb_model(model_path_rgb):
    pipeline_rgb = DiffusionVASPipeline.from_pretrained(
        model_path_rgb, dtype=torch.float16
    ).to("cuda")
    pipeline_rgb.enable_model_cpu_offload()
    pipeline_rgb.set_progress_bar_config(disable=True)

    return pipeline_rgb


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


def load_and_transform_masks(image_folder, resolution=(512, 1024), obj_id=None):
    # Define mask transformation: resize, to tensor, repeat grayscale channel, normalize
    mask_transform = transforms.Compose([
        transforms.Resize(resolution),  # Resize to resolution
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel to 3 channels
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize
    ])

    # List and sort image file paths in the folder
    image_paths = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ])

    processed_frames = []  # List to store transformed frames
    original_size = None  # To capture original image size

    for image_path in image_paths:
        if obj_id is not None:
            image = Image.open(image_path).convert('P')  # Open image and convert to grayscale
            binary_image = image.point(lambda p: 255 if p == obj_id else 0)  # Isolate specific object ID
        else:
            image = Image.open(image_path).convert('L')  # Open image and convert to grayscale
            binary_image = image.point(lambda p: 255 if p > 128 else 0)  # Binarize image
        if original_size is None:
            original_size = binary_image.size[::-1]  # Save original size as (height, width)
        transformed_frame = mask_transform(binary_image)  # Apply transformation
        processed_frames.append(transformed_frame)  # Append to list

    mask_tensor = torch.stack(processed_frames).unsqueeze(0)  # Stack frames and add batch dimension
    return mask_tensor, original_size  # Return tensor and original size


def load_and_transform_rgbs(image_folder, resolution=(512, 1024)):
    """Load RGB images from a folder, transform them, and return as tensor, original size, and raw images."""
    # Define RGB transformation: resize, to tensor, and normalize
    rgb_transform = transforms.Compose([
        transforms.Resize(resolution),  # Resize to resolution
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize
    ])

    # List and sort image file paths in the folder
    image_paths = sorted([
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ])

    transformed_frames = []  # List for transformed frames
    raw_images = []  # List for raw images
    original_size = None  # To capture original image size

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        raw_images.append(np.array(image))
        if original_size is None:
            original_size = image.size[::-1]
        transformed_frame = rgb_transform(image)
        transformed_frames.append(transformed_frame)

    rgb_tensor = torch.stack(transformed_frames).unsqueeze(0)

    return rgb_tensor, original_size, np.array(raw_images)


def rgb_to_depth(rgb_tensor, depth_model):

    # Remove the batch dimension (shape becomes [num_frames, 3, height, width])
    rgb_images = rgb_tensor.squeeze(0)
    rgb_images = (((rgb_images + 1.0) / 2.0) * 255)

    depth_maps = []

    # Loop through each frame in the tensor
    for i in range(rgb_images.shape[0]):
        rgb_image_np = rgb_images[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        depth_map = depth_model.infer_image(rgb_image_np)
        depth_maps.append(depth_map)

    depth_maps_np = np.array(depth_maps)
    depth_maps_np = (depth_maps_np - depth_maps_np.min()) / (depth_maps_np.max() - depth_maps_np.min())

    depth_maps_np = depth_maps_np * 2 - 1
    depth_tensor = torch.tensor(depth_maps_np, dtype=torch.float32)

    depth_tensor_3channel = depth_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: [num_frames, 3, height, width]
    depth_tensor_3channel = depth_tensor_3channel.unsqueeze(0)

    return depth_tensor_3channel


def overlay_mask_on_image(rgb_img, mask, cmap_idx=None, random_color=False, boundary_thickness=3, darken_factor=2):
    # Ensure the input image is RGB and in the range [0, 1]
    assert rgb_img.shape[-1] == 3, "Expected RGB image with 3 channels"
    # assert rgb_img.min() >= 0 and rgb_img.max() <= 1, "Expected rgb_img values in the range [0, 1]"

    # Select a color for the mask overlay
    cmap = plt.get_cmap("tab10")

    cmap_idx = 4
    if cmap_idx is None and random_color:
        cmap_idx = np.random.randint(0, cmap.N)  # Randomly choose a colormap index if not provided

    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    boundary_color = color[:3] * darken_factor  # Darken the color by the darken_factor
    boundary_color = np.concatenate([boundary_color, [1.0]])  # Make boundary fully opaque

    # Create a boundary mask
    dilated_mask = binary_dilation(mask, iterations=boundary_thickness)
    boundary_mask = dilated_mask & ~mask

    # Create a colored mask in the range [0, 1]
    mask_image = np.zeros_like(rgb_img, dtype=np.float32)
    boundary_image = np.zeros_like(rgb_img, dtype=np.float32)

    for i in range(3):  # Apply the mask and boundary to each channel
        mask_image[..., i] = mask * color[i]
        boundary_image[..., i] = boundary_mask * boundary_color[i]

    # Combine the RGB image with the colored mask and boundary
    overlayed_image = np.clip(rgb_img * 0.5 + mask_image + boundary_image, 0, 1)

    return overlayed_image


def main(args):

    generator = torch.manual_seed(23)

    model_path_mask = args.model_path_mask
    pipeline_mask = init_amodal_segmentation_model(model_path_mask)

    model_path_rgb = args.model_path_rgb
    pipeline_rgb = init_rgb_model(model_path_rgb)
    
    depth_encoder = args.depth_encoder
    model_path_depth = args.model_path_depth + f"/depth_anything_v2_{depth_encoder}.pth"
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    data_path = args.data_path
    seq_name = args.seq_name
    seq_path = os.path.join(data_path, seq_name)

    data_output_path = args.data_output_path
    output_seq_path = os.path.join(data_output_path, seq_name)
    os.makedirs(f"{data_output_path}/{seq_name}", exist_ok=True)
    
    # output gif paths
    modal_masks_overlay_path = f"{output_seq_path}/modal_masks_overlay.gif"
    pred_amodal_masks_path = f"{output_seq_path}/pred_amodal_masks.gif"
    pred_amodal_masks_overlay_path = f"{output_seq_path}/pred_amodal_masks_overlay.gif"

    modal_rgb_path = f"{output_seq_path}/modal_rgb.gif"
    modal_rgb_overlay_path = f"{output_seq_path}/modal_rgb_overlay.gif"
    
    pred_amodal_rgb_path = f"{output_seq_path}/pred_amodal_rgb.gif"
    pred_amodal_rgb_overlay_path = f"{output_seq_path}/pred_amodal_rgb_overlay.gif"
    
    # load input modal masks and rgb images
    pred_res = (256, 512) # sometimes a higher resolution (e.g.,512x1024) might produce better results
    
    start, end = 15, 79  # process 25 frames from frame 20 to frame 44 (inclusive)
    obj_id = 1
    save_path = "o"
    save_path_images = f"{save_path}/images"
    save_path_masks = f"{save_path}/masks"
    os.makedirs(save_path_images, exist_ok=True)
    os.makedirs(save_path_masks, exist_ok=True)
    
    modal_pixels, ori_shape = load_and_transform_masks(seq_path + "/masks", resolution=pred_res, obj_id=obj_id)
    rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(seq_path + "/images", resolution=pred_res)
    
    modal_pixels = modal_pixels[:, start:end, :, :, :]
    rgb_pixels = rgb_pixels[:, start:end, :, :, :]
    raw_rgb_pixels = raw_rgb_pixels[start:end, :, :, :]

    depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    print("amodal segmentation by diffusion-vas ...")
    # predict amodal masks (amodal segmentation)
    pred_amodal_masks = pipeline_mask(
        modal_pixels,
        depth_pixels,
        height=pred_res[0],
        width=pred_res[1],
        num_frames=64,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]

    pred_amodal_masks = [np.array(img) for img in pred_amodal_masks]

    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    
    # save pred_amodal_masks
    modal_mask_union = (modal_pixels[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
    pred_amodal_masks = np.logical_or(pred_amodal_masks, modal_mask_union).astype('uint8')

    pred_amodal_masks_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)
                                       for frame in pred_amodal_masks])
    imageio.mimsave(pred_amodal_masks_path, (pred_amodal_masks_save * 255).astype(np.uint8), fps=8)
    
    
    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
        2).repeat(1, 1, 3, 1, 1)

    modal_obj_mask = (modal_pixels > 0).float()
    modal_background = 1 - modal_obj_mask
    rgb_pixels = (rgb_pixels + 1) / 2

    tmp_cmap_idx = np.random.randint(0, plt.get_cmap("tab10").N)
    rgb_pixels_save = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in
         rgb_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])

    amodal_masks_overlay = []
    for i in range(25):
        tmp_rgb_amodal = overlay_mask_on_image(rgb_pixels_save[i], pred_amodal_masks_save[i].astype(np.uint8),
                                               cmap_idx=tmp_cmap_idx)
        amodal_masks_overlay.append(tmp_rgb_amodal)
    modal_mask_union = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for frame in
         modal_obj_mask[0, :, 0, :, :].cpu().numpy().astype(np.uint8)])

    # save amodal_masks_overlay
    amodal_masks_overlay_np = np.stack(amodal_masks_overlay, axis=0)
    imageio.mimsave(pred_amodal_masks_overlay_path, (amodal_masks_overlay_np * 255).astype(np.uint8), fps=8)

    modal_masks_overlay = []
    for i in range(25):
        tmp_rgb_modal = overlay_mask_on_image(rgb_pixels_save[i], modal_mask_union[i].astype(np.uint8),
                                              cmap_idx=tmp_cmap_idx)
        modal_masks_overlay.append(tmp_rgb_modal)

    # save modal_masks_overlay
    modal_masks_overlay_np = np.stack(modal_masks_overlay, axis=0)
    imageio.mimsave(modal_masks_overlay_path, (modal_masks_overlay_np * 255).astype(np.uint8), fps=8)

    # save modal_rgb
    modal_rgb_pixels = rgb_pixels * modal_obj_mask + modal_background
    modal_rgb_pixels_save = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR) for frame in
         modal_rgb_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    imageio.mimsave(modal_rgb_path, (modal_rgb_pixels_save * 255).astype(np.uint8), fps=8)

    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

    print("content completion by diffusion-vas ...")
    # predict amodal rgb (content completion)
    pred_amodal_rgb = pipeline_rgb(
        modal_rgb_pixels,
        pred_amodal_masks_tensor,
        height=pred_res[0],  # my_res[0]
        width=pred_res[1],  # my_res[1]
        num_frames=25,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=generator,
    ).frames[0]

    pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

    # save pred_amodal_rgb
    pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
    pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                     for frame in pred_amodal_rgb])
    imageio.mimsave(pred_amodal_rgb_path, pred_amodal_rgb_save, fps=8)

    # save pred_amodal_rgb_overlay
    transparency_factor = 0.5
    white_background = np.ones_like(raw_rgb_pixels) * 255
    raw_rgb_semi_transparent = np.clip(
        raw_rgb_pixels * transparency_factor + white_background * (1 - transparency_factor), 0, 255
    ).astype(np.uint8)
    pred_amodal_rgb_overlay = np.where(pred_amodal_masks_save[..., None] == 1, pred_amodal_rgb_save, raw_rgb_semi_transparent)
    imageio.mimsave(pred_amodal_rgb_overlay_path, pred_amodal_rgb_overlay, fps=8)

    # save modal_rgb_overlay
    modal_pixels = np.array(
        [cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for frame in
         modal_pixels[0].cpu().numpy().transpose(0, 2, 3, 1)])
    modal_rgb_overlay = np.where(np.array((modal_pixels > 0)[:, :, :, :]) == 1, raw_rgb_pixels, raw_rgb_semi_transparent)
    imageio.mimsave(modal_rgb_overlay_path, modal_rgb_overlay, format='GIF', fps=8)


    for img in pred_amodal_rgb_save:
        cv2.imwrite(f"{save_path_images}/{len(os.listdir(save_path_images)):05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    amodal_masks_overlay_np = (pred_amodal_masks_save * 255).astype(np.uint8)
    for mask in amodal_masks_overlay_np:
        cv2.imwrite(f"{save_path_masks}/{len(os.listdir(save_path_masks)):05d}.png", mask)
