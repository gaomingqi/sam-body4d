import cv2
from PIL import Image
import numpy as np


def is_super_long_or_wide(mask, label, ratio=0.7):
    # Find foreground pixels of the specified label
    ys, xs = np.where(mask == label)
    if len(xs) == 0:
        return False  # No foreground found
    
    # Image dimensions
    h, w = mask.shape
    
    # Bounding box of the foreground object
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    
    # Check if width or height of bounding box is large enough
    # relative to the entire image (e.g., ≥ 90%)
    return (bbox_w / w >= ratio) or (bbox_h / h >= ratio)


def resize_mask_with_unique_label(mask, target_h, target_w, label):
    """
    Resize a binary/segmentation mask to (target_h, target_w)
    ensuring only the given 'label' exists in the output mask.
    Nearest-neighbor used to avoid mixing labels.
    """

    # Convert to PIL for nearest-neighbor resize (no interpolation blending)
    mask_img = Image.fromarray((mask == label).astype(np.uint8))

    # Resize
    mask_resized = mask_img.resize((target_w, target_h), Image.NEAREST)

    # Convert back and fill only with the given label
    mask_out = np.array(mask_resized, dtype=np.uint8)
    mask_out = (mask_out > 0).astype(np.uint8) * label  # ensure only unique label

    return mask_out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected foreground component in a binary mask.
    Input:
        mask: HxW array, values 0 (background) / 1 (foreground)
    Output:
        same shape, only largest component kept as 1, others set to 0
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)

    # no foreground
    if num_labels <= 1:
        return np.zeros_like(mask)

    # count each label (skip background 0)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = counts.argmax()

    largest_mask = (labels == largest_label).astype(mask.dtype)
    return largest_mask


def is_skinny_mask(mask, ratio_threshold=1/5):
    """
    Determine whether the foreground region in the mask is very skinny,
    based on the bounding box aspect ratio.

    Args:
        mask (np.ndarray): Binary or multi-value mask of shape (H, W).
                           Foreground pixels are defined as > 0.
        ratio_threshold (float): Threshold for min(h, w) / max(h, w).
                                 Default = 1/5.

    Returns:
        bool: True if the object is skinny (ratio < threshold), else False.
    """

    # Get coordinates of foreground pixels
    ys, xs = np.where(mask > 0)

    # If no foreground exists
    if len(xs) == 0:
        return False

    # Compute bounding box height and width
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1

    # Compute the smaller-to-larger side ratio
    ratio = min(h, w) / max(h, w)

    # Check if the object is considered skinny
    return ratio < ratio_threshold


def bbox_from_mask(mask):
    """
    Compute normalized bounding box for a boolean/binary mask.
    Returns bbox parameters in range [0, 1].

    Args:
        mask (np.ndarray): Mask of shape (H, W).
                           True/1 = foreground, False/0 = background.

    Returns:
        tuple or None: (cx_norm, cy_norm, width_norm, height_norm)
            - cx_norm, cy_norm: normalized bbox center
            - width_norm, height_norm: normalized bbox size
        None: if no foreground exists
    """
    H, W = mask.shape[:2]

    # Foreground pixel locations
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    # Bounding box extents
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    cx = x_min + width / 2.0
    cy = y_min + height / 2.0

    # Normalize to [0, 1]
    return (
        cx / W,
        cy / H,
        width / W,
        height / H
    )


def are_bboxes_similar(bbox1, bbox2,
                       size_ratio_thresh=0.2,
                       center_ratio_thresh=4):
    """
    Check whether two bboxes are similar.
    Size (width/height) is the primary criterion, center (cx, cy) is secondary.

    Args:
        bbox1, bbox2 (tuple): (cx, cy, width, height).
        size_ratio_thresh (float): Max allowed relative difference for width/height.
        center_ratio_thresh (float): Max allowed center distance normalized
                                     by average bbox size.

    Returns:
        bool: True if bboxes are considered similar, False otherwise.
    """
    if bbox1 is None or bbox2 is None:
        return False

    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    # --- 1) Size similarity (primary) ---
    ratio1 = max(w1, h1) / (min(w1, h1) + 1e-6)
    ratio2 = max(w2, h2) / (min(w2, h2) + 1e-6)

    ratio_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2)

    if ratio_diff > size_ratio_thresh:
        return False

    # --- 2) Center similarity (secondary) ---
    # Normalize center distance by average of bbox sizes
    avg_size = (w1 + h1 + w2 + h2) / 4.0
    center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    center_norm = center_dist / (avg_size + 1e-6)

    if center_norm > center_ratio_thresh:
        return False

    return True
