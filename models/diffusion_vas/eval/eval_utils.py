import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)       # Python's built-in random module
    np.random.seed(seed)    # NumPy's random module
    torch.manual_seed(seed) # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        

def compute_iou(box1, box2):
    # Unpack coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute the coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Compute the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

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
    return [x_min, y_min, width, height]