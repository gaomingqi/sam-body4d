import cv2
import numpy as np
from typing import List

def images_to_mp4(images: List[np.ndarray], output_path: str, fps: int = 25):
    if len(images) == 0:
        raise ValueError("The image list is empty.")

    first = images[0]
    if first.ndim == 2:
        h, w = first.shape
    elif first.ndim == 3:
        h, w, _ = first.shape
    else:
        raise ValueError("Images must be either HxW or HxWx3 numpy arrays.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img in images:
        if img.ndim == 2:
            frame = img
            if frame.shape != (h, w):
                frame = cv2.resize(frame, (w, h))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            frame = img
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image shape.")

        writer.write(frame)

    writer.release()
    print(f"Video saved to: {output_path}")
