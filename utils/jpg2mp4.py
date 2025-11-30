import os
import glob
import cv2

def jpg_folder_to_mp4(folder: str, output_filename: str, fps: int = 25):
    """
    把某个文件夹下的 JPG 图片按文件名排序后，保存为 MP4 视频。

    参数
    ----
    folder : str
        存放 JPG 图片的文件夹路径。
    output_filename : str
        输出 MP4 文件路径（例如 'output.mp4'，可以带完整路径）。
    fps : int, 默认 25
        视频帧率。
    """
    # 找到所有 jpg 图片（区分大小写的都处理一下）
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(folder, p)))

    if not img_paths:
        raise ValueError(f"No JPG images found in folder: {folder}")

    # 按文件名排序
    img_paths = sorted(img_paths)

    # 读第一张图确定分辨率
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        raise ValueError(f"Failed to read image: {img_paths[0]}")
    h, w = first_img.shape[:2]

    # 初始化视频写入器（mp4v 编码）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: skip unreadable image {path}")
            continue
        # 如果分辨率和第一张不一样，强制缩放一下
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"Saved video to: {output_filename}")
