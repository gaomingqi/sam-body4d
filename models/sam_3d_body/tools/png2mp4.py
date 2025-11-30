import cv2
import numpy as np

def concat_pngs_side_by_side(pngs_left, pngs_right, output_path, fps=30):
    """
    将两组 PNG 左右拼接并生成 MP4 视频。

    Args:
        pngs_left (list[str]): 左侧 PNG 路径列表
        pngs_right (list[str]): 右侧 PNG 路径列表
        output_path (str): 输出 MP4 路径
        fps (int): 帧率
    """

    n = min(len(pngs_left), len(pngs_right))
    if n == 0:
        raise ValueError("输入 PNG 列表不能为空")

    # 读第一帧，确定大小
    left0 = cv2.imread(pngs_left[0])
    right0 = cv2.imread(pngs_right[0])

    if left0 is None or right0 is None:
        raise ValueError("无法读取 PNG 图片")

    # 如果大小不同，右侧 resize 到左侧尺寸
    h, w, _ = left0.shape
    right0 = cv2.resize(right0, (w, h))

    # 拼接后的输出分辨率
    out_w = w * 2
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for i in range(n):
        left = cv2.imread(pngs_left[i])
        right = cv2.imread(pngs_right[i])

        if left is None or right is None:
            print(f"警告: 无法读取第 {i} 帧，跳过")
            continue

        # 尺寸对齐
        right = cv2.resize(right, (w, h))

        # 左右拼接
        concat = np.concatenate([left, right], axis=1)

        writer.write(concat)

    writer.release()
    print(f"视频保存到：{output_path}")


if __name__ == "__main__":
    import os, glob
    root = "path to image folder"
    png_list = glob.glob(os.path.join(root, '*'))
    png_list.sort()

    png1 = [os.path.join(p, 'mask_'+p.split('_')[-1]+'_bbox_000.png') for p in png_list]
    png2 = [os.path.join(p, 'mask_'+p.split('_')[-1]+'_overlay_000.png') for p in png_list]

    concat_pngs_side_by_side(png1, png2, 'mini.mp4', fps=30)