from PIL import Image
from PIL import ImageDraw


def draw_point_marker(image: Image.Image, x: int, y: int, point_type: str) -> Image.Image:
    """
    Draw a circular marker with soft color fill:
        - Positive:  light green fill + white border + white "+"
        - Negative:  light red   fill + white border + white "-"

    Marker size auto-scales with image size.

    Args:
        image: PIL Image(RGB)
        x, y: coordinates (int)
        point_type: "positive" or "negative"

    Returns:
        PIL Image with marker drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # image size
    w, h = img.size

    # ===== 自适应大小 =====
    base = min(w, h)
    radius = max(6, int(base * 0.015))        # 圆半径 ~1.5% 短边
    line_w = max(2, radius // 4)              # 加号/减号线宽
    border_w = max(2, radius // 5)            # 外圈白边宽度

    # clamp
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # ===== 颜色设定 =====
    if point_type.lower() == "positive":
        fill_color = (180, 255, 180)   # 淡绿色
    else:
        fill_color = (255, 180, 180)   # 淡红色

    border_color = (255, 255, 255)     # 白色
    sign_color = (255, 255, 255)       # 白色

    # ===== 画圆（填充 + 白边）=====
    bbox = [x - radius, y - radius, x + radius, y + radius]

    # 填充
    draw.ellipse(bbox, fill=fill_color)

    # 外圈白边（叠加一层 outline）
    draw.ellipse(bbox, outline=border_color, width=border_w)

    # ===== 画加号 / 减号（白色线）=====
    # 横线
    draw.line(
        (x - radius + 3, y, x + radius - 3, y),
        fill=sign_color,
        width=line_w,
    )

    # 竖线（只有 positive 画）
    if point_type.lower() == "positive":
        draw.line(
            (x, y - radius + 3, x, y + radius - 3),
            fill=sign_color,
            width=line_w,
        )

    return img
