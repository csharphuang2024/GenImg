import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def parse_color(value):
    value = value.strip().lower()
    if value == "transparent":
        return (0, 0, 0, 0)
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(channel * 2 for channel in value)
    if len(value) == 6:
        value += "ff"
    if len(value) != 8:
        raise ValueError(f"无效颜色: {value}")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4, 6))


def parse_instructions(text):
    canvas = (64, 64, "transparent")
    commands = []
    for raw_line in text.replace("<bos>", "").replace("<eos>", "").splitlines():
        for raw_command in raw_line.split("|"):
            parts = raw_command.strip().strip("`").split()
            if not parts:
                continue
            kind = parts[0].lower()
            if kind == "canvas" and len(parts) >= 4:
                canvas = (int(parts[1]), int(parts[2]), parts[3])
            elif kind in {"ellipse", "tri", "line", "rect", "dot", "curve"}:
                commands.append(parts)
    if not commands:
        raise ValueError("没有找到可绘制的 ellipse、tri、line、rect、dot 或 curve 指令")
    width, height, background = canvas
    if width < 1 or height < 1 or width > 4096 or height > 4096:
        raise ValueError("画布尺寸必须在 1 到 4096 之间")
    return canvas, commands


def validate_instructions(text):
    rows = [part.split() for line in text.splitlines() for part in line.split("|") if part.strip()]
    if not rows or rows[0][0] != "canvas":
        raise ValueError("完整指令必须以 canvas 开始")
    counts = {"canvas": (4,), "ellipse": (6,), "rect": (6,), "tri": (8,),
              "line": (6, 7), "dot": (4, 5), "curve": (12,)}
    color_indices = {"canvas": 3, "ellipse": 5, "rect": 5, "tri": 7, "line": 5, "dot": 3, "curve": 11}
    for index, parts in enumerate(rows):
        kind = parts[0]
        if kind not in counts or len(parts) not in counts[kind] or (kind == "canvas" and index != 0):
            raise ValueError("未知、重复或参数数量错误的绘图指令")
        color_index = color_indices[kind]
        parse_color(parts[color_index])
        numbers = [float(value) for position, value in enumerate(parts[1:], start=1) if position != color_index]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("绘图参数必须是有限数值")
        if kind == "ellipse" and min(numbers[2:4]) <= 0:
            raise ValueError("椭圆半径必须大于 0")
        if kind == "rect" and (numbers[0] > numbers[2] or numbers[1] > numbers[3]):
            raise ValueError("矩形边界顺序错误")
        if kind == "curve" and min(numbers[-2:]) <= 0:
            raise ValueError("曲线半径必须大于 0")
        if kind in {"line", "dot"} and len(parts) == max(counts[kind]) and numbers[-1] <= 0:
            raise ValueError("线宽或点半径必须大于 0")
    return parse_instructions(text)


def render(text, output=None, size=1024, outline_width=0.8, background=None, *, fit_content=True):
    (width, height, canvas_background), commands = parse_instructions(text)
    if size < 64 or size > 4096:
        raise ValueError("输出尺寸必须在 64 到 4096 之间")

    supersample = 2
    scale = min(size / width, size / height) * supersample
    image_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
    silhouette = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(layer)
    mask_draw = ImageDraw.Draw(silhouette)

    def point(x, y):
        return round(float(x) * scale), round(float(y) * scale)

    def box(x1, y1, x2, y2):
        return (*point(x1, y1), *point(x2, y2))

    def rounded_stroke(control_points, start_radius, end_radius, color):
        if start_radius <= 0 or end_radius <= 0:
            raise ValueError("曲线半径必须大于 0")
        if not all(math.isfinite(value) for value in (*control_points, start_radius, end_radius)):
            raise ValueError("曲线参数必须是有限数值")
        start_x, start_y, control1_x, control1_y, control2_x, control2_y, end_x, end_y = control_points
        samples = []
        for index in range(65):
            progress = index / 64
            remaining = 1 - progress
            center_x = remaining**3 * start_x + 3 * remaining**2 * progress * control1_x + 3 * remaining * progress**2 * control2_x + progress**3 * end_x
            center_y = remaining**3 * start_y + 3 * remaining**2 * progress * control1_y + 3 * remaining * progress**2 * control2_y + progress**3 * end_y
            radius = start_radius + (end_radius - start_radius) * progress
            samples.append((center_x, center_y, radius))

        left_edge, right_edge = [], []
        for index, (center_x, center_y, radius) in enumerate(samples):
            previous = samples[max(0, index - 1)]
            following = samples[min(len(samples) - 1, index + 1)]
            tangent_x, tangent_y = following[0] - previous[0], following[1] - previous[1]
            length = math.hypot(tangent_x, tangent_y)
            normal_x, normal_y = (-tangent_y / length, tangent_x / length) if length else (0, 1)
            left_edge.append(point(center_x + normal_x * radius, center_y + normal_y * radius))
            right_edge.append(point(center_x - normal_x * radius, center_y - normal_y * radius))
        polygon = left_edge + list(reversed(right_edge))
        draw.polygon(polygon, fill=color)
        mask_draw.polygon(polygon, fill=255)
        for center_x, center_y, radius in (samples[0], samples[-1]):
            bounds = box(center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            draw.ellipse(bounds, fill=color)
            mask_draw.ellipse(bounds, fill=255)

    for parts in commands:
        kind = parts[0].lower()
        try:
            if kind == "ellipse":
                x, y, rx, ry = map(float, parts[1:5])
                color = parse_color(parts[5])
                bounds = box(x - rx, y - ry, x + rx, y + ry)
                draw.ellipse(bounds, fill=color)
                mask_draw.ellipse(bounds, fill=255)
            elif kind == "rect":
                x1, y1, x2, y2 = map(float, parts[1:5])
                color = parse_color(parts[5])
                bounds = box(x1, y1, x2, y2)
                draw.rectangle(bounds, fill=color)
                mask_draw.rectangle(bounds, fill=255)
            elif kind == "tri":
                values = list(map(float, parts[1:7]))
                color = parse_color(parts[7])
                points = [point(values[index], values[index + 1]) for index in range(0, 6, 2)]
                draw.polygon(points, fill=color)
                mask_draw.polygon(points, fill=255)
            elif kind == "line":
                x1, y1, x2, y2 = map(float, parts[1:5])
                color = parse_color(parts[5])
                line_width = max(1, round(float(parts[6] if len(parts) > 6 else 1) * scale))
                points = [point(x1, y1), point(x2, y2)]
                draw.line(points, fill=color, width=line_width, joint="curve")
                mask_draw.line(points, fill=255, width=line_width, joint="curve")
                radius = float(parts[6] if len(parts) > 6 else 1) / 2
                for center_x, center_y in ((x1, y1), (x2, y2)):
                    bounds = box(center_x - radius, center_y - radius, center_x + radius, center_y + radius)
                    draw.ellipse(bounds, fill=color)
                    mask_draw.ellipse(bounds, fill=255)
            elif kind == "curve":
                control_points = list(map(float, parts[1:9]))
                start_radius, end_radius = map(float, parts[9:11])
                color = parse_color(parts[11])
                rounded_stroke(control_points, start_radius, end_radius, color)
            elif kind == "dot":
                x, y = map(float, parts[1:3])
                color = parse_color(parts[3])
                radius = float(parts[4]) if len(parts) > 4 else 1.0
                bounds = box(x - radius, y - radius, x + radius, y + radius)
                draw.ellipse(bounds, fill=color)
                mask_draw.ellipse(bounds, fill=255)
        except (IndexError, TypeError, ValueError) as error:
            print(f"跳过无效指令 {' '.join(parts)}: {error}", file=sys.stderr)

    if outline_width > 0:
        outline_radius = max(1, round(outline_width * scale))
        kernel = outline_radius * 2 + 1
        if kernel % 2 == 0:
            kernel += 1
        expanded = silhouette.filter(ImageFilter.MaxFilter(kernel))
        outline = Image.new("RGBA", image_size, (25, 27, 34, 255))
        outline.putalpha(expanded)
        layer = Image.alpha_composite(outline, layer)

    if canvas_background != "transparent":
        background_image = Image.new("RGBA", image_size, parse_color(canvas_background))
        layer = Image.alpha_composite(background_image, layer)
    if background is not None:
        background_image = Image.new("RGBA", image_size, parse_color(background))
        layer = Image.alpha_composite(background_image, layer)

    if fit_content and canvas_background == "transparent" and background is None:
        bounds = layer.getchannel("A").getbbox()
        if bounds:
            padding = round(size * 0.06 * supersample)
            left = max(0, bounds[0] - padding)
            top = max(0, bounds[1] - padding)
            right = min(image_size[0], bounds[2] + padding)
            bottom = min(image_size[1], bounds[3] + padding)
            layer = layer.crop((left, top, right, bottom))

    layer.thumbnail((size, size), Image.Resampling.LANCZOS)
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    final.alpha_composite(layer, ((size - layer.width) // 2, (size - layer.height) // 2))
    if output is not None:
        final.save(output)
    return final


def main():
    parser = argparse.ArgumentParser(description="将几何绘图指令渲染为高分辨率 PNG")
    parser.add_argument("--instructions", help="指令文本文件；不指定时从标准输入读取")
    parser.add_argument("--output", default="output.png", help="输出 PNG 路径")
    parser.add_argument("--size", type=int, default=1024, help="正方形输出尺寸，默认 1024")
    parser.add_argument("--outline-width", type=float, default=0.8, help="描边宽度，0 表示关闭")
    parser.add_argument("--background", help="覆盖画布背景色；transparent 表示透明")
    parser.add_argument("--fixed-canvas", action="store_true", help="保留画布坐标，不按内容裁剪，适合动画")
    args = parser.parse_args()

    if args.instructions:
        text = Path(args.instructions).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("请通过 --instructions 文件或标准输入提供绘图指令")
    try:
        image = render(text, args.output, args.size, args.outline_width, args.background,
                   fit_content=not args.fixed_canvas)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"已保存 {args.output} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()