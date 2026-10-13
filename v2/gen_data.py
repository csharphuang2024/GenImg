import itertools
import math
from pathlib import Path

from elements import iter_element_samples, iter_person_samples

# ================== 参数池 ==================
COLORS = {
    "橘色": "#e8923f",
    "黑色": "#000000",
    "白色": "#ffffff",
    "灰色": "#808080",
    "棕色": "#8b4513",
    "蓝色": "#3f7fe8",
    "红色": "#e83f3f",
    "绿色": "#3fe87f",
    "黄色": "#e8d43f",
    "粉色": "#e83f9f",
}

ANIMALS = {
    "猫":  {"body": (10, 6), "head": 6, "ears": "point", "tail": "curl", "face": "whiskers"},
    "狗":  {"body": (11, 7), "head": 7, "ears": "flop", "tail": "wag", "face": "muzzle"},
    "兔子":{"body": (9, 6),  "head": 6, "ears": "long", "tail": "puff", "face": "muzzle"},
    "老鼠":{"body": (7, 4),  "head": 4, "ears": "round", "tail": "curl", "face": "whiskers"},
    "熊":  {"body": (12, 8), "head": 8, "ears": "round", "tail": "puff", "face": "muzzle"},
    "猪":  {"body": (11, 7), "head": 7, "ears": "flop", "tail": "curl", "face": "snout"},
    "狐狸":{"body": (10, 6), "head": 6, "ears": "point", "tail": "bushy", "face": "muzzle"},
    "鸭子":{"body": (9, 6),  "head": 5, "ears": "none", "tail": "none", "face": "beak"},
}

ACTIONS = {
    "站着": {"dy": 0,  "legs": [(32-4, 40, 32-6, 48), (32+4, 40, 32+6, 48)]},
    "坐着": {"dy": 4,  "legs": [(32-4, 42, 32-8, 48), (32+4, 42, 32+8, 48)]},
    "跳跃": {"dy": -6, "legs": [(32-4, 36, 32-10, 42), (32+4, 36, 32+10, 42)]},
    "奔跑": {"dy": -2, "legs": [(32-4, 38, 32-10, 46), (32+4, 38, 32+12, 42)]},
    "趴着": {"dy": 6,  "legs": [(32-6, 44, 32-10, 48), (32+6, 44, 32+10, 48)]},
}

CANVAS = 64
BG = "#ffffff"
CX, CY = 32, 32   # 固定中心


def make_command(animal, action, color_hex, *, pose=None):
    """Create a layered cartoon sprite from the supported geometric primitives."""
    info = ANIMALS[animal]
    act = ACTIONS[action]
    body_rx, body_ry = info["body"]
    head_r = info["head"]

    body_cy = CY + (act["dy"] if pose is None else pose["dy"])
    head_cy = body_cy - body_ry - head_r + 3

    def tint(hex_color, amount):
        channels = [int(hex_color[index:index + 2], 16) for index in (1, 3, 5)]
        mixed = [round(channel + (255 - channel) * amount) for channel in channels]
        return "#" + "".join(f"{channel:02x}" for channel in mixed)

    base = color_hex
    light = tint(base, 0.62)
    dark = "#" + "".join(
        f"{round(int(base[index:index + 2], 16) * 0.68):02x}" for index in (1, 3, 5)
    )
    pink = "#ef91a6"
    ink = "#292632"
    lines = [f"canvas {CANVAS} {CANVAS} transparent"]
    drawing_tail = False

    def curve(points, start_radius, end_radius, color):
        if drawing_tail and pose is not None:
            angle = pose["tail_sway"]
            rotated = []
            for index in range(0, len(points), 2):
                offset_x, offset_y = points[index]-tail_x, points[index+1]-tail_y
                rotated.extend((tail_x+offset_x*math.cos(angle)-offset_y*math.sin(angle),
                                tail_y+offset_x*math.sin(angle)+offset_y*math.cos(angle)))
            points = rotated
        values = " ".join(f"{value:.2f}" for value in (*points, start_radius, end_radius))
        lines.append(f"curve {values} {color}")

    def paw(center_x, center_y, side, color, rear=False):
        if animal == "鸭子":
            lines.append(f"tri {center_x} {center_y-1} {center_x-3} {center_y+2} {center_x+3} {center_y+2} #f4a62a")
            lines.append(f"ellipse {center_x} {center_y+1} 3 1.2 #f4a62a")
        elif animal == "猪":
            lines.append(f"ellipse {center_x} {center_y} 2 1.7 {dark}")
            lines.append(f"line {center_x} {center_y} {center_x} {center_y+1.5} {light} 0.4")
        else:
            radius_x = 3.2 if rear and animal == "兔子" else 2.4
            lines.append(f"ellipse {center_x} {center_y} {radius_x} 1.7 {color}")
            lines.append(f"ellipse {center_x-side*0.4} {center_y-0.5} {radius_x*0.7:.2f} 0.8 {light}")
            for offset in (-0.7, 0.7):
                lines.append(f"line {center_x+offset} {center_y+0.6} {center_x+offset} {center_y+1.2} {dark} 0.3")

    def limb(root, joint, foot, side, rear=False):
        root_x, root_y = root
        joint_x, joint_y = joint
        foot_x, foot_y = foot
        points = (root_x, root_y, joint_x, joint_y, joint_x, joint_y, foot_x, foot_y-0.7)
        if rear:
            lines.append(f"ellipse {root_x} {root_y+1} 2.8 3.2 {dark}")
        limb_color = "#f4a62a" if animal == "鸭子" else dark
        curve(points, 2.2 if rear else 1.8, 1.0, limb_color)
        if not rear:
            curve(points, 1.35, 0.75, base)
        paw(foot_x, foot_y, side, base, rear)

    for root_x, unused_y, foot_x, foot_y in (act["legs"] if pose is None else pose["legs"]):
        side = -1 if root_x < CX else 1
        root_y = body_cy + body_ry * 0.4
        joint = (root_x + side * 3, (root_y + foot_y) / 2)
        limb((root_x, root_y), joint, (foot_x, foot_y), side, rear=True)

    tail_x = CX + body_rx - 2
    tail_y = body_cy + 1
    drawing_tail = True
    if animal == "猫":
        curve((tail_x, tail_y, tail_x+7, tail_y+3, tail_x+10, tail_y-4, tail_x+9, tail_y-8), 2.4, 1.4, dark)
        curve((tail_x+9, tail_y-8, tail_x+9, tail_y-12, tail_x+5, tail_y-13, tail_x+5, tail_y-9), 1.4, 0.6, base)
    elif animal == "老鼠":
        curve((tail_x, tail_y, tail_x+7, tail_y+6, tail_x+14, tail_y+1, tail_x+12, tail_y-4), 1.2, 0.7, pink)
        curve((tail_x+12, tail_y-4, tail_x+11, tail_y-8, tail_x+7, tail_y-7, tail_x+8, tail_y-4), 0.7, 0.3, pink)
    elif animal == "猪":
        curve((tail_x, tail_y, tail_x+4, tail_y+1, tail_x+8, tail_y-1, tail_x+8, tail_y-4), 1.2, 0.9, pink)
        curve((tail_x+8, tail_y-4, tail_x+8, tail_y-8, tail_x+3, tail_y-8, tail_x+3, tail_y-4), 0.9, 0.7, pink)
        curve((tail_x+3, tail_y-4, tail_x+3, tail_y-1, tail_x+6, tail_y-1, tail_x+6, tail_y-4), 0.7, 0.4, pink)
    elif info["tail"] == "wag":
        curve((tail_x, tail_y, tail_x+5, tail_y+1, tail_x+9, tail_y-6, tail_x+7, tail_y-10), 2.8, 0.5, dark)
    elif info["tail"] == "bushy":
        curve((tail_x, tail_y, tail_x+4, tail_y+4, tail_x+13, tail_y, tail_x+10, tail_y-8), 2.5, 4.0, dark)
        curve((tail_x+10, tail_y-8, tail_x+9, tail_y-11, tail_x+6, tail_y-12, tail_x+6, tail_y-14), 4.0, 0.4, light)
    elif info["tail"] == "puff":
        lines.append(f"ellipse {tail_x+3} {tail_y-2} 3.2 3 {base}")
        lines.append(f"ellipse {tail_x+3.4} {tail_y-3} 2.2 1.8 {light}")

    drawing_tail = False
    if info["ears"] == "point":
        lines.append(f"tri {CX-head_r+1} {head_cy-head_r+2} {CX-head_r+2} {head_cy-head_r-5} {CX-1} {head_cy-head_r+1} {base}")
        lines.append(f"tri {CX+head_r-1} {head_cy-head_r+2} {CX+head_r-2} {head_cy-head_r-5} {CX+1} {head_cy-head_r+1} {base}")
        lines.append(f"tri {CX-head_r+2} {head_cy-head_r} {CX-head_r+3} {head_cy-head_r-3} {CX-3} {head_cy-head_r} {pink}")
        lines.append(f"tri {CX+head_r-2} {head_cy-head_r} {CX+head_r-3} {head_cy-head_r-3} {CX+3} {head_cy-head_r} {pink}")
    elif info["ears"] == "long":
        lines.append(f"ellipse {CX-3} {head_cy-head_r-3} 2.5 8 {base}")
        lines.append(f"ellipse {CX+3} {head_cy-head_r-3} 2.5 8 {base}")
        lines.append(f"ellipse {CX-3} {head_cy-head_r-3} 1 6 {pink}")
        lines.append(f"ellipse {CX+3} {head_cy-head_r-3} 1 6 {pink}")
    elif info["ears"] == "round":
        lines.append(f"ellipse {CX-head_r+1} {head_cy-head_r+1} 3 3 {base}")
        lines.append(f"ellipse {CX+head_r-1} {head_cy-head_r+1} 3 3 {base}")
        lines.append(f"ellipse {CX-head_r+1} {head_cy-head_r+1} 1.5 1.5 {pink}")
        lines.append(f"ellipse {CX+head_r-1} {head_cy-head_r+1} 1.5 1.5 {pink}")
    elif info["ears"] == "flop":
        lines.append(f"ellipse {CX-head_r-2} {head_cy+1} 3 5 {dark}")
        lines.append(f"ellipse {CX+head_r+2} {head_cy+1} 3 5 {dark}")

    lines.append(f"ellipse {CX} {body_cy} {body_rx} {body_ry} {base}")
    lines.append(f"ellipse {CX} {body_cy+1} {body_rx*0.48:.1f} {body_ry*0.68:.1f} {light}")

    if animal in ("猫", "熊", "狐狸"):
        lines.append(f"ellipse {CX-5} {body_cy-1} 1.5 2 {dark}")
        lines.append(f"ellipse {CX+5} {body_cy-1} 1.5 2 {dark}")
    elif animal == "狗":
        lines.append(f"ellipse {CX-4} {body_cy-2} 2 2.5 {light}")

    for side in (-1, 1):
        root = (CX + side * body_rx * 0.65, body_cy - 1)
        if animal == "鸭子":
            lines.append(f"ellipse {root[0]} {body_cy+1} 2.5 4 {dark}")
            lines.append(f"ellipse {root[0]-side*0.4} {body_cy} 1.5 2.8 {base}")
            continue
        if action == "跳跃":
            foot = (CX + side * (body_rx + 3), body_cy - 3)
            joint = (root[0] + side * 4, body_cy + 2)
        elif action == "奔跑":
            foot = (CX + side * 12, body_cy + (8 if side < 0 else -3))
            joint = (root[0] + side * 3, body_cy + 3)
        elif action == "趴着":
            foot = (CX + side * (body_rx + 2), body_cy + body_ry + 2)
            joint = (root[0] + side * 4, body_cy + body_ry)
        else:
            foot = (CX + side * 6, body_cy + body_ry + (2 if action == "坐着" else 5))
            joint = (root[0] + side * 2, body_cy + body_ry * 0.7)
        if pose is not None:
            front = pose["front"][0 if side < 0 else 1]
            foot, joint = front["foot"], front["joint"]
        limb(root, joint, foot, side)

    if info["ears"] == "none":
        lines.append(f"ellipse {CX} {head_cy+head_r-1} {head_r*0.8:.1f} {head_r*0.38:.1f} {light}")
    else:
        lines.append(f"ellipse {CX} {head_cy} {head_r} {head_r} {base}")

    if info["face"] == "beak":
        lines.append(f"tri {CX-2} {head_cy+2} {CX+5} {head_cy+3} {CX-2} {head_cy+5} #f4a62a")
    elif info["face"] == "snout":
        lines.append(f"ellipse {CX} {head_cy+2} 3.5 2.5 {pink}")
        lines.append(f"dot {CX-1} {head_cy+2} {ink} 0.7")
        lines.append(f"dot {CX+1} {head_cy+2} {ink} 0.7")
    elif info["face"] == "muzzle":
        lines.append(f"ellipse {CX-2} {head_cy+2} 2.3 1.8 {light}")
        lines.append(f"ellipse {CX+2} {head_cy+2} 2.3 1.8 {light}")
        lines.append(f"ellipse {CX} {head_cy+1} 1.1 0.9 {ink}")
    else:
        lines.append(f"ellipse {CX} {head_cy+2} 1.2 0.9 {ink}")

    for eye_x in (CX-2.5, CX+2.5):
        lines.append(f"ellipse {eye_x} {head_cy-1} 1.25 1.6 #ffffff")
        lines.append(f"dot {eye_x} {head_cy-0.7} {ink} 0.8")
        lines.append(f"dot {eye_x-0.25} {head_cy-1} #ffffff 0.3")
    lines.append(f"line {CX} {head_cy+2} {CX} {head_cy+3.5} {ink} 1")
    lines.append(f"line {CX} {head_cy+3.5} {CX-1.5} {head_cy+4} {ink} 1")
    lines.append(f"line {CX} {head_cy+3.5} {CX+1.5} {head_cy+4} {ink} 1")
    lines.append(f"ellipse {CX-4.5} {head_cy+2} 1.5 0.8 {pink}")
    lines.append(f"ellipse {CX+4.5} {head_cy+2} 1.5 0.8 {pink}")

    if info["face"] == "whiskers":
        for side in (-1, 1):
            for offset in (-1, 1):
                start_x = CX + side * 4
                end_x = CX + side * 8
                y = head_cy + 2 + offset
                lines.append(f"line {start_x} {y} {end_x} {y+offset} {dark} 1")

    return "|".join(lines)


def iter_samples():
    for animal, action, color_name in itertools.product(ANIMALS.keys(), ACTIONS.keys(), COLORS.keys()):
        color_hex = COLORS[color_name]
        cmd = make_command(animal, action, color_hex)
        prompt = f"{color_name}{animal}{action}"
        yield prompt, cmd
    yield from iter_element_samples(COLORS)
    yield from iter_person_samples(COLORS)


def generate(out="data.txt"):
    rows = [f"<bos>{prompt}<sep>{cmd}<eos>" for prompt, cmd in iter_samples()]

    output_path = Path(out)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"已生成 {len(rows)} 条数据 -> {output_path}")


if __name__ == "__main__":
    generate("data.txt")