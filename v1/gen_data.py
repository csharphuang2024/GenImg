import itertools

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
    "猫":  {"body": (10, 6),  "head": 6,  "ear": True},
    "狗":  {"body": (11, 7),  "head": 7,  "ear": True},
    "兔子":{"body": (9, 6),   "head": 6,  "ear": True},
    "老鼠":{"body": (7, 4),   "head": 4,  "ear": True},
    "熊":  {"body": (12, 8),  "head": 8,  "ear": True},
    "猪":  {"body": (11, 7),  "head": 7,  "ear": True},
    "狐狸":{"body": (10, 6),  "head": 6,  "ear": True},
    "鸭子":{"body": (9, 6),   "head": 5,  "ear": False},
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


def make_command(animal, action, color_hex):
    """根据动物、动作、颜色生成几何指令（结构完全固定）"""
    info = ANIMALS[animal]
    act = ACTIONS[action]
    body_rx, body_ry = info["body"]
    head_r = info["head"]

    body_cy = CY + act["dy"]
    head_cy = body_cy - body_ry - head_r + 3

    lines = [f"canvas {CANVAS} {CANVAS} {BG}"]

    # 后腿（固定 2 条）
    for (x1, y1, x2, y2) in act["legs"]:
        lines.append(f"line {x1} {y1} {x2} {y2} {color_hex} 3")

    # 身体（固定 1 个椭圆）
    lines.append(f"ellipse {CX} {body_cy} {body_rx} {body_ry} {color_hex}")

    # 前腿（固定 2 条）
    lines.append(f"line {CX-3} {body_cy} {CX-5} {body_cy+body_ry+4} {color_hex} 3")
    lines.append(f"line {CX+3} {body_cy} {CX+5} {body_cy+body_ry+4} {color_hex} 3")

    # 头（固定 1 个圆）
    lines.append(f"ellipse {CX} {head_cy} {head_r} {head_r} {color_hex}")

    # 耳朵（有的话固定 2 个三角）
    if info["ear"]:
        lines.append(f"tri {CX-head_r+2} {head_cy-head_r+2} {CX-head_r+6} {head_cy-head_r-4} {CX-2} {head_cy-head_r} {color_hex}")
        lines.append(f"tri {CX+head_r-2} {head_cy-head_r+2} {CX+head_r-6} {head_cy-head_r-4} {CX+2} {head_cy-head_r} {color_hex}")

    # 眼睛、鼻子（固定）
    lines.append(f"dot {CX-3} {head_cy-2} #000000")
    lines.append(f"dot {CX+3} {head_cy-2} #000000")
    lines.append(f"dot {CX} {head_cy+2} #000000")

    return "|".join(lines)


def generate(out="data.txt"):
    rows = []
    for animal, action, color_name in itertools.product(ANIMALS.keys(), ACTIONS.keys(), COLORS.keys()):
        color_hex = COLORS[color_name]
        cmd = make_command(animal, action, color_hex)
        # 统一句式：颜色 + 动物 + 动作
        prompt = f"{color_name}{animal}{action}"
        rows.append(f"<bos>{prompt}<sep>{cmd}<eos>")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"已生成 {len(rows)} 条数据 -> {out}")


if __name__ == "__main__":
    generate("data.txt")