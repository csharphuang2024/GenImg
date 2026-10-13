from PIL import Image, ImageDraw, ImageChops, ImageFilter

text = """
#0fffff
line 28 36 22 42 #000000 3
line 36 38 40 48 #000000 3
ellipse 32 30 9 6 #000000
line 29 30 27 40 #000000 3
line 35 30 37 40 #000000 3
ellipse 32 21 6 4 #000000
tri 29 17 #000000
dot #000000
tri 34 17 32 11 34 15 #000000
"""

OUT = "output.png"
PAD = 64
MARGIN = 4
TARGET = 2048
BLUR = 0.4

OUTLINE = True
OUTLINE_WIDTH = 3
OUTLINE_BLUR = 1.0
OUTLINE_COLOR = (0, 0, 0)

W = H = 64
bg = "#ffffff"
shapes, dots = [], []

for line in text.strip().splitlines():
    line = line.strip().strip("`")
    p = line.split()
    if not p:
        continue
    if p[0] == "canvas" and len(p) >= 4:
        W, H, bg = int(p[1]), int(p[2]), p[3]
    elif p[0] == "dot":
        dots.append(p)
    elif p[0] in ("ellipse", "tri", "line", "rect"):
        shapes.append(p)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def fill(s):
    return hex_to_rgb(s.split(",")[0])


# ---------- 0. 内容判定：指令里有没有形状 ----------
has_content = len(shapes) > 0 or len(dots) > 0
if not has_content:
    raise SystemExit("指令里没有任何形状，检查输入")

# ---------- 1. 画形状（背景实心，形状覆盖在上面）----------
bg_rgb = hex_to_rgb(bg)
big = Image.new("RGB", (W + 2 * PAD, H + 2 * PAD), bg_rgb)
d = ImageDraw.Draw(big)
for p in shapes:
    try:
        k = p[0]
        if k == "ellipse":
            cx, cy, rx, ry = [float(v) for v in p[1:5]]
            cx += PAD
            cy += PAD
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill(p[5]))
        elif k == "rect":
            x1, y1, x2, y2 = [float(v) + PAD for v in p[1:5]]
            d.rectangle([x1, y1, x2, y2], fill=fill(p[5]))
        elif k == "tri":
            v = [float(t) + PAD for t in p[1:7]]
            d.polygon(v, fill=fill(p[7]))
        elif k == "line":
            v = [float(t) + PAD for t in p[1:5]]
            w = int(p[6]) if len(p) > 6 else 1
            d.line(v, fill=fill(p[5]), width=w)
    except Exception as e:
        print("跳过出错的行:", " ".join(p), e)

# ---------- 2. 裁剪：用"画布中心"作为裁剪中心 ----------
cx0 = W // 2 + PAD
cy0 = H // 2 + PAD
side = max(W, H) + 2 * MARGIN
left = cx0 - side // 2
top = cy0 - side // 2
img = big.crop((left, top, left + side, top + side))
print("裁剪后尺寸:", img.size)

# ---------- 3. 画 dot ----------
d2 = ImageDraw.Draw(img)
for p in dots:
    try:
        x = int(float(p[1]) + PAD - left)
        y = int(float(p[2]) + PAD - top)
        d2.point((x, y), fill=fill(p[3]))
    except Exception as e:
        print("跳过出错的 dot:", " ".join(p), e)

# ---------- 4. 输出到目标尺寸 ----------
side_now = img.size[0]
if side_now != TARGET:
    img = img.resize((TARGET, TARGET), Image.LANCZOS)
if BLUR > 0:
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR))

# ---------- 5. 形态学膨胀描边 ----------
if OUTLINE and OUTLINE_WIDTH > 0:
    mask = Image.new("L", img.size, 0)
    mp = mask.load()
    ip = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            px = ip[x, y]
            if abs(px[0]-bg_rgb[0]) + abs(px[1]-bg_rgb[1]) + abs(px[2]-bg_rgb[2]) >= 40:
                mp[x, y] = 255

    dilated = mask.filter(ImageFilter.MaxFilter(2 * OUTLINE_WIDTH + 1))
    edge = ImageChops.subtract(dilated, mask)
    if OUTLINE_BLUR > 0:
        edge = edge.filter(ImageFilter.GaussianBlur(radius=OUTLINE_BLUR))

    outline_layer = Image.new("RGB", img.size, OUTLINE_COLOR)
    img = Image.composite(outline_layer, img, edge)

img.save(OUT)
print("已保存", OUT, img.size)