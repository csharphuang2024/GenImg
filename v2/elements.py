import itertools
import math
import random
import re


ELEMENTS = {
    "花": ("五瓣花", "郁金香", "雏菊"),
    "草": ("草丛", "长草"),
    "树": ("阔叶树", "松树", "枯树"),
    "木": ("原木", "木板"),
    "房屋": ("小屋", "楼房", "城堡"),
    "汽车": ("小汽车", "货车"),
    "河流": ("直河", "弯河"),
    "太阳": ("晴日", "夕阳"),
    "月亮": ("弯月", "满月"),
    "星星": ("五角星", "四角星"),
    "云": ("积云", "雨云"),
    "山": ("青山", "雪山"),
    "石头": ("圆石", "岩石"),
    "蘑菇": ("圆顶蘑菇", "斑点蘑菇"),
}

IDENTITIES = ("男孩", "女孩", "男人", "女人")
EXPRESSIONS = ("开心", "平静", "难过", "生气", "惊讶", "眨眼", "苦笑")
HUMAN_OPTIONS = {
    "hair": ("短发", "长发", "马尾", "卷发"),
    "hair_color": ("黑发", "棕发", "金发"),
    "skin": ("浅肤", "麦肤", "深肤"),
    "top": ("短袖", "长袖", "外套"),
    "bottom": ("长裤", "短裤"),
    "shoes": ("运动鞋", "靴子"),
    "build": ("普通身材", "纤细身材", "壮实身材"),
    "eyes": ("大眼睛", "小眼睛"),
    "face": ("圆脸", "长脸"),
    "nose": ("小鼻子", "圆鼻子"),
    "neck": ("普通脖子", "细脖子", "粗脖子"),
    "hands": ("自然手", "张开手", "握拳"),
    "feet": ("自然脚姿", "并脚", "分脚"),
}
SKIN_COLORS = {"浅肤": "#f5c9a6", "麦肤": "#c98c61", "深肤": "#895a42"}
HAIR_COLORS = {"黑发": "#28252e", "棕发": "#694331", "金发": "#e4b453"}
POSES = {
    "站着": {"elbows": ((49, 65), (79, 65)), "wrists": ((50, 79), (78, 79)), "knees": ((58, 94), (70, 94)), "ankles": ((57, 110), (71, 110)), "dy": 0},
    "坐着": {"elbows": ((49, 67), (79, 67)), "wrists": ((53, 78), (75, 78)), "knees": ((45, 87), (83, 87)), "ankles": ((46, 104), (82, 104)), "dy": 5},
    "奔跑": {"elbows": ((43, 61), (84, 43)), "wrists": ((53, 64), (74, 38)), "knees": ((50, 91), (82, 91)), "ankles": ((39, 107), (90, 81)), "dy": 0},
    "跳跃": {"elbows": ((44, 40), (84, 40)), "wrists": ((39, 25), (89, 25)), "knees": ((48, 92), (80, 92)), "ankles": ((40, 106), (88, 106)), "dy": -5},
    "挥手": {"elbows": ((49, 65), (86, 43)), "wrists": ((50, 79), (85, 26)), "knees": ((58, 94), (70, 94)), "ankles": ((57, 110), (71, 110)), "dy": 0},
    "行走": {"elbows": ((48, 65), (80, 62)), "wrists": ((46, 77), (75, 74)), "knees": ((53, 94), (76, 91)), "ankles": ((47, 110), (80, 106)), "dy": 0},
    "抱臂": {"elbows": ((47, 65), (81, 67)), "wrists": ((72, 61), (56, 62)), "knees": ((58, 94), (70, 94)), "ankles": ((57, 110), (71, 110)), "dy": 0},
    "叉腰": {"elbows": ((44, 64), (84, 64)), "wrists": ((55, 71), (73, 71)), "knees": ((56, 94), (72, 94)), "ankles": ((53, 110), (75, 110)), "dy": 0},
    "蹲下": {"elbows": ((47, 64), (81, 64)), "wrists": ((46, 77), (82, 77)), "knees": ((45, 94), (83, 94)), "ankles": ((57, 105), (71, 105)), "dy": 7},
}
INK = "#34323e"


def shade(color, amount):
    channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    if amount >= 0:
        channels = [round(channel + (255 - channel) * amount) for channel in channels]
    else:
        channels = [round(channel * (1 + amount)) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


class Drawing:
    def __init__(self):
        self.commands = ["canvas 128 128 transparent"]

    def add(self, kind, values, color):
        coordinates = " ".join(f"{value:.2f}" for value in values)
        self.commands.append(f"{kind} {coordinates} {color}")

    def ellipse(self, center_x, center_y, radius_x, radius_y, color):
        self.add("ellipse", (center_x, center_y, radius_x, radius_y), color)

    def rect(self, left, top, right, bottom, color):
        self.add("rect", (left, top, right, bottom), color)

    def polygon(self, points, color):
        for index in range(1, len(points) - 1):
            self.add("tri", (*points[0], *points[index], *points[index + 1]), color)

    def line(self, start, end, color, width=1):
        coordinates = " ".join(f"{value:.2f}" for value in (*start, *end))
        self.commands.append(f"line {coordinates} {color} {width:.2f}")

    def curve(self, points, start_radius, end_radius, color):
        self.add("curve", (*points, start_radius, end_radius), color)

    def segment(self, start, end, start_radius, end_radius, color):
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        self.curve((*start, start[0] + delta_x / 3, start[1] + delta_y / 3,
                    end[0] - delta_x / 3, end[1] - delta_y / 3, *end), start_radius, end_radius, color)

    def text(self):
        return "|".join(self.commands)


def make_element(subject, variant, color):
    if subject not in ELEMENTS or variant not in ELEMENTS[subject]:
        raise ValueError("不支持的元素或造型")
    drawing = Drawing()
    light, dark = shade(color, 0.4), shade(color, -0.25)
    green, leaf = "#398951", "#7fc866"
    wood, window = "#ad7651", "#b5e5ed"

    if subject == "花":
        drawing.curve((64, 105, 55, 86, 73, 67, 64, 48), 3, 2, green)
        drawing.curve((63, 85, 52, 73, 40, 72, 35, 68), 5, 0.5, leaf)
        drawing.curve((63, 91, 77, 78, 88, 80, 93, 74), 5, 0.5, green)
        if variant == "郁金香":
            drawing.ellipse(64, 48, 19, 17, color)
            drawing.polygon(((45, 47), (45, 26), (58, 36), (64, 23), (71, 36), (83, 26), (83, 47)), color)
            drawing.curve((64, 58, 56, 48, 57, 38, 60, 34), 1, 0.5, light)
        else:
            petals = 5 if variant == "五瓣花" else 10
            for index in range(petals):
                angle = 2 * math.pi * index / petals - math.pi / 2
                drawing.ellipse(64 + 16 * math.cos(angle), 43 + 16 * math.sin(angle), 10, 11, color)
            drawing.ellipse(64, 43, 10, 10, "#f6d25e")
            drawing.ellipse(62, 40, 4, 3, "#fff0af")
    elif subject == "草":
        for index in range(7):
            start_x = 40 + index * 8
            height = 42 + (index % 3) * (12 if variant == "长草" else 7)
            end_x = start_x + (index - 3) * 4
            drawing.curve((start_x, 106, start_x-5, 89, end_x-6, 116-height, end_x, 106-height), 4, 0.4, color if index % 2 else dark)
    elif subject == "树":
        drawing.curve((64, 108, 60, 84, 66, 59, 64, 36), 7, 4, wood)
        if variant != "松树":
            drawing.segment((62, 69), (44, 44), 3, 1, wood)
            drawing.segment((66, 59), (86, 40), 3, 1, wood)
        if variant == "松树":
            for top, bottom, half_width in ((15, 55, 23), (34, 78, 33), (55, 96, 42)):
                drawing.polygon(((64, top), (64-half_width, bottom), (64+half_width, bottom)), color)
                drawing.line((64-half_width+8, bottom-4), (64+half_width-8, bottom-4), dark, 1)
        elif variant == "阔叶树":
            for center_x, center_y, radius in ((44, 48, 24), (81, 49, 25), (63, 28, 24), (63, 56, 25)):
                drawing.ellipse(center_x, center_y, radius, radius*0.88, color)
            drawing.ellipse(49, 34, 12, 9, light)
            drawing.ellipse(83, 49, 8, 6, light)
        else:
            drawing.segment((43, 45), (40, 25), 2, 0.6, wood)
            drawing.segment((85, 41), (97, 26), 2, 0.6, wood)
        drawing.curve((62, 106, 58, 91, 64, 85, 61, 78), 0.8, 0.5, "#744a36")
    elif subject == "木":
        if variant == "原木":
            drawing.rect(30, 46, 93, 85, color)
            drawing.ellipse(30, 65, 15, 20, dark)
            drawing.ellipse(93, 65, 15, 20, "#dfb47c")
            for radius in (12, 8, 4):
                drawing.curve((93-radius, 66, 93-radius, 48, 93+radius, 48, 93+radius, 66), 0.7, 0.7, dark)
                drawing.curve((93+radius, 66, 93+radius, 83, 93-radius, 83, 93-radius, 66), 0.7, 0.7, dark)
            drawing.line((40, 54), (78, 56), light, 1.5)
            drawing.line((38, 75), (77, 72), dark, 1)
        else:
            drawing.rect(21, 42, 106, 87, color)
            for height in (52, 63, 77):
                drawing.curve((26, height, 47, height-4, 76, height+4, 100, height), 0.7, 0.7, dark)
            drawing.ellipse(74, 65, 7, 3, dark)
            drawing.ellipse(74, 65, 4, 1.5, light)
    elif subject == "房屋":
        if variant == "城堡":
            drawing.rect(24, 38, 103, 107, color)
            for center_x in (29, 97):
                drawing.rect(center_x-10, 29, center_x+10, 107, light)
                for offset in (-7, 3):
                    drawing.rect(center_x+offset, 23, center_x+offset+5, 35, light)
            drawing.ellipse(64, 88, 10, 13, dark)
            drawing.rect(54, 88, 74, 107, dark)
        else:
            drawing.rect(28, 51 if variant == "小屋" else 26, 100, 106, color)
            if variant == "小屋":
                drawing.rect(82, 26, 91, 50, wood)
                drawing.polygon(((19, 55), (64, 20), (109, 55)), dark)
                drawing.line((27, 51), (64, 24), light, 1.3)
            else:
                drawing.rect(25, 22, 103, 29, dark)
            drawing.rect(58, 78, 73, 106, wood)
            drawing.ellipse(69, 92, 1, 1, "#f2d071")
        for center_x in (42, 85):
            for center_y in ((65, 43) if variant == "楼房" else (66,)):
                drawing.rect(center_x-6, center_y-7, center_x+6, center_y+7, window)
                drawing.line((center_x, center_y-7), (center_x, center_y+7), "#f5f5ed", 1)
                drawing.line((center_x-6, center_y), (center_x+6, center_y), "#f5f5ed", 1)
    elif subject == "汽车":
        drawing.rect(16, 58, 112, 87, color)
        if variant == "小汽车":
            drawing.polygon(((31, 60), (44, 35), (83, 35), (99, 60)), color)
            drawing.polygon(((39, 58), (48, 40), (63, 40), (63, 58)), window)
            drawing.polygon(((68, 40), (80, 40), (91, 58), (68, 58)), window)
        else:
            drawing.rect(18, 32, 76, 71, dark)
            drawing.polygon(((78, 58), (78, 40), (99, 40), (111, 58)), color)
            drawing.polygon(((82, 55), (82, 44), (97, 44), (104, 55)), window)
        drawing.line((18, 84), (111, 84), dark, 2)
        drawing.rect(104, 64, 113, 69, "#ffed9d")
        drawing.line((68, 65), (74, 65), dark, 1)
        for center_x in (35, 94):
            drawing.ellipse(center_x, 87, 13, 13, INK)
            drawing.ellipse(center_x, 87, 7, 7, "#d8dce2")
            drawing.ellipse(center_x, 87, 3, 3, "#818b96")
    elif subject == "河流":
        controls = (57, 24, 98, 46, 23, 74, 61, 98) if variant == "弯河" else (63, 24, 64, 44, 64, 83, 64, 98)
        drawing.curve(controls, 16, 24, "#7aab72")
        drawing.curve(controls, 12, 20, color)
        for center_x, center_y in ((65, 39), (57, 64), (52, 88)):
            drawing.curve((center_x-7, center_y, center_x-2, center_y-2, center_x+3, center_y+2, center_x+9, center_y), 0.7, 0.7, light)
    elif subject == "太阳":
        for index in range(12):
            angle = index * math.pi / 6
            drawing.line((64+34*math.cos(angle), 64+34*math.sin(angle)), (64+44*math.cos(angle), 64+44*math.sin(angle)), color, 3)
        drawing.ellipse(64, 64, 27, 27, color)
        drawing.ellipse(57, 56, 12, 9, light)
        if variant == "夕阳":
            drawing.curve((39, 78, 52, 83, 78, 83, 89, 78), 2, 2, "#e77b51")
    elif subject == "月亮":
        if variant == "满月":
            drawing.ellipse(64, 64, 33, 33, color)
            for center_x, center_y, radius in ((49, 52, 7), (79, 77, 9), (55, 86, 4)):
                drawing.ellipse(center_x, center_y, radius, radius*0.7, dark)
        else:
            outer = [(64+34*math.cos(math.radians(angle)), 64+34*math.sin(math.radians(angle))) for angle in range(65, 296, 10)]
            inner = [(80+24*math.cos(math.radians(angle)), 64+30*math.sin(math.radians(angle))) for angle in range(255, 104, -10)]
            anchor = (42, 64)
            boundary = outer + inner
            for start, end in zip(boundary, boundary[1:] + boundary[:1]):
                drawing.polygon((anchor, start, end), color)
    elif subject == "星星":
        corners = 5 if variant == "五角星" else 4
        points = []
        for index in range(corners*2):
            radius = 39 if index % 2 == 0 else 17
            angle = index*math.pi/corners-math.pi/2
            points.append((64+radius*math.cos(angle), 64+radius*math.sin(angle)))
        for start, end in zip(points, points[1:] + points[:1]):
            drawing.polygon(((64, 64), start, end), color)
        drawing.ellipse(56, 53, 4, 3, light)
    elif subject == "云":
        for center_x, center_y, radius in ((37, 65, 21), (59, 48, 24), (86, 61, 24)):
            drawing.ellipse(center_x, center_y, radius, radius*0.8, color)
        drawing.rect(34, 61, 92, 80, color)
        if variant == "雨云":
            for center_x in (42, 64, 86):
                drawing.segment((center_x, 92), (center_x-4, 104), 1.7, 1, "#62add4")
    elif subject == "山":
        drawing.polygon(((16, 107), (49, 38), (85, 107)), dark)
        drawing.polygon(((40, 107), (82, 21), (117, 107)), color)
        if variant == "雪山":
            drawing.polygon(((82, 21), (66, 54), (78, 49), (86, 57), (95, 51)), "#f5f7fa")
    elif subject == "石头":
        if variant == "圆石":
            drawing.ellipse(64, 71, 36, 28, color)
        else:
            drawing.polygon(((23, 90), (30, 55), (57, 34), (91, 47), (108, 86), (85, 103), (40, 105)), color)
            drawing.polygon(((57, 34), (68, 66), (91, 47)), light)
        drawing.curve((40, 84, 58, 91, 82, 86, 92, 78), 1.3, 0.8, dark)
    elif subject == "蘑菇":
        drawing.curve((63, 104, 61, 88, 66, 72, 64, 56), 10, 7, "#e8d8b3")
        drawing.ellipse(64, 51, 37, 25, color)
        drawing.ellipse(64, 67, 35, 7, dark)
        if variant == "斑点蘑菇":
            for center_x, center_y, radius in ((44, 47, 6), (68, 37, 5), (84, 51, 7)):
                drawing.ellipse(center_x, center_y, radius, radius*0.7, "#fff4df")
    return drawing.text()


def default_person():
    return {
        "subject": "人物", "identity": "男孩", "action": "站着", "expression": "开心",
        "hair": "短发", "hair_color": "棕发", "skin": "浅肤", "top": "短袖",
        "bottom": "长裤", "shoes": "运动鞋", "build": "普通身材", "eyes": "大眼睛",
        "face": "圆脸", "nose": "小鼻子", "top_color": "蓝色", "bottom_color": "灰色",
        "shoe_color": "白色", "neck": "普通脖子", "hands": "自然手", "feet": "自然脚姿",
    }


def person_prompt(spec):
    keys = ("identity", "action", "expression", *HUMAN_OPTIONS)
    parts = ["人物", *(spec[key] for key in keys)]
    parts.extend((spec["top_color"]+"上衣", spec["bottom_color"]+"裤子", spec["shoe_color"]+"鞋子"))
    return " ".join(parts)


def make_person(spec, palette, *, pose=None, face_state=None, hair_sway=0):
    drawing = Drawing()
    for key, options in HUMAN_OPTIONS.items():
        if spec[key] not in options:
            raise ValueError(f"不支持的人物参数: {key}")
    if spec["identity"] not in IDENTITIES or spec["action"] not in POSES or spec["expression"] not in EXPRESSIONS:
        raise ValueError("不支持的人物类型、动作或表情")
    pose = POSES[spec["action"]] if pose is None else pose
    skin = SKIN_COLORS[spec["skin"]]
    skin_shadow = shade(skin, -0.17)
    top, trousers, shoes = (palette[spec[key]] for key in ("top_color", "bottom_color", "shoe_color"))
    hair = HAIR_COLORS[spec["hair_color"]]
    body_scale = {"普通身材": 1, "纤细身材": 0.84, "壮实身材": 1.18}[spec["build"]]
    child = spec["identity"] in ("男孩", "女孩")
    head_radius_x = 15 if child else 12
    head_radius_y = 17 if child else 14
    if spec["face"] == "长脸":
        head_radius_x -= 1.5
        head_radius_y += 1.5
    head_y = 28 + pose["dy"]

    def location(point):
        return 64 + (point[0]-64)*body_scale, point[1]+pose["dy"]

    def segment(start, end, start_radius, end_radius, color):
        drawing.segment(location(start), location(end), start_radius*body_scale, end_radius*body_scale, color)

    def joint(point, radius, color):
        drawing.ellipse(*location(point), radius*body_scale, radius, color)

    def hand(wrist, elbow, side):
        wrist_x, wrist_y = location(wrist)
        elbow_x, elbow_y = location(elbow)
        distance = math.hypot(wrist_x-elbow_x, wrist_y-elbow_y)
        direction_x, direction_y = (wrist_x-elbow_x)/distance, (wrist_y-elbow_y)/distance
        normal_x, normal_y = -direction_y, direction_x
        palm_x, palm_y = wrist_x+direction_x*2, wrist_y+direction_y*2
        drawing.ellipse(palm_x, palm_y, 2.4, 3, skin)
        open_hand = spec["hands"] == "张开手" or (
            spec["hands"] == "自然手" and spec["action"] in ("挥手", "跳跃")
        )
        if open_hand:
            for finger_index in range(4):
                spread = (finger_index-1.5)*1.3
                start = (palm_x+normal_x*spread, palm_y+normal_y*spread)
                reach = 4.2-abs(finger_index-1.5)*0.7
                end = (start[0]+direction_x*reach+normal_x*spread*0.25,
                       start[1]+direction_y*reach+normal_y*spread*0.25)
                drawing.segment(start, end, 0.65, 0.45, skin)
        elif spec["hands"] == "自然手":
            for spread in (-0.9, 0, 0.9):
                start = (palm_x+normal_x*spread, palm_y+normal_y*spread)
                drawing.curve((*start, start[0]+direction_x*2, start[1]+direction_y*2,
                               start[0]+direction_x*3+normal_x*0.6, start[1]+direction_y*3+normal_y*0.6,
                               start[0]+direction_x*3, start[1]+direction_y*3), 0.6, 0.4, skin)
        else:
            drawing.ellipse(palm_x+direction_x, palm_y+direction_y, 2.5, 2.4, skin)
            for offset in (-0.8, 0.8):
                drawing.line((palm_x+offset, palm_y+1), (palm_x+offset, palm_y+2), skin_shadow, 0.35)
        drawing.segment((palm_x+side*1.4, palm_y), (palm_x+side*3.2, palm_y+direction_y*1.3), 0.9, 0.6, skin)

    for side_index, side in enumerate((-1, 1)):
        hip = (58 if side < 0 else 70, 78)
        knee, ankle = pose["knees"][side_index], pose["ankles"][side_index]
        stance_scale = {"自然脚姿": 1, "并脚": 0.5, "分脚": 1.3}[spec["feet"]]
        knee = (64+(knee[0]-64)*stance_scale, knee[1])
        ankle = (64+(ankle[0]-64)*stance_scale, ankle[1])
        long_trousers = spec["bottom"] == "长裤"
        segment(hip, knee, 4.3, 3.2, trousers)
        if not long_trousers:
            segment(((hip[0]+knee[0])/2, (hip[1]+knee[1])/2), knee, 2.8, 2.6, skin)
        segment(knee, ankle, 3.1 if long_trousers else 2.6, 2.3, trousers if long_trousers else skin)
        joint(knee, 2.9, trousers if long_trousers else skin_shadow)
        crease_x, crease_y = location(knee)
        drawing.curve((crease_x-2, crease_y-1, crease_x, crease_y-2, crease_x+1, crease_y, crease_x+2, crease_y), 0.35, 0.25, shade(trousers, 0.35) if long_trousers else skin)
        ankle_x, ankle_y = location(ankle)
        if spec["shoes"] == "靴子":
            drawing.segment((ankle_x, ankle_y-6), (ankle_x, ankle_y+2), 3, 2.8, shoes)
            drawing.line((ankle_x-2.5, ankle_y-5), (ankle_x+2.5, ankle_y-5), shade(shoes, -0.3), 0.7)
        drawing.ellipse(ankle_x+side*2, ankle_y+3, 5.6, 3, shoes)
        drawing.ellipse(ankle_x-side, ankle_y+1, 2.7, 2.8, shoes)
        drawing.line((ankle_x-side*3, ankle_y+5), (ankle_x+side*7, ankle_y+5), INK, 0.8)
        if spec["shoes"] == "运动鞋":
            for offset in (-0.5, 1):
                drawing.line((ankle_x+offset-1, ankle_y+1), (ankle_x+offset+1, ankle_y+2), shade(shoes, -0.4), 0.4)

    if spec["hair"] == "长发":
        drawing.ellipse(64, head_y+8, head_radius_x+3, head_radius_y+7, hair)
    elif spec["hair"] == "马尾":
        drawing.curve((64+head_radius_x, head_y-6, 90+hair_sway*0.3, head_y-15, 88+hair_sway, head_y+15, 91+hair_sway, head_y+21), 5, 1.3, hair)
        drawing.ellipse(64+head_radius_x+1, head_y-4, 2.4, 2, shade(top, 0.3))

    neck_y = head_y + head_radius_y
    neck_scale = {"普通脖子": 1, "细脖子": 0.75, "粗脖子": 1.2}[spec["neck"]]
    drawing.segment((64, neck_y-3), (64, 53+pose["dy"]), 3.2*neck_scale, 3.6*neck_scale, skin_shadow)
    drawing.segment((63.3, neck_y-2), (63.3, 52+pose["dy"]), 2.3*neck_scale, 2.8*neck_scale, skin)
    drawing.polygon(tuple(location(point) for point in ((53, 49), (75, 49), (73, 61), (73, 79), (55, 79), (55, 61))), top)
    drawing.line(location((55, 78)), location((73, 78)), shade(top, -0.25), 1)
    drawing.curve((58, 50+pose["dy"], 60, 55+pose["dy"], 68, 55+pose["dy"], 70, 50+pose["dy"]), 0.7, 0.7, shade(top, 0.6))
    if spec["top"] == "外套":
        drawing.line(location((64, 55)), location((64, 77)), shade(top, -0.4), 0.8)
        for height in (60, 66, 72):
            drawing.ellipse(*location((66, height)), 0.7, 0.7, "#efdf9a")
        for pocket_x in (58, 70):
            drawing.line(location((pocket_x-2, 69)), location((pocket_x+2, 69)), shade(top, -0.3), 0.5)
    else:
        drawing.ellipse(*location((61, 62)), 2.5, 2.5, shade(top, 0.4))
        drawing.line(location((68, 72)), location((71, 75)), shade(top, -0.2), 0.5)

    for side_index, side in enumerate((-1, 1)):
        shoulder = (54 if side < 0 else 74, 52)
        elbow, wrist = pose["elbows"][side_index], pose["wrists"][side_index]
        joint(shoulder, 3.6, top)
        segment(shoulder, elbow, 3.3, 2.7, skin if spec["top"] == "短袖" else top)
        segment(elbow, wrist, 2.7, 1.6, skin if spec["top"] == "短袖" else top)
        joint(elbow, 2.7, skin if spec["top"] == "短袖" else top)
        if spec["top"] == "短袖":
            sleeve_end = (shoulder[0]+(elbow[0]-shoulder[0])*0.43, shoulder[1]+(elbow[1]-shoulder[1])*0.43)
            segment(shoulder, sleeve_end, 3.5, 3.2, top)
        else:
            segment((wrist[0], wrist[1]-0.5), wrist, 2.0, 1.9, shade(top, 0.45))
        elbow_x, elbow_y = location(elbow)
        drawing.line((elbow_x-0.8, elbow_y-0.5), (elbow_x+1.2, elbow_y+0.8), skin_shadow if spec["top"] == "短袖" else shade(top, -0.25), 0.4)
        hand(wrist, elbow, side)

    for side in (-1, 1):
        drawing.ellipse(64+side*head_radius_x, head_y+1, 2.8, 4, skin_shadow)
        drawing.ellipse(64+side*head_radius_x, head_y+1, 1.5, 2.4, skin)
    drawing.ellipse(64, head_y, head_radius_x, head_radius_y, skin)
    drawing.ellipse(64-head_radius_x*0.55, head_y+6, 2.1, 1.2, shade(skin_shadow, 0.3))
    drawing.ellipse(64+head_radius_x*0.55, head_y+6, 2.1, 1.2, shade(skin_shadow, 0.3))

    eye_radius = 2.4 if spec["eyes"] == "大眼睛" else 1.7
    expression = spec["expression"]
    for side in (-1, 1):
        eye_x, eye_y = 64+side*5, head_y+1
        closure = max(face_state.get("blink", 0), face_state.get("wink", 0) if side > 0 else 0) if face_state is not None else (1 if expression == "眨眼" and side > 0 else 0)
        if closure > 0.95:
            drawing.curve((eye_x-2.5, eye_y, eye_x-1, eye_y-1.5, eye_x+1, eye_y-1.5, eye_x+2.5, eye_y), 0.55, 0.55, INK)
        else:
            surprise = face_state["surprise"] if face_state is not None else (1 if expression == "惊讶" else 0)
            eye_height = max(0.2, (eye_radius*1.3 + 0.7*surprise) * (1-closure))
            drawing.ellipse(eye_x, eye_y, eye_radius, eye_height, "#fffaf5")
            drawing.ellipse(eye_x, eye_y+0.25, eye_radius*0.53, eye_height*0.7, "#72513e")
            drawing.ellipse(eye_x, eye_y+0.4, eye_radius*0.3, eye_height*0.47, INK)
            drawing.ellipse(eye_x-0.5, eye_y-0.6, 0.5, 0.6, "#ffffff")
            drawing.curve((eye_x-eye_radius, eye_y-1, eye_x-1, eye_y-eye_height, eye_x+1, eye_y-eye_height, eye_x+eye_radius, eye_y-1), 0.4, 0.4, INK)
        brow_y = eye_y - 5
        if face_state is not None:
            slope = -side*1.7*face_state["anger"] + side*1.4*face_state["sad"]
        else:
            slope = -side*1.7 if expression == "生气" else side*1.4 if expression in ("难过", "苦笑") else 0
        drawing.curve((eye_x-2.7, brow_y-slope, eye_x-1, brow_y-0.7, eye_x+1, brow_y-0.7, eye_x+2.7, brow_y+slope), 0.6, 0.4, hair)

    if spec["nose"] == "圆鼻子":
        drawing.ellipse(64, head_y+5, 1.7, 1.4, skin_shadow)
        drawing.ellipse(63.7, head_y+4.6, 0.8, 0.5, skin)
    else:
        drawing.curve((64, head_y+2, 62.5, head_y+4, 62.7, head_y+5.5, 64.7, head_y+5.5), 0.4, 0.3, skin_shadow)
    mouth_y = head_y + 9
    if face_state is not None:
        opening = face_state["open"]
        if opening > 0.05:
            mouth_radius = 3.6 - 1.3*face_state["surprise"]
            mouth_height = max(0.25, opening*2.5 + face_state["surprise"]*0.5)
            drawing.ellipse(64, mouth_y, mouth_radius, mouth_height, "#79444c")
            drawing.ellipse(64, mouth_y+mouth_height*0.55, mouth_radius*0.6, max(0.15, mouth_height*0.3), "#e68f97")
        else:
            smile = face_state["smile"]
            drawing.curve((61, mouth_y, 62, mouth_y+2*smile, 66, mouth_y+2*smile, 67, mouth_y), 0.45, 0.45, "#79444c")
    elif expression in ("开心", "眨眼"):
        drawing.ellipse(64, mouth_y, 3.6, 2.3, "#79444c")
        drawing.ellipse(64, mouth_y+1, 2.1, 0.8, "#e68f97")
        drawing.line((61.4, mouth_y-0.8), (66.6, mouth_y-0.8), "#fff8ed", 0.7)
    elif expression == "惊讶":
        drawing.ellipse(64, mouth_y, 2.3, 3, "#79444c")
        drawing.ellipse(64, mouth_y+1.6, 1.3, 0.6, "#e68f97")
    elif expression == "苦笑":
        drawing.curve((60.5, mouth_y-0.5, 62, mouth_y+1.6, 66, mouth_y+2, 67.5, mouth_y+0.6), 0.5, 0.4, "#79444c")
    elif expression == "难过":
        drawing.curve((60.5, mouth_y+1, 62, mouth_y-2, 66, mouth_y-2, 67.5, mouth_y+1), 0.5, 0.5, "#79444c")
    else:
        drawing.curve((61, mouth_y, 63, mouth_y+0.4, 65, mouth_y+0.4, 67, mouth_y), 0.45, 0.45, "#79444c")

    drawing.ellipse(64, head_y-head_radius_y+3, head_radius_x+0.8, 6, hair)
    if spec["hair"] == "卷发":
        for index in range(7):
            angle = math.pi + index*math.pi/6
            drawing.ellipse(64+head_radius_x*math.cos(angle), head_y+head_radius_y*math.sin(angle), 4, 4, hair)
    else:
        drawing.curve((64-head_radius_x+1, head_y-9, 54, head_y-13, 63, head_y-8, 59, head_y-5), 3.8, 0.7, hair)
        drawing.curve((64+head_radius_x-1, head_y-9, 73, head_y-13, 68, head_y-7, 73, head_y-5), 3, 0.7, hair)
    if spec["hair"] == "长发":
        for side in (-1, 1):
            drawing.curve((64+side*head_radius_x, head_y-6, 64+side*(head_radius_x+2), head_y+3, 64+side*(head_radius_x+3), head_y+12, 64+side*(head_radius_x+1), head_y+20), 2.6, 1, hair)
    drawing.curve((57, head_y-head_radius_y+2, 62, head_y-head_radius_y, 67, head_y-head_radius_y, 70, head_y-head_radius_y+2), 0.5, 0.4, shade(hair, 0.3))
    return drawing.text()


def iter_element_samples(palette):
    for subject, variants in ELEMENTS.items():
        for variant, color_name in itertools.product(variants, palette):
            yield f"{color_name}{subject}{variant}", make_element(subject, variant, palette[color_name])


def iter_person_samples(palette):
    generator = random.Random(20261013)
    color_names = tuple(palette)
    for identity, action, expression, color_name in itertools.product(IDENTITIES, POSES, EXPRESSIONS, color_names):
        spec = default_person()
        spec.update(identity=identity, action=action, expression=expression, top_color=color_name)
        for key, options in HUMAN_OPTIONS.items():
            spec[key] = generator.choice(options)
        spec["bottom_color"] = generator.choice(color_names)
        spec["shoe_color"] = generator.choice(color_names)
        yield person_prompt(spec), make_person(spec, palette)


ELEMENT_ALIASES = {
    "花": ("花朵", "鲜花"), "草": ("草地",), "树": ("树木", "大树"),
    "木": ("木头",), "房屋": ("房子", "住宅"), "汽车": ("车子", "轿车"),
    "河流": ("小河", "河水"), "太阳": (), "月亮": ("月球",), "星星": ("星形",),
    "云": ("云朵",), "山": ("山峰",), "石头": ("石块",), "蘑菇": (),
}
DEFAULT_ELEMENT_COLORS = {
    "花": "粉色", "草": "绿色", "树": "绿色", "木": "棕色", "房屋": "橘色",
    "汽车": "蓝色", "河流": "蓝色", "太阳": "黄色", "月亮": "黄色", "星星": "黄色",
    "云": "白色", "山": "绿色", "石头": "灰色", "蘑菇": "红色",
}
PERSON_ACTION_ALIASES = {
    "站立": "站着", "站姿": "站着", "坐下": "坐着", "坐姿": "坐着",
    "跑步": "奔跑", "跑着": "奔跑", "跳起来": "跳跃", "打招呼": "挥手",
    "走路": "行走", "交叉手臂": "抱臂", "蹲着": "蹲下",
}
EXPRESSION_ALIASES = {
    "微笑": "开心", "高兴": "开心", "快乐": "开心", "伤心": "难过",
    "愤怒": "生气", "吃惊": "惊讶", "无表情": "平静",
}


def parse_new_prompt(prompt, palette):
    text = "".join(prompt.split())
    candidates = []
    for subject, variants in ELEMENTS.items():
        for name in (subject, *variants, *ELEMENT_ALIASES[subject]):
            for match in re.finditer(re.escape(name), text):
                candidates.append((match.start(), match.end(), subject))
    selected = []
    for start, end, subject in sorted(candidates, key=lambda item: item[1]-item[0], reverse=True):
        if not any(left <= start and end <= right for left, right, _ in selected):
            selected.append((start, end, subject))
    subjects = {subject for _, _, subject in selected}
    is_person = "人" in text or any(name in text for name in (*IDENTITIES, "男生", "女生"))
    if len(subjects) > 1 or (is_person and subjects):
        raise ValueError("目前一次仅支持一个人物或元素；请将多个主体分别生成")
    if not is_person and not subjects:
        return None

    color_aliases = {**{name: name for name in palette}, "橙色": "橘色", "粉红色": "粉色"}
    color_pattern = "|".join(re.escape(name) for name in sorted(color_aliases, key=len, reverse=True))
    if not is_person:
        subject = next(iter(subjects))
        match = re.search(color_pattern, text)
        color_name = color_aliases[match.group()] if match else DEFAULT_ELEMENT_COLORS[subject]
        variants = [variant for variant in ELEMENTS[subject] if variant in text]
        if len(variants) > 1:
            raise ValueError("请一次仅指定一种元素造型")
        variant = variants[0] if variants else ELEMENTS[subject][0]
        return f"{color_name}{subject}{variant}", {
            "subject": subject, "variant": variant, "color": color_name,
        }

    spec = default_person()
    spec["identity"] = next((name for name in IDENTITIES if name in text), "女孩" if "女生" in text else "男孩")
    if spec["identity"] in ("女孩", "女人"):
        spec["hair"] = "马尾"
    spec["action"] = next((name for name in POSES if name in text), None)
    if spec["action"] is None:
        spec["action"] = next((value for name, value in PERSON_ACTION_ALIASES.items() if name in text), "站着")
    spec["expression"] = next((name for name in EXPRESSIONS if name in text), None)
    if spec["expression"] is None:
        spec["expression"] = next((value for name, value in EXPRESSION_ALIASES.items() if name in text), "开心")
    for key, options in HUMAN_OPTIONS.items():
        spec[key] = next((value for value in options if value in text), spec[key])
    for name, value in (("浅色皮肤", "浅肤"), ("小麦色皮肤", "麦肤"), ("深色皮肤", "深肤")):
        if name in text:
            spec["skin"] = value
    for name, value in (("黑色", "黑发"), ("棕色", "棕发"), ("金色", "金发")):
        if re.search(re.escape(name)+r"(?:的)?(?:头发|短发|长发|马尾|卷发)", text):
            spec["hair_color"] = value
    generic_color = re.search(r"("+color_pattern+r")(?:的)?(?:人物|男孩|女孩|男人|女人|人)", text)
    if generic_color:
        spec["top_color"] = color_aliases[generic_color.group(1)]
    garment_names = {
        "top_color": "上衣|衣服|短袖|长袖|外套",
        "bottom_color": "裤子|长裤|短裤",
        "shoe_color": "鞋子|运动鞋|靴子|鞋",
    }
    for key, garment in garment_names.items():
        match = re.search(r"("+color_pattern+r")(?:的)?(?:"+garment+r")", text)
        if match is None:
            match = re.search(r"(?:"+garment+r")(?:颜色|为|是|[:：])?("+color_pattern+r")", text)
        if match:
            spec[key] = color_aliases[match.group(1)]
    return person_prompt(spec), spec