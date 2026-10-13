import copy
import math

from elements import POSES


ACTION_NAMES = {
    "static": "站着", "idle": "站着", "stand": "站着", "walk": "行走",
    "run": "奔跑", "jump": "跳跃", "wave": "挥手", "sit": "坐着",
    "crouch": "蹲下", "arms_crossed": "抱臂", "hands_on_hips": "叉腰",
    "wind": "摇摆",
}
ACTION_ALIASES = {**{value: key for key, value in ACTION_NAMES.items()},
                  "站着": "idle", "走路": "walk", "跑步": "run", "摇晃": "wind"}
EMOTIONS = {
    "开心": {"smile": 1, "open": 0.7, "anger": 0, "sad": 0, "surprise": 0, "wink": 0},
    "平静": {"smile": 0, "open": 0, "anger": 0, "sad": 0, "surprise": 0, "wink": 0},
    "难过": {"smile": -1, "open": 0, "anger": 0, "sad": 1, "surprise": 0, "wink": 0},
    "生气": {"smile": -0.2, "open": 0, "anger": 1, "sad": 0, "surprise": 0, "wink": 0},
    "惊讶": {"smile": 0, "open": 1, "anger": 0, "sad": 0, "surprise": 1, "wink": 0},
    "眨眼": {"smile": 1, "open": 0.7, "anger": 0, "sad": 0, "surprise": 0, "wink": 1},
    "苦笑": {"smile": 0.4, "open": 0.2, "anger": 0, "sad": 0.7, "surprise": 0, "wink": 0},
}


def normalize_action(action):
    normalized = ACTION_ALIASES.get(action, action)
    if normalized not in ACTION_NAMES:
        raise ValueError(f"不支持动画动作 {action}；暂不支持 360 度转身或自动打闹编排")
    return normalized


def smoothstep(progress):
    progress = max(0.0, min(1.0, progress))
    return progress * progress * (3 - 2 * progress)


def solve_joint(start, target, upper_length, lower_length, bend):
    delta_x, delta_y = target[0] - start[0], target[1] - start[1]
    distance = math.hypot(delta_x, delta_y)
    direction = (delta_x / distance, delta_y / distance) if distance else (0, 1)
    reach = max(abs(upper_length - lower_length) + 1e-5,
                min(distance, upper_length + lower_length - 1e-5))
    endpoint = (start[0] + direction[0] * reach, start[1] + direction[1] * reach)
    along = (upper_length**2 - lower_length**2 + reach**2) / (2 * reach)
    perpendicular = math.sqrt(max(0.0, upper_length**2 - along**2)) * bend
    joint = (start[0] + direction[0] * along - direction[1] * perpendicular,
             start[1] + direction[1] * along + direction[0] * perpendicular)
    return joint, endpoint


def human_pose(action, phase):
    action = normalize_action(action)
    angle = 2 * math.pi * (phase % 1)
    if action in {"static", "sit", "crouch", "arms_crossed", "hands_on_hips"}:
        return copy.deepcopy(POSES[ACTION_NAMES[action]])
    if action == "wind":
        raise ValueError("人物不能使用 wind 动作")
    pose = {"elbows": [], "wrists": [], "knees": [], "ankles": [], "dy": 0.0}
    if action == "run":
        pose["dy"] = -1.3 * (1 - math.cos(2 * angle))
    elif action == "walk":
        pose["dy"] = -0.5 * (1 - math.cos(2 * angle))
    elif action == "jump":
        pose["dy"] = -6 * (1 - math.cos(angle))
    else:
        pose["dy"] = -0.25 * math.sin(angle)

    for side in (-1, 1):
        leg_angle = angle + (math.pi if side > 0 else 0)
        swing = math.sin(leg_angle)
        hip = (58 if side < 0 else 70, 78)
        if action in {"walk", "run"}:
            lift = (11 if action == "run" else 6) * max(0, swing)**2
            target = (hip[0] + side * (3 + 3 * math.cos(leg_angle)), 110 - pose["dy"] - lift)
        elif action == "jump":
            tuck = (1 - math.cos(angle)) / 2
            target = (hip[0] + side * (3 + 6 * tuck), 110 - 7 * tuck)
        else:
            target = (hip[0] + side, 110 - pose["dy"])
        knee, ankle = solve_joint(hip, target, 18, 18, -side)
        pose["knees"].append(knee)
        pose["ankles"].append(ankle)

        shoulder = (54 if side < 0 else 74, 52)
        if action in {"run", "walk"}:
            arm_swing = -swing
            amplitude = 9 if action == "run" else 4
            target = (shoulder[0] + side * (3 + max(0, arm_swing) * 4),
                      (69 if action == "run" else 76) - amplitude * arm_swing)
        elif action == "wave" and side > 0:
            target = (85 + 3 * math.sin(angle), 27 + math.cos(angle))
        elif action == "jump":
            lift = (1 - math.cos(angle)) / 2
            target = (shoulder[0] + side * (4 + 10 * lift), 78 - 52 * lift)
        else:
            target = (shoulder[0] + side * 4, 79)
        elbow, wrist = solve_joint(shoulder, target, 16, 15, -side)
        pose["elbows"].append(elbow)
        pose["wrists"].append(wrist)
    return pose


def blend_pose(first, second, weight):
    if weight <= 0:
        return copy.deepcopy(first)
    if weight >= 1:
        return copy.deepcopy(second)
    pose = {"dy": first["dy"] + (second["dy"] - first["dy"]) * weight}
    for middle_field, end_field, roots in (("elbows", "wrists", ((54, 52), (74, 52))),
                                          ("knees", "ankles", ((58, 78), (70, 78)))):
        pose[middle_field], pose[end_field] = [], []
        for index, root in enumerate(roots):
            blended_start = root
            for field, first_start, second_start in (
                    (middle_field, root, root),
                    (end_field, first[middle_field][index], second[middle_field][index])):
                first_vector = tuple(end-start for start, end in zip(first_start, first[field][index]))
                second_vector = tuple(end-start for start, end in zip(second_start, second[field][index]))
                first_angle = math.atan2(first_vector[1], first_vector[0])
                second_angle = math.atan2(second_vector[1], second_vector[0])
                difference = (second_angle-first_angle+math.pi) % (2*math.pi)-math.pi
                angle = first_angle+difference*weight
                length = math.hypot(*first_vector)+(math.hypot(*second_vector)-math.hypot(*first_vector))*weight
                endpoint = (blended_start[0]+length*math.cos(angle), blended_start[1]+length*math.sin(angle))
                pose[field].append(endpoint)
                blended_start = endpoint
    return pose


def blend_emotion(first, second, weight):
    return {key: first[key] + (second[key] - first[key]) * weight for key in first}


def blink_amount(time, offset=0):
    distance = (time + offset + 1.85) % 3.7 - 1.85
    return 0.5 + 0.5 * math.cos(math.pi * distance / 0.09) if abs(distance) < 0.09 else 0.0


def animal_pose(action, phase, body_rx, body_ry):
    action = normalize_action(action)
    if action not in {"idle", "walk", "run", "jump"}:
        raise ValueError("动物目前支持 idle、walk、run、jump 和静态姿态")
    angle = 2 * math.pi * (phase % 1)
    moving = action in {"run", "walk"}
    bounce = -0.6*(1-math.cos(2*angle)) if moving else -0.2*math.sin(angle)
    if action == "jump":
        bounce = -3*(1-math.cos(angle))
    pose = {"dy": bounce, "legs": [], "front": [], "tail_sway": 0.15*math.sin(angle)}
    for side in (-1, 1):
        leg_angle = angle + (math.pi if side > 0 else 0)
        stride = math.cos(leg_angle) if moving else 0
        lift = max(0, math.sin(leg_angle))**2 if moving else 0
        rear_x = 32+side*(6+2*stride)
        rear_y = 48-4*lift + (bounce if action == "jump" else 0)
        pose["legs"].append((32+side*4, 40, rear_x, rear_y))
        front_lift = max(0, -math.sin(leg_angle))**2 if moving else 0
        pose["front"].append({
            "foot": (32+side*(6-2*stride), 32+body_ry+5-3*front_lift+(bounce if action == "jump" else 0)),
            "joint": (32+side*(body_rx*0.65+2), 32+bounce+body_ry*0.7),
        })
    return pose