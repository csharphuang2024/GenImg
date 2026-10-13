import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from elements import EXPRESSIONS, POSES, make_element, make_person
from gen_data import ANIMALS, COLORS, make_command
from genimg import parse_color, parse_instructions, render, validate_instructions
from infer import canonicalize_prompt, template_command
from motion import (ACTION_NAMES, EMOTIONS, animal_pose, blend_emotion, blend_pose,
                    blink_amount, human_pose, normalize_action, smoothstep)


BASE_DIR = Path(__file__).resolve().parent
POINT_FIELDS = {"ellipse": (1,), "rect": (1, 3), "tri": (1, 3, 5),
                "line": (1, 3), "dot": (1,), "curve": (1, 3, 5, 7)}


def number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数值")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{label} 超出允许范围")
    return float(value)


def coordinates(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} 必须是 [x, y]")
    return tuple(number(component, label) for component in value)


def check_fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{label} 包含未知字段或不是对象")


def warp_instructions(text, transform, padding=0):
    (width, height, _), commands = parse_instructions(text)
    output = [f"canvas {width+2*padding} {height+2*padding} transparent"]
    for command in commands:
        parts = command.copy()
        for index in POINT_FIELDS[parts[0]]:
            point_x, point_y = transform(float(parts[index]), float(parts[index+1]))
            parts[index], parts[index+1] = f"{point_x+padding:.4f}", f"{point_y+padding:.4f}"
        output.append(" ".join(parts))
    return "|".join(output)


def blend_instructions(first, second, weight):
    canvas, first_commands = parse_instructions(first)
    _, second_commands = parse_instructions(second)
    if len(first_commands) != len(second_commands):
        raise ValueError("动作过渡必须使用相同角色拓扑")
    output = [f"canvas {canvas[0]} {canvas[1]} transparent"]
    for before, after in zip(first_commands, second_commands):
        if len(before) != len(after) or before[0] != after[0]:
            raise ValueError("动作过渡指令拓扑不匹配")
        parts = [after[0]]
        for left, right in zip(before[1:], after[1:]):
            parts.append(right if right.startswith("#") else f"{float(left)+(float(right)-float(left))*weight:.4f}")
        output.append(" ".join(parts))
    return "|".join(output)


class Scene:
    def __init__(self, definition, base_dir=BASE_DIR):
        self.definition = copy.deepcopy(definition)
        check_fields(self.definition, {"version", "width", "height", "fps", "duration", "characters", "background", "actors"}, "场景")
        if self.definition.get("version") != 1:
            raise ValueError("场景 version 必须是 1")
        self.width = self.integer(self.definition.get("width", 960), "width", 64, 2048)
        self.height = self.integer(self.definition.get("height", 540), "height", 64, 2048)
        self.fps = self.integer(self.definition.get("fps", 24), "fps", 1, 60)
        self.duration = number(self.definition.get("duration", 3), "duration", 0.1, 120)
        self.frame_count = math.ceil(self.duration*self.fps)
        if self.frame_count > 3600:
            raise ValueError("单次导出最多支持 3600 帧，请拆分镜头")
        self.base_dir = Path(base_dir)
        self.characters = {}
        self.ids = set()
        self.sprite_cache = {}
        definitions = self.definition.get("characters", {})
        if not isinstance(definitions, dict):
            raise ValueError("characters 必须是按角色 ID 索引的对象")
        for character_id, character in definitions.items():
            self.register_character(character_id, character)

        background = self.definition.get("background", {})
        check_fields(background, {"color", "ground_color", "horizon", "image", "items"}, "background")
        self.background = Image.new("RGBA", (self.width, self.height), parse_color(background.get("color", "#bfe6f5")))
        if "image" in background:
            source = Path(background["image"])
            if not source.is_absolute():
                source = self.base_dir/source
            with Image.open(source) as image:
                if image.size != self.background.size:
                    raise ValueError("背景图片尺寸必须与场景 width、height 一致")
                self.background = image.convert("RGBA")
        if "ground_color" in background:
            horizon = number(background.get("horizon", 0.78), "horizon", 0, 1)
            ImageDraw.Draw(self.background).rectangle((0, round(self.height*horizon), self.width, self.height), fill=parse_color(background["ground_color"]))
        items = background.get("items", [])
        actors = self.definition.get("actors", [])
        if not isinstance(items, list) or not isinstance(actors, list) or len(items)+len(actors) > 32:
            raise ValueError("items 和 actors 必须是数组，合计最多 32 个对象")
        for index, item in enumerate(items):
            actor = self.prepare_actor(item, f"background-{index}", background=True)
            self.place(self.background, actor, 0)
        self.actors = [self.prepare_actor(actor, f"actor-{index}") for index, actor in enumerate(actors)]
        if not self.actors:
            raise ValueError("场景至少需要一个 actor")
        self.background_hash = hashlib.sha256(self.background.tobytes()).hexdigest()

    @staticmethod
    def integer(value, label, minimum, maximum):
        numeric = number(value, label, minimum, maximum)
        if not numeric.is_integer():
            raise ValueError(f"{label} 必须是整数")
        return int(numeric)

    def register_character(self, character_id, definition):
        if not isinstance(character_id, str) or not character_id or character_id in self.characters:
            raise ValueError("角色 ID 必须是非空且不重复的字符串")
        check_fields(definition, {"prompt", "attributes"}, "角色定义")
        if "attributes" in definition:
            attributes = copy.deepcopy(definition["attributes"])
            if isinstance(attributes, list):
                attributes = tuple(attributes)
            canonical = definition.get("prompt", "")
        else:
            prompt = definition.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("角色定义需要 prompt 或已保存的 attributes")
            canonical, attributes = canonicalize_prompt(prompt)
        try:
            command = template_command(attributes)
            validate_instructions(command)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("角色属性缺失或无效") from error
        digest = hashlib.sha256(json.dumps(attributes, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        self.characters[character_id] = {"canonical": canonical, "attributes": attributes,
                                         "appearance_sha256": digest, "base_command": command}

    def prepare_actor(self, definition, default_id, background=False):
        check_fields(definition, {"id", "character", "prompt", "position", "height", "action", "rate", "phase", "path", "clips", "flip", "blink"}, "actor")
        actor_id = definition.get("id", default_id)
        if not isinstance(actor_id, str) or not actor_id or actor_id in self.ids:
            raise ValueError("actor ID 必须是非空且不重复的字符串")
        self.ids.add(actor_id)
        if ("character" in definition) == ("prompt" in definition):
            raise ValueError("actor 必须且只能指定 character 或 prompt")
        character_id = definition.get("character")
        if character_id is None:
            character_id = f"inline:{actor_id}"
            self.register_character(character_id, {"prompt": definition["prompt"]})
        if character_id not in self.characters:
            raise ValueError(f"找不到角色 {character_id}")
        attributes = self.characters[character_id]["attributes"]
        if isinstance(attributes, dict):
            kind = "person" if attributes["subject"] == "人物" else "element"
        else:
            kind = "animal"
        allowed = set(ACTION_NAMES)-{"wind"} if kind == "person" else {"static", "idle", "run", "walk", "jump"} if kind == "animal" else {"static"}
        if kind == "element" and attributes["subject"] in {"树", "草", "花"}:
            allowed.add("wind")
        inferred = "static"
        if kind == "person":
            inferred = normalize_action(attributes["action"])
        elif kind == "animal" and attributes[2] in ACTION_NAMES.values():
            inferred = normalize_action(attributes[2])
        default_action = inferred if inferred in allowed and not background else "static"
        action = normalize_action(definition.get("action", default_action))
        if action not in allowed or background and action != "static":
            raise ValueError(f"{kind} 不支持动作 {action}；摇摆的植物应放入 actors")
        if background and ("path" in definition or "clips" in definition):
            raise ValueError("背景项必须固定；运动对象应放入 actors")
        height = number(definition.get("height", self.height*0.45), "actor height", 8, 1024)
        position = coordinates(definition.get("position", [self.width*0.5, self.height*0.82]), "position")
        rate = number(definition.get("rate", 1), "rate", 0.05, 6)
        phase = number(definition.get("phase", 0), "phase")
        for field in ("flip", "blink"):
            if field in definition and not isinstance(definition[field], bool):
                raise ValueError(f"{field} 必须是布尔值")
        clips = definition.get("clips", [{"start": 0, "end": self.duration, "action": action}])
        if not isinstance(clips, list) or not clips:
            raise ValueError("clips 必须是非空数组")
        normalized_clips = []
        previous_end = 0.0
        for clip in clips:
            check_fields(clip, {"start", "end", "action", "expression", "rate", "transition"}, "clip")
            start = number(clip.get("start"), "clip start", 0, self.duration)
            end = number(clip.get("end"), "clip end", 0, self.duration)
            clip_action = normalize_action(clip.get("action", action))
            if not math.isclose(start, previous_end) or end <= start or clip_action not in allowed:
                raise ValueError("clips 必须按时间连续排列、无重叠无空档，且动作适用于此角色")
            expression = clip.get("expression", attributes["expression"] if kind == "person" else None)
            if "expression" in clip and kind != "person" or kind == "person" and expression not in EXPRESSIONS:
                raise ValueError("expression 仅支持人物的已知表情")
            normalized_clips.append({"start": start, "end": end, "action": clip_action, "expression": expression,
                                     "rate": number(clip.get("rate", rate), "clip rate", 0.05, 6),
                                     "transition": number(clip.get("transition", min(0.15, (end-start)/2)), "transition", 0, end-start)})
            previous_end = end
        if not math.isclose(previous_end, self.duration):
            raise ValueError("clips 必须覆盖场景完整 duration")
        path = definition.get("path", [])
        if not isinstance(path, list):
            raise ValueError("path 必须是数组")
        normalized_path = []
        previous_time = -1.0
        for key in path:
            check_fields(key, {"time", "position"}, "path key")
            time = number(key.get("time"), "path time", 0, self.duration)
            if time <= previous_time:
                raise ValueError("path 时间必须严格递增")
            normalized_path.append((time, coordinates(key.get("position"), "path position")))
            previous_time = time
        if normalized_path and (len(normalized_path) < 2 or normalized_path[0][0] != 0):
            raise ValueError("path 至少包含两个点，并从 time=0 开始")
        logical = 64 if kind == "animal" else 128
        anchor = (32, 50) if kind == "animal" else (64, 116) if kind == "person" else (64, 112)
        return {"id": actor_id, "character": character_id, "kind": kind, "height": height,
                "position": position, "path": normalized_path, "clips": normalized_clips,
                "phase": phase, "action": action, "rate": rate, "flip": definition.get("flip", False),
                "blink": definition.get("blink", not background and action != "static"),
                "logical": logical, "anchor": anchor}

    def state(self, actor, time):
        clips = actor["clips"]
        index = next((index for index, clip in enumerate(clips) if time < clip["end"]), len(clips)-1)
        clip = clips[index]
        phase = (time-clip["start"])*clip["rate"]+actor["phase"]
        weight = 1.0 if index == 0 or clip["transition"] == 0 else smoothstep((time-clip["start"])/clip["transition"])
        previous = clips[max(0, index-1)]
        previous_phase = (time-previous["start"])*previous["rate"]+actor["phase"]
        return clip, phase, previous, previous_phase, weight

    def command(self, actor, time):
        attributes = copy.deepcopy(self.characters[actor["character"]]["attributes"])
        clip, phase, previous, previous_phase, weight = self.state(actor, time)
        action = clip["action"]
        if actor["kind"] == "person":
            def pose_for(selected_action, selected_phase):
                return copy.deepcopy(POSES[attributes["action"]]) if selected_action == "static" else human_pose(selected_action, selected_phase)
            pose = pose_for(action, phase)
            emotion = EMOTIONS[clip["expression"]].copy()
            if weight < 1:
                pose = blend_pose(pose_for(previous["action"], previous_phase), pose, weight)
                emotion = blend_emotion(EMOTIONS[previous["expression"]], emotion, weight)
            offset = int(hashlib.sha256(actor["id"].encode("utf-8")).hexdigest()[:8], 16) % 300 / 100 + 0.2
            emotion["blink"] = blink_amount(time, offset) if actor["blink"] else 0
            attributes["expression"] = clip["expression"]
            return make_person(attributes, COLORS, pose=pose, face_state=emotion,
                               hair_sway=((0 if previous["action"] == "static" else 1.2*math.sin(2*math.pi*previous_phase))*(1-weight)
                                          +(0 if action == "static" else 1.2*math.sin(2*math.pi*phase))*weight))
        if actor["kind"] == "animal":
            color, animal, original_action = attributes
            def command_for(selected_action, selected_phase):
                if selected_action == "static":
                    return make_command(animal, original_action, COLORS[color])
                pose = animal_pose(selected_action, selected_phase, *ANIMALS[animal]["body"])
                return make_command(animal, "站着", COLORS[color], pose=pose)
            text = command_for(action, phase)
            return blend_instructions(command_for(previous["action"], previous_phase), text, weight) if weight < 1 else text
        text = make_element(attributes["subject"], attributes["variant"], COLORS[attributes["color"]])
        if action == "wind" or weight < 1 and previous["action"] == "wind":
            amplitude = ((4*math.sin(2*math.pi*previous_phase) if previous["action"] == "wind" else 0)*(1-weight)
                         +(4*math.sin(2*math.pi*phase) if action == "wind" else 0)*weight)
            text = warp_instructions(text, lambda point_x, point_y: (point_x+amplitude*(max(0, 108-point_y)/108)**2, point_y))
        return text

    def sprite(self, actor, time):
        cached = len(actor["clips"]) == 1 and actor["clips"][0]["action"] == "static" and not actor["blink"]
        if cached and actor["id"] in self.sprite_cache:
            return self.sprite_cache[actor["id"]]
        padding = actor["logical"]//8
        text = warp_instructions(self.command(actor, time), lambda point_x, point_y: (point_x, point_y), padding)
        tile_size = round(actor["height"]*(actor["logical"]+2*padding)/actor["logical"])
        image = render(text, size=max(64, tile_size), fit_content=False)
        if image.width != tile_size:
            image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        if actor["flip"]:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if cached:
            self.sprite_cache[actor["id"]] = image
        return image

    def position(self, actor, time):
        path = actor["path"]
        if not path:
            return actor["position"]
        for (start, first), (end, second) in zip(path, path[1:]):
            if time <= end:
                weight = max(0, (time-start)/(end-start))
                return tuple(left+(right-left)*weight for left, right in zip(first, second))
        return path[-1][1]

    def place(self, frame, actor, time):
        image = self.sprite(actor, time)
        padding = actor["logical"]//8
        scale = image.height/(actor["logical"]+2*padding)
        anchor_x, anchor_y = actor["anchor"]
        offset_x = (anchor_x+padding)*scale
        if actor["flip"]:
            offset_x = image.width-offset_x
        offset_y = (anchor_y+padding)*scale
        position_x, position_y = self.position(actor, time)
        frame.alpha_composite(image, (round(position_x-offset_x), round(position_y-offset_y)))

    def frame(self, time):
        number(time, "frame time", 0, self.duration)
        frame = self.background.copy()
        for actor in self.actors:
            self.place(frame, actor, time)
        return frame

    def export(self, directory, gif=False):
        directory = Path(directory)
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("输出目录必须为空或尚不存在，避免覆盖已有帧")
        directory.mkdir(parents=True, exist_ok=True)
        self.background.save(directory/"background.png")
        frozen_scene = copy.deepcopy(self.definition)
        frozen_scene["background"] = {"image": "background.png"}
        frozen_scene["characters"] = {key: {"attributes": character["attributes"]} for key, character in self.characters.items()}
        frozen_scene["actors"] = []
        for original, actor in zip(self.definition["actors"], self.actors):
            definition = copy.deepcopy(original)
            definition.pop("prompt", None)
            definition["character"] = actor["character"]
            frozen_scene["actors"].append(definition)
        (directory/"scene.json").write_text(json.dumps(frozen_scene, ensure_ascii=False, indent=2), encoding="utf-8")
        preview, durations = [], []
        stride = max(1, math.ceil(self.frame_count/180), math.ceil(self.fps/12))
        palette = None
        for index in range(self.frame_count):
            image = self.frame(index/self.fps)
            image.save(directory/f"frame_{index:06d}.png")
            if gif and index % stride == 0:
                thumbnail = Image.new("RGB", image.size, "white")
                thumbnail.paste(image, mask=image.getchannel("A"))
                thumbnail.thumbnail((640, 360), Image.Resampling.LANCZOS)
                if palette is None:
                    palette = thumbnail.quantize(colors=256)
                preview.append(thumbnail.quantize(palette=palette, dither=Image.Dither.NONE))
                next_index = min(self.frame_count, index+stride)
                durations.append(max(10, round(next_index*100/self.fps)*10-round(index*100/self.fps)*10))
        if preview:
            preview[0].save(directory/"preview.gif", save_all=True, append_images=preview[1:], loop=0,
                            duration=durations, disposal=2, optimize=False)
        metadata = {"backend": "deterministic-2d-rig", "fps": self.fps, "frames": self.frame_count,
                    "duration": self.frame_count/self.fps, "size": [self.width, self.height],
                    "background_sha256": self.background_hash,
                    "characters": {key: {field: value for field, value in character.items() if field != "base_command"}
                                   for key, character in self.characters.items()},
                    "preview_stride": stride if gif else None}
        (directory/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata


def main():
    parser = argparse.ArgumentParser(description="以固定角色和场景生成连续 2D 动画帧，不逐帧调用模型")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", help="JSON 场景文件")
    source.add_argument("--prompt", help="单个角色提示词，无需先训练")
    parser.add_argument("--action", default="run", help="单角色动作，如 run、walk、wave、jump、wind")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", default=str(BASE_DIR/"animation_out"), help="新的或空的输出目录")
    parser.add_argument("--gif", action="store_true", help="额外输出低分辨率 GIF 预览，无需 FFmpeg")
    args = parser.parse_args()
    try:
        if args.scene:
            path = Path(args.scene).resolve()
            definition = json.loads(path.read_text(encoding="utf-8-sig"))
            base_dir = path.parent
        else:
            definition = {"version": 1, "width": args.width, "height": args.height,
                          "fps": args.fps, "duration": args.duration,
                          "background": {"color": "#bfe6f5", "ground_color": "#96c879", "horizon": 0.83},
                          "actors": [{"id": "hero", "prompt": args.prompt, "action": args.action,
                                      "height": args.height*0.65, "position": [args.width*0.5, args.height*0.83]}]}
            base_dir = BASE_DIR
        metadata = Scene(definition, base_dir).export(args.output, args.gif)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"已保存 {metadata['frames']} 帧到 {args.output}，{metadata['fps']} fps；未训练或逐帧调用模型")


if __name__ == "__main__":
    main()