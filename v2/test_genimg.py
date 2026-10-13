import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image, ImageChops

from animate import Scene
from gen_data import ACTIONS, ANIMALS, COLORS, iter_samples, make_command
from elements import ELEMENTS, EXPRESSIONS, HUMAN_OPTIONS, IDENTITIES, POSES, default_person, make_element, make_person, person_prompt
from genimg import parse_instructions, render, validate_instructions
from infer import canonicalize_prompt, generate, template_command
from model import tokenize
from motion import animal_pose, blend_pose, human_pose


class FilmTests(unittest.TestCase):
    def test_generated_sprites_are_used_in_all_shots(self):
        from make_film import ASSET_PROMPTS, background_frame, film_frame, prepare_sprites, title_font

        assets = {name: Image.new("RGBA", (32, 64), "#ff00ff") for name in ASSET_PROMPTS}
        sprites = prepare_sprites(assets, 320, 180)
        background = background_frame(sprites, 320, 180)
        font, chinese = title_font(180)
        for shot in range(3):
            first = film_frame(sprites, background, shot, 0, 0, font, chinese)
            last = film_frame(sprites, background, shot, 1, 2, font, chinese)
            self.assertEqual(first.size, (320, 180))
            self.assertEqual(first.mode, "RGB")
            self.assertIsNotNone(ImageChops.difference(first, last).getbbox())
            self.assertGreater(dict((color, count) for count, color in first.getcolors(320 * 180)).get((255, 0, 255), 0), 0)

    def test_fallback_is_recorded_and_strict_mode_stops(self):
        from make_film import build_assets

        command = "canvas 64 64 transparent|ellipse 32 32 10 10 #ff0000"

        def fallback(*args):
            print("template fallback", file=__import__("sys").stderr)
            return command

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            directory = Path(temporary)
            with patch("make_film.load_model", return_value=(None, {}, {}, {})), \
                    patch("make_film.ASSET_PROMPTS", {"cat": "橘色猫站着"}), \
                    patch("make_film.generate", side_effect=fallback):
                assets, records = build_assets(directory / "assets")
                self.assertEqual(records["cat"]["source"], "template")
                self.assertTrue((directory / "assets" / "cat.png").exists())
                self.assertEqual(assets["cat"].mode, "RGBA")
                with self.assertRaisesRegex(ValueError, "严格模式"):
                    build_assets(directory / "strict", strict=True)


class RenderTests(unittest.TestCase):
    def render_memory(self, text):
        output = io.BytesIO()
        output.name = "check.png"
        return render(text, output, size=64, outline_width=0, background="transparent")

    def test_curve_bends_and_tapers(self):
        image = self.render_memory(
            "canvas 64 64 transparent|curve 10 45 10 10 50 10 50 45 4 1 #e8923f"
        )
        alpha = image.getchannel("A")
        self.assertGreater(alpha.getpixel((10, 45)), 200)
        self.assertGreater(alpha.getpixel((50, 45)), 150)
        self.assertGreater(alpha.getpixel((30, 19)), 100)
        self.assertEqual(alpha.getpixel((30, 45)), 0)
        self.assertGreater(alpha.getpixel((7, 45)), 100)
        self.assertLess(alpha.getpixel((47, 45)), 80)

    def test_old_lines_have_round_caps(self):
        image = self.render_memory("canvas 64 64 transparent|line 10 30 40 30 #e8923f 6")
        alpha = image.getchannel("A")
        self.assertGreater(alpha.getpixel((8, 30)), 150)
        self.assertLess(alpha.getpixel((7, 27)), 80)

    def test_curve_accepts_both_separators(self):
        text = "canvas 64 64 transparent|curve 10 45 10 10 50 10 50 45 4 1 #e8923f"
        self.assertEqual(parse_instructions(text), parse_instructions(text.replace("|", "\n")))

    def test_all_templates_render_without_skipped_commands(self):
        for animal in ANIMALS:
            for action in ACTIONS:
                for color in COLORS.values():
                    with self.subTest(animal=animal, action=action, color=color):
                        command = make_command(animal, action, color)
                        _, commands = parse_instructions(command)
                        self.assertTrue(any(parts[0] == "curve" for parts in commands))
                        errors = io.StringIO()
                        with contextlib.redirect_stderr(errors):
                            image = self.render_memory(command)
                        self.assertEqual(errors.getvalue(), "")
                        self.assertIsNotNone(image.getchannel("A").getbbox())
                        self.assertEqual(image.mode, "RGBA")

    def test_invalid_curve_does_not_hide_valid_shapes(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            image = self.render_memory(
                "canvas 64 64 transparent|curve 10 45 10 10 50 10 50 45 0 1 #e8923f|"
                "ellipse 32 32 4 4 #ffffff"
            )
        self.assertTrue(errors.getvalue())
        self.assertGreater(image.getchannel("A").getpixel((32, 32)), 200)

    def test_new_long_commands_are_not_cut_at_384_tokens(self):
        samples = []
        color_name = next(iter(COLORS))
        for animal in ANIMALS:
            for action in ACTIONS:
                prompt = f"{color_name}{animal}{action}"
                command = make_command(animal, action, COLORS[color_name])
                tokens = tokenize(f"<bos>{prompt}<sep>{command}<eos>")
                samples.append((prompt, tokens))
        prompt, tokens = max(samples, key=lambda sample: len(sample[1]))
        separator = tokens.index("<sep>")
        self.assertGreater(len(tokens[separator + 1:]), 384)
        self.assert_recorded_generation(prompt, " ".join(tokens[separator + 1:-1]))

    def assert_recorded_generation(self, prompt, command):
        tokens = tokenize(f"<bos>{prompt}<sep>{command}<eos>")
        special = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
        vocab = special + sorted(set(tokens) - set(special))
        token_to_id = {token: index for index, token in enumerate(vocab)}
        id_to_token = dict(enumerate(vocab))
        separator = tokens.index("<sep>")
        response = tokens[separator + 1:]

        class RecordedModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, inputs):
                logits = torch.full((1, 1, len(vocab)), -10.0)
                logits[0, 0, token_to_id[response[self.calls]]] = 10.0
                self.calls += 1
                return logits

        model = RecordedModel()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = generate(prompt, model, token_to_id, id_to_token, {"max_len": len(tokens)})
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(model.calls, len(response))
        self.assertEqual(result, " ".join(response[:-1]))

    def test_person_longer_than_512_tokens_decodes_without_fallback(self):
        prompt, command = next((prompt, command) for prompt, command in iter_samples() if prompt.startswith("人物"))
        self.assertGreater(len(tokenize(command)), 512)
        self.assert_recorded_generation(prompt, command)

    def test_short_element_decodes_without_fallback(self):
        prompt = "白色云积云"
        command = template_command(canonicalize_prompt(prompt)[1])
        self.assertLess(command.count("|"), 30)
        self.assert_recorded_generation(prompt, command)

    def test_every_training_sample_round_trips_and_renders(self):
        seen = set()
        for prompt, command in iter_samples():
            with self.subTest(prompt=prompt):
                self.assertNotIn(prompt, seen)
                seen.add(prompt)
                canonical, attributes = canonicalize_prompt(prompt)
                self.assertEqual(canonical, prompt)
                self.assertEqual(template_command(attributes), command)
                validate_instructions(command)
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    image = self.render_memory(command)
                self.assertEqual(errors.getvalue(), "")
                self.assertIsNotNone(image.getchannel("A").getbbox())
        expected = len(ANIMALS)*len(ACTIONS)*len(COLORS)
        expected += sum(map(len, ELEMENTS.values()))*len(COLORS)
        expected += len(IDENTITIES)*len(POSES)*len(EXPRESSIONS)*len(COLORS)
        self.assertEqual(len(seen), expected)

    def test_natural_person_prompt_has_independent_colors(self):
        _, spec = canonicalize_prompt("开心的女孩正在挥手，长发，深色皮肤，蓝色长袖，红色短裤，白色运动鞋，张开手，细脖子")
        self.assertEqual(spec["identity"], "女孩")
        self.assertEqual(spec["action"], "挥手")
        self.assertEqual(spec["expression"], "开心")
        self.assertEqual(spec["hair"], "长发")
        self.assertEqual(spec["skin"], "深肤")
        self.assertEqual(spec["top"], "长袖")
        self.assertEqual(spec["bottom"], "短裤")
        self.assertEqual(spec["top_color"], "蓝色")
        self.assertEqual(spec["bottom_color"], "红色")
        self.assertEqual(spec["shoe_color"], "白色")
        self.assertEqual(spec["hands"], "张开手")
        self.assertEqual(spec["neck"], "细脖子")

    def test_element_aliases_and_defaults(self):
        for prompt, subject, variant in (("一根原木", "木", "原木"), ("绿色树木", "树", "阔叶树"),
                                         ("雪山", "山", "雪山"), ("蓝色小河", "河流", "直河"),
                                         ("橙色货车", "汽车", "货车")):
            with self.subTest(prompt=prompt):
                _, spec = canonicalize_prompt(prompt)
                self.assertEqual(spec["subject"], subject)
                self.assertEqual(spec["variant"], variant)
        for prompt in ("太阳和月亮", "女孩和房屋", "男孩和猫"):
            with self.assertRaises(ValueError):
                canonicalize_prompt(prompt)

    def test_each_person_control_changes_rendered_pixels(self):
        baseline = self.render_memory(make_person(default_person(), COLORS)).tobytes()
        groups = {**HUMAN_OPTIONS, "expression": EXPRESSIONS, "action": tuple(POSES),
                  "top_color": tuple(COLORS), "bottom_color": tuple(COLORS), "shoe_color": tuple(COLORS)}
        for key, options in groups.items():
            for value in options:
                if value == default_person()[key]:
                    continue
                with self.subTest(key=key, value=value):
                    spec = default_person()
                    spec[key] = value
                    image = self.render_memory(make_person(spec, COLORS))
                    self.assertNotEqual(image.tobytes(), baseline)

    def test_element_variants_change_pixels(self):
        for subject, variants in ELEMENTS.items():
            images = {self.render_memory(make_element(subject, variant, "#3f7fe8")).tobytes() for variant in variants}
            self.assertEqual(len(images), len(variants), subject)

    def test_crescent_has_transparent_interior(self):
        output = io.BytesIO()
        output.name = "check.png"
        image = render(make_element("月亮", "弯月", "#e8d43f"), output, size=128,
                       outline_width=0, background="transparent")
        alpha = image.getchannel("A")
        self.assertGreater(alpha.getpixel((40, 64)), 200)
        self.assertEqual(alpha.getpixel((70, 64)), 0)

    def test_strict_validation_rejects_malformed_model_output(self):
        for text in ("canvas 128 128 transparent|curve 1 2 3 #ffffff",
                     "canvas 128 128 transparent|ellipse 32 32 nan 2 #ffffff",
                     "canvas 128 128 transparent|unknown 32 32 #ffffff",
                     "canvas 128 128 transparent|ellipse 32 32 -2 2 #ffffff"):
            with self.assertRaises(ValueError):
                validate_instructions(text)


class AnimationTests(unittest.TestCase):
    def definition(self):
        return {"version": 1, "width": 320, "height": 180, "fps": 8, "duration": 1,
                "characters": {"hero": {"prompt": person_prompt(default_person())}},
                "background": {"color": "#bfe6f5", "ground_color": "#96c879", "horizon": 0.85},
                "actors": [{"id": "hero", "character": "hero", "height": 140,
                            "position": [160, 155], "action": "run", "blink": False}]}

    def assert_bones(self, pose):
        for middle, end, roots, lengths in (
                ("elbows", "wrists", ((54, 52), (74, 52)), (16, 15)),
                ("knees", "ankles", ((58, 78), (70, 78)), (18, 18))):
            for index, root in enumerate(roots):
                self.assertAlmostEqual(math.dist(root, pose[middle][index]), lengths[0], places=6)
                self.assertAlmostEqual(math.dist(pose[middle][index], pose[end][index]), lengths[1], places=6)

    def test_fixed_canvas_does_not_refit_moving_limb(self):
        base = "canvas 128 128 transparent|ellipse 64 90 8 10 #3f7fe8"
        first = render(base, size=128, fit_content=False)
        second = render(base+"|ellipse 64 10 3 3 #e83f3f", size=128, fit_content=False)
        self.assertEqual(first.crop((40, 70, 88, 115)).tobytes(), second.crop((40, 70, 88, 115)).tobytes())

    def test_human_loops_have_fixed_bones_and_exact_endpoints(self):
        for action in ("run", "walk", "jump", "wave", "idle"):
            frames = []
            for index in range(13):
                pose = human_pose(action, index/12)
                self.assert_bones(pose)
                command = make_person(default_person(), COLORS, pose=pose)
                validate_instructions(command)
                frames.append(render(command, size=64, fit_content=False).tobytes())
            self.assertEqual(frames[0], frames[-1], action)
            self.assertGreater(len(set(frames)), 3, action)

    def test_human_transition_preserves_bone_lengths(self):
        first, second = human_pose("run", 0.2), human_pose("wave", 0.7)
        for index in range(11):
            self.assert_bones(blend_pose(first, second, index/10))
        self.assertEqual(blend_pose(first, second, 0), first)
        self.assertEqual(blend_pose(first, second, 1), second)

    def test_all_animal_loops_render_and_close(self):
        for animal, info in ANIMALS.items():
            for action in ("idle", "walk", "run", "jump"):
                frames = []
                for index in range(9):
                    command = make_command(animal, "站着", COLORS["橘色"],
                                           pose=animal_pose(action, index/8, *info["body"]))
                    validate_instructions(command)
                    frames.append(render(command, size=64, fit_content=False).tobytes())
                self.assertEqual(frames[0], frames[-1], (animal, action))
                self.assertGreater(len(set(frames)), 2)

    def test_scene_determinism_identity_and_cached_background(self):
        definition = self.definition()
        original = copy.deepcopy(definition)
        with patch("torch.load", side_effect=AssertionError("animation must not load a model")):
            scene = Scene(definition)
            background = scene.background.tobytes()
            characters = copy.deepcopy(scene.characters)
            frames = [scene.frame(time) for time in (0, .125, .25, .75)]
            self.assertEqual(scene.frame(.25).tobytes(), frames[2].tobytes())
            self.assertGreater(len({frame.tobytes() for frame in frames}), 2)
            self.assertEqual(scene.background.tobytes(), background)
            self.assertEqual(scene.characters, characters)
            self.assertEqual(definition, original)
            for frame in frames:
                self.assertEqual(frame.crop((0, 0, 60, 180)).tobytes(), scene.background.crop((0, 0, 60, 180)).tobytes())
                colors = frame.getcolors(maxcolors=frame.width*frame.height)
                self.assertGreater(sum(count for count, color in colors if color[:3] == (63, 127, 232)), 50)

    def test_human_action_and_expression_switch_is_continuous(self):
        definition = self.definition()
        definition["actors"][0]["clips"] = [
            {"start": 0, "end": .5, "action": "run", "expression": "开心"},
            {"start": .5, "end": 1, "action": "wave", "expression": "生气", "transition": .2}]
        scene = Scene(definition)
        before, after = scene.frame(.5-1e-7), scene.frame(.5)
        self.assertIsNone(ImageChops.difference(before, after).convert("RGB").getbbox())
        self.assertNotEqual(scene.frame(.8).tobytes(), after.tobytes())

    def test_path_positions_are_explicit_and_camera_is_fixed(self):
        definition = self.definition()
        definition["actors"][0]["path"] = [{"time": 0, "position": [100, 155]}, {"time": 1, "position": [220, 155]}]
        scene = Scene(definition)
        self.assertEqual(scene.position(scene.actors[0], .5), (160, 155))
        self.assertEqual(scene.frame(.5).size, (320, 180))

    def test_padded_sprites_do_not_clip_jump_hair(self):
        for hair in HUMAN_OPTIONS["hair"]:
            spec = default_person()
            spec["hair"], spec["face"] = hair, "长脸"
            definition = self.definition()
            definition["characters"]["hero"] = {"prompt": person_prompt(spec)}
            definition["actors"][0]["action"] = "jump"
            scene = Scene(definition)
            sizes = set()
            for time in (0, .25, .5, .75):
                sprite = scene.sprite(scene.actors[0], time)
                sizes.add(sprite.size)
                bounds = sprite.getchannel("A").getbbox()
                self.assertIsNotNone(bounds)
                self.assertGreater(bounds[1], 0)
                self.assertLess(bounds[3], sprite.height)
            self.assertEqual(len(sizes), 1)

    def test_background_items_are_cached_and_wind_root_is_fixed(self):
        definition = self.definition()
        definition["actors"] = [{"id": "tree", "prompt": "绿色阔叶树", "action": "wind", "height": 140, "position": [160, 160]}]
        definition["background"]["items"] = [{"id": "sun", "prompt": "黄色太阳", "height": 50, "position": [40, 60]}]
        scene = Scene(definition)
        with patch("animate.make_element", wraps=make_element) as factory:
            first = scene.frame(.25)
            second = scene.frame(.75)
            self.assertEqual(factory.call_count, 2)
        self.assertNotEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.crop((147, 154, 173, 175)).tobytes(), second.crop((147, 154, 173, 175)).tobytes())

    def test_scene_export_and_frozen_reload(self):
        definition = self.definition()
        definition["background"]["items"] = [{"prompt": "黄色太阳", "height": 50, "position": [40, 60]}]
        scene = Scene(definition)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, ignore_cleanup_errors=True) as directory:
            metadata = scene.export(directory, gif=True)
            self.assertEqual(metadata["frames"], 8)
            self.assertEqual(len(list(Path(directory).glob("frame_*.png"))), 8)
            frozen = json.loads((Path(directory)/"scene.json").read_text(encoding="utf-8"))
            rebuilt = Scene(frozen, directory)
            self.assertEqual(rebuilt.frame(.25).tobytes(), scene.frame(.25).tobytes())
            with Image.open(Path(directory)/"preview.gif") as preview:
                self.assertGreater(preview.n_frames, 1)
            with self.assertRaises(ValueError):
                scene.export(directory)

    def test_scene_rejects_invalid_or_unsupported_controls(self):
        mutations = [lambda scene: scene.update(fps=0), lambda scene: scene.update(width=100.5),
                     lambda scene: scene.update(duration=float("nan")),
                     lambda scene: scene["actors"][0].update(action="360度转身"),
                     lambda scene: scene["actors"][0].update(action="wind"),
                     lambda scene: scene["actors"][0].update(flip="yes"),
                     lambda scene: scene["actors"][0].update(clips=[{"start": .1, "end": 1}]),
                     lambda scene: scene["actors"][0].update(clips=[{"start": 0, "end": .7}]),
                     lambda scene: scene["actors"][0].update(path=[{"time": .1, "position": [0, 0]}]),
                     lambda scene: scene["actors"].append(copy.deepcopy(scene["actors"][0])),
                     lambda scene: scene["actors"][0].update(unknown=True)]
        for mutate in mutations:
            definition = self.definition()
            mutate(definition)
            with self.assertRaises(ValueError):
                Scene(definition)

    def test_demo_scene_renders_without_model(self):
        path = Path(__file__).with_name("scene.example.json")
        definition = json.loads(path.read_text(encoding="utf-8"))
        with patch("torch.load", side_effect=AssertionError("must not load model")):
            scene = Scene(definition, path.parent)
            frames = [scene.frame(time).tobytes() for time in (0, 1, 2, 3, 3.9)]
        self.assertEqual(len(set(frames)), 5)


if __name__ == "__main__":
    unittest.main()