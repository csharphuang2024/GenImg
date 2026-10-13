import argparse
import contextlib
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
import torch

from genimg import render
from infer import generate, load_model


ASSET_PROMPTS = {
    "girl": "开心的女孩挥手，马尾，蓝色长袖，红色长裤",
    "cat": "橘色猫站着",
    "flower": "红色郁金香",
    "tree": "绿色阔叶树",
    "house": "橘色小屋",
    "sun": "黄色太阳",
    "cloud": "白色积云",
}
SHOT_TITLES = [
    ("公园里的新朋友", "A new friend"),
    ("小猫发现了一朵花", "A flower in the park"),
    ("一起分享美好的一天", "A lovely day together"),
]


def find_encoder():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("缺少视频编码器，请执行 python -m pip install imageio-ffmpeg") from error


def build_assets(directory, strict=False):
    directory.mkdir()
    model, token_to_id, id_to_token, config = load_model()
    assets = {}
    records = {}
    for name, prompt in ASSET_PROMPTS.items():
        print(f"生成素材 {name}: {prompt}", flush=True)
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            command = generate(prompt, model, token_to_id, id_to_token, config)
        warning = warnings.getvalue().strip()
        if warning:
            print(f"  {warning}", file=sys.stderr, flush=True)
            if strict:
                raise ValueError(f"素材 {name} 回退到模板，严格模式停止制作")
        image = render(command, size=512, outline_width=0.8, background="transparent")
        bounds = image.getchannel("A").getbbox()
        if bounds is None:
            raise ValueError(f"素材 {name} 没有可见内容")
        image = image.crop(bounds)
        image.save(directory / f"{name}.png")
        (directory / f"{name}.txt").write_text(command, encoding="utf-8")
        assets[name] = image
        records[name] = {"prompt": prompt, "source": "template" if warning else "model",
                         "warning": warning, "size": list(image.size)}
        (directory / "manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return assets, records


def prepare_sprites(assets, width, height):
    heights = {"girl": 0.48, "cat": 0.19, "flower": 0.15, "tree": 0.60,
               "house": 0.27, "sun": 0.16, "cloud": 0.13}
    sprites = {}
    for name, fraction in heights.items():
        image = assets[name]
        scale = min(height * fraction / image.height, width * 0.30 / image.width)
        sprites[name] = image.resize((max(1, round(image.width * scale)),
                                      max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return sprites


def place(image, sprite, center_x, bottom, angle=0, flip=False):
    if flip:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if angle:
        sprite = sprite.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    image.alpha_composite(sprite, (round(center_x - sprite.width / 2), round(bottom - sprite.height)))


def background_frame(sprites, width, height):
    image = Image.new("RGBA", (width, height), "#c3edf5")
    draw = ImageDraw.Draw(image)
    draw.ellipse((-width * 0.25, height * 0.43, width * 0.80, height * 1.20), fill="#92ce88")
    draw.ellipse((width * 0.35, height * 0.48, width * 1.30, height * 1.25), fill="#70bfa0")
    draw.rectangle((0, height * 0.75, width, height), fill="#73b85c")
    draw.polygon([(0, height), (width * 0.43, height * 0.75),
                  (width * 0.62, height * 0.75), (width, height)], fill="#d5dce0")
    place(image, sprites["sun"], width * 0.83, height * 0.23)
    place(image, sprites["house"], width * 0.74, height * 0.72)
    place(image, sprites["tree"], width * 0.12, height * 0.79)
    return image


def title_font(height):
    font_size = max(12, round(height * 0.045))
    for filename in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
        if path.exists():
            return ImageFont.truetype(str(path), font_size), True
    return ImageFont.load_default(size=font_size), False


def film_frame(sprites, background, shot, progress, time, font, chinese=True):
    image = background.copy()
    width, height = image.size
    progress = min(1, max(0, progress))
    smooth = progress * progress * (3 - 2 * progress)
    ground = height * 0.90
    place(image, sprites["cloud"], width * (0.35 + 0.025 * math.sin(time * 0.25)), height * 0.22)
    place(image, sprites["cloud"], width * (0.61 + 0.025 * math.sin(time * 0.25 + 1)), height * 0.32)
    place(image, sprites["flower"], width * 0.80, ground, angle=3 * math.sin(time * 2))
    if shot == 0:
        girl_x = width * (0.29 + 0.10 * smooth)
        cat_x = width * (0.93 - 0.36 * smooth)
        cat_bottom = ground - abs(math.sin(time * 7)) * height * 0.012
        flip = True
    elif shot == 1:
        girl_x = width * (0.39 + 0.17 * smooth)
        cat_x = width * (0.57 + 0.17 * smooth)
        cat_bottom = ground - math.sin(math.pi * progress) ** 4 * height * 0.10
        flip = False
    else:
        girl_x, cat_x = width * 0.56, width * 0.74
        cat_bottom = ground - (0.5 + 0.5 * math.sin(time * 3)) * height * 0.008
        flip = True
    draw = ImageDraw.Draw(image)
    for center, radius in ((girl_x, sprites["girl"].width * 0.45),
                           (cat_x, sprites["cat"].width * 0.42)):
        draw.ellipse((center - radius, ground - height * 0.008,
                      center + radius, ground + height * 0.016), fill="#568b50")
    place(image, sprites["girl"], girl_x, ground - math.sin(time * 3) * height * 0.005,
          angle=1.5 * math.sin(time * 2))
    place(image, sprites["cat"], cat_x, cat_bottom, angle=2 * math.sin(time * 4), flip=flip)
    draw = ImageDraw.Draw(image)
    title = SHOT_TITLES[shot][0 if chinese else 1]
    bounds = draw.textbbox((0, 0), title, font=font)
    text_width = bounds[2] - bounds[0]
    draw.rectangle((0, 0, width, height * 0.10), fill="#f3f7f8")
    draw.text(((width - text_width) / 2, height * 0.022 - bounds[1]), title, font=font, fill="#254339")
    return image.convert("RGB")


def encode_film(executable, directory, sprites, width, height, fps, seconds_per_shot, save_frames=False):
    background = background_frame(sprites, width, height)
    background.save(directory / "background.png")
    font, chinese = title_font(height)
    frames_per_shot = round(seconds_per_shot * fps)
    frame_count = frames_per_shot * len(SHOT_TITLES)
    if save_frames:
        (directory / "frames").mkdir()
    temporary = directory / "film.partial.mp4"
    target = directory / "film.mp4"
    command = [executable, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    with (directory / "encoder.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index in range(frame_count):
                shot, local_index = divmod(index, frames_per_shot)
                if local_index == 0:
                    print(f"合成镜头 {shot + 1}/{len(SHOT_TITLES)}", flush=True)
                frame = film_frame(sprites, background, shot, local_index / max(1, frames_per_shot - 1),
                                   index / fps, font, chinese)
                process.stdin.write(frame.tobytes())
                if save_frames:
                    frame.save(directory / "frames" / f"frame_{index:06d}.png")
            process.stdin.close()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            raise
    if returncode or not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"视频编码失败，请查看 {directory / 'encoder.log'}")
    temporary.rename(target)
    return target, frame_count


def main():
    parser = argparse.ArgumentParser(description="用现有模型生成素材，自动合成女孩与小猫的三镜头 MP4 短片")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent /
                        f"film_{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-shot", type=float, default=6)
    parser.add_argument("--strict-model", action="store_true", help="任何素材回退到模板时停止")
    parser.add_argument("--save-frames", action="store_true", help="额外保存每一帧 PNG")
    args = parser.parse_args()
    if (not 128 <= args.width <= 1920 or not 128 <= args.height <= 1080
            or args.width % 2 or args.height % 2):
        parser.error("画面尺寸必须为偶数，宽 128..1920，高 128..1080")
    if not 1 <= args.fps <= 60 or not math.isfinite(args.seconds_per_shot) or not 1 <= args.seconds_per_shot <= 30:
        parser.error("fps 必须为 1..60，每个镜头时长必须为 1..30 秒")
    if args.output.exists():
        parser.error("输出目录已存在，请选择新目录，避免覆盖已有短片")
    try:
        executable = find_encoder()
        args.output.mkdir(parents=True)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        assets, records = build_assets(args.output / "assets", args.strict_model)
        sprites = prepare_sprites(assets, args.width, args.height)
        target, frame_count = encode_film(executable, args.output, sprites, args.width, args.height,
                                          args.fps, args.seconds_per_shot, args.save_frames)
        metadata = {"title": "女孩与小猫", "width": args.width, "height": args.height,
                    "fps": args.fps, "frames": frame_count, "duration": frame_count / args.fps,
                    "audio": False, "motion": "固定模型素材的位移、摆动和跳跃，不是骨骼动画",
                    "shots": [title[0] for title in SHOT_TITLES], "assets": records}
        (args.output / "story.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        fallback_count = sum(record["source"] == "template" for record in records.values())
        print(f"已保存 {target.resolve()} ({frame_count / args.fps:.1f} 秒，无音轨)")
        print(f"素材来源：模型 {len(records) - fallback_count}，模板回退 {fallback_count}；详见 assets/manifest.json")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"制作失败：{error}\n")


if __name__ == "__main__":
    main()