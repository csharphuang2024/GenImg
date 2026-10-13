import argparse
import sys
from pathlib import Path

import torch

from gen_data import ACTIONS, ANIMALS, COLORS, make_command
from elements import make_element, make_person, parse_new_prompt
from genimg import validate_instructions
from model import TinySeq2Seq, tokenize


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tiny_model.pt"
SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<sep>", "<unk>"}
COLOR_ALIASES = {
    "橙色": "橘色", "橙": "橘色", "棕": "棕色", "灰": "灰色",
    "粉红色": "粉色", "粉": "粉色", "白": "白色", "黑": "黑色",
    "蓝": "蓝色", "红": "红色", "绿": "绿色", "黄": "黄色",
}
ACTION_ALIASES = {
    "站立": "站着", "站": "站着", "坐": "坐着", "跳起来": "跳跃",
    "跳": "跳跃", "跑步": "奔跑", "跑着": "奔跑", "跑": "奔跑",
    "趴下": "趴着", "趴": "趴着",
}


def canonicalize_prompt(prompt):
    new_prompt = parse_new_prompt(prompt, COLORS)
    if new_prompt is not None:
        if any(animal in prompt for animal in ANIMALS):
            raise ValueError("目前一次仅支持一个主体；请将人物、元素和动物分别生成")
        return new_prompt
    color = next((name for alias, name in COLOR_ALIASES.items() if alias in prompt), None)
    if color is None:
        color = next((name for name in COLORS if name in prompt), "橘色")
    animal = next((name for name in sorted(ANIMALS, key=len, reverse=True) if name in prompt), None)
    action = next((name for alias, name in ACTION_ALIASES.items() if alias in prompt), None)
    if action is None:
        action = next((name for name in ACTIONS if name in prompt), None)
    missing = []
    if animal is None:
        missing.append("动物: " + "、".join(ANIMALS))
    if action is None:
        missing.append("动作: " + "、".join(ACTIONS))
    if missing:
        raise ValueError("无法识别 " + "；".join(missing))
    return f"{color}{animal}{action}", (color, animal, action)


def template_command(attributes):
    if isinstance(attributes, dict):
        if attributes["subject"] == "人物":
            return make_person(attributes, COLORS)
        return make_element(attributes["subject"], attributes["variant"], COLORS[attributes["color"]])
    color, animal, action = attributes
    return make_command(animal, action, COLORS[color])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型 {MODEL_PATH}，请先运行 python train.py")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if checkpoint.get("tokenizer") != "dsl-token-v1":
        raise ValueError("模型格式过旧；请在 v2 目录中重新运行 python train.py")
    token_to_id = checkpoint["stoi"]
    id_to_token = {int(index): token for index, token in checkpoint["itos"].items()}
    model = TinySeq2Seq(len(token_to_id), **checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, token_to_id, id_to_token, checkpoint["config"]


def generate(prompt, model, token_to_id, id_to_token, config, max_new=None):
    canonical, attributes = canonicalize_prompt(prompt)
    unknown_id = token_to_id["<unk>"]
    prompt_tokens = tokenize(canonical)
    if any(token not in token_to_id for token in prompt_tokens):
        raise ValueError("当前模型词表不支持这个提示词；请用新版数据重新训练，或使用 --template 预览")
    ids = [token_to_id["<bos>"]]
    ids.extend(token_to_id.get(token, unknown_id) for token in prompt_tokens)
    ids.append(token_to_id["<sep>"])
    eos_id = token_to_id["<eos>"]
    completed = False
    remaining = config["max_len"] - len(ids)
    if remaining <= 0:
        raise ValueError("提示词超出当前模型上下文，请重新训练或缩短描述")
    if max_new is not None and max_new <= 0:
        raise ValueError("--max-new 必须大于 0")
    budget = remaining if max_new is None else min(max_new, remaining)

    with torch.inference_mode():
        for _ in range(budget):
            logits = model(torch.tensor([ids], dtype=torch.long))
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
            if next_id == eos_id:
                completed = True
                break

    generated_tokens = [id_to_token[index] for index in ids]
    try:
        separator = generated_tokens.index("<sep>")
        tail = generated_tokens[separator + 1:]
        if "<eos>" in tail:
            tail = tail[:tail.index("<eos>")]
        command = " ".join(token for token in tail if token not in SPECIAL_TOKENS).strip()
        if not completed:
            raise ValueError("model output incomplete")
        validate_instructions(command)
        return command
    except (ValueError, IndexError):
        print("模型指令未完成或语法无效，已回退到模板；本次图片不是完整模型推理结果。", file=sys.stderr)
        return template_command(attributes)


def main():
    parser = argparse.ArgumentParser(description="根据中文描述生成动物、人物和常用元素绘图指令")
    parser.add_argument("prompt", nargs="*", help="例如：一只可爱的蓝色猫正在跳跃")
    parser.add_argument("--max-new", type=int, help="最多生成的 token 数，默认使用模型剩余上下文")
    parser.add_argument("--template", action="store_true", help="直接使用新版角色模板预览，不加载模型")
    args = parser.parse_args()

    if not args.template:
        try:
            model, token_to_id, id_to_token, config = load_model()
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))

    while True:
        prompt = " ".join(args.prompt).strip() if args.prompt else input("提示词（回车退出）: ").strip()
        if not prompt:
            break
        try:
            if args.template:
                _, attributes = canonicalize_prompt(prompt)
                command = template_command(attributes)
            else:
                command = generate(prompt, model, token_to_id, id_to_token, config, args.max_new)
            print(command, flush=True)
        except ValueError as error:
            if args.prompt:
                parser.error(str(error))
            print(f"提示词错误: {error}", file=sys.stderr, flush=True)
        if args.prompt:
            break


if __name__ == "__main__":
    main()