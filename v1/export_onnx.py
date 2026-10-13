import json
import torch
from model import TinySeq2Seq

# ---------- 1. 加载 checkpoint ----------
ckpt = torch.load("tiny_model.pt", map_location="cpu")
stoi = ckpt["stoi"]
itos = ckpt["itos"]
cfg = ckpt["config"]

print(f"词表大小: {len(stoi)}")
print(f"配置: {cfg}")

# ---------- 2. 重建模型 ----------
model = TinySeq2Seq(len(stoi), **cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# ---------- 3. 构造假输入 ----------
# 用真实的词表 id，避免 embedding 越界
dummy_input = torch.randint(0, len(stoi), (1, 64), dtype=torch.long)

# ---------- 4. 导出 ONNX ----------
torch.onnx.export(
    model,
    dummy_input,
    "tiny_model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits":    {0: "batch", 1: "seq_len"},
    },
    opset_version=14,
)

print("已导出 tiny_model.onnx")

# ---------- 5. 保存词表和配置 ----------
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {"stoi": stoi, "itos": itos, "config": cfg},
        f,
        ensure_ascii=False,
    )
print("已保存 vocab.json")