import json
import numpy as np
import onnxruntime as ort

# ---------- 1. 加载词表和配置 ----------
with open("vocab.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

stoi = meta["stoi"]
# JSON 的 key 都变成了字符串，需要转回 int
itos = {int(k): v for k, v in meta["itos"].items()}
cfg = meta["config"]
MAX_LEN = cfg["max_len"]

print(f"词表大小: {len(stoi)}, max_len: {MAX_LEN}")

# ---------- 2. 加载 ONNX 模型 ----------
sess = ort.InferenceSession(
    "tiny_model.onnx",
    providers=["CPUExecutionProvider"],
)


# ---------- 3. 贪心解码 ----------
def generate(prompt, max_new=256):
    ids = [stoi["<bos>"]] + [stoi.get(c, 0) for c in prompt] + [stoi["<sep>"]]

    for _ in range(max_new):
        x = np.array([ids[-MAX_LEN:]], dtype=np.int64)
        logits = sess.run(None, {"input_ids": x})[0]   # (1, seq, vocab)
        next_id = int(np.argmax(logits[0, -1]))
        ids.append(next_id)
        if itos[next_id] == "<eos>":
            break

    # 只取 <sep> 之后的部分
    out_ids = ids[ids.index(stoi["<sep>"]) + 1:]
    text = "".join(
        itos[i] for i in out_ids
        if itos[i] not in ("<eos>", "<pad>")
    )
    return text


# ---------- 4. 交互 ----------
if __name__ == "__main__":
    while True:
        p = input("提示词: ").strip()
        if not p:
            break
        print("指令:", generate(p))
        print()