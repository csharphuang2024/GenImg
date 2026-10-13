import torch
from model import TinySeq2Seq

ckpt = torch.load("tiny_model.pt", map_location="cpu")
stoi = ckpt["stoi"]
itos = ckpt["itos"]
cfg = ckpt["config"]
MAX_LEN = cfg["max_len"]

model = TinySeq2Seq(len(stoi), **cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

print(f"词表大小: {len(stoi)}, max_len: {MAX_LEN}")

SPECIAL = {"<eos>", "<pad>", "<sep>", "<bos>"}


def generate(prompt, max_new=256):
    ids = [stoi["<bos>"]] + [stoi.get(c, 0) for c in prompt] + [stoi["<sep>"]]

    for _ in range(max_new):
        x = torch.tensor([ids[-MAX_LEN:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_id = int(logits[0, -1].argmax().item())
        ids.append(next_id)
        if itos[next_id] == "<eos>":
            break

    full = "".join(itos[i] for i in ids)
    for sp in ("<bos>", "<eos>", "<pad>"):
        full = full.replace(sp, "")
    parts = full.split("<sep>")
    tail = parts[-1] if parts else ""

    if not tail.strip():
        for p in reversed(parts):
            if p.strip():
                tail = p
                break

    idx = tail.find("canvas")
    if idx == -1:
        idx = tail.find("#")
    if idx > 0:
        tail = tail[idx:]

    segs = tail.split("|")
    clean = []
    for s in segs:
        if s.startswith("canvas") or "#" in s:
            clean.append(s)
        else:
            break
    return "|".join(clean)


if __name__ == "__main__":
    while True:
        p = input("提示词: ").strip()
        if not p:
            break
        out = generate(p)
        print("指令:")
        print(out.replace("|", "\n"))
        print()