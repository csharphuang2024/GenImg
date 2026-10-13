import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TinySeq2Seq

# ---------- 1. 读数据 ----------
with open("data.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

if not lines:
    raise SystemExit("data.txt 为空，先跑 python gen_data.py")

print(f"数据条数: {len(lines)}")
print(f"最长一行: {max(len(l) for l in lines)} 字符")

# ---------- 2. 词表 ----------
special = ["<pad>", "<bos>", "<eos>", "<sep>"]
chars = sorted(set("".join(lines)))
vocab = special + [c for c in chars if c not in special]
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(vocab)
print(f"词表大小: {VOCAB_SIZE}")

# ---------- 3. 编码 ----------
MAX_LEN = 256

def encode(s):
    ids = [stoi["<bos>"]] + [stoi.get(c, 0) for c in s] + [stoi["<eos>"]]
    return ids[:MAX_LEN]

class SeqDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        for l in lines:
            s = l.replace("<bos>", "").replace("<eos>", "")
            self.data.append(encode(s))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate(batch):
    max_len = max(len(x) for x in batch)
    padded = [x + [stoi["<pad>"]] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded, dtype=torch.long)

# ---------- 4. 模型 ----------
device = "cpu"
model = TinySeq2Seq(VOCAB_SIZE, d_model=128, nhead=4, num_layers=2, max_len=MAX_LEN).to(device)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])

dataset = SeqDataset(lines)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

# ---------- 5. 训练 ----------
EPOCHS = 100
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, loss={avg:.4f}")

torch.save({
    "model": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {"d_model": 128, "nhead": 4, "num_layers": 2, "max_len": MAX_LEN},
}, "tiny_model.pt")
print("已保存 tiny_model.pt")