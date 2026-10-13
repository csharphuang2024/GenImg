from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TinySeq2Seq, tokenize


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.txt"
MODEL_PATH = BASE_DIR / "tiny_model.pt"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 160


class CommandDataset(Dataset):
    def __init__(self, rows, token_to_id):
        self.samples = []
        unknown_id = token_to_id["<unk>"]
        for row in rows:
            content = row.removeprefix("<bos>").removesuffix("<eos>")
            tokens = ["<bos>", *tokenize(content), "<eos>"]
            self.samples.append([token_to_id.get(token, unknown_id) for token in tokens])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"找不到训练数据: {DATA_PATH}，请先运行 python gen_data.py")

    rows = [row.strip() for row in DATA_PATH.read_text(encoding="utf-8").splitlines() if row.strip()]
    if not rows:
        raise SystemExit("data.txt 为空，请先运行 python gen_data.py")

    tokenized_rows = [tokenize(row.removeprefix("<bos>").removesuffix("<eos>")) for row in rows]
    vocab = SPECIAL_TOKENS + sorted({token for row in tokenized_rows for token in row} - set(SPECIAL_TOKENS))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    max_len = max(len(row) for row in tokenized_rows) + 2

    print(f"数据条数: {len(rows)}，最长 token 序列: {max_len}，词表大小: {len(vocab)}")
    dataset = CommandDataset(rows, token_to_id)
    pad_id = token_to_id["<pad>"]

    def collate(batch):
        batch_max_len = max(map(len, batch))
        padded = [sample + [pad_id] * (batch_max_len - len(sample)) for sample in batch]
        return torch.tensor(padded, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySeq2Seq(
        len(vocab), d_model=D_MODEL, nhead=N_HEAD,
        num_layers=NUM_LAYERS, max_len=max_len,
    ).to(device)
    print(f"设备: {device}，参数量: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs, src_key_padding_mask=inputs.eq(pad_id))
            loss = loss_fn(logits.reshape(-1, len(vocab)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, loss={total_loss / len(loader):.4f}")

    torch.save({
        "model": model.cpu().state_dict(),
        "stoi": token_to_id,
        "itos": id_to_token,
        "config": {
            "d_model": D_MODEL,
            "nhead": N_HEAD,
            "num_layers": NUM_LAYERS,
            "max_len": max_len,
        },
        "tokenizer": "dsl-token-v1",
    }, MODEL_PATH)
    print(f"已保存模型: {MODEL_PATH}")


if __name__ == "__main__":
    main()