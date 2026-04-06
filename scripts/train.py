"""Pick/Decline 二値分類の fine-tuning (manual training loop)"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODELS = {
    "modernbert": "sbintuitions/modernbert-ja-70m",
    "deberta": "ku-nlp/deberta-v3-base-japanese",
}
EMPTY_TITLE_TOKEN = "[TITLE_EMPTY]"
LABEL2ID = {"Decline": 0, "Pick": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class PickClassifier(nn.Module):
    def __init__(self, encoder, num_labels=2):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.pooler = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        pooled = torch.tanh(self.pooler(cls_output))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def load_data():
    df = pd.read_csv("data/train/train.csv")
    df = df[df["pick"].isin(["Pick", "Decline"])].reset_index(drop=True)
    df["TITLE"] = df["TITLE"].fillna("").astype(str)
    df["BODY"] = df["BODY"].fillna("").astype(str)
    df["label"] = df["pick"].map(LABEL2ID)
    return df


def tokenize_fn(examples, tokenizer, max_length):
    titles = [t if t.strip() else EMPTY_TITLE_TOKEN for t in examples["TITLE"]]
    bodies = list(examples["BODY"])
    return tokenizer(titles, bodies, truncation=True, max_length=max_length)


def dynamic_pad_collate(batch):
    """バッチ内の最長に合わせてパディング"""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    # token_type_ids がある場合も対応
    has_token_type = "token_type_ids" in batch[0]
    token_type_ids = [] if has_token_type else None

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        if has_token_type:
            token_type_ids.append(b["token_type_ids"] + [0] * pad_len)
        labels.append(b["label"])

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if has_token_type:
        result["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
    return result


def build_model(model_name, tokenizer):
    """エンコーダ + 分類ヘッドを構築"""
    encoder = AutoModel.from_pretrained(model_name)

    # DeBERTa v3: transformers v5.xで _weight キーがロードされない問題を修正
    if "deberta" in model_name:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt_path = hf_hub_download(model_name, "model.safetensors")
        state_dict = load_file(ckpt_path)
        key_old = "deberta.embeddings.word_embeddings._weight"
        if key_old in state_dict:
            with torch.no_grad():
                encoder.embeddings.word_embeddings.weight.copy_(state_dict[key_old])
            print(f"Fixed DeBERTa embedding: loaded from {key_old}")

    encoder.resize_token_embeddings(len(tokenizer))
    model = PickClassifier(encoder, num_labels=2)
    return model


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="modernbert", choices=MODELS.keys())
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    if args.output_dir is None:
        args.output_dir = f"models/{args.model}-pick-classifier"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")

    if args.cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # データ
    df = load_data()
    print(f"Total: {len(df)} (Pick: {(df['label']==1).sum()}, Decline: {(df['label']==0).sum()})")

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # トークナイザー & モデル
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [EMPTY_TITLE_TOKEN]})

    model = build_model(model_name, tokenizer)
    model.to(device)

    # データセット
    train_ds = Dataset.from_pandas(train_df[["TITLE", "BODY", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["TITLE", "BODY", "label"]].reset_index(drop=True))
    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer, args.max_length), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(x, tokenizer, args.max_length), batched=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=dynamic_pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dynamic_pad_collate)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 学習
    best_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

        val_labels, val_preds = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if device.type == "mps":
            torch.mps.empty_cache()

    # ベストモデルを保存
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    from safetensors.torch import save_file as save_safetensors
    save_safetensors(best_state, str(output_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(output_dir))
    # config保存
    config = {
        "model_name": model_name,
        "num_labels": 2,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "max_length": args.max_length,
        "title_empty_token": EMPTY_TITLE_TOKEN,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Model saved to {output_dir}")

    # テストデータで評価
    print("\n=== Test Evaluation ===")
    test_df = pd.read_csv("data/test/twitter_test.csv")
    if "title_original" in test_df.columns:
        test_df["TITLE"] = test_df["title_original"].fillna("").astype(str)
        test_df["BODY"] = test_df["body_original"].fillna("").astype(str)
    else:
        test_df["TITLE"] = test_df["TITLE"].fillna("").astype(str)
        test_df["BODY"] = test_df["BODY"].fillna("").astype(str)
    test_df["label"] = test_df["pick"].map(LABEL2ID)

    test_ds = Dataset.from_pandas(test_df[["TITLE", "BODY", "label"]].reset_index(drop=True))
    test_ds = test_ds.map(lambda x: tokenize_fn(x, tokenizer, args.max_length), batched=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dynamic_pad_collate)

    test_labels, test_preds = evaluate(model, test_loader, device)
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.1%}")
    print(classification_report(test_labels, test_preds, target_names=["Decline", "Pick"]))


if __name__ == "__main__":
    main()
