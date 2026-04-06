"""Pick/Decline 二値分類の fine-tuning"""

import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODELS = {
    "modernbert": "sbintuitions/modernbert-ja-70m",
    "deberta": "ku-nlp/deberta-v3-base-japanese",
}
EMPTY_TITLE_TOKEN = "[TITLE_EMPTY]"
LABEL2ID = {"Decline": 0, "Pick": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_data():
    """訓練データを読み込む"""
    df = pd.read_csv("data/train/train.csv")
    df = df[df["pick"].isin(["Pick", "Decline"])].reset_index(drop=True)
    df["TITLE"] = df["TITLE"].fillna("").astype(str)
    df["BODY"] = df["BODY"].fillna("").astype(str)
    df["label"] = df["pick"].map(LABEL2ID)
    return df


def build_input_text(title: str, body: str) -> str:
    if not title or title.strip() == "":
        title = EMPTY_TITLE_TOKEN
    return f"{title} [SEP] {body}"


def tokenize_fn(examples, tokenizer):
    titles = [
        t if t.strip() else EMPTY_TITLE_TOKEN
        for t in examples["TITLE"]
    ]
    bodies = list(examples["BODY"])
    return tokenizer(titles, bodies, truncation=True, max_length=384)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="modernbert", choices=MODELS.keys())
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="0 for CE loss, >0 for focal loss")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    if args.output_dir is None:
        args.output_dir = f"models/{args.model}-pick-classifier"

    print(f"Model: {model_name}")

    # デバイス
    if args.cpu:
        device = "cpu"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # データ
    df = load_data()
    print(f"Total: {len(df)} (Pick: {(df['label']==1).sum()}, Decline: {(df['label']==0).sum()})")

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df[["TITLE", "BODY", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["TITLE", "BODY", "label"]].reset_index(drop=True))

    # トークナイザー & モデル
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # DeBERTa v3はresize_token_embeddingsとの相性が悪いので、
    # 特殊トークン追加はmodernbertのみ
    if "modernbert" in model_name:
        tokenizer.add_special_tokens({"additional_special_tokens": [EMPTY_TITLE_TOKEN]})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # DeBERTa v3 の embedding weight ロード確認
    if "deberta" in model_name:
        emb = model.deberta.embeddings.word_embeddings.weight
        print(f"Embedding stats: mean={emb.mean().item():.4f}, std={emb.std().item():.4f}, min={emb.min().item():.4f}, max={emb.max().item():.4f}")
        if emb.std().item() < 0.001 or emb.std().item() > 10:
            print("WARNING: Embeddings may not be loaded correctly!")

    if "modernbert" in model_name:
        model.resize_token_embeddings(len(tokenizer))

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=device == "cuda",
        no_cuda=args.cpu,
        use_cpu=args.cpu,
        report_to="none",
    )

    focal_gamma = args.focal_gamma
    label_smoothing = args.label_smoothing

    if focal_gamma > 0 or label_smoothing > 0:
        print(f"Focal gamma: {focal_gamma}, Label smoothing: {label_smoothing}")

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # Label smoothing
                n_classes = logits.size(-1)
                if label_smoothing > 0:
                    targets = (1 - label_smoothing) * torch.nn.functional.one_hot(labels, n_classes).float() \
                        + label_smoothing / n_classes
                else:
                    targets = torch.nn.functional.one_hot(labels, n_classes).float()
                # Focal weight
                if focal_gamma > 0:
                    probs = log_probs.exp()
                    focal_weight = (1 - probs) ** focal_gamma
                    loss = -(focal_weight * targets * log_probs).sum(dim=-1).mean()
                else:
                    loss = -(targets * log_probs).sum(dim=-1).mean()
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

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
    test_ds = test_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    preds_output = trainer.predict(test_ds)
    preds = preds_output.predictions.argmax(axis=-1)
    labels = preds_output.label_ids

    print(f"Accuracy: {accuracy_score(labels, preds):.1%}")
    print(classification_report(labels, preds, target_names=["Decline", "Pick"]))


if __name__ == "__main__":
    main()
