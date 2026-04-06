"""学習済みモデルでテストデータを評価する"""

import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

EMPTY_TITLE_TOKEN = "[TITLE_EMPTY]"
TEST_FILES = ["data/test/twitter_test.csv", "data/test/canary.csv"]


def find_model_dir():
    """models/ 以下の最新モデルを自動検出"""
    candidates = glob.glob("models/*/config.json")
    if not candidates:
        raise FileNotFoundError("models/ にモデルが見つかりません")
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def build_input_text(title: str, body: str) -> str:
    if not title or title.strip() == "":
        title = EMPTY_TITLE_TOKEN
    return f"{title} [SEP] {body}"


def evaluate(clf, csv_path: str):
    df = pd.read_csv(csv_path)
    title_col = "title_original" if "title_original" in df.columns else "TITLE"
    body_col = "body_original" if "body_original" in df.columns else "BODY"

    texts = [
        build_input_text(
            str(row[title_col]) if pd.notna(row[title_col]) else "",
            str(row[body_col]) if pd.notna(row[body_col]) else "",
        )
        for _, row in df.iterrows()
    ]
    labels = df["pick"].tolist()

    results = clf(texts, batch_size=64)
    preds = [r["label"] for r in results]

    print(f"Accuracy: {accuracy_score(labels, preds):.1%}")
    print(classification_report(labels, preds, target_names=["Decline", "Pick"]))

    errors = [(i, labels[i], preds[i]) for i in range(len(labels)) if labels[i] != preds[i]]
    print(f"Errors: {len(errors)}")
    for i, expected, got in errors:
        body = str(df.iloc[i][body_col])[:80].replace("\n", " ")
        print(f"  [{i}] expected={expected} got={got} | {body}")


def main():
    model_dir = find_model_dir()
    print(f"Model: {model_dir}")
    clf = pipeline("text-classification", model=model_dir, device="cpu", truncation=True, max_length=384)

    for csv_path in TEST_FILES:
        if os.path.exists(csv_path):
            print(f"\n{'='*60}")
            print(f"  {csv_path}")
            print(f"{'='*60}")
            evaluate(clf, csv_path)


if __name__ == "__main__":
    main()
