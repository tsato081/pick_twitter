"""学習済みモデルでテストデータを評価する"""

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

EMPTY_TITLE_TOKEN = "[TITLE_EMPTY]"


def build_input_text(title: str, body: str) -> str:
    if not title or title.strip() == "":
        title = EMPTY_TITLE_TOKEN
    return f"{title} [SEP] {body}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/pick-classifier")
    parser.add_argument("--test_csv", type=str, default="data/test/twitter_test.csv")
    parser.add_argument("--title_col", type=str, default="title_original")
    parser.add_argument("--body_col", type=str, default="body_original")
    parser.add_argument("--label_col", type=str, default="pick")
    args = parser.parse_args()

    clf = pipeline("text-classification", model=args.model_dir, device="cpu", truncation=True, max_length=384)

    df = pd.read_csv(args.test_csv)
    texts = [
        build_input_text(
            str(row[args.title_col]) if pd.notna(row[args.title_col]) else "",
            str(row[args.body_col]) if pd.notna(row[args.body_col]) else "",
        )
        for _, row in df.iterrows()
    ]
    labels = df[args.label_col].tolist()

    results = clf(texts, batch_size=64)
    preds = [r["label"] for r in results]

    print(f"Accuracy: {accuracy_score(labels, preds):.1%}")
    print(classification_report(labels, preds, target_names=["Decline", "Pick"]))

    # エラー一覧
    errors = [(i, labels[i], preds[i]) for i in range(len(labels)) if labels[i] != preds[i]]
    print(f"Errors: {len(errors)}")
    for i, expected, got in errors:
        body = str(df.iloc[i][args.body_col])[:80].replace("\n", " ")
        print(f"  [{i}] expected={expected} got={got} | {body}")


if __name__ == "__main__":
    main()
