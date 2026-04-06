"""学習済みモデルをHuggingFace Hubにpushする"""

import argparse

from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/pick-classifier")
    parser.add_argument("--repo_id", type=str, default="teru00801/pick-twitter-classifier")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
