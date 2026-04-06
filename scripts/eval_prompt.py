"""テストデータに対してGPT-5.4-miniでPick/Decline判定を行い、精度を評価する"""

import asyncio
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_TEMPLATE = Path("prompts/pick_classify.txt").read_text()
SYSTEM_PROMPT = PROMPT_TEMPLATE.split("---")[0].strip()

client = OpenAI()
MODEL = "gpt-5.4"
MAX_CONCURRENT = 20


async def classify(sem: asyncio.Semaphore, title: str, body: str) -> str:
    user_msg = f"タイトル: {title}\n本文: {body}"
    async with sem:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=256,
        )
    answer = resp.choices[0].message.content.strip()
    if "Pick" in answer:
        return "Pick"
    elif "Decline" in answer:
        return "Decline"
    return answer


async def main():
    df = pd.read_csv("data/test/twitter_test.csv")
    titles = df["title_original"].fillna("").astype(str).tolist()
    bodies = df["body_original"].fillna("").astype(str).tolist()

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    print(f"Evaluating {len(df)} samples with {MODEL}...")
    start = time.time()

    tasks = [classify(sem, t, b) for t, b in zip(titles, bodies)]
    preds = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    df["pred"] = preds
    correct = (df["pick"] == df["pred"]).sum()
    total = len(df)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

    for label in ["Pick", "Decline"]:
        subset = df[df["pick"] == label]
        c = (subset["pick"] == subset["pred"]).sum()
        print(f"  {label}: {c}/{len(subset)} = {c/len(subset):.1%}")

    errors = df[df["pick"] != df["pred"]]
    print(f"\nErrors: {len(errors)}")
    for _, row in errors.iterrows():
        body_short = str(row["body_original"])[:80].replace("\n", " ")
        print(f"  [{row['quality_test_id']}] expected={row['pick']} got={row['pred']} | {body_short}")

    df.to_csv("data/test/twitter_test_eval.csv", index=False)
    print("\nSaved to data/test/twitter_test_eval.csv")


if __name__ == "__main__":
    asyncio.run(main())
