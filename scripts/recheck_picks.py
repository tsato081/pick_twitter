"""twitter_labeled.csvのPick 194件を GPT-5.4 で再チェックする"""

import asyncio
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

STAGE2_PROMPT = Path("prompts/pick_stage2.txt").read_text().split("---")[0].strip()

client = OpenAI()
MODEL = "gpt-5.4"
MAX_CONCURRENT = 20


def sanitize(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:3000]


async def call_llm(sem: asyncio.Semaphore, title: str, body: str) -> str:
    user_msg = f"タイトル: {sanitize(title)}\n本文: {sanitize(body)}"
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": STAGE2_PROMPT},
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
            except Exception as e:
                if attempt == 2:
                    print(f"  Error: {e}")
                    return "Error"
                await asyncio.sleep(2 ** attempt)


async def main():
    df = pd.read_csv("data/train/twitter_labeled.csv")
    picks = df[df["pick"] == "Pick"].copy()
    print(f"Rechecking {len(picks)} Pick samples with {MODEL}...", flush=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    start = time.time()

    titles = picks["TITLE"].fillna("").astype(str).tolist()
    bodies = picks["BODY"].fillna("").astype(str).tolist()

    tasks = [call_llm(sem, t, b) for t, b in zip(titles, bodies)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    picks["recheck"] = results

    kept = (picks["recheck"] == "Pick").sum()
    flipped = (picks["recheck"] == "Decline").sum()
    print(f"Done in {elapsed:.1f}s", flush=True)
    print(f"Kept as Pick: {kept}, Flipped to Decline: {flipped}", flush=True)

    # Flipped ones
    flipped_df = picks[picks["recheck"] == "Decline"]
    print(f"\nFlipped to Decline:", flush=True)
    for _, row in flipped_df.iterrows():
        body = str(row["BODY"])[:120].replace("\n", " ")
        print(f"  {body}", flush=True)

    # 更新: flippedをDeclineに変更してtwitter_labeled.csvを上書き
    flip_idx = picks[picks["recheck"] == "Decline"].index
    df.loc[flip_idx, "pick"] = "Decline"
    df.to_csv("data/train/twitter_labeled.csv", index=False)
    print(f"\nUpdated data/train/twitter_labeled.csv ({len(flip_idx)} labels flipped)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
