"""2段階判定でrawデータにPick/Declineラベルを付与する"""

import asyncio
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

STAGE1_PROMPT = Path("prompts/pick_stage1.txt").read_text().split("---")[0].strip()
STAGE2_PROMPT = Path("prompts/pick_stage2.txt").read_text().split("---")[0].strip()

client = OpenAI()
MODEL = "gpt-5.4-mini"
MAX_CONCURRENT = 30


def sanitize(text: str) -> str:
    """制御文字を除去し、長すぎるテキストを切り詰める"""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:3000]


async def call_llm(sem: asyncio.Semaphore, system: str, user_msg: str) -> str:
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
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
                    print(f"  Error after 3 attempts: {e}")
                    return "Error"
                await asyncio.sleep(2 ** attempt)


async def classify(sem: asyncio.Semaphore, title: str, body: str) -> tuple[str, str]:
    user_msg = f"タイトル: {sanitize(title)}\n本文: {sanitize(body)}"
    s1 = await call_llm(sem, STAGE1_PROMPT, user_msg)
    if s1 != "Pick":
        return s1, "s1"
    s2 = await call_llm(sem, STAGE2_PROMPT, user_msg)
    return s2, "s2"


async def main():
    df = pd.read_csv("data/raw/search_collected_v2.csv")
    df = df.reset_index(drop=True)
    print(f"Total: {len(df)} samples", flush=True)

    titles = df["TITLE"].fillna("").astype(str).tolist()
    bodies = df["BODY"].fillna("").astype(str).tolist()

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    start = time.time()

    tasks = [classify(sem, t, b) for t, b in zip(titles, bodies)]
    results = await asyncio.gather(*tasks)
    preds = [r[0] for r in results]
    stages = [r[1] for r in results]

    elapsed = time.time() - start

    df["pick"] = preds
    df["stage"] = stages

    pick_count = (df["pick"] == "Pick").sum()
    decline_count = (df["pick"] == "Decline").sum()
    error_count = (df["pick"] == "Error").sum()
    print(f"Done in {elapsed:.1f}s", flush=True)
    print(f"Pick: {pick_count}, Decline: {decline_count}, Error: {error_count}", flush=True)
    print(f"Stage1 Decline: {sum(1 for s in stages if s == 's1')}, Passed to Stage2: {sum(1 for s in stages if s == 's2')}", flush=True)

    df.to_csv("data/train/search_v2_labeled.csv", index=False)
    print("Saved to data/train/search_v2_labeled.csv", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
