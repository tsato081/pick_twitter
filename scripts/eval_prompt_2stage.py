"""2段階判定: Stage1(広く拾う) → Stage2(絞る)でPick/Decline判定"""

import asyncio
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
MAX_CONCURRENT = 20


async def call_llm(sem: asyncio.Semaphore, system: str, user_msg: str) -> str:
    async with sem:
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


async def classify(sem: asyncio.Semaphore, title: str, body: str) -> tuple[str, str]:
    user_msg = f"タイトル: {title}\n本文: {body}"

    # Stage 1
    s1 = await call_llm(sem, STAGE1_PROMPT, user_msg)
    if s1 != "Pick":
        return s1, "s1"

    # Stage 2 (only if Stage 1 said Pick)
    s2 = await call_llm(sem, STAGE2_PROMPT, user_msg)
    return s2, "s2"


async def main():
    df = pd.read_csv("data/test/twitter_test.csv")
    titles = df["title_original"].fillna("").astype(str).tolist()
    bodies = df["body_original"].fillna("").astype(str).tolist()

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    print(f"Evaluating {len(df)} samples with {MODEL} (2-stage)...")
    start = time.time()

    tasks = [classify(sem, t, b) for t, b in zip(titles, bodies)]
    results = await asyncio.gather(*tasks)
    preds = [r[0] for r in results]
    stages = [r[1] for r in results]

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    df["pred"] = preds
    df["stage"] = stages

    # Stage stats
    s1_decline = sum(1 for s in stages if s == "s1")
    s2_total = sum(1 for s in stages if s == "s2")
    print(f"\nStage1 Decline: {s1_decline}, Passed to Stage2: {s2_total}")

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
        print(f"  [{row['quality_test_id']}] expected={row['pick']} got={row['pred']} stage={row['stage']} | {body_short}")

    df.to_csv("data/test/twitter_test_eval_2stage.csv", index=False)
    print("\nSaved to data/test/twitter_test_eval_2stage.csv")


if __name__ == "__main__":
    asyncio.run(main())
