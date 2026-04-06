"""OpenAI Search APIでカテゴリ別にTwitter投稿を大量収集する"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = "gpt-5.4-mini"

# 具体的な企業名+カテゴリで多様な検索クエリを生成
SEARCH_QUERIES = [
    # カネ
    "site:x.com 倒産 株式会社",
    "site:x.com 破産手続き 開始決定",
    "site:x.com 民事再生 申請",
    "site:x.com 赤字 決算 億円",
    "site:x.com 業績悪化 下方修正",
    "site:x.com 債務超過 経営",
    "site:x.com 粉飾決算 不正会計",
    "site:x.com 賃金未払い 会社",
    "site:x.com 横領 社員 逮捕",
    "site:x.com 詐欺 会社 被害",
    "site:x.com 資金調達 億円 スタートアップ",
    "site:x.com 増収増益 決算 好調",
    "site:x.com 上方修正 業績",
    "site:x.com 価格改定 値上げ 企業",
    # ヒト
    "site:x.com パワハラ 会社 告発",
    "site:x.com セクハラ 企業 処分",
    "site:x.com リストラ 人員削減 早期退職",
    "site:x.com 労災 死亡事故 工場",
    "site:x.com 従業員 不祥事 解雇",
    "site:x.com ストライキ 労組 賃上げ",
    "site:x.com 新卒採用 大量採用 企業",
    "site:x.com 賃上げ ベースアップ 春闘",
    "site:x.com 違法残業 過労死 企業",
    "site:x.com 役員変更 社長交代 就任",
    # モノ
    "site:x.com 異物混入 回収 食品",
    "site:x.com 品質不正 データ改ざん メーカー",
    "site:x.com 工場火災 爆発 企業",
    "site:x.com リコール 不具合 自動車",
    "site:x.com 新商品 発売 企業",
    "site:x.com 新店舗オープン 開店",
    "site:x.com 生産停止 出荷停止 メーカー",
    "site:x.com 新工場 建設 竣工",
    "site:x.com 商品表示 不備 回収",
    # 情報
    "site:x.com サイバー攻撃 ランサムウェア 企業",
    "site:x.com 個人情報漏洩 不正アクセス",
    "site:x.com システム障害 サービス停止 企業",
    "site:x.com 情報漏洩 顧客データ",
    "site:x.com 通信障害 復旧 キャリア",
    # 経営
    "site:x.com M&A 買収 合意",
    "site:x.com 子会社化 TOB 株式取得",
    "site:x.com 社名変更 新社名",
    "site:x.com 本社移転 移転先",
    "site:x.com 社長 死去 訃報 企業",
    "site:x.com 代表取締役 逮捕 容疑",
    "site:x.com 行政処分 業務停止命令",
    "site:x.com 事業撤退 サービス終了",
    "site:x.com 業務提携 協業 発表",
    "site:x.com IPO 上場 承認",
    "site:x.com 事業縮小 店舗閉鎖",
    "site:x.com 海外進出 海外拠点",
    "site:x.com 不適切発言 炎上 企業",
    "site:x.com 訴訟 損害賠償 企業",
    # Decline用
    "site:x.com 今日のランチ おいしい",
    "site:x.com 映画 感想 おすすめ",
    "site:x.com ゲーム 攻略 クリア",
    "site:x.com 推し活 ライブ 最高",
    "site:x.com 散歩 天気 気持ちいい",
    "site:x.com 料理 作った レシピ",
    "site:x.com 読書 感想 小説",
    "site:x.com 旅行 観光 楽しかった",
]


def search_tweets(query: str, max_results: int = 20) -> list[dict]:
    prompt = f"""以下の検索クエリでTwitter(x.com)上の日本語投稿を探し、{max_results}件の投稿テキストをJSON配列で返してください。

検索クエリ: {query}

以下のJSON形式のみを出力:
[{{"text": "投稿本文"}}, ...]"""

    try:
        response = client.responses.create(
            model=MODEL,
            tools=[{
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "JP",
                },
            }],
            input=prompt,
        )

        text = ""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text = content.text

        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            elif "[" in text:
                start = text.index("[")
                end = text.rindex("]") + 1
                json_str = text[start:end]
            else:
                return []

            results = json.loads(json_str)
            if isinstance(results, list):
                return [r for r in results if isinstance(r, dict) and "text" in r]
            return []
        except (json.JSONDecodeError, ValueError):
            return []
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        return []


def sanitize(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def main():
    all_results = []
    total = len(SEARCH_QUERIES)

    for i, query in enumerate(SEARCH_QUERIES):
        print(f"[{i+1}/{total}] {query}", flush=True)
        results = search_tweets(query, max_results=20)
        for r in results:
            r["search_query"] = query
        all_results.extend(results)
        print(f"  -> {len(results)} tweets", flush=True)
        time.sleep(0.5)

    df = pd.DataFrame(all_results)
    if len(df) == 0:
        print("No results found!")
        return

    df["text"] = df["text"].apply(sanitize)
    df = df[df["text"].str.len() > 10]  # 短すぎるの除外
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df = df.rename(columns={"text": "BODY"})
    df["TITLE"] = ""

    print(f"\nTotal unique tweets: {len(df)}", flush=True)

    output_path = Path("data/raw/search_collected_v2.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
