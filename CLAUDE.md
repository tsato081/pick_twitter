sbintuitions/modernbert-ja-70mをベースに、Twitterのセンチメントを分類するモデルを作るrepo.

# tasks
分類対象：リンクにtwitter.com/x.comを含むもの。データはすべてこの中から取ってきたものを使う。
入力：title, body（テストデータは上記の対象記事のtitle_original, body_original）
出力：'Pick', 'Decline'の二値
分類基準：
- 特定の企業の名前が出ている
- 上の企業に対して、批判的なコメントがある

# rules
- uv環境を前提にすること。
- csvファイルは必ずpandasを使って読むこと。 `wc -l`などのコマンドは使用しない。
- 学習はローカルでは行わない。クラウド上のmps環境で行う。
- 環境変数はload_dotenvを使う。
- `AsyncOpenAI()`は使用しない。asyncioで実行すること。
- OpenAI APIを含むスクリプトの実行は必ずバックグラウンドで行うこと。

# references
- definitions/category_definitions.py: 87カテゴリの定義（CATEGORY_DEFINE_LIST）
- definitions/category_flow.md: カテゴリ分類のワークフロー・ルールブック（別repoで使用していたもの）
- docs/openai_gpt54_api.md: GPT-5.4/5.4-mini APIの叩き方（モデルID、料金、コード例）
- docs/openai_search_api.md: OpenAI Web Search APIの使い方（Responses API/Chat Completions API）
