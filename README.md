# pick_twitter

sbintuitions/modernbert-ja-70m をベースに、Twitter投稿のPick/Decline二値分類モデルを作る。

## Quick Start (クラウド学習)

### 1. 環境構築

```bash
git clone <repo-url>
cd pick_twitter
uv sync
```

### 2. データ取得

```bash
# HuggingFace からラベル付きデータをダウンロード (private repo)
# HF_TOKEN を .env に設定済みの前提
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from datasets import load_dataset
ds = load_dataset('teru00801/pick-twitter')
ds['train'].to_csv('data/train/train.csv', index=False)
ds['test'].to_csv('data/test/twitter_test.csv', index=False)
ds['canary'].to_csv('data/test/canary.csv', index=False)
print('Done')
"
```

### 3. 学習

```bash
uv run python scripts/train.py \
  --epochs 5 \
  --batch_size 32 \
  --lr 2e-5 \
  --output_dir models/pick-classifier
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--epochs` | 5 | エポック数 |
| `--batch_size` | 32 | バッチサイズ (メモリに応じて調整) |
| `--lr` | 2e-5 | 学習率 |
| `--output_dir` | models/pick-classifier | モデル保存先 |

MPS/CUDA/CPUは自動検出。学習後にテストデータ (`data/test/twitter_test.csv`) で自動評価。

### 4. テストデータでの評価

学習完了時に自動で実行されます。別途単体で実行する場合:

```bash
uv run python scripts/evaluate.py
```

`models/` 以下の最新モデルを自動検出し、`twitter_test.csv` と `canary.csv` の両方で評価します。

### 5. モデルをHuggingFaceにpush

精度を確認した上で実行:

```bash
uv run python scripts/push_model.py --repo_id teru00801/pick-twitter-classifier
```
