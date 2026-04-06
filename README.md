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

データを `data/train/` に配置:
- `empty_title_critical_labeled.csv` (6,495件, Pick多め)
- `twitter_labeled.csv` (10,000件, Decline多め)

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

学習完了時に自動で実行されます。出力例:
```
Accuracy: XX.X%
              precision    recall  f1-score   support
     Decline       ...
        Pick       ...
```
