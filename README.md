# PyTorch Lightning + Hydra + Wandb テンプレート

PyTorch, Lightning, Hydra, Wandb を使用した、スケーラブルで管理しやすいディープラーニングプロジェクトのためのテンプレートリポジトリです。

## ✨ 特徴

- **Reproducibility (再現性):** `uv` による依存関係の固定と、`Hydra` による設定管理で、誰でも同じ実験結果を再現できます。
- **Scalability (拡張性):** `src` レイアウトとモジュール化された設定により、プロジェクトが大規模になってもコードベースをクリーンに保ちます。
- **Best Practices:** `ruff` によるリンティングとフォーマット、`pytest` によるテスト、GitHub Actions によるCI/CDなど、モダンな開発プラクティスを導入済みです。
- **Easy to Use (使いやすさ):** MNISTの分類タスクをサンプルとして実装済み。すぐに動かして、テンプレートの動作を確認できます。

## 🚀 使い方

### 1. セットアップ

まず、このリポジトリをクローンまたはテンプレートとして使用し、ローカルに展開します。

次に、`uv` を使って仮想環境を作成し、依存関係をインストールします。`uv` がインストールされていない場合は、先にインストールしてください。

```bash
# 仮想環境を作成 (初回のみ)
uv venv

# 仮想環境をアクティベート
source .venv/bin/activate

# 依存関係をインストール
uv pip install -e ".[dev]"
```

### 2. 学習の実行

`train.py` スクリプトを実行することで、モデルの学習を開始できます。Hydraのおかげで、コマンドラインから設定を簡単に上書きできます。

```bash
# デフォルト設定で学習を実行
python src/train.py

# 設定を上書きして学習を実行 (例: バッチサイズとエポック数を変更)
python src/train.py datamodule.batch_size=128 trainer.max_epochs=20

# マルチラン (例: オプティマイザの学習率を複数試す)
python src/train.py --multirun model.optimizer_cfg.lr=0.001,0.01
```

学習の進捗と結果は、[Weights & Biases](https://wandb.ai) に自動的に記録されます。

### 3. テストの実行

プロジェクトのテストは `pytest` を使って実行できます。

```bash
uv run pytest
```

### 4. コード品質チェック

コードのリンティングとフォーマットは `ruff` を使って行います。

```bash
# コードのチェック
uv run ruff check .

# コードのフォーマット
uv run ruff format .
```

## 📂 プロジェクト構造

```text
.
├── .github/workflows/      # GitHub Actions のワークフロー
├── configs/                # Hydra の設定ファイル
│   ├── datamodule/
│   ├── model/
│   ├── trainer/
│   └── ...
├── data/                   # (Git管理外) データセット
├── notebooks/              # 実験用のJupyter Notebook
├── outputs/                # (Git管理外) Hydra の出力 (ログ、チェックポイントなど)
├── src/                    # ソースコード
│   ├── dl_template/
│   │   ├── datamodules/    # LightningDataModules
│   │   └── models/         # LightningModules
│   └── train.py            # 学習実行スクリプト
├── tests/                  # テストコード
├── .gitignore
├── pyproject.toml          # プロジェクト設定と依存関係 (uv + ruff)
└── README.md
```
