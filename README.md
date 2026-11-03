# ZeroWeight

**v0.1.0 — PyTorchの途中値と重み変更を調べる、小さな共通ツール。**

入力を揃えて2つのモデルを実行し、各観測点の値・重み・出力の差を記録する。学習・重み探索は行わない。この公開リポジトリには研究モデルの重み、改善した重み、それらを再現する生成コード・実験結果は含めない。

## 実行

Python 3.12。検証環境: PyTorch 2.10.0+cu128、NumPy 2.5.4（テストはCPU実行）。

```sh
python -m venv .venv
# 仮想環境を有効にして実行
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m zeroweight --output reports/toy.json --save-tensors
python -m zeroweight --change none --output reports/unchanged.json
```

サンプルは2出力の単純な計算だけを使う。片方のバイアスを変え、最初の差が `logits` に現れることを示す。研究用の生成式や改善モデルを必要としない。

## 自分の推論処理へ組み込む

```python
from zeroweight.inspect import Trace, compare_models, save_report

def forward(weights, tokens, *, trace=None):
    # 既存の推論処理で、調べたい場所に以下を追加する
    hidden = your_hidden_computation(weights, tokens)
    if trace is not None:
        trace("layer1.hidden", hidden)
    logits = your_output_computation(weights, hidden)
    if trace is not None:
        trace("logits", logits)
    return logits  # (batch, positions, vocabulary)

trace = Trace(patterns=("layer1.*", "logits"))
logits = forward(original_weights, tokens, trace=trace)
print(trace.tensors["layer1.hidden"])
trace.save("trace.pt")

report, before_trace, after_trace = compare_models(
    original_weights, edited_weights, tokens, forward,
    atol=1e-6, rtol=1e-5,
)
save_report(report, "comparison.json")
```

上は組み込み例。`your_*` とモデル・入力は利用側で定義する。辞書形式の重みを受け取り、`trace` を渡せる推論関数なら使える。アーキテクチャが異なる場合は共通の観測点名を付ける。

## 出力

- `Trace.tensors`: 観測点ごとの実際の途中値。CPUへ複製し、勾配から切り離す。
- `weights`: 名前ごとの重みの差、形、型、最大・平均絶対差。
- `activations`: 実行順に並ぶ観測点の差。
- `first_changed_activation`: 観測した点の中で最初に差が出た場所。原因の証明ではない。
- `outputs`: logitsの差。`argmax_changed_positions` は同じ入力の全位置で選択が変わった数であり、課題の正解率ではない。
- 重み・入力のSHA-256、入力トークン、`cfg` があれば構成。重みのハッシュはテンソル名・型・形・値が対象。コード・位置符号・トークナイザーは含まないため、再現時はコードのコミットと環境も別途保存する。

## 制約

記録容量は1回の `Trace` あたり既定64MiB。上限超過は明示的に失敗する。大きな入力では名前フィルタや小さなバッチを使う。上限は記録の複製に対するもので、モデルの推論メモリ全体を制限しない。自己回帰生成では各forwardに新しいTraceを使う。同名への上書きは拒否する。

コールバックは渡された値を変更しないこと。付属Traceは複製する。比較時のNaN/Infは一致扱いにしない。数値が等しく型だけ違う場合は `equal` と型の違いを併記する。比較は実数の浮動小数テンソルを想定し、複素数や巨大な整数の厳密比較用ではない。比較はCPUのfloat64へ変換するため大規模モデルではメモリ負荷に注意する。

実験の精度は、モデルごとに同じ問題から自己回帰生成して別途評価する。学習済みモデルは利用側で評価モードにし、乱数・前処理を揃える。`compare_models` は学習をしないが、利用側の関数による状態変更は抑止しない。

`.pt` はローカルの途中値保存用。信頼できるファイルだけ `torch.load(path, weights_only=True)` で読む。出力レポートにも入力データが含まれるため、公開する前に内容を確認する。

## 公開範囲の確認

公開対象は共通ツール・一般サンプル・テスト・説明・検査スクリプトのみ。研究フォルダーはGit管理から除外する。公開前に以下を実行する。

```sh
python scripts/check_public_files.py
```

この検査はステージ済みのファイル一覧を許可リストと照合する。内容の秘密検査を完全に代替するものではない。
