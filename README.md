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
