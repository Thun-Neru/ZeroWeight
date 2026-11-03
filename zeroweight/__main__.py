"""Small synthetic example; contains no research model or improved weights."""
import argparse
import copy
from pathlib import Path

import torch

from .inspect import compare_models, save_report


def toy_forward(weights, tokens, *, trace=None):
    x = tokens.to(weights["scale"].dtype).unsqueeze(-1)
    hidden = x * weights["scale"]
    logits = hidden + weights["bias"]
    if trace is not None:
        trace("hidden", hidden)
        trace("logits", logits)
    return logits


def main():
    parser = argparse.ArgumentParser(description="途中値・重み変更比較の一般サンプル（研究モデル不使用）")
    parser.add_argument("--output", type=Path, default=Path("reports/toy.json"))
    parser.add_argument("--change", choices=["none", "bias"], default="bias")
    parser.add_argument("--save-tensors", action="store_true")
    args = parser.parse_args()
    before = {"scale": torch.tensor([1., -1.]), "bias": torch.zeros(2)}
    after = copy.deepcopy(before)
    if args.change == "bias":
        after["bias"][1] = 3
    tokens = torch.tensor([[0, 1, 2]])
    report, left, right = compare_models(before, after, tokens, toy_forward)
    report["example"] = "synthetic_only"
    report["environment"] = {"torch": torch.__version__, "device": "cpu"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, args.output)
    if args.save_tensors:
        left.save(args.output.with_suffix(".before.pt"))
        right.save(args.output.with_suffix(".after.pt"))
    print(f"Report: {args.output}; first difference: {report['first_changed_activation']}")


if __name__ == "__main__":
    main()
