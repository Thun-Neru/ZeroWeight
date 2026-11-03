"""Opt-in activation snapshots and same-input model comparisons (CPU reports)."""
from fnmatch import fnmatchcase
import hashlib
import json
import math
from pathlib import Path

import torch


class Trace:
    """Detached snapshots, filtered before copying, bounded to prevent accidental OOM.

    Names follow forward execution order. Use a new Trace per decoding step.
    Callback implementations must not mutate the supplied tensors; this recorder copies.
    """
    def __init__(self, patterns=("*",), max_bytes=64 * 1024 * 1024):
        self.patterns = tuple(patterns)
        self.max_bytes = max_bytes
        self.tensors = {}
        self.bytes = 0

    def __call__(self, name, value):
        if not any(fnmatchcase(name, pattern) for pattern in self.patterns):
            return
        if name in self.tensors:
            raise ValueError(f"Duplicate trace name: {name}; use a fresh Trace per forward")
        size = value.numel() * value.element_size()
        if self.bytes + size > self.max_bytes:
            raise ValueError("Trace size limit exceeded; filter names or use a smaller input")
        self.tensors[name] = value.detach().cpu().clone()
        self.bytes += size

    def save(self, path):
        """Save exact tensors; load trusted files with torch.load(..., weights_only=True)."""
        torch.save(self.tensors, path)


def tensors(tree, prefix=""):
    """Flatten named tensor dictionaries without exposing non-tensor metadata."""
    for name, value in tree.items():
        path = f"{prefix}.{name}" if prefix else name
        if torch.is_tensor(value):
            yield path, value
        elif isinstance(value, dict):
            yield from tensors(value, path)


def fingerprint(tree):
    digest = hashlib.sha256()
    for name, value in tensors(tree):
        v = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(v.dtype), list(v.shape)]).encode())
        digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare_tensors(before, after, *, atol=0.0, rtol=0.0):
    if atol < 0 or rtol < 0 or not all(math.isfinite(v) for v in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    rows = []
    for name in dict.fromkeys([*before, *after]):
        a, b = before.get(name), after.get(name)
        if a is None or b is None:
            rows.append({"name": name, "status": "added" if a is None else "removed"})
            continue
        row = {"name": name, "shape_before": list(a.shape), "shape_after": list(b.shape),
               "dtype_before": str(a.dtype), "dtype_after": str(b.dtype)}
        if a.shape != b.shape:
            rows.append({**row, "status": "shape_changed"})
            continue
        x, y = a.detach().cpu().double(), b.detach().cpu().double()
        finite = torch.isfinite(x) & torch.isfinite(y)
        close = torch.isclose(x, y, atol=atol, rtol=rtol, equal_nan=False) & finite
        delta = (x[finite] - y[finite]).abs()
        rows.append({**row, "status": "equal" if bool(close.all()) else "changed",
                     "changed": int((~close).sum()), "total": x.numel(),
                     "nonfinite": int((~finite).sum()),
                     "max_abs": float(delta.max()) if delta.numel() else None,
                     "mean_abs": float(delta.mean()) if delta.numel() else None})
    return rows


def compare_models(before, after, tokens, forward, *, atol=0.0, rtol=0.0,
                   patterns=("*",), max_bytes=64 * 1024 * 1024):
    """Identical input, no fitting/search; forward(W, tokens, trace=...) required.

    Token differences refer to all supplied positions, not task accuracy. For
    autoregressive task accuracy run both models on the same prompts separately.
    """
    left, right = Trace(patterns, max_bytes), Trace(patterns, max_bytes)
    with torch.no_grad():
        a = forward(before, tokens, trace=left)
        b = forward(after, tokens, trace=right)
    activations = compare_tensors(left.tensors, right.tensors, atol=atol, rtol=rtol)
    outputs = compare_tensors({"logits": a}, {"logits": b}, atol=atol, rtol=rtol)[0]
    compatible = a.shape == b.shape
    result = {
        "schema_version": 1, "atol": atol, "rtol": rtol,
        "before_sha256": fingerprint(before), "after_sha256": fingerprint(after),
        "input_sha256": fingerprint({"tokens": tokens}),
        "input_tokens": tokens.detach().cpu().tolist(),
        "config_before": before.get("cfg"), "config_after": after.get("cfg"),
        "weights": compare_tensors(dict(tensors(before)), dict(tensors(after)), atol=atol, rtol=rtol),
        "activations": activations,
        "first_changed_activation": next((r["name"] for r in activations if r["status"] != "equal"), None),
        "outputs": outputs,
        "argmax_changed_positions": int((a.argmax(-1).cpu() != b.argmax(-1).cpu()).sum()) if compatible else None,
        "argmax_total_positions": a.shape[0] * a.shape[1] if compatible else None,
    }
    return result, left, right


def save_report(report, path):
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
