import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from zeroweight.__main__ import toy_forward
from zeroweight.inspect import Trace, compare_models, compare_tensors, save_report


class InspectionTest(unittest.TestCase):
    def test_identity_and_known_edit(self):
        before = {"scale": torch.tensor([1., -1.]), "bias": torch.zeros(2)}
        inputs = torch.tensor([[0, 1, 2]])
        same, _, _ = compare_models(before, copy.deepcopy(before), inputs, toy_forward)
        self.assertIsNone(same["first_changed_activation"])
        self.assertEqual(same["argmax_changed_positions"], 0)
        after = copy.deepcopy(before)
        after["bias"][1] = 3
        report, left, _ = compare_models(before, after, inputs, toy_forward)
        self.assertEqual(report["first_changed_activation"], "logits")
        self.assertEqual(report["argmax_changed_positions"], 2)
        self.assertEqual([r["name"] for r in report["weights"] if r["status"] != "equal"], ["bias"])
        self.assertTrue(torch.equal(left.tensors["logits"], toy_forward(before, inputs)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            save_report(report, path)
            self.assertEqual(json.loads(path.read_text())["input_tokens"], inputs.tolist())
            left.save(Path(tmp) / "trace.pt")
            saved = torch.load(Path(tmp) / "trace.pt", weights_only=True)
            self.assertTrue(torch.equal(saved["hidden"], left.tensors["hidden"]))

    def test_detached_copy_filter_and_budget(self):
        trace = Trace(("hidden",), max_bytes=8)
        value = torch.ones(2, requires_grad=True)
        trace("ignored", torch.ones(100))
        trace("hidden", value)
        self.assertFalse(trace.tensors["hidden"].requires_grad)
        with torch.no_grad():
            value.zero_()
        self.assertEqual(trace.tensors["hidden"].tolist(), [1, 1])
        with self.assertRaises(ValueError):
            trace("hidden", value)
        with self.assertRaises(ValueError):
            Trace(max_bytes=1)("large", value)

    def test_nonfinite_missing_shape_and_tolerances(self):
        a = {"x": torch.tensor([1., float("nan")]), "removed": torch.ones(1), "shape": torch.ones(1)}
        b = {"x": torch.tensor([1.00001, float("nan")]), "added": torch.ones(1), "shape": torch.ones(2)}
        rows = {r["name"]: r for r in compare_tensors(a, b, atol=0.001)}
        self.assertEqual(rows["x"]["changed"], 1)
        self.assertEqual(rows["x"]["nonfinite"], 1)
        self.assertEqual(rows["shape"]["status"], "shape_changed")
        self.assertEqual(rows["removed"]["status"], "removed")
        self.assertEqual(rows["added"]["status"], "added")
        json.dumps(rows, allow_nan=False)
        with self.assertRaises(ValueError):
            compare_tensors({}, {}, atol=float("nan"))

    def test_dtype_and_output_shape_changes(self):
        a = {"scale": torch.tensor([1., -1.]), "bias": torch.zeros(2)}
        b = {k: v.double() for k, v in a.items()}
        report, _, _ = compare_models(a, b, torch.tensor([[1]]), toy_forward)
        self.assertEqual(report["outputs"]["status"], "equal")
        self.assertNotEqual(report["before_sha256"], report["after_sha256"])
        b = {"scale": torch.ones(3), "bias": torch.zeros(3)}
        report, _, _ = compare_models(a, b, torch.tensor([[1]]), toy_forward)
        self.assertEqual(report["outputs"]["status"], "shape_changed")
        self.assertIsNone(report["argmax_changed_positions"])


if __name__ == "__main__":
    unittest.main()
