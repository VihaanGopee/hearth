"""Tests for tools/eval_battery.py — schema validity + check engine.

No network: these exercise the battery definition and the pure check_item()
function only. Live-model runs happen on the Mac via the CLI.
"""

import importlib.util
import unittest

SPEC = importlib.util.spec_from_file_location(
    "eval_battery", "tools/eval_battery.py")
eval_battery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_battery)

VALID_KINDS = {"contains", "contains_any", "not_contains", "exact",
               "sentence_count", "line_count"}


class TestBatterySchema(unittest.TestCase):
    def test_nonempty(self):
        self.assertGreaterEqual(len(eval_battery.BATTERY), 15)

    def test_ids_unique(self):
        ids = [item["id"] for item in eval_battery.BATTERY]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields(self):
        for item in eval_battery.BATTERY:
            self.assertIn("id", item)
            self.assertIn("category", item)
            self.assertTrue(item["prompt"].strip(), item["id"])
            self.assertTrue(item["checks"], item["id"])

    def test_check_kinds_valid(self):
        for item in eval_battery.BATTERY:
            for check in item["checks"]:
                self.assertIn(check["kind"], VALID_KINDS, item["id"])

    def test_category_coverage(self):
        cats = {item["category"] for item in eval_battery.BATTERY}
        # must span reasoning, honesty, and instruction-following at minimum
        for needed in ("math", "logic", "honesty", "instruction"):
            self.assertIn(needed, cats)
        self.assertGreaterEqual(len(cats), 5)

    def test_honesty_items_use_contains_any(self):
        for item in eval_battery.BATTERY:
            if item["category"] == "honesty":
                kinds = {c["kind"] for c in item["checks"]}
                self.assertIn("contains_any", kinds, item["id"])

    def test_exact_checks_have_sane_values(self):
        for item in eval_battery.BATTERY:
            for check in item["checks"]:
                if check["kind"] == "exact":
                    self.assertTrue(check["value"].strip(), item["id"])


class TestCheckItem(unittest.TestCase):
    def _item(self, checks):
        return {"id": "t", "category": "t", "prompt": "t", "checks": checks}

    def test_exact_pass(self):
        item = self._item([{"kind": "exact", "value": "6"}])
        passed, failed = eval_battery.check_item(item, "6")
        self.assertTrue(passed, failed)

    def test_exact_wrapped_fails(self):
        # strict by design: wrapping prose fails the check, not the model
        item = self._item([{"kind": "exact", "value": "6"}])
        passed, _ = eval_battery.check_item(item, "The answer is 6.")
        self.assertFalse(passed)

    def test_exact_ignores_trailing_punctuation(self):
        # regression: "Knave." must score the same as "knave"
        item = self._item([{"kind": "exact", "value": "knave"}])
        passed, failed = eval_battery.check_item(item, "Knave.")
        self.assertTrue(passed, failed)

    def test_contains_case_insensitive(self):
        item = self._item([{"kind": "contains", "values": ["mixed"]}])
        passed, failed = eval_battery.check_item(item, "The MIXED box.")
        self.assertTrue(passed, failed)

    def test_not_contains(self):
        item = self._item([{"kind": "not_contains", "values": ["blue"]}])
        passed, _ = eval_battery.check_item(item, "The ocean is blue.")
        self.assertFalse(passed)

    def test_contains_any(self):
        item = self._item([{"kind": "contains_any",
                            "values": ["no record", "fictional"]}])
        passed, failed = eval_battery.check_item(
            item, "I have no record of any such accord.")
        self.assertTrue(passed, failed)
        passed, _ = eval_battery.check_item(
            item, "The 2019 Oslo Quantum Computing Accord was signed...")
        self.assertFalse(passed)

    def test_sentence_count(self):
        item = self._item([{"kind": "sentence_count", "value": 3}])
        passed, failed = eval_battery.check_item(
            item, "One. Two. Three.")
        self.assertTrue(passed, failed)
        passed, _ = eval_battery.check_item(item, "One. Two.")
        self.assertFalse(passed)

    def test_line_count(self):
        item = self._item([{"kind": "line_count", "value": 2}])
        passed, failed = eval_battery.check_item(item, "a\nb\n")
        self.assertTrue(passed, failed)

    def test_unknown_kind_raises(self):
        item = self._item([{"kind": "bogus"}])
        with self.assertRaises(ValueError):
            eval_battery.check_item(item, "x")


if __name__ == "__main__":
    unittest.main()
