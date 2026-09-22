"""Tests for tools/watch_sweep.py: the automated HF/GitHub watch sweep.

All network access is faked: a `fake_get` closure serves canned JSON keyed
on URL fragments, and a raising variant exercises the per-check error path.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.watch_sweep import (
    CHECKS,
    WatchError,
    _params_billions,
    check_ddalcu_iqmlx,
    check_jangqai_latest,
    check_novamlx_release,
    check_tq_gguf_70b,
    diff_check,
    format_baseline,
    format_report,
    load_snapshot,
    parse_args,
    run_sweep,
    save_snapshot,
)


def _entry(repo_id, downloads=10, lastModified="2026-09-22T00:00:00.000Z"):
    return {"id": repo_id, "downloads": downloads, "lastModified": lastModified}


class FakeGet:
    """Callable returning canned JSON by URL fragment; optional failures."""

    def __init__(self, routes, fail=()):
        self.routes = routes  # list of (fragment, payload)
        self.fail = set(fail)
        self.seen = []

    def __call__(self, url):
        self.seen.append(url)
        for frag in self.fail:
            if frag in url:
                raise WatchError("boom: %s" % url)
        for frag, payload in self.routes:
            if frag in url:
                return payload
        raise AssertionError("no fake route for %s" % url)


def tq_payload():
    return [
        _entry("Anjielon/ODINO-397B-v34a-TQ1_0", 415),
        _entry("BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked", 97),
        _entry("GeorgyGUF/Llama-4-Maverick-17B-128E-Instruct-tq1_0.gguf", 44),
    ]


class TestParamsBillions(unittest.TestCase):
    def test_moe_ids(self):
        self.assertEqual(_params_billions("Qwen3.5-35B-A3B"), 35.0)
        self.assertEqual(_params_billions("Qwen3.5-397B-A17B-TQ1_0-GGUF"), 397.0)

    def test_plain_ids(self):
        self.assertEqual(_params_billions("meta-llama/Llama-3.1-70B"), 70.0)
        self.assertEqual(_params_billions("unsloth/qwen3-8b-GGUF"), 8.0)

    def test_decimal(self):
        self.assertEqual(_params_billions("nvidia/Nemotron-3-Nano-30B-A3B"), 30.0)

    def test_no_match(self):
        self.assertIsNone(_params_billions("foo/bar"))
        self.assertIsNone(_params_billions(None))

    def test_bitnet_b158_does_not_parse(self):
        # "1.58" must not be read as 1.58B and "2B-4T" must parse as 2.0
        self.assertEqual(_params_billions("microsoft/BitNet-b1.58-2B-4T"), 2.0)


class TestTqCheck(unittest.TestCase):
    def test_flags_60b_plus(self):
        get = FakeGet([("search=TQ1_0", tq_payload()), ("search=TQ2_0", tq_payload())])
        d = check_tq_gguf_70b(get)
        self.assertEqual(d["top"]["TQ1_0"], "Anjielon/ODINO-397B-v34a-TQ1_0")
        big = [b for b in d["big_entries"] if b["quant"] == "TQ1_0"]
        # "397B" and the "100B" inside a chunked-GGUF filename both trip
        # the heuristic — same behavior as the manual sweeps (flag, then
        # dismiss on inspection).
        self.assertEqual(len(big), 2)
        self.assertEqual({b["paramsB"] for b in big}, {397.0, 100.0})

    def test_empty_search(self):
        get = FakeGet([("search=", [])])
        d = check_tq_gguf_70b(get)
        self.assertIsNone(d["top"]["TQ1_0"])
        self.assertEqual(d["big_entries"], [])


class TestJangQaiCheck(unittest.TestCase):
    def test_detects_jang_2x_35b(self):
        get = FakeGet(
            [
                (
                    "author=JANGQ-AI",
                    [
                        _entry("JANGQ-AI/Qwen3.6-35B-A3B-JANG_2L"),
                        _entry("JANGQ-AI/Spark-X2.5-4B-JANG_8M"),
                    ],
                )
            ]
        )
        d = check_jangqai_latest(get)
        self.assertEqual(d["jang_2x_35b"], ["JANGQ-AI/Qwen3.6-35B-A3B-JANG_2L"])
        self.assertEqual(d["newest"][0]["id"], "JANGQ-AI/Qwen3.6-35B-A3B-JANG_2L")


class TestNovamlxCheck(unittest.TestCase):
    def test_release_shape(self):
        get = FakeGet(
            [
                (
                    "cnshsliu/novamlx",
                    [{"tag_name": "v0.9.0", "published_at": "2026-09-20"}],
                )
            ]
        )
        d = check_novamlx_release(get)
        self.assertEqual(d["release"]["tag"], "v0.9.0")
        self.assertEqual(d["release"]["published_at"], "2026-09-20")

    def test_no_releases(self):
        get = FakeGet([("cnshsliu/novamlx", [])])
        d = check_novamlx_release(get)
        self.assertIsNone(d["release"]["tag"])

    def test_baseline_renders_tag(self):
        get = FakeGet(
            [("cnshsliu/novamlx", [{"tag_name": "v0.9.0", "published_at": "x"}])]
        )
        bl = format_baseline({"novamlx_release": check_novamlx_release(get)}, {})
        self.assertTrue(any("novamlx_release: v0.9.0" in l for l in bl))

    def test_tag_bump_shows_in_deltas(self):
        # diff_check takes the per-check snapshot dict (as format_report passes it)
        old = {"release": {"tag": "v0.9.0"}}
        new = {"release": {"tag": "v0.10.0"}}
        changed, lines = diff_check("novamlx_release", old, new)
        self.assertTrue(changed)
        self.assertTrue(any("release tag v0.9.0 -> v0.10.0" in l for l in lines))


class TestDdalcuCheck(unittest.TestCase):
    def test_json_serializable(self):
        get = FakeGet(
            [
                (
                    "author=ddalcu",
                    [
                        _entry("ddalcu/Qwen3.8-27B-4bit-iQ-MLX", 500),
                        _entry("ddalcu/Qwen-Image-2.1-MLX-Serve", 5),
                    ],
                )
            ]
        )
        d = check_ddalcu_iqmlx(get)
        self.assertEqual(d["hits"], [])
        # must survive json round-trip (snapshot): tuple -> list
        self.assertEqual(
            d["family_top"], ("ddalcu/Qwen3.8-27B-4bit-iQ-MLX", 500)
        )
        self.assertEqual(
            json.loads(json.dumps(d))["family_top"],
            ["ddalcu/Qwen3.8-27B-4bit-iQ-MLX", 500],
        )


class TestDiffCheck(unittest.TestCase):
    def test_first_run_is_baseline(self):
        changed, lines = diff_check("foo", None, {"x": 1})
        self.assertTrue(changed)
        self.assertIn("foo: baseline recorded", lines)

    def test_new_and_gone_ids(self):
        old = {"newest": [{"id": "a/A"}, {"id": "b/B"}]}
        new = {"newest": [{"id": "a/A"}, {"id": "c/C"}]}
        changed, lines = diff_check("foo", old, new)
        self.assertTrue(changed)
        self.assertIn("foo: NEW c/C", lines)
        self.assertIn("foo: GONE b/B", lines)

    def test_no_change(self):
        d = {"newest": [{"id": "a/A"}], "downloads": 0,
             "release": {"tag": "v1"}, "newest_modified": "2026-09-22"}
        changed, lines = diff_check("foo", d, dict(d))
        self.assertFalse(changed)
        self.assertEqual(lines, [])

    def test_scalar_changes(self):
        old = {"downloads": 0, "release": {"tag": "v1.6.64"},
               "newest_modified": "2026-09-09"}
        new = {"downloads": 12, "release": {"tag": "v1.6.70"},
               "newest_modified": "2026-09-09"}
        changed, lines = diff_check("nemotron_dl", old, new)
        self.assertTrue(changed)
        self.assertTrue(any("nemotron downloads 0 -> 12" in l for l in lines))
        self.assertTrue(any("release tag v1.6.64 -> v1.6.70" in l for l in lines))

    def test_scalar_first_seen_is_not_a_change(self):
        # downloads None -> 0 on first real fetch is baseline, not churn
        changed, lines = diff_check(
            "nemotron_dl", {"downloads": None}, {"downloads": 0}
        )
        self.assertFalse(changed)


class TestRunSweep(unittest.TestCase):
    def _routes(self):
        return [
            ("search=TQ1_0", tq_payload()),
            ("search=TQ2_0", tq_payload()),
            ("author=JANGQ-AI&sort", [_entry("JANGQ-AI/Spark-X2.5-4B-JANG_8M")]),
            ("author=Jundot&sort", [_entry("Jundot/X-oQ3e-mtp")]),
            ("author=ddalcu", []),
            ("Nemotron-3-Nano", {"downloads": 0, "likes": 0}),
            ("author=microsoft&sort", [_entry("microsoft/BitNet-b1.58-2B-4T", 9000)]),
            ("jjang-ai/vmlx", [{"tag_name": "v1.6.64", "published_at": "2026-09-19"}]),
            ("0xZKnw/mlxl3", [{"tag_name": "v1.1.1", "published_at": "2026-09-22"}]),
            ("cnshsliu/novamlx", [{"tag_name": "v0.9.0", "published_at": "2026-09-20"}]),
            ("search=35B-A3B-exl3", [_entry("yeasah/Qwen3.6-35B-A3B-exl3")]),
        ]

    def test_all_ten_checks_run(self):
        results, errors = run_sweep(FakeGet(self._routes()))
        self.assertEqual(errors, {})
        self.assertEqual(len(results), 10)
        self.assertEqual(len(CHECKS), 10)

    def test_one_failing_check_does_not_kill_sweep(self):
        get = FakeGet(self._routes(), fail=("jjang-ai/vmlx",))
        results, errors = run_sweep(get)
        self.assertIn("vmlx_release", errors)
        self.assertEqual(len(results), 9)

    def test_report_and_baseline(self):
        results, errors = run_sweep(FakeGet(self._routes()))
        report = format_report(results, errors, None, full=True)
        self.assertIn("WATCH SWEEP", report)
        self.assertIn("CHANGES:", report)
        self.assertIn("BASELINE:", report)
        self.assertIn("tq: >=60B entries:", report)
        bl = format_baseline(results, errors)
        self.assertEqual(len(bl), 10)
        # release lines render tag + date
        self.assertTrue(any("v1.6.64" in l for l in bl))

    def test_ddalcu_baseline_names_hit_ids(self):
        results = {
            "ddalcu_iqmlx": check_ddalcu_iqmlx(
                FakeGet(
                    [
                        (
                            "author=ddalcu",
                            [_entry("ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit")],
                        )
                    ]
                )
            )
        }
        bl = format_baseline(results, {})
        self.assertIn("ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit", bl[0])


class TestSnapshot(unittest.TestCase):
    def test_save_load_roundtrip(self):
        results, _ = run_sweep(FakeGet(TestRunSweep()._routes()))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.json")
            self.assertIsNone(load_snapshot(path))
            save_snapshot(path, results)
            snap = load_snapshot(path)
            self.assertIn("swept_at", snap)
            self.assertEqual(snap["checks"]["nemotron_dl"]["downloads"], 0)

    def test_delta_against_old_snapshot(self):
        old_results, _ = run_sweep(FakeGet(TestRunSweep()._routes()))
        # new sweep: a JANGQ-AI upload appears, Nemotron gets downloads
        routes = TestRunSweep()._routes()
        routes[2] = (
            "author=JANGQ-AI&sort",
            [
                _entry("JANGQ-AI/Qwen3.6-35B-A3B-JANG_2L", 5),
                _entry("JANGQ-AI/Spark-X2.5-4B-JANG_8M"),
            ],
        )
        routes[5] = ("Nemotron-3-Nano", {"downloads": 7, "likes": 2})
        new_results, errors = run_sweep(FakeGet(routes))
        report = format_report(
            new_results, errors, {"swept_at": "t0", "checks": old_results}
        )
        self.assertIn("jangqai_latest: NEW JANGQ-AI/Qwen3.6-35B-A3B-JANG_2L", report)
        self.assertIn("nemotron_dl: nemotron downloads 0 -> 7", report)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertFalse(args.full)
        self.assertTrue(args.snapshot.endswith("watch_snapshot.json"))
        self.assertEqual(args.timeout, 20)


if __name__ == "__main__":
    unittest.main()
