#!/usr/bin/env python3
"""PlotResonantLimitsTask.external_limit_file: an external_limits item of a limit_plots entry.

A string is a dhi external-limits file, passed on; {file, key} picks curves out of a file of
several references (here one per channel and spin) and writes them as a dhi file next to the
plot.
"""

import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402

CURVE = {"name": "Run 2 DL spin 0", "scan_parameter": "mhh", "limits": {"300.0": 6.6}}
REFERENCES = {
    "DL": {"spin0": CURVE, "spin2": [dict(CURVE, name="a"), dict(CURVE, name="b")]},
    "SL": {},
}


class ExternalLimitFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ana = os.path.join(self.tmp.name, "ana")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(os.path.join(self.ana, "config"))
        os.makedirs(self.out)
        with open(os.path.join(self.ana, "config", "refs.json"), "w") as f:
            json.dump(REFERENCES, f)
        task = fx.si.PlotResonantLimitsTask
        # the method needs only the analysis area of the task
        self.task = types.SimpleNamespace(
            _resolve_config_path=lambda p: task._resolve_config_path(
                types.SimpleNamespace(ana_path=lambda: self.ana), p
            )
        )
        self.select = lambda item: task.external_limit_file(
            self.task, item, {"name": "spin0_plot"}, self.out
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_file_name_is_passed_on(self):
        self.assertEqual(
            self.select("config/run2.json"), os.path.join(self.ana, "config/run2.json")
        )
        self.assertEqual(self.select("/abs/run2.json"), "/abs/run2.json")

    def test_a_key_selects_its_curves(self):
        path = self.select({"file": "config/refs.json", "key": "DL/spin0"})
        self.assertEqual(os.path.dirname(path), self.out)
        self.assertEqual(os.path.basename(path), "spin0_plot_external_DL_spin0.json")
        with open(path) as f:
            self.assertEqual(json.load(f), [CURVE])
        path = self.select({"file": "config/refs.json", "key": "DL/spin2"})
        with open(path) as f:
            self.assertEqual([c["name"] for c in json.load(f)], ["a", "b"])

    def test_errors(self):
        for item, message in (
            (
                {"file": "config/refs.json", "key": "SL/spin0"},
                "no 'spin0' at that level",
            ),
            (
                {"file": "config/refs.json", "key": "DL"},
                "not a dhi external-limit curve",
            ),
            ({"file": "config/refs.json"}, "a file name or"),
        ):
            with self.subTest(item=item), self.assertRaisesRegex(RuntimeError, message):
                self.select(item)


if __name__ == "__main__":
    unittest.main()
