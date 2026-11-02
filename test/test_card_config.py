"""common/card_config.py: a datacard configuration built on another (`extends`) with `variables`."""

import os
import tempfile
import textwrap
import unittest

from StatInference.common.card_config import load_card_config


class CardConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(textwrap.dedent(text))
        return path

    def base(self):
        return self.write(
            "base.yaml",
            """
            variables: { SIGNAL: GluGluToRadion, SPIN: 0 }
            analysis: hh
            channels: [ eE, muMu ]
            processes:
              - process: ${SIGNAL}_bbWW_${MX}
                is_signal: true
              - TT
            uncertainties:
              - name: lumi
                processes: [ "^${SIGNAL}_" ]
                value: 1.01
            limit_plots:
              - name: spin${SPIN}
                files: [ "run2_spin${SPIN}.json" ]
                glob: "${ERA}/*.txt"
            model: { parameters: [ MX ], input_file_pattern: "${ERA}/m${MX}.root" }
            """,
        )

    def test_a_card_without_extends_or_variables_is_read_as_is(self):
        path = self.write(
            "plain.yaml", "analysis: hh\nchannels: [ eE ]\nlabel: '$BIN ${MX}'\n"
        )
        self.assertEqual(
            load_card_config(path),
            {"analysis": "hh", "channels": ["eE"], "label": "$BIN ${MX}"},
        )

    def test_variables_are_substituted_and_other_placeholders_kept(self):
        cfg = load_card_config(self.base())
        self.assertNotIn("variables", cfg)
        self.assertEqual(cfg["processes"][0]["process"], "GluGluToRadion_bbWW_${MX}")
        self.assertEqual(cfg["uncertainties"][0]["processes"], ["^GluGluToRadion_"])
        self.assertEqual(cfg["limit_plots"][0]["name"], "spin0")
        self.assertEqual(cfg["limit_plots"][0]["files"], ["run2_spin0.json"])
        self.assertEqual(cfg["limit_plots"][0]["glob"], "${ERA}/*.txt")
        self.assertEqual(cfg["model"]["input_file_pattern"], "${ERA}/m${MX}.root")

    def test_extends_overrides_a_variable_of_the_base(self):
        self.base()
        path = self.write(
            "sub/spin2.yaml",
            """
            extends: ../base.yaml
            variables: { SIGNAL: GluGluToBulkGraviton, SPIN: 2 }
            """,
        )
        cfg = load_card_config(path)
        self.assertNotIn("extends", cfg)
        self.assertEqual(
            cfg["processes"][0]["process"], "GluGluToBulkGraviton_bbWW_${MX}"
        )
        self.assertEqual(
            cfg["uncertainties"][0]["processes"], ["^GluGluToBulkGraviton_"]
        )
        self.assertEqual(cfg["limit_plots"][0]["name"], "spin2")
        self.assertEqual(cfg["channels"], ["eE", "muMu"])

    def test_merge_rules(self):
        self.base()
        path = self.write(
            "over.yaml",
            """
            extends: base.yaml
            channels: [ eMu ]
            model: { parameters: [ MY ] }
            limit_plots: null
            new_key: 3
            """,
        )
        cfg = load_card_config(path)
        self.assertEqual(cfg["channels"], ["eMu"])  # a list replaces
        self.assertEqual(cfg["model"]["parameters"], ["MY"])  # a mapping merges ...
        self.assertEqual(
            cfg["model"]["input_file_pattern"], "${ERA}/m${MX}.root"
        )  # ... key by key
        self.assertNotIn("limit_plots", cfg)  # null removes
        self.assertEqual(cfg["new_key"], 3)

    def test_a_chain_of_extends(self):
        self.base()
        self.write("mid.yaml", "extends: base.yaml\nanalysis: mid\n")
        cfg = load_card_config(
            self.write("top.yaml", "extends: mid.yaml\nchannels: [ eMu ]\n")
        )
        self.assertEqual((cfg["analysis"], cfg["channels"]), ("mid", ["eMu"]))
        self.assertEqual(cfg["processes"][0]["process"], "GluGluToRadion_bbWW_${MX}")

    def test_errors(self):
        self.write("a.yaml", "extends: b.yaml\n")
        self.write("b.yaml", "extends: a.yaml\n")
        with self.assertRaisesRegex(ValueError, "cycle"):
            load_card_config(os.path.join(self.dir, "a.yaml"))
        with self.assertRaisesRegex(ValueError, "string or a number"):
            load_card_config(self.write("bad.yaml", "variables: { X: [ 1 ] }\n"))
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_card_config(self.write("list.yaml", "- 1\n"))


if __name__ == "__main__":
    unittest.main()
