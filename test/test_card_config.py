"""common/card_config.py: a datacard configuration and its `spins:` axis."""

import os
import tempfile
import textwrap
import unittest

from StatInference.common.card_config import card_spins, load_card_config

CARD = """
spins:
  0: { SIGNAL_MODEL: GluGluToRadion, SPIN_NAME: radion }
  2: { SIGNAL_MODEL: GluGluToBulkGraviton, SPIN_NAME: graviton }
analysis: hh
channels: [ eE, muMu ]
processes:
  - process: ${SIGNAL_MODEL}_bbWW_${MX}
    is_signal: true
  - TT
uncertainties:
  - name: BR_hbb
    processes: [ "^${SIGNAL_MODEL}_" ]
limit_plots:
  - name: spin${SPIN}_limits
    datacards: [ { name: "spin-${SPIN} (${SPIN_NAME})", glob: "${ERA}/*.txt" } ]
    external_limits: [ "run2_spin${SPIN}.json" ]
model: { parameters: [ MX ], input_file_pattern: "${ERA}/m${MX}.root" }
label: "$BIN"
"""


class CardConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="card.yaml"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(textwrap.dedent(text))
        return path

    def test_a_card_without_spins_is_read_as_is(self):
        path = self.write("analysis: hh\nchannels: [ eE ]\nlabel: '$BIN ${MX}'\n")
        cfg = load_card_config(path)
        self.assertEqual(
            cfg, {"analysis": "hh", "channels": ["eE"], "label": "$BIN ${MX}"}
        )
        self.assertEqual(card_spins(cfg), [None])
        with self.assertRaisesRegex(ValueError, "declares no spins"):
            load_card_config(path, "0")

    def test_the_spins_in_their_order(self):
        cfg = load_card_config(self.write(CARD))
        self.assertEqual(card_spins(cfg), ["0", "2"])
        # as written: the spin values are not substituted
        self.assertEqual(cfg["processes"][0]["process"], "${SIGNAL_MODEL}_bbWW_${MX}")

    def test_the_card_of_a_spin(self):
        path = self.write(CARD)
        for spin, model, name in (
            ("0", "GluGluToRadion", "radion"),
            (2, "GluGluToBulkGraviton", "graviton"),
        ):
            cfg = load_card_config(path, spin)
            self.assertNotIn("spins", cfg)
            self.assertEqual(cfg["processes"][0]["process"], f"{model}_bbWW_${{MX}}")
            self.assertEqual(cfg["uncertainties"][0]["processes"], [f"^{model}_"])
            plot = cfg["limit_plots"][0]
            self.assertEqual(plot["name"], f"spin{spin}_limits")
            self.assertEqual(plot["datacards"][0]["name"], f"spin-{spin} ({name})")
            self.assertEqual(plot["external_limits"], [f"run2_spin{spin}.json"])
            # placeholders that are not spin values are left for later
            self.assertEqual(plot["datacards"][0]["glob"], "${ERA}/*.txt")
            self.assertEqual(cfg["model"]["input_file_pattern"], "${ERA}/m${MX}.root")
            self.assertEqual(cfg["label"], "$BIN")
            self.assertEqual(cfg["channels"], ["eE", "muMu"])

    def test_errors(self):
        path = self.write(CARD)
        with self.assertRaisesRegex(ValueError, "no spin 1; it declares 0, 2"):
            load_card_config(path, "1")
        for text, message in (
            ("spins: []\n", "non-empty mapping"),
            ("spins: { 0: [ 1 ] }\n", "must be a mapping"),
            ("spins: { 0: { X: [ 1 ] } }\n", "string or a number"),
            ("spins: { 0: { SPIN: 0 } }\n", "SPIN is the spin itself"),
            ("- 1\n", "not a mapping"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                load_card_config(self.write(text, "bad.yaml"), "0")


if __name__ == "__main__":
    unittest.main()
