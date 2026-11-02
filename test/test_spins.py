#!/usr/bin/env python3
"""A configuration with a `spins:` axis: one card per era and spin.

On the scratch analysis of statinference_fixture, with card_spins.yaml (the preprocessed card
with spins 0 and 2, whose signal and one preprocess argument name the spin):

* PreprocessShapesTask has one branch per card era and spin, {"era", "spin"}, writing
  `Hists_preprocessed/<card>/spin<spin>`, and runs its step with --spin and ${SPIN} filled;
* CreateDatacardsTask one per card era, spin and mass, {"era", "spin", "mass"}, writing
  `Datacards/<card>/spin<spin>/MX_<mass>`, reading the preprocessing of its own era and spin
  and building with --spin;
* ResonantLimitsTask writes each spin's limits under `spin<spin>/`, mirrors a spin's cards
  into `Datacards/spin<spin>/<card>` and merges and combines every spin on its own;
* the configuration of a spin has its values substituted, the configuration as written not.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402
import test_law_chain as chain  # noqa: E402

import luigi  # noqa: E402

VERSION = chain.VERSION
SPINS = list(fx.SPINS)


def cards_of(limits):
    """{(card era, spin, mass): the task that writes that card's mass point}."""
    return {
        (t.branch_data["era"], t.branch_data["spin"], t.branch_data["mass"]): t
        for workflow in chain.card_workflows(limits)
        for t in workflow.get_branch_tasks().values()
    }


class SpinChainTest(chain.ChainTestCase):
    card_file = "config/Datacards/card_spins.yaml"

    def test_the_configuration_of_a_spin(self):
        limits = chain.limits_task()
        self.assertEqual(limits.card_spins(), SPINS)
        raw = limits.get_config_data()
        self.assertIn("spins", raw)
        self.assertEqual(raw["processes"][1]["process"], "${MODEL}_${MX}")
        for spin, model in fx.SPINS.items():
            cfg = limits.get_config_data(spin)
            self.assertNotIn("spins", cfg)
            self.assertEqual(cfg["processes"][1]["process"], f"{model}_${{MX}}")
            self.assertEqual(cfg["preprocess"]["args"][-1], f"${{ERA}}_spin{spin}")

    def test_one_card_per_era_spin_and_mass(self):
        self.assertEqual(
            set(cards_of(chain.limits_task())),
            {(e, s, m) for e in fx.CARD_ERAS for s in SPINS for m in fx.MASSES},
        )

    def test_outputs(self):
        for (era, spin, mass), card in cards_of(chain.limits_task()).items():
            self.assertEqual(
                card.output().abspath,
                self.store(
                    "default", VERSION, "Datacards", era, f"spin{spin}", f"MX_{mass}"
                ),
            )
            (preprocess,) = card.requires()
            self.assertEqual(preprocess.branch_data, {"era": era, "spin": spin})
            self.assertEqual(
                preprocess.output().abspath,
                self.store(
                    "default", VERSION, "Hists_preprocessed", era, f"spin{spin}"
                ),
            )
            self.assertEqual(
                {r.output().abspath for r in luigi.task.flatten(preprocess.requires())},
                self.merged_hists(era),
            )

    def test_preprocess_and_datacard_commands_name_the_spin(self):
        self.write_merged_hists()
        for (era, spin, mass), card in cards_of(chain.limits_task()).items():
            if mass != fx.MASSES[0]:
                continue
            with self.subTest(card=era, spin=spin):
                (preprocess,) = card.requires()
                ((cmd, _, _),) = self.run_payload(preprocess, chain.preprocess_module)
                self.assertEqual(self.arg(cmd, "--era"), era)
                self.assertEqual(self.arg(cmd, "--spin"), spin)
                self.assertEqual(self.arg(cmd, "--tag"), f"{era}_spin{spin}")
                calls = self.run_payload(card, chain.create_datacards_module)
                (maker,) = [c[0] for c in calls if "--eras" in c[0]]
                self.assertEqual(self.arg(maker, "--eras"), era)
                self.assertEqual(self.arg(maker, "--spin"), spin)
                self.assertEqual(self.arg(maker, "--param_values"), str(mass))

    def test_limits_per_spin(self):
        limits = chain.limits_task()
        data = os.path.join(self.tmp, "data", VERSION)
        outputs = limits.output()
        self.assertEqual(set(outputs), {f"spin{s}" for s in SPINS})
        for spin in SPINS:
            self.assertEqual(
                outputs[f"spin{spin}"]["limits"].abspath,
                os.path.join(
                    data, "ResonantLimitsTask", "combined", f"spin{spin}", "limits.npz"
                ),
            )
            self.assertEqual(
                outputs[f"spin{spin}"]["datacards"].abspath,
                os.path.join(data, "Datacards", f"spin{spin}", "combined"),
            )

        for era, spin, mass in cards_of(limits):
            fx.write(
                self.store(
                    "default",
                    VERSION,
                    "Datacards",
                    era,
                    f"spin{spin}",
                    f"MX_{mass}",
                    f"datacard_{era}_{mass}.txt",
                ),
                f"{era} {spin}",
            )

        def combine(cmd, stdout=None, **kwargs):
            stdout.write("combined\n")

        def merge(version, datacards):
            return ("merge", frozenset(datacards))

        with mock.patch.object(
            chain.limits_module, "subprocess", mock.Mock(run=combine)
        ), mock.patch.object(
            chain.limits_module, "MergeResonantLimits", merge
        ), mock.patch.object(
            type(limits), "cmssw_env", new_callable=mock.PropertyMock
        ):
            merges = next(limits.run())

        expected = {}
        for spin in SPINS:
            for era in fx.CARD_ERAS:
                paths = {
                    os.path.join(
                        limits.datacards_dir(era, spin), f"datacard_{era}_{m}.txt"
                    )
                    for m in fx.MASSES
                }
                for path in paths:
                    with open(path) as f:
                        self.assertEqual(f.read(), f"{era} {spin}")
                expected[f"spin{spin}|{era}"] = ("merge", frozenset(paths))
            combined = os.path.join(data, "Datacards", f"spin{spin}", "combined")
            expected[f"spin{spin}|{chain.limits_module.COMBINED_KEY}"] = (
                "merge",
                frozenset(
                    os.path.join(combined, f"combined_{m}.txt") for m in fx.MASSES
                ),
            )
        self.assertEqual(merges, expected)


if __name__ == "__main__":
    unittest.main()
