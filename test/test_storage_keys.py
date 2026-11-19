#!/usr/bin/env python3
"""Where the chain reads and writes when the analysis sets FLAF's storage keys.

test_law_chain.py sets neither key: the chain reads each era's `fs_HistTuple` and writes to
`fs_default`. Here the scratch analysis of statinference_fixture also sets `fs_Hists` (one
for Run3_2022EE of its own) and `fs_StatInference`, on the configuration with a `preprocess:`
step:

* the merged histograms a card or its preprocessing reads are each era's `fs_Hists`, where
  FLAF's HistMergerTask writes them, not its `fs_HistTuple`: `MergedHists` is the very path of
  FLAF's `MergedHistsTask` (the output of HistMergerTask for that era);
* the datacards, the preprocessed shapes and the limit and impact plots are written to
  `fs_StatInference`; the job logs stay on `fs_default`;
* `fs_StatInference` is one for all the eras of a run: an era with another one is refused
  where it is read;
* a remote `fs_StatInference` (a URL, as the productions configure their stores) holds every
  product, with the logs on a local `fs_default`.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402
import test_law_chain as chain  # noqa: E402

import luigi  # noqa: E402

import FLAF.RunKit.law_gfal as law_gfal  # noqa: E402

REMOTE = "davs://storage.example:1094/store/user/flaf/inference"
HIST_STORE = {fx.ERA_B: "merged_2022EE"}


def hist_store(era):
    return HIST_STORE.get(era, "merged")


class StorageKeysTest(chain.ChainTestCase):
    card_file = "config/Datacards/card_preprocess.yaml"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        store = os.path.join(cls.tmp, "store")
        cls.append(
            "config/global.yaml",
            f"fs_Hists: {store}/merged\nfs_StatInference: {store}/inference\n",
        )
        cls.append(
            f"config/{fx.ERA_B}/global.yaml", f"fs_Hists: {store}/merged_2022EE\n"
        )

    @classmethod
    def append(cls, path, text):
        with open(os.path.join(cls.ana, path), "a") as f:
            f.write(text)

    def merged_hist(self, era, variable):
        return self.store(
            hist_store(era),
            chain.HISTS_VERSION,
            "Hists_merged",
            era,
            variable,
            f"{variable}.root",
        )

    def test_the_merged_histograms_are_read_from_fs_Hists(self):
        for (card_era, mass), card in chain.cards_of(chain.limits_task()).items():
            with self.subTest(card_era=card_era, mass=mass):
                preprocess = chain.preprocess_of(card)
                reqs = luigi.task.flatten(preprocess.requires())
                self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
                self.assertEqual(
                    {r.output().abspath for r in reqs}, self.merged_hists(card_era)
                )

    def test_merged_hists_is_the_path_flaf_writes(self):
        for era in fx.ERAS:
            for variable in fx.VARIABLES:
                with self.subTest(era=era, variable=variable):
                    reader = fx.si.MergedHists(
                        **chain.root_params(), hist_era=era, variable=variable
                    )
                    writer = fx.analysis.MergedHistsTask(
                        version=chain.HISTS_VERSION, era=era, variable=variable
                    )
                    self.assertEqual(reader.output().abspath, writer.output().abspath)
                    self.assertEqual(
                        reader.output().abspath, self.merged_hist(era, variable)
                    )

    def test_the_products_are_written_to_fs_StatInference(self):
        limits = chain.limits_task()
        for (card_era, mass), card in chain.cards_of(limits).items():
            with self.subTest(card_era=card_era, mass=mass):
                self.assertEqual(
                    card.output().abspath,
                    self.store(
                        "inference",
                        chain.VERSION,
                        "Datacards",
                        self.card_name,
                        card_era,
                        f"MX_{mass}",
                    ),
                )
                self.assertEqual(
                    chain.preprocess_of(card).output().abspath,
                    self.store(
                        "inference",
                        chain.VERSION,
                        "Hists_preprocessed",
                        self.card_name,
                        card_era,
                    ),
                )
        for cls, directory in (
            (fx.si.PlotResonantLimitsTask, "LimitPlots"),
            (fx.si.PlotPullsAndImpactsTask, "ImpactPlots"),
        ):
            with self.subTest(task=cls.__name__):
                task = cls(**chain.root_params(), card=self.card_name)
                self.assertEqual(
                    task.output().abspath,
                    self.store("inference", chain.VERSION, directory, self.card_name),
                )

    def test_the_job_logs_stay_on_fs_default(self):
        (workflow,) = chain.card_workflows(chain.limits_task())
        self.assertEqual(
            workflow.remote_log_dir_target().abspath,
            self.store("default", chain.VERSION, "logs", "CreateDatacardsTask"),
        )


class RemoteStatInferenceTest(chain.ChainTestCase):
    card_file = "config/Datacards/card_preprocess.yaml"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(os.path.join(cls.ana, "config/global.yaml"), "a") as f:
            f.write(f"fs_StatInference: {REMOTE}\n")
        # no path-cache server and no grid proxy (a CI runner has neither): the filesystem
        # is built and never listed
        cls.patches = [
            mock.patch.object(fx.Setup, "local_path_cache_server", lambda: None),
            mock.patch.object(law_gfal, "get_voms_proxy_info", lambda: {"path": None}),
        ]
        for patch in cls.patches:
            patch.start()

    @classmethod
    def tearDownClass(cls):
        for patch in cls.patches:
            patch.stop()
        super().tearDownClass()

    def test_every_product_is_on_the_remote_store(self):
        limits = chain.limits_task()
        self.assertEqual(limits.fs_StatInference.base, [REMOTE])
        targets = []
        for card in chain.cards_of(limits).values():
            targets += [card.output(), chain.preprocess_of(card).output()]
        for cls in (fx.si.PlotResonantLimitsTask, fx.si.PlotPullsAndImpactsTask):
            targets.append(cls(**chain.root_params(), card=self.card_name).output())
        for target in targets:
            self.assertIs(target.fs, limits.fs_StatInference)
            self.assertTrue(target.uri().startswith(f"{REMOTE}/{chain.VERSION}/"))
        (workflow,) = chain.card_workflows(limits)
        self.assertEqual(
            workflow.remote_log_dir_target().abspath,
            self.store("default", chain.VERSION, "logs", "CreateDatacardsTask"),
        )


class StatInferenceStoreAgreementTest(chain.ChainTestCase):
    card_file = "config/Datacards/card.yaml"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        store = os.path.join(cls.tmp, "store")
        with open(os.path.join(cls.ana, f"config/{fx.ERA_C}/global.yaml"), "a") as f:
            f.write(f"fs_StatInference: {store}/inference_2023\n")

    def test_another_fs_StatInference_in_one_era_is_refused(self):
        (workflow,) = chain.card_workflows(chain.limits_task())
        with self.assertRaises(fx.lc.EraDependentValueError) as caught:
            workflow.get_branch_tasks()[0].output()
        message = str(caught.exception)
        for text in ("fs_StatInference", "inference_2023", fx.ERA_C):
            self.assertIn(text, message)


if __name__ == "__main__":
    unittest.main()
