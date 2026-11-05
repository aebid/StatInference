#!/usr/bin/env python3
"""The card stages over several eras: one workflow whose branches are the cards' mass points.

On the scratch analysis of statinference_fixture (cards Run3_Early = Run3_2022 + Run3_2022EE,
Run3_2022 and Run3_2023, in that order; Run3_2023BPix configured but behind no card):

* `--era` must hold every real era behind the cards: over fewer, the tasks a run starts from
  that are not workflows raise when constructed, a card workflow when its map is built, naming
  the card and the missing eras, and luigi runs nothing; more eras give the same cards. A
  --branches selection of mass points reaches their cards' inputs and preprocessing. A
  configuration without `eras:` raises.
* CreateDatacardsTask has a branch per card and mass point, card by card and mass by mass:
  every mass the configuration declares, or those of the `StatInference` param_values.
* Constructing a card workflow over one era does not raise: FLAF builds single-era
  instances of a remote workflow for its luigi resources.
* The job state and the logs do not depend on `--era`: one job directory, one luigi
  token and one log directory per version and class, the same over more eras.
* Every mass branch of a card in CreateDatacardsTask requires the branch of that card in
  PreprocessShapesTask (one per card); every branch task carries all eras of the run, and
  each merged histogram is read with its own era.
* ResonantLimitsTask requires the one card workflow over the run's eras and stages each card
  from the branches of that card's masses; ResonantLimitsAndHistPlotTask requires one HistPlotTask over the
  run's eras, and its marker is keyed on the version alone.
* A setting a run reads once must agree between its eras: the `StatInference` block, and
  the var_list of a plotted variable between the eras summed in one plot.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402

import luigi  # noqa: E402

VERSION = "v_cards"
HISTS_VERSION = "v_hists"
WITH_EXTRA = fx.ERAS + (fx.EXTRA_ERA,)


class CardTestCase(unittest.TestCase):
    card_file = "config/Datacards/card.yaml"
    #: extra files of the scratch analysis: {path: text}
    extra_files = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="statinference_cards_")
        cls.ana = fx.make_analysis(cls.tmp, cls.card_file)
        for path, text in cls.extra_files.items():
            fx.write(os.path.join(cls.ana, path), text)
        cls.env = mock.patch.dict(
            os.environ,
            {
                "ANALYSIS_PATH": cls.ana,
                "ANALYSIS_DATA_PATH": os.path.join(cls.tmp, "data"),
            },
        )
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        fx.clear_process_caches()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        fx.clear_process_caches()

    def task(self, cls, eras=fx.ERAS, **kwargs):
        return cls(version=VERSION, era=eras, hists_version=HISTS_VERSION, **kwargs)

    def data(self, *path):
        return os.path.join(self.tmp, "data", VERSION, *path)


def card_masses(masses=fx.MASSES):
    """The branch map of the card workflow: every (card era, mass), card by card."""
    return dict(
        enumerate(
            {"era": card_era, "mass": mass}
            for card_era in fx.CARD_ERAS
            for mass in masses
        )
    )


def branches_of(card_era):
    """The branches of the card workflow that build `card_era`."""
    return tuple(
        branch for branch, data in card_masses().items() if data["era"] == card_era
    )


class CardsNeedTheirEras(CardTestCase):
    def test_fewer_eras_name_the_missing_ones(self):
        for cls in (fx.si.CreateDatacardsTask, fx.si.PreprocessShapesTask):
            with self.subTest(cls=cls.__name__):
                workflow = self.task(cls, eras=(fx.ERA_A,))
                with self.assertRaises(ValueError) as raised:
                    workflow.get_branch_map()
                message = str(raised.exception)
                self.assertIn(f"{fx.GROUP}: {fx.ERA_B}", message)
                self.assertIn(f"{fx.ERA_C}: {fx.ERA_C}", message)
                self.assertIn(f"add {fx.ERA_B},{fx.ERA_C} to --era", message)
                self.assertNotIn(f"{fx.ERA_A}: {fx.ERA_A}", message)

    def test_the_cards_are_the_configured_eras(self):
        for eras in (fx.ERAS, WITH_EXTRA):
            with self.subTest(eras=eras):
                workflow = self.task(fx.si.CreateDatacardsTask, eras=eras)
                self.assertEqual(workflow.get_branch_map(), card_masses())


class RestrictedMasses(CardTestCase):
    """global.yaml's `StatInference` param_values restricts the mass points."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = os.path.join(cls.ana, "config", "global.yaml")
        with open(path) as f:
            text = f.read()
        fx.write(
            path, text.replace("  hist_bins:", "  param_values: [ 300 ]\n  hist_bins:")
        )

    def test_only_the_restricted_masses_are_branches(self):
        workflow = self.task(fx.si.CreateDatacardsTask)
        self.assertEqual(workflow.get_branch_map(), card_masses([300]))

    def test_a_run_starting_from_the_limits_stops_at_construction(self):
        """The tasks a run starts from that are not workflows refuse too few eras when
        they are constructed, before luigi schedules anything -- also when the limits
        already exist, so that nothing below is asked to build the card map."""
        eras = (fx.ERA_A, fx.ERA_B)
        for name in (
            "datacards/combined/combined_300.txt",
            "ResonantLimitsTask/combined/limits.npz",
            "ResonantLimitsTask/combined/era_limits/limits_Run3_2022.npz",
        ):
            fx.write(self.data(*name.replace("datacards", "Datacards").split("/")), "")
        for cls in (
            fx.si.ResonantLimitsTask,
            fx.si.ResonantLimitsAndHistPlotTask,
            fx.si.PlotResonantLimitsTask,
            fx.si.PlotPullsAndImpactsTask,
        ):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(ValueError) as raised:
                    self.task(cls, eras=eras)
                self.assertIn(f"add {fx.ERA_C} to --era", str(raised.exception))

    def test_luigi_runs_nothing_over_too_few_eras(self):
        """A card workflow started directly checks when luigi builds its map: over
        --workflow local the error is a failed complete(), and luigi schedules nothing;
        over htcondor luigi also asks the workflow for its process resources, which law's
        remote proxy takes from the map, outside luigi's error handling, so the error ends
        the run."""
        eras = (fx.ERA_A, fx.ERA_B)
        local = self.task(fx.si.CreateDatacardsTask, eras=eras, workflow="local")
        self.assertFalse(
            luigi.build([local], local_scheduler=True, log_level="CRITICAL")
        )
        fx.clear_process_caches()
        remote = self.task(fx.si.CreateDatacardsTask, eras=eras, workflow="htcondor")
        with self.assertRaises(ValueError) as raised:
            luigi.build([remote], local_scheduler=True, log_level="CRITICAL")
        self.assertIn(f"add {fx.ERA_C} to --era", str(raised.exception))
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, "store", "default", VERSION))
        )


class CardsWithoutEras(CardTestCase):
    card_file = "config/Datacards/card_no_eras.yaml"
    extra_files = {
        "config/Datacards/card_no_eras.yaml": fx.card(preprocess=False).replace(
            f"eras: [ {fx.GROUP}, {fx.ERA_A}, {fx.ERA_C} ]\n", ""
        )
    }

    def test_a_configuration_without_eras_raises(self):
        workflow = self.task(fx.si.CreateDatacardsTask)
        with self.assertRaises(ValueError) as raised:
            workflow.get_branch_map()
        self.assertIn("card_no_eras.yaml lists no `eras:`", str(raised.exception))


class StateWithoutEras(CardTestCase):
    def test_one_workflow_over_one_era_can_be_built(self):
        for cls in (fx.si.CreateDatacardsTask, fx.si.PreprocessShapesTask):
            self.task(cls, eras=(fx.ERA_A,))

    def test_job_state_and_logs_do_not_depend_on_the_eras(self):
        for cls in (fx.si.CreateDatacardsTask, fx.si.PreprocessShapesTask):
            with self.subTest(cls=cls.__name__):
                directory = self.data(cls.__name__)
                logs = os.path.join(self.tmp, "store", "default", VERSION, "logs")
                for eras in (fx.ERAS, WITH_EXTRA):
                    workflow = self.task(cls, eras=eras, workflow="htcondor")
                    self.assertEqual(
                        workflow.htcondor_output_directory().abspath, directory
                    )
                    self.assertEqual(
                        workflow.process_resources(), {f"flaf_jobs:{directory}": 1}
                    )
                    self.assertEqual(
                        workflow.remote_log_dir_target().abspath,
                        os.path.join(logs, cls.__name__),
                    )


class CardsPairAndCarryEveryEra(CardTestCase):
    card_file = "config/Datacards/card_preprocess.yaml"

    def test_each_card_reads_the_preprocessed_shapes_of_that_card(self):
        cards = self.task(fx.si.CreateDatacardsTask).get_branch_tasks()
        self.assertEqual(len(cards), len(fx.CARD_ERAS) * len(fx.MASSES))
        outputs = set()
        for card in cards.values():
            (preprocess,) = card.requires()
            self.assertIsInstance(preprocess, fx.si.PreprocessShapesTask)
            self.assertEqual(
                preprocess.branch, fx.CARD_ERAS.index(card.branch_data["era"])
            )
            self.assertEqual(preprocess.branch_data, card.branch_data["era"])
            outputs.add(preprocess.output().abspath)
        self.assertEqual(len(outputs), len(fx.CARD_ERAS))

    def test_the_workflow_requires_the_preprocess_workflow(self):
        reqs = self.task(fx.si.CreateDatacardsTask).workflow_requires()
        self.assertEqual(list(reqs), ["PreprocessShapes"])
        preprocess = reqs["PreprocessShapes"]
        self.assertTrue(preprocess.is_workflow())
        self.assertEqual(preprocess.era, fx.ERAS)

    def test_every_branch_carries_every_era(self):
        for cls in (fx.si.CreateDatacardsTask, fx.si.PreprocessShapesTask):
            for card in self.task(cls).get_branch_tasks().values():
                self.assertEqual(card.era, fx.ERAS)

    def test_each_merged_histogram_is_read_with_its_own_era(self):
        preprocess = self.task(fx.si.PreprocessShapesTask).get_branch_tasks()
        for task in preprocess.values():
            for req in luigi.task.flatten(task.requires()):
                self.assertEqual(req.era, fx.ERAS)
                self.assertIn(req.hist_era, fx.SUB_ERAS[task.branch_data])
                self.assertIn(f"/Hists_merged/{req.hist_era}/", req.output().abspath)


class SelectedCards(CardTestCase):
    card_file = "config/Datacards/card_preprocess.yaml"

    def test_a_selection_reaches_the_preprocessing(self):
        branches = branches_of(fx.CARD_ERAS[1])
        workflow = self.task(fx.si.CreateDatacardsTask, branches=branches)
        self.assertEqual(
            workflow.get_branch_map(), {b: card_masses()[b] for b in branches}
        )
        preprocess = workflow.workflow_requires()["PreprocessShapes"]
        self.assertEqual(preprocess.get_branch_map(), {1: fx.CARD_ERAS[1]})
        reqs = luigi.task.flatten(preprocess.workflow_requires())
        self.assertEqual({r.hist_era for r in reqs}, set(fx.SUB_ERAS[fx.CARD_ERAS[1]]))


class SelectedCardsWithoutPreprocess(CardTestCase):
    def test_a_selection_requires_its_cards_inputs_only(self):
        workflow = self.task(fx.si.CreateDatacardsTask, branches=branches_of(fx.GROUP))
        reqs = luigi.task.flatten(workflow.workflow_requires())
        self.assertEqual({r.hist_era for r in reqs}, set(fx.SUB_ERAS[fx.GROUP]))


class LimitsOverTheCards(CardTestCase):
    def test_one_card_workflow_over_the_runs_eras(self):
        limits = self.task(fx.si.ResonantLimitsTask, eras=WITH_EXTRA)
        cards = limits.requires()
        self.assertIsInstance(cards, fx.si.CreateDatacardsTask)
        self.assertTrue(cards.is_workflow())
        self.assertEqual(cards.era, WITH_EXTRA)
        self.assertEqual(cards.branches, ())
        outputs = limits.era_card_outputs()
        self.assertEqual(list(outputs), fx.CARD_ERAS)
        for card_era, card_outputs in outputs.items():
            self.assertEqual(
                [output.abspath for output in card_outputs],
                [
                    os.path.join(
                        self.tmp,
                        "store",
                        "default",
                        VERSION,
                        "Datacards",
                        card_era,
                        f"MX_{mass}",
                    )
                    for mass in fx.MASSES
                ],
            )

    def test_one_histplot_over_the_runs_eras(self):
        task = self.task(fx.si.ResonantLimitsAndHistPlotTask, eras=WITH_EXTRA)
        (histplot,) = [
            r for r in task.requires() if isinstance(r, fx.analysis.HistPlotTask)
        ]
        self.assertEqual(histplot.era, WITH_EXTRA)
        self.assertEqual(histplot.version, HISTS_VERSION)
        for eras in (fx.ERAS, WITH_EXTRA):
            marker = self.task(fx.si.ResonantLimitsAndHistPlotTask, eras=eras).output()
            self.assertEqual(
                marker.abspath, self.data("ResonantLimitsAndHistPlotTask", "dummy.txt")
            )


class SettingsAgreeBetweenEras(CardTestCase):
    extra_files = {
        f"config/{fx.ERA_C}/global.yaml": (
            "StatInference:\n  config: config/Datacards/card.yaml\n"
            "  param_values: [ 500 ]\n"
        ),
        f"config/{fx.ERA_B}/plot/histograms.yaml": "".join(
            f"{v}:\n  var_list: [ HME, DNN_m{m} ]\n"
            for v, m in zip(fx.VARIABLES, fx.MASSES)
        )
        + "only_in_2022EE:\n  var_list: [ a, b ]\n",
    }

    def test_the_statinference_block_must_agree(self):
        workflow = self.task(fx.si.CreateDatacardsTask)
        with self.assertRaises(fx.lc.EraDependentValueError) as raised:
            workflow.get_branch_map()
        self.assertIn("StatInference differs between the eras", str(raised.exception))

    def test_the_plotted_var_list_must_agree(self):
        card = self.task(fx.si.CreateDatacardsTask, eras=(fx.ERA_A, fx.ERA_B), branch=0)
        variable = fx.VARIABLES[0]
        with self.assertRaises(fx.lc.EraDependentValueError) as raised:
            card.plot_variable(variable, [fx.ERA_A, fx.ERA_B])
        self.assertIn(f"var_list of {variable}", str(raised.exception))
        self.assertEqual(card.plot_variable(variable, [fx.ERA_A]), "HME")
        self.assertEqual(
            card.plot_variable(variable, [fx.ERA_B]), f"DNN_m{fx.MASSES[0]}"
        )
        self.assertEqual(
            card.plot_variable("not_a_variable", [fx.ERA_A, fx.ERA_B]), "not_a_variable"
        )
        # present in one of the summed eras only: they disagree as well
        with self.assertRaises(fx.lc.EraDependentValueError):
            card.plot_variable("only_in_2022EE", [fx.ERA_A, fx.ERA_B])


if __name__ == "__main__":
    unittest.main()
