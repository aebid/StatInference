#!/usr/bin/env python3
"""What the datacard chain reads, writes and runs, per card.

On the real FLAF and law classes over the scratch analysis of statinference_fixture (a group
card Run3_Early = Run3_2022 + Run3_2022EE, a card for its member Run3_2022 and one for
Run3_2023; Run3_2022EE reads its histograms from a store of its own), with and without a
`preprocess:` step:

* one branch per card era the configuration lists and mass point, written to
  `<fs_default>/<v>/Datacards/<card>/MX_<mass>` by the card's payload for that mass alone;
* the merged histograms a card reads are its sub-eras' variables, each from its own era's
  `fs_HistTuple` under --hists-version, required by the card and by its workflow; with
  `preprocess:` the card reads the preprocessed shapes
  `<fs_default>/<v>/Hists_preprocessed/<card>`, which are built from that same set; a
  missing one is named;
* ResonantLimitsTask writes `limits.npz`, `era_limits/` and `Datacards/combined` where it
  always has, mirrors every mass of a card into the one `Datacards/<card>`, merges each
  card's limits on their own and combines only the top-level cards (the group, not its
  member);
* ResonantLimitsAndHistPlotTask requires the HistPlots of every real era behind the cards,
  which the eras of the root task cover, under --hists-version;
* the payload commands name the card era (`--eras`, `--era`, `${ERA}` in `hist_bins` and in
  the preprocess arguments) and the branch's mass (`--param_values`), and the shape plots,
  of that mass's variable only, the card era as the year and its first sub-era as the era;
  no plot without `make_plots`.

`root_params()`, `card_workflows()` and the `*_of()` helpers are the only places that know
which eras the root task takes and how a card's task is reached from the limits; everything
else states what must hold however that changes.
"""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402

import luigi  # noqa: E402

VERSION = "v_chain"
HISTS_VERSION = "v_hists"

create_datacards_module = importlib.import_module(
    "StatInference.law.CreateDatacardsTask"
)
preprocess_module = importlib.import_module("StatInference.law.PreprocessShapesTask")
limits_module = importlib.import_module("StatInference.law.ResonantLimitsTask")


def root_params():
    """The parameters of the task the chain is run from, over every real era behind the
    cards."""
    return dict(version=VERSION, era=fx.ERAS, hists_version=HISTS_VERSION)


def limits_task():
    return fx.si.ResonantLimitsTask(**root_params())


def limits_and_histplot_task():
    return fx.si.ResonantLimitsAndHistPlotTask(**root_params())


def card_workflows(limits):
    """The CreateDatacardsTask workflows `limits` requires: one, whose branches are the
    cards' mass points."""
    return luigi.task.flatten(limits.requires())


def cards_of(limits):
    """{(card era, mass): the task that writes that card's mass point}."""
    return {
        (task.branch_data["era"], task.branch_data["mass"]): task
        for workflow in card_workflows(limits)
        for task in workflow.get_branch_tasks().values()
    }


def preprocess_of(card):
    """The task that writes the preprocessed shapes `card` reads."""
    (preprocess,) = card.requires()
    return preprocess


def histplot_eras(task):
    """{era: HistPlotTask version} over the HistPlotTask requirements of `task`."""
    eras = {}
    for req in luigi.task.flatten(task.requires()):
        if isinstance(req, fx.analysis.HistPlotTask):
            for era in req.era:
                eras[era] = req.version
    return eras


class _ShapeFile:
    """A ROOT file holding every directory asked for."""

    def IsZombie(self):
        return False

    def Get(self, path):
        return True

    def Close(self):
        pass


class _ROOT:
    class TFile:
        @staticmethod
        def Open(path):
            return _ShapeFile()


def every_card_mass():
    """{(card era, mass)} over every card and mass point of the configuration."""
    return {(card_era, mass) for card_era in fx.CARD_ERAS for mass in fx.MASSES}


class ChainTestCase(unittest.TestCase):
    card_file = None  # relative to the analysis

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="statinference_chain_")
        cls.ana = fx.make_analysis(cls.tmp, cls.card_file)
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
        for directory in ("store", "data"):
            shutil.rmtree(os.path.join(self.tmp, directory), ignore_errors=True)

    def store(self, *path):
        return os.path.join(self.tmp, "store", *path)

    def merged_hist(self, era, variable):
        return os.path.join(
            self.tmp,
            fx.hist_store(era),
            HISTS_VERSION,
            "Hists_merged",
            era,
            variable,
            f"{variable}.root",
        )

    @staticmethod
    def variables_of(mass):
        """The input variables of `mass`, or of every mass for None."""
        return fx.VARIABLES if mass is None else [fx.VARIABLES[fx.MASSES.index(mass)]]

    def merged_hists(self, card_era, mass=None):
        return {
            self.merged_hist(era, variable)
            for era in fx.SUB_ERAS[card_era]
            for variable in self.variables_of(mass)
        }

    def every_merged_hist(self):
        return set().union(*(self.merged_hists(e) for e in fx.CARD_ERAS))

    def staged_layout(self, card_era, mass=None):
        return {
            os.path.join(era, variable, f"{variable}.root")
            for era in fx.SUB_ERAS[card_era]
            for variable in self.variables_of(mass)
        }

    def write_merged_hists(self):
        for path in self.every_merged_hist():
            fx.write(path, "")

    def run_payload(self, task, module, fail=lambda cmd: False):
        """Run `task` with its commands recorded instead of executed: [(cmd, env, files)],
        `files` being the layout of the command's --input directory at the time. A command
        with --output writes the file `payload_output` there; a command `fail` accepts
        raises as a failed ps_call does."""
        calls = []

        def ps_call(cmd, env=None, **kwargs):
            if fail(cmd):
                raise RuntimeError(f"command failed: {cmd[1]}")
            files = None
            if "--input" in cmd:
                base = cmd[cmd.index("--input") + 1]
                files = {
                    os.path.relpath(os.path.join(root, name), base)
                    for root, _, names in os.walk(base)
                    for name in names
                }
            if "--output" in cmd:
                fx.write(os.path.join(self.arg(cmd, "--output"), "payload_output"), "")
            calls.append((list(cmd), env, files))

        with mock.patch.object(module, "ps_call", ps_call), mock.patch.object(
            type(task), "cmssw_env", new_callable=mock.PropertyMock
        ) as cmssw_env, mock.patch.object(
            create_datacards_module, "importROOT", lambda: _ROOT
        ), mock.patch.object(
            fx.si.CreateDatacardsTask, "_stitch_grids"
        ):
            cmssw_env.return_value = {"CMSSW_ENV": "1"}
            task.run()
        return calls

    @staticmethod
    def arg(cmd, flag):
        return cmd[cmd.index(flag) + 1]


class PlainChainTest(ChainTestCase):
    """The configuration without a `preprocess:` step."""

    card_file = "config/Datacards/card.yaml"

    def test_one_branch_per_configured_era_and_mass(self):
        self.assertEqual(set(cards_of(limits_task())), every_card_mass())

    def test_card_outputs(self):
        for (card_era, mass), card in cards_of(limits_task()).items():
            self.assertEqual(
                card.output().abspath,
                self.store("default", VERSION, "Datacards", card_era, f"MX_{mass}"),
            )

    def test_card_inputs_are_the_sub_eras_merged_histograms(self):
        """Of the branch's own mass: a mass branch does not stage the other masses'."""
        for (card_era, mass), card in cards_of(limits_task()).items():
            reqs = luigi.task.flatten(card.requires())
            self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
            self.assertEqual(
                {r.output().abspath for r in reqs}, self.merged_hists(card_era, mass)
            )

    def test_workflows_require_every_cards_inputs(self):
        reqs = [
            r
            for workflow in card_workflows(limits_task())
            for r in luigi.task.flatten(workflow.workflow_requires())
        ]
        self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
        self.assertEqual({r.output().abspath for r in reqs}, self.every_merged_hist())

    def test_limit_outputs(self):
        limits = limits_task()
        data = os.path.join(self.tmp, "data", VERSION)
        outputs = limits.output()
        self.assertEqual(
            outputs["limits"].abspath,
            os.path.join(data, "ResonantLimitsTask", "combined", "limits.npz"),
        )
        self.assertEqual(
            outputs["era_limits"].abspath,
            os.path.join(data, "ResonantLimitsTask", "combined", "era_limits"),
        )
        self.assertEqual(
            outputs["datacards"].abspath, os.path.join(data, "Datacards", "combined")
        )
        for card_era in fx.CARD_ERAS:
            self.assertEqual(
                limits.datacards_dir(card_era),
                os.path.join(data, "Datacards", card_era),
            )

    def test_limits_merge_each_card_and_combine_the_top_level_ones(self):
        """ResonantLimitsTask.run() up to its dynamic dependencies: every mass of a card
        mirrored from its own output into the one directory of the card, one merge per
        card, and the combination of the cards that are not a member of a group."""
        limits = limits_task()
        for card_era, mass in every_card_mass():
            fx.write(
                self.store(
                    "default",
                    VERSION,
                    "Datacards",
                    card_era,
                    f"MX_{mass}",
                    f"datacard_{card_era}_{mass}.txt",
                ),
                card_era,
            )
        combine_calls = []

        def combine(cmd, stdout=None, **kwargs):
            combine_calls.append(cmd)
            stdout.write("combined\n")

        def merge(version, datacards):
            return ("merge", version, frozenset(datacards))

        with mock.patch.object(
            limits_module, "subprocess", mock.Mock(run=combine)
        ), mock.patch.object(
            limits_module, "MergeResonantLimits", merge
        ), mock.patch.object(
            type(limits), "cmssw_env", new_callable=mock.PropertyMock
        ):
            merges = next(limits.run())

        cards = {
            card_era: {
                os.path.join(
                    limits.datacards_dir(card_era), f"datacard_{card_era}_{mass}.txt"
                )
                for mass in fx.MASSES
            }
            for card_era in fx.CARD_ERAS
        }
        for card_era, paths in cards.items():
            # the masses side by side, as when one task wrote them all
            self.assertEqual(
                {
                    os.path.join(limits.datacards_dir(card_era), name)
                    for name in os.listdir(limits.datacards_dir(card_era))
                },
                paths,
            )
            for path in paths:
                with open(path) as f:
                    self.assertEqual(f.read(), card_era)
        combined = os.path.join(self.tmp, "data", VERSION, "Datacards", "combined")
        self.assertEqual(
            merges,
            {
                **{e: ("merge", VERSION, frozenset(p)) for e, p in cards.items()},
                limits_module.COMBINED_KEY: (
                    "merge",
                    VERSION,
                    frozenset(
                        os.path.join(combined, f"combined_{m}.txt") for m in fx.MASSES
                    ),
                ),
            },
        )
        top_level = [fx.GROUP, fx.ERA_C]
        self.assertEqual(
            sorted(combine_calls),
            sorted(
                ["combineCards.py"]
                + [
                    f"{e}={os.path.join(limits.datacards_dir(e), f'datacard_{e}_{m}.txt')}"
                    for e in top_level
                ]
                for m in fx.MASSES
            ),
        )

    def test_histplots_of_every_real_era_behind_the_cards(self):
        task = limits_and_histplot_task()
        self.assertEqual(histplot_eras(task), {era: HISTS_VERSION for era in fx.ERAS})
        (limits,) = [
            r
            for r in luigi.task.flatten(task.requires())
            if isinstance(r, fx.si.ResonantLimitsTask)
        ]
        self.assertEqual(
            limits.output()["limits"].abspath, limits_task().output()["limits"].abspath
        )

    def test_payload_commands(self):
        self.write_merged_hists()
        ana = self.ana
        for (card_era, mass), card in cards_of(limits_task()).items():
            with self.subTest(card=card_era, mass=mass):
                calls = self.run_payload(card, create_datacards_module)
                (maker,) = [c for c in calls if "--eras" in c[0]]
                cmd, env, files = maker
                self.assertEqual(
                    cmd[1],
                    os.path.join(
                        ana, "StatInference", "dc_make", "create_datacards.py"
                    ),
                )
                self.assertEqual(self.arg(cmd, "--eras"), card_era)
                self.assertEqual(
                    self.arg(cmd, "--config"),
                    os.path.join(ana, "config", "Datacards", "card.yaml"),
                )
                self.assertEqual(
                    self.arg(cmd, "--hist-bins"),
                    os.path.join(ana, "config", "Datacards", f"bins_{card_era}.json"),
                )
                self.assertEqual(cmd.count("--param_values"), 1)
                self.assertEqual(self.arg(cmd, "--param_values"), str(mass))
                self.assertEqual(env, {"CMSSW_ENV": "1"})
                self.assertEqual(files, self.staged_layout(card_era, mass))
                self.assertIn("payload_output", os.listdir(card.output().abspath))

                # the variable of this mass only
                (plot,) = [c[0] for c in calls if "--all_keys" in c[0]]
                self.assertEqual(
                    os.path.basename(self.arg(plot, "--inFile")),
                    f"_summed_{fx.VARIABLES[fx.MASSES.index(mass)]}.root",
                )
                self.assertEqual(self.arg(plot, "--year"), card_era)
                self.assertEqual(self.arg(plot, "--era"), fx.SUB_ERAS[card_era][0])
                self.assertEqual(self.arg(plot, "--LAWrunVersion"), VERSION)
                # HistPlotter builds its Setup from these, so no config file
                self.assertEqual(self.arg(plot, "--ana_path"), ana)
                self.assertNotIn("--globalConfig", plot)
                # the surviving axis of the 2D variable: y without a binning record
                self.assertEqual(self.arg(plot, "--var"), "HME")

    def test_no_plots_without_make_plots(self):
        self.write_merged_hists()
        for (card_era, mass), card in cards_of(limits_task()).items():
            with self.subTest(card=card_era, mass=mass):
                card = fx.si.CreateDatacardsTask.req(
                    card, make_plots=False, _skip_task_excludes=True
                )
                calls = self.run_payload(card, create_datacards_module)
                self.assertEqual(
                    [c[0][1] for c in calls],
                    [
                        os.path.join(
                            self.ana, "StatInference", "dc_make", "create_datacards.py"
                        )
                    ],
                )

    def test_a_failed_plot_fails_the_card(self):
        self.write_merged_hists()
        card = cards_of(limits_task())[(fx.GROUP, fx.MASSES[0])]
        with self.assertRaises(RuntimeError) as raised:
            self.run_payload(
                card, create_datacards_module, fail=lambda cmd: "--all_keys" in cmd
            )
        message = str(raised.exception)
        self.assertIn(f"shape plotting failed for {fx.VARIABLES[0]}", message)
        self.assertIn("--make-plots False", message)

    def test_a_missing_input_is_named(self):
        self.write_merged_hists()
        missing = self.merged_hist(fx.ERA_B, fx.VARIABLES[0])
        os.remove(missing)
        card = cards_of(limits_task())[(fx.GROUP, fx.MASSES[0])]
        with self.assertRaises(RuntimeError) as raised:
            self.run_payload(card, create_datacards_module)
        message = str(raised.exception)
        self.assertIn(f"{fx.ERA_B}/{fx.VARIABLES[0]}", message)
        self.assertIn(f"hists_version='{HISTS_VERSION}'", message)


class PreprocessChainTest(ChainTestCase):
    """The configuration with a `preprocess:` step."""

    card_file = "config/Datacards/card_preprocess.yaml"

    def test_card_reads_its_preprocessed_shapes(self):
        for (card_era, _), card in cards_of(limits_task()).items():
            preprocess = preprocess_of(card)
            # every mass of a card reads the one preprocessing branch of that card
            self.assertEqual(preprocess.branch, fx.CARD_ERAS.index(card_era))
            self.assertEqual(
                preprocess.output().abspath,
                self.store("default", VERSION, "Hists_preprocessed", card_era),
            )
            reqs = luigi.task.flatten(preprocess.requires())
            self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
            self.assertEqual(
                {r.output().abspath for r in reqs}, self.merged_hists(card_era)
            )

    def test_workflows_require_every_cards_inputs(self):
        preprocess = [
            r
            for workflow in card_workflows(limits_task())
            for r in luigi.task.flatten(workflow.workflow_requires())
        ]
        self.assertTrue(
            all(isinstance(r, fx.si.PreprocessShapesTask) for r in preprocess)
        )
        reqs = [
            r
            for workflow in preprocess
            for r in luigi.task.flatten(workflow.workflow_requires())
        ]
        self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
        self.assertEqual({r.output().abspath for r in reqs}, self.every_merged_hist())

    def test_preprocess_command(self):
        self.write_merged_hists()
        ana = self.ana
        preprocess_of_card = {
            card_era: preprocess_of(card)
            for (card_era, _), card in cards_of(limits_task()).items()
        }
        for card_era, preprocess in preprocess_of_card.items():
            with self.subTest(card=card_era):
                (call,) = self.run_payload(preprocess, preprocess_module)
                cmd, env, files = call
                self.assertEqual(
                    cmd[:3],
                    [
                        "python3",
                        "-u",
                        os.path.join(ana, "StatInference", "bin_opt_2d", "rebin_2d.py"),
                    ],
                )
                self.assertEqual(self.arg(cmd, "--era"), card_era)
                self.assertEqual(
                    self.arg(cmd, "--config"),
                    os.path.join(ana, "config", "Datacards", "card_preprocess.yaml"),
                )
                self.assertEqual(
                    self.arg(cmd, "--binning-config"),
                    os.path.join(
                        ana, "config", "Datacards", f"binning_{card_era}.yaml"
                    ),
                )
                self.assertEqual(self.arg(cmd, "--tag"), card_era)
                self.assertEqual(env, {"CMSSW_ENV": "1"})
                self.assertEqual(files, self.staged_layout(card_era))
                self.assertIn("payload_output", os.listdir(preprocess.output().abspath))

    def test_create_datacards_reads_the_preprocessed_shapes(self):
        for (card_era, mass), card in cards_of(limits_task()).items():
            with self.subTest(card=card_era, mass=mass):
                preprocessed = preprocess_of(card).output().abspath
                fx.write(os.path.join(preprocessed, card_era, "marker.root"), "")
                calls = self.run_payload(card, create_datacards_module)
                (maker,) = [c for c in calls if "--eras" in c[0]]
                cmd, _, files = maker
                self.assertEqual(self.arg(cmd, "--eras"), card_era)
                self.assertEqual(files, {os.path.join(card_era, "marker.root")})


class StageMassDirTest(unittest.TestCase):
    """ResonantLimitsTask._merge_mass_dir: how one mass's output joins its era's mirror."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.dest = os.path.join(self.tmp, "era")
        self.merge = limits_module.ResonantLimitsTask._merge_mass_dir

    def mass_dir(self, mass, files):
        src = os.path.join(self.tmp, f"src_{mass}")
        for rel in files:
            path = os.path.join(src, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"{mass}\n")
        return src

    def tree(self):
        return {
            os.path.relpath(os.path.join(root, name), self.dest)
            for root, _, files in os.walk(self.dest)
            for name in files
        }

    def test_masses_side_by_side_and_plots_per_mass(self):
        for mass in (300, 1000):
            src = self.mass_dir(
                mass,
                [
                    f"datacard_{mass}.txt",
                    f"era/eE/datacard_{mass}.txt",
                    "plots/DNN_eE_SR_res2b.pdf",
                ],
            )
            self.merge(src, self.dest, f"MX_{mass}")
        self.assertEqual(
            self.tree(),
            {
                "datacard_300.txt",
                "datacard_1000.txt",
                "era/eE/datacard_300.txt",
                "era/eE/datacard_1000.txt",
                "plots/MX_300/DNN_eE_SR_res2b.pdf",
                "plots/MX_1000/DNN_eE_SR_res2b.pdf",
            },
        )

    def test_a_path_written_by_two_masses_raises(self):
        self.merge(self.mass_dir(300, ["shapes.root"]), self.dest, "MX_300")
        with self.assertRaisesRegex(RuntimeError, "shapes.root.*another mass point"):
            self.merge(self.mass_dir(1000, ["shapes.root"]), self.dest, "MX_1000")
        with open(os.path.join(self.dest, "shapes.root")) as f:
            self.assertEqual(f.read(), "300\n")


if __name__ == "__main__":
    unittest.main()
