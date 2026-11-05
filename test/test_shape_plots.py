#!/usr/bin/env python3
"""The shape plots of CreateDatacardsTask for categories read through `input_categories`.

The maker reads a mapped datacard category from its input directory, with the model point
substituted (`SR/res2b` from `SR_SL_M${MX}/res2b`); the plots must show the same histograms.
On the scratch analysis of statinference_fixture with a single-lepton-like card (two mass
points, two mapped categories and one read under its own name), the branch of each mass plots
that mass's variable alone, and HistPlotter is asked for the input directory of each mapped
category, also when a directory under the configured name exists too; the plot files and the grids keep the datacard bin's name; with nothing to
plot the paths looked up, an unreadable summed file or the lack of a category read from one
directory are named and the summed shapes removed; the plotter gets the task's --user-custom; a
category the points of one variable read from different directories is not plotted.
"""

import importlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402

VERSION = "v_plots"
HISTS_VERSION = "v_hists"
MASSES = [300, 500]
CHANNELS = ["e", "mu"]

CARD = """
analysis: test
type: resonant
model:
  parameters: [ MX ]
  param_dependent_bkg: true
  input_file_pattern: ${ERA}/TwoStageDNN_M${MX}_binary/TwoStageDNN_M${MX}_binary.root
eras: [ Run3_2022 ]
channels: [ e, mu ]
categories: [ SR/res2b, SR/recovery, OS_Iso/inclusive ]
input_categories:
  SR/res2b: SR_SL_M${MX}/res2b
  SR/recovery: SR_SL_M${MX}/recovery
processes:
  - process: data_obs
    is_data: true
  - process: Signal_${MX}
    param_values: [ 300, 500 ]
    is_signal: true
  - TT
"""

create_datacards_module = importlib.import_module(
    "StatInference.law.CreateDatacardsTask"
)


def variable(mass):
    return f"TwoStageDNN_M{mass}_binary"


class _ShapeFile:
    def __init__(self, directories):
        self.directories = directories

    def IsZombie(self):
        return False

    def Get(self, path):
        return path in self.directories

    def Close(self):
        pass


def fake_root(directories_of):
    """A ROOT whose summed shape file of a variable holds `directories_of[variable]`."""

    class TFile:
        @staticmethod
        def Open(path):
            name = os.path.basename(path)[len("_summed_") : -len(".root")]
            return _ShapeFile(directories_of.get(name, set()))

    class ROOT:
        pass

    ROOT.TFile = TFile
    return ROOT


def sl_directories(mass, with_configured_name=False):
    """What a single-lepton summed shape file of `mass` holds: every mass's signal region,
    the OS_Iso region and, optionally, an `SR` region under the configured name."""
    regions = [f"SR_SL_M{m}" for m in MASSES] + (["SR"] if with_configured_name else [])
    directories = set()
    for channel in CHANNELS:
        directories.add(channel)
        for region in regions:
            directories.add(f"{channel}/{region}")
            for cat in ("res2b", "recovery"):
                directories.add(f"{channel}/{region}/{cat}")
        directories.add(f"{channel}/OS_Iso")
        directories.add(f"{channel}/OS_Iso/inclusive")
    return directories


class ShapePlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="statinference_plots_")
        cls.ana = fx.make_analysis(cls.tmp, "config/Datacards/card_sl.yaml")
        fx.write(os.path.join(cls.ana, "config", "Datacards", "card_sl.yaml"), CARD)
        fx.write(
            os.path.join(cls.ana, "config", "user_custom_plots.yaml"), "# no override\n"
        )
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
        for mass in MASSES:
            var = variable(mass)
            fx.write(
                os.path.join(
                    self.tmp,
                    fx.hist_store(fx.ERA_A),
                    HISTS_VERSION,
                    "Hists_merged",
                    fx.ERA_A,
                    var,
                    f"{var}.root",
                ),
                "",
            )

    def card(self, mass, **params):
        """The branch of the one card that builds `mass`."""
        card = fx.si.CreateDatacardsTask(
            version=VERSION,
            era=fx.ERA_A,
            hists_version=HISTS_VERSION,
            branch=MASSES.index(mass),
            **params,
        )
        self.assertEqual(card.branch_data, {"era": fx.ERA_A, "mass": mass})
        return card

    def plot(self, card, directories_of, root=None):
        """Run `card` with its commands recorded: (HistPlotter commands, the grids' panel
        maps, the summed files left in plots/, printed lines)."""
        calls, grids, printed = [], [], []

        def ps_call(cmd, env=None, **kwargs):
            if cmd[0] == "hadd":
                fx.write(cmd[2], "")
            calls.append(list(cmd))

        def stitch_grids(task, cfg, plots_dir, var, panel_of_key):
            grids.append((var, dict(panel_of_key)))

        with mock.patch.object(
            create_datacards_module, "ps_call", ps_call
        ), mock.patch.object(
            type(card), "cmssw_env", new_callable=mock.PropertyMock
        ), mock.patch.object(
            create_datacards_module,
            "importROOT",
            lambda: root if root is not None else fake_root(directories_of),
        ), mock.patch.object(
            fx.si.CreateDatacardsTask, "_stitch_grids", stitch_grids
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            card.run()
        plots_dir = os.path.join(card.output().abspath, "plots")
        left = sorted(n for n in os.listdir(plots_dir) if n.startswith("_summed_"))
        plotter = [c for c in calls if "--all_keys" in c]
        return plotter, grids, left, printed

    @staticmethod
    def arg(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    @staticmethod
    def expected(mass):
        """(HistPlotter keys, output file names, grid panels) of `mass`'s variable."""
        keys, outputs, panels = [], [], {}
        for region, cat, in_region in (
            ("SR", "res2b", f"SR_SL_M{mass}"),
            ("SR", "recovery", f"SR_SL_M{mass}"),
            ("OS_Iso", "inclusive", "OS_Iso"),
        ):
            for channel in CHANNELS:
                keys.append(f"{channel}:{cat}:{in_region}")
                out = f"{variable(mass)}_{channel}_{region}_{cat}.pdf"
                outputs.append(out)
                panels[f"{channel}:{cat}:{region}"] = out
        return keys, outputs, panels

    def check_plots(self, card, directories_of):
        """The plots of `card`: those of its own mass's variable only."""
        mass = card.branch_data["mass"]
        plotter, grids, left, _ = self.plot(card, directories_of)
        self.assertEqual(len(plotter), 1)
        for cmd, (grid_var, panels) in zip(plotter, grids):
            var = os.path.basename(self.arg(cmd, "--inFile"))[len("_summed_") : -5]
            self.assertEqual(var, variable(mass))
            keys, outputs, expected_panels = self.expected(mass)
            self.assertEqual(self.arg(cmd, "--all_keys").split(","), keys)
            # in the plots/ directory of the card's output, while the task writes it
            files = self.arg(cmd, "--all_outFiles").split(",")
            self.assertEqual([os.path.basename(f) for f in files], outputs)
            self.assertEqual(
                {os.path.basename(os.path.dirname(f)) for f in files}, {"plots"}
            )
            self.assertEqual(grid_var, var)
            self.assertEqual(
                {k: os.path.basename(v) for k, v in panels.items()}, expected_panels
            )
            self.assertEqual(set(panels.values()), set(files))
        self.assertEqual(left, [])
        return plotter

    def test_mapped_categories_are_plotted_from_their_input_directory(self):
        for mass in MASSES:
            with self.subTest(mass=mass):
                self.check_plots(
                    self.card(mass), {variable(m): sl_directories(m) for m in MASSES}
                )

    def test_the_input_directory_wins_over_one_of_the_configured_name(self):
        for mass in MASSES:
            with self.subTest(mass=mass):
                self.check_plots(
                    self.card(mass),
                    {
                        variable(m): sl_directories(m, with_configured_name=True)
                        for m in MASSES
                    },
                )

    def test_nothing_to_plot_names_the_paths_looked_up(self):
        for mass in MASSES:
            with self.subTest(mass=mass):
                plotter, _, left, printed = self.plot(self.card(mass), {})
                self.assertEqual(plotter, [])
                self.assertEqual(left, [])
                # one warning, for this mass's variable
                (warning,) = [
                    line
                    for line in printed
                    if line.startswith("WARNING: no shapes to plot for")
                ]
                self.assertTrue(
                    warning.startswith(
                        f"WARNING: no shapes to plot for {variable(mass)}:"
                    )
                )
                for channel in CHANNELS:
                    self.assertIn(f"{channel}/SR_SL_M{mass}/res2b", warning)
                    self.assertIn(f"{channel}/OS_Iso/inclusive", warning)

    def test_an_unreadable_summed_file_is_named(self):
        class Unreadable:
            class TFile:
                @staticmethod
                def Open(path):
                    return None

        for mass in MASSES:
            with self.subTest(mass=mass):
                plotter, _, left, printed = self.plot(
                    self.card(mass), {}, root=Unreadable
                )
                self.assertEqual(plotter, [])
                self.assertEqual(left, [])
                var = variable(mass)
                pattern = (
                    f"WARNING: no shapes to plot for {var}: "
                    f".*/plots/_summed_{var}\\.root cannot be read"
                )
                self.assertTrue(
                    any(re.fullmatch(pattern, line) for line in printed), printed
                )

    def test_no_category_from_one_directory_is_named(self):
        with mock.patch.object(
            fx.si.CreateDatacardsTask,
            "_present_keys",
            staticmethod(lambda shape_file, cfg, points: ({}, [])),
        ):
            for mass in MASSES:
                with self.subTest(mass=mass):
                    plotter, _, left, printed = self.plot(self.card(mass), {})
                    self.assertEqual(plotter, [])
                    self.assertEqual(left, [])
                    self.assertIn(
                        f"WARNING: no shapes to plot for {variable(mass)}: "
                        "no category is read from one directory",
                        printed,
                    )

    def test_the_plotter_gets_the_user_custom(self):
        directories = {variable(m): sl_directories(m) for m in MASSES}
        mass = MASSES[0]
        for cmd in self.check_plots(self.card(mass), directories):
            self.assertNotIn("--user-custom", cmd)
        fx.clear_process_caches()
        user_custom = os.path.join(self.ana, "config", "user_custom_plots.yaml")
        for cmd in self.check_plots(
            self.card(mass, user_custom=user_custom), directories
        ):
            self.assertEqual(self.arg(cmd, "--user-custom"), user_custom)

    def test_a_category_read_from_several_directories_is_not_plotted(self):
        card = self.card(MASSES[0])
        cfg = card.get_config_data()
        printed = []
        with mock.patch.object(
            create_datacards_module,
            "importROOT",
            lambda: fake_root({"shared": sl_directories(300)}),
        ), mock.patch(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        ):
            present, looked_up = card._present_keys(
                "_summed_shared.root", cfg, [{"MX": 300}, {"MX": 500}]
            )
        self.assertEqual(
            present,
            {
                (channel, "OS_Iso", "inclusive"): "OS_Iso/inclusive"
                for channel in CHANNELS
            },
        )
        self.assertEqual(looked_up, [f"{c}/OS_Iso/inclusive" for c in CHANNELS])
        self.assertEqual(len([line for line in printed if "not plotted" in line]), 2)


if __name__ == "__main__":
    unittest.main()
