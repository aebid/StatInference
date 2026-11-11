#!/usr/bin/env python3
"""Datacard configurations: cards with includes, and the list of cards to build.

* load_card_config(): a card without `include:` reads exactly as yaml.safe_load() reads it.
  With includes, the files merge in order (includes, then the card): named lists are joined,
  and every other top-level key comes from the one file that sets it. Every ambiguity
  raises, naming the files: a key set in two files (also with identical values), an entry
  name defined twice (across files or within one), an unnamed entry in a card with
  includes, an include that includes, is included twice, is the card itself or is missing.
  Include paths are relative to the including file.
* resolve_cards(): {card name: file}, in order, file paths relative to the base directory;
  a repeated file, a name unfit for a directory, a missing card, no cards raise.
* The law chain over global.yaml's StatInference `cards:` of two cards -- one plain, one
  split with an include, with different eras and only one of them preprocessed -- has a
  branch per card, era and mass; each card's outputs live under its name; the per-card
  tasks without --card stand for every card; an unknown card, a block setting both
  `cards:` and `config:` (also when one is empty), and an empty `cards:` or `config:` raise.
"""

import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statinference_fixture as fx  # noqa: E402

import luigi  # noqa: E402
import yaml  # noqa: E402

from StatInference.common.card_config import (  # noqa: E402
    CardConfigError,
    card_name_of,
    dump_card_config,
    load_card_config,
    load_card_config_with_sources,
    resolve_cards,
)


class FilesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="statinference_card_config_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def file(self, name, text):
        path = os.path.join(self.tmp, name)
        fx.write(path, textwrap.dedent(text))
        return path

    def assert_raises_naming(self, call, *needles):
        with self.assertRaises(CardConfigError) as raised:
            call()
        message = str(raised.exception)
        for needle in needles:
            self.assertIn(needle, message)
        return message


COMMON = """
    model:
      parameters: [ MX ]
    channels: [ eE, muMu ]
    processes:
      - process: data_obs
        is_data: true
      - TT
    uncertainties:
      - name: lumi
        type: lnN
        value: 1.02
"""

SPIN0 = """
    include: [ common.yaml ]
    eras: [ Run3_2022 ]
    processes:
      - process: Radion_${MX}
        is_signal: true
        param_values: [ 300 ]
    uncertainties:
      - name: BR_hbb
        type: lnN
        processes: [ ^Radion_ ]
        value: 1.01
    limit_plots:
      - name: spin0
"""


class Merging(FilesTestCase):
    def test_a_card_without_includes_reads_as_it_is(self):
        text = fx.card(preprocess=True)
        path = self.file("card.yaml", text)
        self.assertEqual(load_card_config(path), yaml.safe_load(text))

    def test_includes_merge_in_order(self):
        self.file("common.yaml", COMMON)
        merged, sources = load_card_config_with_sources(self.file("spin0.yaml", SPIN0))
        self.assertNotIn("include", merged)
        self.assertEqual(merged["model"], {"parameters": ["MX"]})
        self.assertEqual(merged["eras"], ["Run3_2022"])
        # includes first, then the card
        self.assertEqual(
            [p if isinstance(p, str) else p["process"] for p in merged["processes"]],
            ["data_obs", "TT", "Radion_${MX}"],
        )
        self.assertEqual(
            [u["name"] for u in merged["uncertainties"]], ["lumi", "BR_hbb"]
        )
        self.assertEqual(sources["model"], os.path.join(self.tmp, "common.yaml"))
        self.assertEqual(
            sources[("uncertainties", "BR_hbb")], os.path.join(self.tmp, "spin0.yaml")
        )

    def test_a_key_set_in_an_include_and_the_card_raises(self):
        self.file("common.yaml", COMMON)
        card = self.file("spin0.yaml", SPIN0 + "    channels: [ eE ]\n")
        self.assert_raises_naming(
            lambda: load_card_config(card),
            "`channels:`",
            os.path.join(self.tmp, "common.yaml"),
            card,
        )

    def test_a_key_set_twice_raises_even_with_the_same_value(self):
        self.file("common.yaml", COMMON)
        self.file("other.yaml", "channels: [ eE, muMu ]\n")
        card = self.file(
            "spin0.yaml",
            SPIN0.replace("[ common.yaml ]", "[ common.yaml, other.yaml ]"),
        )
        self.assert_raises_naming(
            lambda: load_card_config(card), "`channels:`", "other.yaml", "common.yaml"
        )

    def test_a_mapping_is_not_merged_below_the_top_level(self):
        self.file("common.yaml", COMMON)
        card = self.file(
            "spin0.yaml", SPIN0 + "    model:\n      input_file_pattern: x.root\n"
        )
        self.assert_raises_naming(lambda: load_card_config(card), "`model:`")

    def test_an_entry_defined_in_two_files_raises(self):
        self.file("common.yaml", COMMON)
        card = self.file("spin0.yaml", SPIN0.replace("- name: BR_hbb", "- name: lumi"))
        self.assert_raises_naming(
            lambda: load_card_config(card),
            "`uncertainties` entry 'lumi'",
            "common.yaml",
            "spin0.yaml",
        )

    def test_an_identical_entry_in_two_files_raises(self):
        self.file("common.yaml", COMMON)
        card = self.file(
            "spin0.yaml",
            SPIN0.replace("    uncertainties:\n", "      - TT\n    uncertainties:\n"),
        )
        self.assert_raises_naming(
            lambda: load_card_config(card), "`processes` entry 'TT'"
        )

    def test_an_entry_defined_twice_in_one_file_raises(self):
        card = self.file(
            "card.yaml",
            """
            uncertainties:
              - name: lumi
              - name: lumi
            """,
        )
        self.assert_raises_naming(
            lambda: load_card_config(card), "'lumi' is defined twice in"
        )

    def test_an_unnamed_entry_raises_only_in_a_card_with_includes(self):
        unnamed = "    impact_plots:\n      - masses: [ 300 ]\n"
        plain = self.file("plain.yaml", unnamed)
        self.assertEqual(load_card_config(plain)["impact_plots"], [{"masses": [300]}])
        self.file("common.yaml", COMMON)
        card = self.file("spin0.yaml", SPIN0 + unnamed)
        self.assert_raises_naming(
            lambda: load_card_config(card), "`impact_plots` has no `name`"
        )

    def test_a_named_list_must_be_a_list(self):
        card = self.file("card.yaml", "uncertainties:\n  lumi: {}\n")
        self.assert_raises_naming(
            lambda: load_card_config(card), "`uncertainties:` must be a list"
        )

    def test_include_paths_are_relative_to_the_including_file(self):
        self.file("shared/common.yaml", COMMON)
        card = self.file(
            "DL/spin0.yaml", SPIN0.replace("common.yaml", "../shared/common.yaml")
        )
        self.assertEqual(load_card_config(card)["channels"], ["eE", "muMu"])

    def test_includes_do_not_nest(self):
        self.file("deeper.yaml", "analysis: x\n")
        self.file("common.yaml", COMMON + "    include: [ deeper.yaml ]\n")
        card = self.file("spin0.yaml", SPIN0)
        self.assert_raises_naming(
            lambda: load_card_config(card), "cannot include further files"
        )

    def test_an_include_listed_twice_raises(self):
        self.file("common.yaml", COMMON)
        card = self.file(
            "spin0.yaml",
            SPIN0.replace("[ common.yaml ]", "[ common.yaml, ./common.yaml ]"),
        )
        self.assert_raises_naming(lambda: load_card_config(card), "already included")

    def test_a_card_including_itself_raises(self):
        card = self.file("spin0.yaml", "include: [ spin0.yaml ]\n")
        self.assert_raises_naming(lambda: load_card_config(card), "the card itself")

    def test_a_missing_include_raises_naming_the_card(self):
        card = self.file("spin0.yaml", SPIN0)
        self.assert_raises_naming(
            lambda: load_card_config(card),
            f"{card}: `include:` lists common.yaml, but there is no such file",
        )

    def test_an_empty_include_raises(self):
        card = self.file("spin0.yaml", 'include: [ "" ]\n')
        self.assert_raises_naming(lambda: load_card_config(card), "an empty file name")

    def test_a_key_set_twice_in_one_file_raises(self):
        """YAML itself keeps the last of two equal keys; the loader refuses, naming the
        file and both lines -- at the top level and inside an entry."""
        card = self.file(
            "card.yaml",
            """
            uncertainties:
              - name: lumi
            channels: [ eE ]
            uncertainties:
              - name: pileup
            """,
        )
        self.assert_raises_naming(
            lambda: load_card_config(card),
            f"{card}: key 'uncertainties' is set twice in one mapping, on lines 2 and 5",
        )
        nested = self.file(
            "nested.yaml",
            """
            uncertainties:
              - name: lumi
                type: lnN
                type: shape
            """,
        )
        self.assert_raises_naming(
            lambda: load_card_config(nested), "key 'type' is set twice"
        )

    def test_a_merge_key_may_be_overridden(self):
        """`<<:` brings in an anchor's keys; a key written next to it overrides the merged
        one, as YAML intends -- only explicitly written keys are checked."""
        card = self.file(
            "card.yaml",
            """
            limit_plots:
              - name: a
                plot_params: &style { y_log: true, x_max: 1500 }
              - name: b
                plot_params:
                  <<: *style
                  x_max: 1000
            """,
        )
        plots = load_card_config(card)["limit_plots"]
        self.assertEqual(plots[1]["plot_params"], {"y_log": True, "x_max": 1000})

    def test_the_dump_reads_back_as_the_merged_card(self):
        self.file("common.yaml", COMMON)
        card = self.file("spin0.yaml", SPIN0)
        out = os.path.join(self.tmp, "merged.yaml")
        merged = dump_card_config(card, out, relative_to=self.tmp)
        self.assertEqual(load_card_config(out), merged)
        with open(out) as f:
            header = f.read().split("\n")[:4]
        self.assertEqual(
            header,
            [
                "# The datacard configuration spin0.yaml",
                "# merged with the files it includes, in this order:",
                "#   common.yaml",
                "#   spin0.yaml",
            ],
        )


class Cards(FilesTestCase):
    def resolve(self, cards):
        return resolve_cards(cards, self.tmp, "StatInference `cards:`")

    def test_the_cards_in_order_relative_to_the_base(self):
        a = self.file("Datacards/DL/a.yaml", "analysis: a\n")
        b = self.file("Datacards/b.yaml", "analysis: b\n")
        cards = {"DL_b": "Datacards/b.yaml", "DL_a": "Datacards/DL/a.yaml"}
        self.assertEqual(list(self.resolve(cards).items()), [("DL_b", b), ("DL_a", a)])

    def test_the_same_file_twice_raises(self):
        self.file("a.yaml", "analysis: a\n")
        self.assert_raises_naming(
            lambda: self.resolve({"a": "a.yaml", "b": "./a.yaml"}),
            "'a' and 'b' are the same file",
        )

    def test_two_keys_reading_as_one_name_raise(self):
        self.file("a.yaml", "analysis: a\n")
        self.file("b.yaml", "analysis: b\n")
        self.assert_raises_naming(
            lambda: self.resolve({1: "a.yaml", "1": "b.yaml"}),
            "two cards are named '1'",
        )

    def test_a_card_name_with_a_trailing_newline_raises(self):
        self.file("a.yaml", "analysis: a\n")
        self.assert_raises_naming(lambda: self.resolve({"a\n": "a.yaml"}), "card name")

    def test_a_config_cards_file_name_must_be_a_card_name(self):
        self.assert_raises_naming(
            lambda: card_name_of("/x/my card.yaml"), "card name 'my card'"
        )
        self.assertEqual(card_name_of("/x/dl_spin0.yaml"), "dl_spin0")

    def test_a_card_name_must_be_a_directory_name(self):
        self.file("a.yaml", "analysis: a\n")
        self.assert_raises_naming(lambda: self.resolve({"a/b": "a.yaml"}), "'a/b'")

    def test_a_missing_card_raises(self):
        self.assert_raises_naming(lambda: self.resolve({"a": "a.yaml"}), "no such file")

    def test_no_cards_raises(self):
        self.assert_raises_naming(lambda: self.resolve({}), "must map")


VERSION = "v_cards"
HISTS_VERSION = "v_hists"

#: card "B": the fixture card's settings minus its eras and processes, in a shared file,
#: plus its own eras (only Run3_2023) and processes -- and no preprocessing step, so card
#: "A" (card_preprocess.yaml) is the only preprocessed one
SHARED = "\n".join(
    line
    for line in fx.card(preprocess=False).split("processes:")[0].split("\n")
    if not line.startswith("eras:")
)
CARD_B = f"""
include: [ ../shared.yaml ]
eras: [ {fx.ERA_C} ]
processes:
  - process: data_obs
    is_data: true
  - process: Signal_${{MX}}
    param_values: [ {", ".join(str(m) for m in fx.MASSES)} ]
    is_signal: true
  - TT
limit_plots:
  - name: b_limits
    datacards:
      - {{ name: "B", glob: "*.txt" }}
impact_plots:
  - name: b_impacts
    masses: [ {fx.MASSES[0]} ]
"""
#: the StatInference block's cards, relative to the analysis
CARDS = (
    "cards:\n"
    "    A: config/Datacards/card_preprocess.yaml\n"
    "    B: config/Datacards/B/card_b.yaml"
)

#: (card, era) of every card's eras, in the order of `cards:`
CARD_ERAS = [("A", era) for era in fx.CARD_ERAS] + [("B", fx.ERA_C)]


class CardsChain(unittest.TestCase):
    statinference = CARDS

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="statinference_cards_")
        cls.ana = fx.make_analysis(cls.tmp, "config/Datacards/card.yaml")
        datacards = os.path.join(cls.ana, "config", "Datacards")
        fx.write(os.path.join(datacards, "shared.yaml"), SHARED)
        fx.write(os.path.join(datacards, "B", "card_b.yaml"), CARD_B)
        global_yaml = os.path.join(cls.ana, "config", "global.yaml")
        with open(global_yaml) as f:
            text = f.read()
        fx.write(
            global_yaml,
            text.replace(
                "  config: config/Datacards/card.yaml", f"  {cls.statinference}"
            ),
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

    def task(self, cls, eras=fx.ERAS, **kwargs):
        return cls(version=VERSION, era=eras, hists_version=HISTS_VERSION, **kwargs)

    def store(self, *path):
        return os.path.join(self.tmp, "store", "default", VERSION, *path)

    def test_a_branch_per_card_era_and_mass(self):
        workflow = self.task(fx.si.CreateDatacardsTask)
        self.assertEqual(
            list(workflow.get_branch_map().values()),
            [
                {"card": card, "era": era, "mass": mass}
                for card, era in CARD_ERAS
                for mass in fx.MASSES
            ],
        )

    def test_the_split_card_reads_merged(self):
        task = self.task(fx.si.CreateDatacardsTask)
        b = task.get_config_data("B")
        self.assertEqual(b["eras"], [fx.ERA_C])
        self.assertEqual(b["channels"], ["eE", "muMu"])
        self.assertEqual(task.get_config_data("A")["eras"], fx.CARD_ERAS)

    def test_outputs_per_card(self):
        for branch in self.task(fx.si.CreateDatacardsTask).get_branch_tasks().values():
            data = branch.branch_data
            self.assertEqual(
                branch.output().abspath,
                self.store(
                    "Datacards", data["card"], data["era"], f"MX_{data['mass']}"
                ),
            )

    def test_only_the_preprocessed_card_is_preprocessed(self):
        preprocess = self.task(fx.si.PreprocessShapesTask)
        self.assertEqual(
            list(preprocess.get_branch_map().values()),
            [{"card": "A", "era": era} for era in fx.CARD_ERAS],
        )
        for branch in self.task(fx.si.CreateDatacardsTask).get_branch_tasks().values():
            reqs = luigi.task.flatten(branch.requires())
            if branch.branch_data["card"] == "A":
                (req,) = reqs
                self.assertIsInstance(req, fx.si.PreprocessShapesTask)
                self.assertEqual(
                    req.branch_data, {"card": "A", "era": branch.branch_data["era"]}
                )
            else:
                self.assertTrue(all(isinstance(r, fx.si.MergedHists) for r in reqs))
                self.assertEqual({r.hist_era for r in reqs}, {fx.ERA_C})

    def test_the_limits_of_every_card_without_card(self):
        limits = self.task(fx.si.ResonantLimitsTask)
        self.assertTrue(limits.all_cards)
        reqs = limits.requires()
        self.assertEqual(list(reqs), ["A", "B"])
        for card, task in reqs.items():
            self.assertEqual(task.card, card)
            self.assertEqual(
                task.output()["limits"].abspath,
                os.path.join(
                    self.tmp,
                    "data",
                    VERSION,
                    "ResonantLimitsTask",
                    card,
                    "combined",
                    "limits.npz",
                ),
            )
        self.assertEqual(
            {card: out["limits"].abspath for card, out in limits.output().items()},
            {card: t.output()["limits"].abspath for card, t in reqs.items()},
        )

    def test_the_limits_of_one_card(self):
        limits = self.task(fx.si.ResonantLimitsTask, card="B")
        self.assertFalse(limits.all_cards)
        self.assertIsInstance(limits.requires(), fx.si.CreateDatacardsTask)
        self.assertEqual(list(limits.era_card_outputs()), [fx.ERA_C])
        self.assertEqual(
            limits.datacards_dir(fx.ERA_C),
            os.path.join(self.tmp, "data", VERSION, "Datacards", "B", fx.ERA_C),
        )

    def test_one_card_builds_only_its_own_branches(self):
        """--card X requires X's datacard branches only, and through them only X's
        preprocessing (A) or merged histograms (B): another card is neither built nor
        waited for."""
        all_branches = self.task(fx.si.CreateDatacardsTask).get_branch_map()
        for card in ("A", "B"):
            with self.subTest(card=card):
                limits = self.task(fx.si.ResonantLimitsTask, card=card)
                workflow = limits.requires()
                own = {i for i, d in all_branches.items() if d["card"] == card}
                self.assertEqual(set(workflow.get_branch_map()), own)
                reqs = workflow.workflow_requires()
                if card == "A":
                    preprocess = reqs["PreprocessShapes"]
                    self.assertEqual(
                        {d["card"] for d in preprocess.get_branch_map().values()}, {"A"}
                    )
                else:
                    self.assertNotIn("PreprocessShapes", reqs)
                    merged = luigi.task.flatten(reqs)
                    self.assertEqual({r.hist_era for r in merged}, {fx.ERA_C})
        # the plot tasks reach the datacards through the card's limits
        plots = self.task(fx.si.PlotResonantLimitsTask, card="B")
        (limits,) = plots.requires()
        self.assertEqual(
            set(limits.requires().get_branch_map()),
            {i for i, d in all_branches.items() if d["card"] == "B"},
        )

    def test_a_card_task_checks_every_cards_eras_at_construction(self):
        """B needs only Run3_2023, but the datacard workflow builds its map over every
        card, so A's eras are required too -- refused when the task is constructed, not
        during scheduling (where a keep_alive worker would wait for ever)."""
        with self.assertRaises(ValueError) as raised:
            self.task(fx.si.ResonantLimitsTask, card="B", eras=(fx.ERA_C,))
        self.assertIn(f"A {fx.GROUP}: {fx.ERA_A}, {fx.ERA_B}", str(raised.exception))

    def test_plots_per_card(self):
        for cls, name in (
            (fx.si.PlotResonantLimitsTask, "LimitPlots"),
            (fx.si.PlotPullsAndImpactsTask, "ImpactPlots"),
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(
                    self.task(cls, card="A").output().abspath, self.store(name, "A")
                )
                # A declares no plots: the all-cards wrapper leaves it out
                self.assertEqual(list(self.task(cls).requires()), ["B"])

    def test_a_plot_wrapper_with_no_card_to_plot_raises(self):
        for cls in (fx.si.PlotResonantLimitsTask, fx.si.PlotPullsAndImpactsTask):
            with self.subTest(cls=cls.__name__), mock.patch.object(
                cls, "card_has_work", return_value=False
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self.task(cls).requires()
                self.assertIn(
                    "no card has anything to do (A, B)", str(raised.exception)
                )

    def test_impact_plots_carry_the_card_in_their_name(self):
        task = self.task(fx.si.PlotPullsAndImpactsTask, card="B")
        entry = task.get_config_data()["impact_plots"][0]
        # dhi is a stand-in here, so its parameter list cannot be checked
        with mock.patch.object(
            type(task), "card_for", return_value="card.txt"
        ), mock.patch.object(type(task), "validate_plot_params"):
            spec = task.build_plot_spec(entry, "combined", fx.MASSES[0])
        self.assertEqual(spec["plot_postfix"], "B")

    def test_an_include_cards_script_gets_the_merged_card(self):
        """CreateDatacardsTask.run() for B, whose card includes a shared file: the
        script's --config is the merged card -- no `include:` left, equal to what the
        loader reads -- written into the task's output."""
        import importlib

        module = importlib.import_module("StatInference.law.CreateDatacardsTask")
        workflow = self.task(fx.si.CreateDatacardsTask, make_plots=False)
        (branch,) = [
            t
            for t in workflow.get_branch_tasks().values()
            if t.branch_data["card"] == "B"
        ][:1]
        for req in luigi.task.flatten(branch.requires()):
            fx.write(req.output().abspath, "")
        calls = []

        def ps_call(cmd, **kwargs):
            config = cmd[cmd.index("--config") + 1]
            with open(config) as f:
                calls.append((cmd, yaml.safe_load(f)))
            fx.write(os.path.join(cmd[cmd.index("--output") + 1], "payload"), "")

        with mock.patch.object(module, "ps_call", ps_call), mock.patch.object(
            type(branch), "cmssw_env", new_callable=mock.PropertyMock
        ):
            branch.run()
        ((cmd, given),) = calls
        self.assertNotIn("include", given)
        self.assertEqual(given, load_card_config(branch.datacard_config_path("B")))
        self.assertEqual(
            os.path.basename(cmd[cmd.index("--config") + 1]), "card_config.yaml"
        )
        self.assertIn("card_config.yaml", os.listdir(branch.output().abspath))

    def test_an_unknown_card_raises(self):
        with self.assertRaises(ValueError) as raised:
            self.task(fx.si.ResonantLimitsTask, card="C")
        self.assertIn("no card 'C'", str(raised.exception))

    def test_missing_eras_name_the_card(self):
        workflow = self.task(fx.si.CreateDatacardsTask, eras=(fx.ERA_A, fx.ERA_B))
        with self.assertRaises(ValueError) as raised:
            workflow.get_branch_map()
        self.assertIn(f"B {fx.ERA_C}: {fx.ERA_C}", str(raised.exception))


class RequestedMasses(unittest.TestCase):
    """global.yaml's `param_values` serves every card: each card is built for the
    requested masses it declares; a mass no card declares, or a card left with none,
    raises."""

    def run_with(self, param_values, check):
        class Block(CardsChain):
            pass

        Block.statinference = CARDS + f"\n  param_values: {param_values}"
        Block.setUpClass()
        try:
            fx.clear_process_caches()
            check(
                lambda cls, **kw: cls(
                    version=VERSION, era=fx.ERAS, hists_version=HISTS_VERSION, **kw
                )
            )
        finally:
            Block.tearDownClass()

    def test_each_card_gets_the_requested_masses_it_declares(self):
        def check(task):
            masses = {
                d["mass"]
                for d in task(fx.si.CreateDatacardsTask).get_branch_map().values()
            }
            self.assertEqual(masses, {fx.MASSES[0]})

        self.run_with(f"[ {fx.MASSES[0]} ]", check)

    def test_a_mass_no_card_declares_raises(self):
        def check(task):
            with self.assertRaises(ValueError) as raised:
                task(fx.si.CreateDatacardsTask).get_branch_map()
            self.assertIn("lists [999], which no card declares", str(raised.exception))

        self.run_with(f"[ {fx.MASSES[0]}, 999 ]", check)

    def test_a_card_left_with_no_mass_raises(self):
        def check(task):
            limits = task(fx.si.ResonantLimitsTask, card="B")
            with mock.patch.object(
                fx.si.CreateDatacardsTask,
                "get_masses",
                lambda self, card=None: [999] if card == "A" else fx.MASSES,
            ):
                fx.clear_process_caches()
                with self.assertRaises(ValueError) as raised:
                    limits.requires()
            self.assertIn("card 'B' has no mass point to build", str(raised.exception))

        self.run_with("[ 999 ]", check)


class AmbiguousStatInferenceBlock(unittest.TestCase):
    """The cards are chosen by which keys the block sets, not by their values: an empty
    `cards:` is reported, never read as unset."""

    def error(self, statinference):
        class Block(CardsChain):
            pass

        Block.statinference = statinference
        Block.setUpClass()
        try:
            fx.clear_process_caches()
            task = fx.si.CreateDatacardsTask(
                version=VERSION, era=fx.ERAS, hists_version=HISTS_VERSION
            )
            with self.assertRaises(CardConfigError) as raised:
                task.get_branch_map()
            return str(raised.exception)
        finally:
            Block.tearDownClass()

    def test_both_raise(self):
        self.assertIn(
            "sets both `cards:` and `config:`",
            self.error(CARDS + "\n  config: config/Datacards/card.yaml"),
        )

    def test_an_empty_cards_next_to_config_still_raises(self):
        self.assertIn(
            "sets both `cards:` and `config:`",
            self.error("cards: {}\n  config: config/Datacards/card.yaml"),
        )

    def test_an_empty_cards_raises(self):
        for empty in ("cards: {}", "cards:"):
            with self.subTest(empty=empty):
                self.assertIn("must map each card name to its file", self.error(empty))

    def test_an_empty_config_raises(self):
        self.assertIn("`config:` must name one card file", self.error('config: ""'))


if __name__ == "__main__":
    unittest.main()
