import contextlib
import law
import os

from string import Template

from FLAF.RunKit.run_tools import ps_call
from FLAF.run_tools.law_customizations import HTCondorWorkflow, copy_param

from .StatInferenceTask import StatInferenceTask


class PreprocessShapesTask(StatInferenceTask, HTCondorWorkflow, law.LocalWorkflow):
    """Run the datacard configuration's `preprocess:` step over the merged histograms.

    A hook, not a rebinning. The configuration names a script and any arguments it wants;
    this task supplies --input, --output, --era and --config and knows nothing else about
    what the step does. --config is the datacard configuration, which any step working on
    these shapes needs anyway: it is where the processes, the model and the categories
    are. HH->bbWW plugs in bin_opt_2d/rebin_2d.py to cut its 2D DNN-vs-HME shapes
    into per-slice 1D ones, but an analysis that needs some other transformation writes its
    own script, and one that needs none declares no `preprocess:` block at all -- then this
    task is never scheduled and CreateDatacardsTask reads the merged histograms directly.

    Whatever the step writes must be laid out as "<era>/<variable>/<variable>.root", which
    is what the model's input_file_pattern resolves against and what the merged histograms
    already look like. That is the whole contract: the datacard step cannot tell whether it
    is reading preprocessed shapes or raw ones.

    One branch per card, era and mass -- branch data {"card": card, "era": era, "mass":
    mass}, card by card in the order of `cards:`, each card's eras in the order of its
    `eras:`, each era's masses in the order of card_masses() -- for the cards that declare
    a `preprocess:` step. A branch stages only the input of its own mass (mass_hist_reqs)
    and writes "<param>_<mass>/<era>/<variable>/<variable>.root", so the branches run as
    independent jobs. CreateDatacardsTask's branch of a card, era and mass requires the
    branch of the same card, era and mass here. ${ERA} and ${<param>} (e.g. ${MX}) in the
    script's arguments are replaced by the branch's era and mass, so the card can restrict
    the script to that mass (EqualSignalBinning: --masses ${MX}); a script given only one
    mass's input must be. The script is given the card as one merged file (see
    StatInferenceTask.write_merged_card), which is also kept in the output.
    """

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 4.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    @staticmethod
    def branch_data_of(task):
        """[branch data] of this task's map, built for any StatInference task `task` (so
        CreateDatacardsTask finds the branch of its own card, era and mass)."""
        task.check_requested_masses()
        return [
            {"card": card, "era": era, "mass": mass}
            for card, era in task.card_eras()
            if task.preprocess_config(card)
            for mass in task.card_masses(card)
        ]

    def model_param(self, card):
        return self.get_config_data(card)["model"]["parameters"][0]

    def input_hist_reqs(self, branches):
        """{key: MergedHists} of the branches (card, era, mass) `branches`, each file
        once: only the input variables of each branch's own mass."""
        reqs = {}
        for data in branches:
            card = data["card"]
            for (era, variable), req in self.mass_hist_reqs(
                self.sub_eras_of([data["era"]], card), [data["mass"]], card
            ).items():
                reqs.setdefault(f"MergedHists_{era}_{variable}", req)
        return reqs

    def stage_workflow_requires(self):
        # Merged with the base class's rather than replacing them: FLAF's HTCondorWorkflow
        # puts the software bundles a submitted job unpacks in there, and returning only
        # our own inputs drops them, so the jobs start without the bundle they need.
        reqs = super().stage_workflow_requires()
        # the selected branches only (the map honours --branches)
        reqs.update(self.input_hist_reqs(self.get_branch_map().values()))
        return reqs

    def requires(self):
        return list(self.input_hist_reqs([self.branch_data]).values())

    def create_branch_map(self):
        return dict(enumerate(self.branch_data_of(self)))

    def output(self):
        return self.output_dir_target(
            self.version,
            "Hists_preprocessed",
            self.branch_data["card"],
            self.branch_data["era"],
            f"{self.model_param(self.branch_data['card'])}_{self.branch_data['mass']}",
        )

    def run(self):
        card, card_era = self.branch_data["card"], self.branch_data["era"]
        mass = self.branch_data["mass"]
        cfg = self.preprocess_config(card)
        if not cfg:
            raise RuntimeError(
                f"{self.datacard_config_path(card)} declares no 'preprocess' block, so "
                "this task has nothing to run. CreateDatacardsTask reads the merged "
                "histograms directly in that case and does not require it."
            )
        script = os.path.join(self.ana_path(), cfg["script"])
        with contextlib.ExitStack() as stack:
            targets = {
                key: req.output()
                for key, req in self.mass_hist_reqs(
                    self.get_sub_eras(card_era, card), [mass], card
                ).items()
            }
            self.check_inputs(targets)
            base_dir_local = stack.enter_context(self.stage_inputs(targets))
            local_output = stack.enter_context(self.output().localize("w"))
            merged_card = self.write_merged_card(
                card, os.path.join(local_output.abspath, self.MERGED_CARD_FILE)
            )
            cmd = [
                "python3",
                "-u",
                script,
                "--input",
                base_dir_local.abspath,
                "--output",
                local_output.abspath,
                "--era",
                card_era,
                "--config",
                merged_card,
            ]
            # ${ERA} and ${<param>} in an argument name the era and mass being produced,
            # the same way the model's input_file_pattern does. An argument beginning with "config/" is resolved
            # against the analysis area, so a configuration can point at its own files
            # without absolute paths; every other argument is passed through untouched.
            for arg in cfg.get("args", []):
                arg = Template(str(arg)).safe_substitute(
                    ERA=card_era, **{self.model_param(card): mass}
                )
                if arg.startswith("config/"):
                    arg = os.path.join(self.ana_path(), arg)
                cmd.append(arg)
            ps_call(cmd, env=self.cmssw_env, verbose=1)
