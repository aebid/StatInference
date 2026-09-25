import contextlib
import law
import luigi
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
    """

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 4.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    # As on CreateDatacardsTask: preprocessing a meta-era needs its members, and
    # self.period must stay a real era because FLAF's Setup resolves it against config/.
    meta_era = luigi.Parameter(default="")

    def get_sub_periods(self):
        return self.get_era_groups().get(self.datacard_era, [self.period])

    def input_hist_reqs(self):
        return {
            f"MergedHists_{era}_{variable}": req
            for (era, variable), req in self.merged_hist_reqs(
                self.get_sub_periods()
            ).items()
        }

    def workflow_requires(self):
        # Merged with the base class's rather than replacing them: FLAF's HTCondorWorkflow
        # puts the software bundles a submitted job unpacks in there, and returning only
        # our own inputs drops them, so the jobs start without the bundle they need.
        reqs = super().workflow_requires()
        reqs.update(self.input_hist_reqs())
        return reqs

    def requires(self):
        return list(self.input_hist_reqs().values())

    def create_branch_map(self):
        return {0: None}

    def output(self):
        return self.output_dir_target(
            *self.version_parts(), "Hists_preprocessed", self.datacard_era
        )

    def run(self):
        cfg = self.preprocess_config()
        if not cfg:
            raise RuntimeError(
                f"{self.datacard_config_path()} declares no 'preprocess' block, so this "
                "task has nothing to run. CreateDatacardsTask reads the merged histograms "
                "directly in that case and does not require it."
            )
        script = os.path.join(self.ana_path(), cfg["script"])
        with contextlib.ExitStack() as stack:
            targets = {
                key: req.output()
                for key, req in self.merged_hist_reqs(self.get_sub_periods()).items()
            }
            self.check_inputs(targets)
            base_dir_local = stack.enter_context(self.stage_inputs(targets))
            local_output = stack.enter_context(self.output().localize("w"))
            cmd = [
                "python3",
                "-u",
                script,
                "--input",
                base_dir_local.abspath,
                "--output",
                local_output.abspath,
                "--era",
                self.datacard_era,
                "--config",
                self.datacard_config_path(),
            ]
            # ${ERA} in an argument names the era being produced, the same way the model's
            # input_file_pattern does. An argument beginning with "config/" is resolved
            # against the analysis area, so a configuration can point at its own files
            # without absolute paths; every other argument is passed through untouched.
            for arg in cfg.get("args", []):
                arg = Template(str(arg)).safe_substitute(ERA=self.datacard_era)
                if arg.startswith("config/"):
                    arg = os.path.join(self.ana_path(), arg)
                cmd.append(arg)
            ps_call(cmd, env=self.cmssw_env, verbose=1)
