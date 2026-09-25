import law
import luigi

from .StatInferenceTask import StatInferenceTask


class MergedHists(StatInferenceTask, law.ExternalTask):
    """One merged histogram file, addressed by path under --hists-version.

    External on purpose. The limit-setting chain consumes histograms it does not make, so
    a missing input is reported as a missing dependency rather than pulling the whole
    AnaTuple production graph (AnaTupleFileListTask -> HistFromNtupleProducerTask ->
    HistMergerTask) into the run and rebuilding its branch maps just to discover the file
    is already there.

    The path mirrors HistMergerTask.output() in FLAF/Analysis/tasks.py -- that is the
    contract between the two halves, and the only thing this chain needs from FLAF.
    """

    variable = luigi.Parameter(description="Hists_merged variable (directory) name")

    # The input file is the same whichever configuration reads it.
    exclude_params_req = StatInferenceTask.exclude_params_req | {
        "datacard_config",
        "datacard_tag",
    }

    def output(self):
        return self.remote_target(
            self.input_hists_version,
            "Hists_merged",
            self.period,
            self.variable,
            f"{self.variable}.root",
            fs=self.fs_HistTuple,
        )
