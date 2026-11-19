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
    # The era the file belongs to, one of `era`: a task over several eras requires the
    # files of each, each from that era's own fs_Hists (which falls back to fs_HistTuple).
    hist_era = luigi.Parameter(description="era of the merged histograms")

    def output(self):
        return self.remote_target(
            self.input_hists_version,
            "Hists_merged",
            self.hist_era,
            self.variable,
            f"{self.variable}.root",
            fs=self.setup_for(self.hist_era).get_fs("Hists"),
        )
