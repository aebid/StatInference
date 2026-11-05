import contextlib
import law
import luigi
import os
import shutil
import yaml

from string import Template
from FLAF.run_tools.law_customizations import Task, era_csv


class StatInferenceTask(Task):
    """Shared datacard-configuration access for the limit-setting chain.

    Everything downstream of the merged histograms -- rebinning, datacards, limits, limit
    plots -- is driven by the datacard configuration named in global.yaml's
    ``StatInference.config``, not by global.yaml's own variable lists. This base class is
    the single place that file is read, so the four tasks below cannot disagree about
    which eras, masses or variables the analysis consists of.

    A task here may run over several eras (`--era A,B,...`), which must include every real
    era behind the configuration's cards. The output paths are keyed on the version and the
    card, the job state on the version and the class, never on `--era`: a card is the same
    product whichever eras the run also covers.
    """

    allow_multi_era = True
    # Check the eras of the run against the cards when the task is constructed. On for the
    # tasks a run starts from that are not workflows, so the error stops `law run` before
    # luigi schedules anything (a dependency that fails while scheduling leaves its pending
    # parent, and a worker with keep_alive, waiting for ever). The card workflows check when
    # they build their map: FLAF constructs them over single eras for their resource tokens.
    check_card_eras = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.check_card_eras:
            self.card_eras()

    # The Hists_merged tree to read. Separate from `version`, which names what this chain
    # *writes*: re-binning or re-fitting under a new version must not require the input
    # histograms to be reproduced (or copied) under that name. Not significant -- it
    # identifies an input, not a product, and law's store paths are about products.
    hists_version = luigi.Parameter(
        default="",
        significant=False,
        description="version of the Hists_merged tree to read; defaults to --version",
    )

    @property
    def input_hists_version(self):
        return self.hists_version or self.version

    def output_dir_target(self, *path):
        """remote_dir_target() that also works when fs_default is a local directory.

        FLAF's remote_dir_target() handles a str fs and a remote fs but not a
        law.LocalFileSystem: it falls through to WLCGDirectoryTarget(), which raises
        "fs must be a RemoteFileSystem instance". Its sibling remote_target() (the file
        variant) already carries the branch mirrored here. The CI configures fs_default
        as a plain local path, so every directory output in this chain has to cope --
        without this, output() raises and luigi reports the task as having "error in
        complete() method".
        """
        fs = self.fs_default
        if isinstance(fs, law.LocalFileSystem):
            return law.LocalDirectoryTarget(os.path.join(*path), fs=fs)
        return self.remote_dir_target(*path)

    def statinference_settings(self):
        """global.yaml's `StatInference` block, which every era of the task must share."""
        return self.common(
            lambda setup: setup.global_params["StatInference"], "StatInference"
        )

    def datacard_config_path(self):
        return os.path.join(self.ana_path(), self.statinference_settings()["config"])

    def get_config_data(self):
        # Cached per instance: requires()/workflow_requires() are re-entered many times
        # during graph construction and each call would otherwise re-parse the yaml.
        if getattr(self, "_config_data", None) is None:
            with open(self.datacard_config_path(), "r") as f:
                self._config_data = yaml.safe_load(f)
        return self._config_data

    def store_parts(self):
        """Keyed on the version and the class, not on `--era`.

        The card workflows hold every card of the configuration as one branch each, so their
        HTCondor job file (htcondor_output_directory() is local_path()) belongs to the
        configuration, not to the eras of the run: a later run over more eras resumes the
        same jobs. FLAF's default keys on the task's one era and refuses several.
        """
        return (self.version, self.__class__.__name__)

    def remote_log_dir_target(self):
        """Job logs of a version and class, for the same reason as store_parts()."""
        return self.output_dir_target(self.version, "logs", self.__class__.__name__)

    def datacards_dir(self, era):
        """Local directory holding an era's datacards.

        The cards are produced on fs_default but must be real local files by the time
        combine sees them; ResonantLimitsTask mirrors them here and everything downstream
        (dhi's --multi-datacards globbing, the overlay plots) resolves against this path.
        """
        return os.path.join(self.ana_data_path(), self.version, "Datacards", era)

    def preprocess_config(self):
        """The datacard configuration's `preprocess:` block, or None.

        None means the chain reads the merged histograms as they are -- an analysis that
        needs no transformation declares nothing and PreprocessShapesTask never runs.
        """
        return self.get_config_data().get("preprocess") or None

    def get_era_groups(self):
        return self.get_config_data().get("era_groups", {})

    def get_all_eras(self):
        """Every era the configuration lists, each of which gets its own datacards, its
        own limit and its own plots.

        A group era and its members all appear here: they are separate measurements of
        different datasets, not a double count. Only combining them into one card would
        be -- see get_top_level_eras().
        """
        eras = self.get_config_data().get("eras")
        if not eras:
            raise ValueError(
                f"{self.datacard_config_path()} lists no `eras:`; the cards to build are "
                "named there"
            )
        return eras

    def get_sub_eras(self, card_era):
        """The real eras whose histograms feed the card of `card_era`: the members of a
        group era, else the era itself."""
        return self.get_era_groups().get(card_era, [card_era])

    def card_eras(self):
        """get_all_eras(), once every real era behind those cards is one of this task's
        eras: a card is built from all of its sub-eras or not at all."""
        cards = self.get_all_eras()
        missing = {}
        for card in cards:
            absent = [era for era in self.get_sub_eras(card) if era not in self.era]
            if absent:
                missing[card] = absent
        if missing:
            needed = sorted({era for absent in missing.values() for era in absent})
            raise ValueError(
                f"{self.get_task_family()}(era={era_csv(self.era)}): the cards of "
                f"{self.datacard_config_path()} need "
                + "; ".join(
                    f"{card}: {', '.join(absent)}" for card, absent in missing.items()
                )
                + f", which --era does not include; add {era_csv(needed)} to --era"
            )
        return cards

    def get_top_level_eras(self):
        """The eras that may be *combined* with each other: group eras plus any standalone
        real era.

        Members of a group are excluded because the group already is their combination, so
        a card built from both would count those events twice. This is a strictly smaller
        set than get_all_eras(), and only the cross-era combination uses it.
        """
        grouped = {e for sub in self.get_era_groups().values() for e in sub}
        return [e for e in self.get_all_eras() if e not in grouped]

    def get_campaign(self, era):
        """dhi campaign key for an era, from the config's ``campaigns`` map.

        It cannot be derived from the era name. 'Run3_Early' happens to lowercase into
        dhi's 'run3_early', but 'Run3_2022' would give 'run3_2022', which is not a key of
        dhi.config.campaign_labels -- and dhi draws an unknown key verbatim rather than
        failing, so a guessed value ends up printed on the plot as the luminosity label.
        """
        return self.get_config_data().get("campaigns", {}).get(era)

    def get_masses(self):
        """Every parameter value the configuration declares, across all processes.

        Not restricted to signals: with ``param_dependent_bkg`` a background can be
        parameterised too, and its histograms live in the same per-mass input file.
        """
        masses = set()
        for proc in self.get_config_data().get("processes", []):
            if not isinstance(proc, dict):
                # A bare string entry names a background directly and carries no
                # settings -- the shape dc_make's load_config functions also accept.
                continue
            for value in proc.get("param_values") or []:
                masses.add(value)
        if not masses:
            raise RuntimeError(
                f"No process in {self.datacard_config_path()} declares param_values, so "
                "there is no mass point to build inputs for."
            )
        return sorted(masses)

    def get_variable_points(self):
        """{variable: [model point, ...]}: the mass points whose input is that
        Hists_merged variable, each as {parameter: value}, from the datacard config's
        input_file_pattern."""
        data = self.get_config_data()
        model = data["model"]
        pattern = model["input_file_pattern"]
        param_name = model["parameters"][0]

        points = {}
        for mass in self.get_masses():
            rel = Template(pattern).safe_substitute({"ERA": "ERA", param_name: mass})
            # "<era>/<variable>/<variable>.root": the variable is the directory holding
            # the file, not the file itself.
            variable = os.path.basename(os.path.dirname(rel))
            points.setdefault(variable, []).append({param_name: mass})
        return points

    def get_required_variables(self):
        """The Hists_merged variables this chain reads: one per mass point (the 2D
        shape the datacards are built from), derived from the datacard config's
        input_file_pattern.

        This is the *complete* input list -- there is no intersection with global.yaml's
        histTuple_flavor variable list anywhere downstream, so a mass point present here
        is read whether or not the active flavour happens to mention it.
        """
        return sorted(self.get_variable_points())

    def merged_hist_reqs(self, eras):
        """{(era, variable): MergedHists} -- every merged input file for those eras.

        Read by the card tasks over the sub-eras of the cards they build.
        """
        # Deferred: MergedHists subclasses this class, so importing it at module level
        # would be circular.
        from .MergedHists import MergedHists

        return {
            (era, variable): MergedHists.req(self, hist_era=era, variable=variable)
            for era in eras
            for variable in self.get_required_variables()
        }

    def sub_eras_of(self, card_eras):
        """The real eras behind the cards of `card_eras`, each once, in order."""
        return list(
            dict.fromkeys(era for card in card_eras for era in self.get_sub_eras(card))
        )

    @contextlib.contextmanager
    def stage_inputs(self, targets):
        """Assemble the required merged histograms into the "<era>/<variable>/
        <variable>.root" layout that Model.getInputFileName resolves input_file_pattern
        against, and yield the directory holding it.

        Each input is localized individually so only the files this task actually reads
        cross the network -- the merged tree holds every variable of the active
        histTuple_flavor (~106 for 'default') across every era.
        """
        with contextlib.ExitStack() as stack:
            staging = law.LocalDirectoryTarget(is_tmp=True)
            staging.touch()
            stack.callback(lambda: staging.remove(silent=True))

            for (era, variable), target in targets.items():
                dest_dir = os.path.join(staging.abspath, era, variable)
                os.makedirs(dest_dir, exist_ok=True)
                local_inp = stack.enter_context(target.localize("r"))
                shutil.copy2(
                    local_inp.abspath, os.path.join(dest_dir, f"{variable}.root")
                )

            yield staging

    def check_inputs(self, targets):
        """Report every missing input at once, naming the version they were looked for
        under. law already refuses to run with an incomplete MergedHists dependency; this
        makes the condor log self-explanatory when the task is forced anyway, instead of
        failing on whichever file ROOT happened to open first."""
        missing = sorted(
            f"{era}/{var}" for (era, var), t in targets.items() if not t.exists()
        )
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(targets)} merged histograms are missing under "
                f"hists_version='{self.input_hists_version}' "
                f"(<fs_HistTuple>/{self.input_hists_version}/Hists_merged/<era>/<var>/<var>.root): "
                + ", ".join(missing)
            )
