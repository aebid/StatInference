import contextlib
import law
import luigi
import os
import re
import shutil
import yaml

from string import Template
from FLAF.run_tools.law_customizations import Task


class StatInferenceTask(Task):
    """Shared datacard-configuration access for the limit-setting chain.

    Everything downstream of the merged histograms -- rebinning, datacards, limits, limit
    plots -- is driven by the datacard configuration named in global.yaml's
    ``StatInference.config``, not by global.yaml's own variable lists. This base class is
    the single place that file is read, so the four tasks below cannot disagree about
    which eras, masses or variables the analysis consists of.
    """

    # The Hists_merged tree to read. Separate from `version`, which names what this chain
    # *writes*: re-binning or re-fitting under a new version must not require the input
    # histograms to be reproduced (or copied) under that name. Not significant -- it
    # identifies an input, not a product, and law's store paths are about products.
    hists_version = luigi.Parameter(
        default="",
        significant=False,
        description="version of the Hists_merged tree to read; defaults to --version",
    )

    # The datacard configuration to run, when it is not global.yaml's StatInference.config.
    # A combination configuration sets it on each of its members, which is how two
    # configurations run side by side in one invocation. Significant: two configurations
    # are two different sets of products.
    datacard_config = luigi.Parameter(
        default="",
        description="datacard configuration to use instead of StatInference.config",
    )
    # Names this configuration's products under --version (<version>/<tag>/...), so the
    # members of a combination do not overwrite each other. Empty keeps the products
    # directly under --version, which is what every single-configuration run has always
    # written -- and so what its completed outputs are found under.
    datacard_tag = luigi.Parameter(
        default="",
        description="sub-directory of --version holding this configuration's products",
    )

    @property
    def input_hists_version(self):
        return self.hists_version or self.version

    def version_parts(self):
        """The leading path components of every product: --version, then the tag.

        A combination configuration's own products (its combined cards, limits and plots)
        default to the tag "combined", beside its members' own sub-directories.
        """
        tag = self.datacard_tag or ("combined" if self.is_combination() else "")
        return (self.version, tag) if tag else (self.version,)

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

    def datacard_config_path(self):
        path = self.datacard_config or self.global_params["StatInference"]["config"]
        return path if os.path.isabs(path) else os.path.join(self.ana_path(), path)

    def get_config_data(self):
        # Cached per instance: requires()/workflow_requires() are re-entered many times
        # during graph construction and each call would otherwise re-parse the yaml.
        if getattr(self, "_config_data", None) is None:
            with open(self.datacard_config_path(), "r") as f:
                self._config_data = yaml.safe_load(f)
        return self._config_data

    def is_combination(self):
        """Whether the configuration combines others (`members:`) rather than building
        datacards itself."""
        return bool(self.get_config_data().get("members"))

    def members(self):
        """{name: spec} of a combination configuration, validated.

        Each spec names the member's datacard configuration and, because the members are
        usually built from different histogram productions, the Hists_merged version and
        the user_custom overlay (which is where fs_HistTuple is set) to read them with.
        """
        members = self.get_config_data().get("members") or {}
        known = {"config", "hists_version", "user_custom", "customisations"}
        for name, spec in members.items():
            if not isinstance(spec, dict) or "config" not in spec:
                raise RuntimeError(
                    f"{self.datacard_config_path()}: member '{name}' must be a mapping "
                    "with at least 'config'"
                )
            unknown = sorted(set(spec) - known)
            if unknown:
                raise RuntimeError(
                    f"{self.datacard_config_path()}: member '{name}' has unknown keys "
                    f"{unknown}; known are {sorted(known)}"
                )
            if not re.fullmatch(r"[A-Za-z0-9_]+", str(name)) or name == "combined":
                raise RuntimeError(
                    f"{self.datacard_config_path()}: member name '{name}' must be "
                    "alphanumeric and not 'combined' -- it names a directory and a "
                    "combineCards label"
                )
        return members

    def member_req(self, name, cls, **kwargs):
        """`cls` for member `name`: its configuration, under its own tag, reading its own
        histograms. Everything else (version, period, workflow, ...) is inherited.

        FLAF builds a Setup per (user_custom, customisations), so a member with its own
        user_custom reads from its own fs_HistTuple within the same invocation.
        """
        spec = self.members()[name]
        params = dict(
            datacard_config=spec["config"],
            datacard_tag=name,
            hists_version=spec.get("hists_version", self.hists_version),
            user_custom=spec.get("user_custom", self.user_custom),
            customisations=spec.get("customisations", self.customisations),
        )
        params.update(kwargs)
        return cls.req(self, **params)

    @property
    def datacard_era(self):
        """The era this task's products belong to: its meta-era if it has one, else
        `period`. Equal to `period` for every task that does not declare `meta_era`.
        """
        return getattr(self, "meta_era", "") or self.period

    def store_parts(self):
        """Keyed on datacard_era, not period.

        A meta-era instance is constructed with `period` set to one of its own sub-eras
        -- FLAF's Setup only knows real periods -- so keying on `period` gave
        CreateDatacardsTask(period=Run3_2022) and
        CreateDatacardsTask(period=Run3_2022, meta_era=Run3_Early) the same store path.
        The declared outputs never collided (they key on datacard_era already), but
        law's HTCondor control files live here: htcondor_output_directory() is
        local_path(), and the submission file is named only after the branches, which
        for both is {0: None}. A run needing both -- any configuration listing a group
        era and its members -- had the second workflow resume from the first's
        submission data, poll jobs that were not its own, and finish incomplete.

        Identical to the old value for every task without a meta_era.
        """
        return (*self.version_parts(), self.__class__.__name__, self.datacard_era)

    def datacards_dir(self, era):
        """Local directory holding an era's datacards.

        The cards are produced on fs_default but must be real local files by the time
        combine sees them; ResonantLimitsTask mirrors them here and everything downstream
        (dhi's --multi-datacards globbing, the overlay plots) resolves against this path.
        """
        return os.path.join(
            self.ana_data_path(), *self.version_parts(), "Datacards", era
        )

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
        return self.get_config_data().get("eras", [self.period])

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

    def get_required_variables(self):
        """The Hists_merged variables this chain reads: one per mass point (the 2D
        shape the datacards are built from), derived from the datacard config's
        input_file_pattern.

        This is the *complete* input list -- there is no intersection with global.yaml's
        histTuple_flavor variable list anywhere downstream, so a mass point present here
        is read whether or not the active flavour happens to mention it.
        """
        data = self.get_config_data()
        model = data["model"]
        pattern = model["input_file_pattern"]
        param_name = model["parameters"][0]

        variables = set()
        for mass in self.get_masses():
            rel = Template(pattern).safe_substitute({"ERA": "ERA", param_name: mass})
            # "<era>/<variable>/<variable>.root": the variable is the directory holding
            # the file, not the file itself.
            variables.add(os.path.basename(os.path.dirname(rel)))
        return sorted(variables)

    def merged_hist_reqs(self, eras):
        """{(era, variable): MergedHists} -- every merged input file for those eras.

        Read by CreateDatacardsTask over the sub-periods of the era it builds cards for.
        """
        # Deferred: MergedHists subclasses this class, so importing it at module level
        # would be circular.
        from .MergedHists import MergedHists

        return {
            (era, variable): MergedHists.req(self, period=era, variable=variable)
            for era in eras
            for variable in self.get_required_variables()
        }

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
