import glob
import json
import law
import luigi
import os
import re

from string import Template

from FLAF.RunKit.run_tools import ps_call
from dhi.tasks.pulls_impacts import MergePullsAndImpacts, PlotPullsAndImpacts
from dhi.tasks.resonant import MergeResonantLimits

from .DhiPlotMixin import DhiPlotMixin
from .StatInferenceTask import StatInferenceTask
from .ResonantLimitsTask import ResonantLimitsTask


class PlotPullsAndImpactsTask(DhiPlotMixin, StatInferenceTask):
    """The nuisance pull and impact plots, declared in the datacard configuration.

    Each entry of the config's ``impact_plots`` block becomes one dhi PlotPullsAndImpacts
    task per mass point it lists. Like the limit plots, the results are copied to
    fs_default beside the rest of the chain's products; dhi keeps its own copy under
    inference/data/store, which is what makes a redraw cheap.

    The input is the combined card ResonantLimitsTask already writes -- one per mass,
    every era in it, at ``<data>/<version>/Datacards/combined/combined_<mass>.txt``. Their
    shape paths are absolute, which is what makes them usable here at all: PullsAndImpacts
    runs combine in a temporary directory.

    This is a task of its own rather than something the limit chain drags in because it is
    far more expensive than everything before it. PullsAndImpacts is a per-parameter
    workflow: roughly two combine fits per nuisance per mass point, so a card with 76
    nuisances costs ~150 fits for a single mass.
    """

    # Not a workflow itself. The parameter exists so that --workflow reaches the tasks
    # below that are -- law's req() only forwards parameters both tasks declare -- and
    # NO_STR leaves each of those at its own default.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    def requires(self):
        # ResonantLimitsTask is what writes the combined cards this reads.
        if self.is_combination():
            # Not wired yet: the combined card lives with CombinedResonantLimitsTask, and
            # card_for()/card_family() only know a single configuration's layout. Refused
            # rather than left to run impacts on whichever cards happen to match.
            raise RuntimeError(
                f"{self.datacard_config_path()} is a combination; impacts are not yet "
                "supported for it. Run them on a member with --datacard-config."
            )
        return [ResonantLimitsTask.req(self)]

    def output(self):
        return self.output_dir_target(*self.version_parts(), "ImpactPlots")

    def entry_eras(self, entry):
        """The eras an entry is drawn for.

        An entry with a `glob:` addresses cards inside one era's directory, so it is
        drawn once per era the configuration lists. The default combined card already
        contains every era, so it is drawn once and labelled "combined".
        """
        return self.get_all_eras() if entry.get("glob") else ["combined"]

    def card_for(self, entry, era, mass):
        """The single datacard an entry's plot is built from.

        PullsAndImpacts fits one card, not a glob -- every parameter gets its own
        combine job against that one workspace -- so an entry's `glob:` and mass have to
        resolve to exactly one file.
        """
        name = entry.get("name") or "impacts"
        pattern = entry.get("glob")
        if not pattern:
            # The combined card ResonantLimitsTask writes: one per mass, every era in it.
            base = self.datacards_dir("combined")
            candidates = glob.glob(os.path.join(base, "combined_*.txt"))
        else:
            base = self.datacards_dir(era)
            candidates = glob.glob(
                os.path.join(base, Template(pattern).safe_substitute(ERA=era))
            )

        # Same convention dhi's datacard_pattern uses: the mass is the trailing number.
        matched = [
            p
            for p in candidates
            if (m := re.search(r"_(\d+)\.txt$", os.path.basename(p)))
            and int(m.group(1)) == int(mass)
        ]
        if len(matched) != 1:
            found = sorted(os.path.basename(p) for p in candidates)
            raise RuntimeError(
                f"impact_plots entry '{name}' ({era}, mass {mass}): expected exactly one "
                f"datacard, found {len(matched)}. Looked in {base} for "
                f"{pattern or 'combined_*.txt'}; it holds: {', '.join(found) or 'nothing'}."
            )
        return matched[0]

    def card_family(self, entry, era):
        """The datacard set an entry's card belongs to, as a MergeResonantLimits glob.

        The limit is merged over a family of per-mass cards, so the family is what has to
        be named to look one up -- the same glob the entry selects its own card from,
        without the mass filter.
        """
        pattern = entry.get("glob")
        if pattern:
            return (
                os.path.join(
                    self.datacards_dir(era), Template(pattern).safe_substitute(ERA=era)
                ),
            )

        # The combined card's own family, combined_*.txt, is a datacard set nobody has
        # limits for -- asking for them starts a fresh workspace and fit per mass. The
        # limits that do exist are the ones the limit plots merged, whose "Combined" curve
        # globs an era's cards; with a single top-level era that is the same measurement
        # the combined card makes.
        eras = self.get_top_level_eras()
        if len(eras) != 1:
            raise RuntimeError(
                f"impact_plots: poi_value 'limit' needs one top-level era to take the "
                f"limit from, but the configuration has {eras}. Give the entry a number "
                "instead."
            )
        return (os.path.join(self.datacards_dir(eras[0]), "*.txt"),)

    def expected_limit(self, entry, era, mass):
        """The expected limit on r at a mass, from the limits already computed for it.

        The path comes from dhi's own task rather than being assembled here: the store
        directory is a hash of the resolved datacard set. MergeResonantLimits is run if it
        has not been already -- ResonantLimitsTask has produced the per-mass limits this
        merges, so this is a merge and not a fit.
        """
        import numpy as np

        sequence = self.card_family(entry, era)
        target = MergeResonantLimits(
            version=self.version, datacards=tuple(sequence)
        ).output()
        if not os.path.exists(target.path):
            ps_call(
                [
                    "law",
                    "run",
                    "MergeResonantLimits",
                    "--version",
                    self.version,
                    "--datacards",
                    ",".join(sequence),
                ],
                cwd=self.ana_path(),
                verbose=1,
            )
        if not os.path.exists(target.path):
            raise RuntimeError(
                f"impact_plots: no merged limits at {target.path} for {list(sequence)}, "
                "so poi_value: limit cannot be resolved. Set a number instead."
            )
        data = np.load(target.path, allow_pickle=True)["data"]
        for row in data:
            if int(row["mhh"]) == int(mass):
                return float(row["limit"])
        raise RuntimeError(
            f"impact_plots: the merged limits at {target.path} carry no mass {mass}; "
            f"they hold {sorted({int(r['mhh']) for r in data})}."
        )

    # The parameter the ranking is of. dhi's own default for a resonant search.
    poi_name = "r"

    def poi_value(self, entry, era, mass):
        """The value of r the Asimov dataset is built at, or None to leave dhi's default.

        This matters more than it looks. dhi builds the Asimov at r=1 unless told
        otherwise, and r=1 is not a meaningful reference at either end of this scan: where
        the limit is ~7 it is an almost invisible signal, so the ranking is really the
        background-only one, and where the limit is ~0.16 it is several times more signal
        than could be excluded, which pins r and collapses every impact towards zero.
        Ranking at the limit puts the fit where the measurement actually is.

        Note this cannot go through dhi's `parameter_values`: for a resonant search
        hh_model is NO_STR, and POITask then hard-codes both the joined parameter values
        ('""') and the output postfix (r=1.0), so the value would be silently dropped. It
        has to reach combine as --expectSignal through PullsAndImpacts' custom_args.

        WARNING: custom_args is *not* part of any dhi output path. It is a significant
        luigi parameter, so it changes the task id, but dhi builds its file names from
        store_parts() and get_output_postfix() alone -- neither of which reads it, and its
        own parameter description says as much ("they might not be encoded into output
        file paths"). So changing poi_value or fit_args and re-running under the same
        version silently reuses the previous fits, and --redraw then redraws from those.
        Until that is fixed (by deriving the dhi version from a hash of the custom args),
        a changed fit setting needs a new --version or a hand-cleared
        inference/data/store.
        """
        value = entry.get("poi_value")
        if value is None:
            return None
        if value == "limit":
            return self.expected_limit(entry, era, mass)
        return float(value)

    def build_plot_spec(self, entry, era, mass):
        """PlotPullsAndImpacts parameters for an entry and mass point.

        Kept as a plain dict so the same values can both construct the task (to learn
        where it will write) and be rendered into a command line (to make it write).
        """
        name = entry.get("name") or "impacts"
        plot_params = dict(entry.get("plot_params") or {})
        self.validate_plot_params(
            PlotPullsAndImpacts, plot_params, f"impact_plots entry '{name}'"
        )

        # The combined cards carry a few hundred autoMCStats bins, so mc_stats puts ~460
        # parameters on the plot. parameters_per_page defaults to -1, meaning a single
        # page, and the result is an unreadable hairline strip rather than an error.
        if (
            plot_params.get("mc_stats")
            and plot_params.get("parameters_per_page", -1) < 1
        ):
            raise RuntimeError(
                f"impact_plots entry '{name}': mc_stats puts every autoMCStats bin on the "
                "plot, and parameters_per_page defaults to -1 (one page), which is "
                "unreadable. Set parameters_per_page (25 works) or drop mc_stats."
            )

        return dict(
            version=self.version,
            datacards=(self.card_for(entry, era, mass),),
            mass=float(mass),
            # PullsAndImpacts is a plain POITask, so hh_model defaults to the
            # non-resonant model_default -- which would fit r together with kl, kt, CV and
            # C2V, and build its own workspace under a hh_model__model_default/ store path
            # rather than reusing the one the resonant chain already made. dhi's own
            # resonant tasks pin hh_model to NO_STR for the same reason
            # (dhi/tasks/resonant.py:35); allow_empty_hh_model is already True on the
            # PullsAndImpacts base, so this is simply saying the analysis is resonant.
            # An entry may still override it through plot_params.
            **{"hh_model": law.NO_STR, **plot_params},
        )

    def report_dropped_parameters(self, entry, era, mass, spec):
        """Warn about nuisances robustHesse removed from the fit.

        robustHesse drops a parameter it cannot invert, logs "Dropping <name> from the
        hessian" and carries on successfully; the nuisance is then simply absent from the
        plot and the merged JSON, with nothing on the plot marking its absence. So the
        card's own nuisance list is the reference: anything in it that the merged JSON
        does not carry was dropped.
        """
        merged = MergePullsAndImpacts(
            **{
                k: v
                for k, v in spec.items()
                if k in self.known_plot_params(MergePullsAndImpacts)
            }
        ).output()
        if not os.path.exists(merged.path):
            return
        with open(merged.path) as f:
            fitted = {p["name"] for p in json.load(f).get("params", [])}

        card_params = set()
        with open(self.card_for(entry, era, mass)) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] in ("shape", "lnN"):
                    card_params.add(parts[0])

        dropped = sorted(card_params - fitted)
        if dropped:
            name = entry.get("name") or "impacts"
            print(
                f"WARNING: impact_plots entry '{name}' ({era}, mass {mass}): "
                f"{len(dropped)} "
                f"nuisance(s) are in the card but not in the fit, so they are missing "
                f"from the plot without being marked: {', '.join(dropped)}. "
                "With --method robust this is robustHesse dropping what it could not "
                "invert; the ranking is not complete."
            )

    def run(self):
        entries = self.get_config_data().get("impact_plots", [])
        if not entries:
            raise RuntimeError(
                f"{self.datacard_config_path()} declares no 'impact_plots' block, so "
                "there is nothing to plot."
            )

        with self.output().localize("w") as local_output:
            produced = []
            for entry in entries:
                name = entry.get("name") or "impacts"
                masses = entry.get("masses")
                if not masses:
                    raise RuntimeError(
                        f"impact_plots entry '{name}': no 'masses'. Each one costs about "
                        "two combine fits per nuisance, so they are listed explicitly "
                        "rather than defaulting to every mass in the model."
                    )

                for era in self.entry_eras(entry):
                    for mass in masses:
                        rel = os.path.join(name, era, str(mass))
                        dest_dir = os.path.join(local_output.abspath, rel)
                        os.makedirs(dest_dir, exist_ok=True)

                        spec = self.build_plot_spec(entry, era, mass)
                        extra = list(entry.get("dhi_args") or ())
                        # Everything that has to reach combine itself goes through
                        # PullsAndImpacts' custom_args, which is a single string, so the
                        # signal strength and any fit options are joined into one.
                        custom = []
                        poi = self.poi_value(entry, era, mass)
                        if poi is not None:
                            custom.append(f"--expectSignal={poi:.4g}")
                        custom += [str(a) for a in (entry.get("fit_args") or ())]
                        if custom:
                            extra.append(
                                "--PullsAndImpacts-custom-args=" + " ".join(custom)
                            )
                        basenames = self.draw_plot(
                            PlotPullsAndImpacts,
                            spec,
                            name,
                            f"{era} mass {mass}",
                            dest_dir,
                            extra_args=extra,
                        )
                        self.report_dropped_parameters(entry, era, mass, spec)
                        produced.extend(os.path.join(rel, b) for b in basenames)

            with open(os.path.join(local_output.abspath, "impacts.json"), "w") as f:
                json.dump(sorted(produced), f, indent=2)
