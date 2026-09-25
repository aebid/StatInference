import glob
import json
import law
import luigi
import math
import os
import re

from string import Template
from FLAF.RunKit.run_tools import ps_call
from dhi.config import campaign_labels, campaign_lumis
from dhi.tasks.resonant import (
    MergeResonantLimits,
    PlotMultipleResonantLimits,
    PlotResonantLimits,
)

from .DhiPlotMixin import DhiPlotMixin
from .StatInferenceTask import StatInferenceTask
from .ResonantLimitsTask import ResonantLimitsTask
from .CombinedResonantLimitsTask import CombinedResonantLimitsTask


class PlotResonantLimitsTask(DhiPlotMixin, StatInferenceTask):
    """The limit plots, declared in the datacard configuration.

    Each entry of the config's ``limit_plots`` block becomes one dhi
    PlotMultipleResonantLimits task: a list of datacard globs with the labels they should
    carry, optional external limit curves, and the plot styling. Globs are relative to the
    era's datacards directory and may use ``${ERA}``; the per-channel and per-category
    sub-directories they address come from DatacardMaker.writeDatacards.

    An entry with ``bands: true`` additionally gets one dhi PlotResonantLimits plot per
    curve -- the standard single-curve plot with the +-1/+-2 sigma bands, which the overlay
    cannot show. External limits and luminosity projections are separate curves rather than
    datacards, so they appear only on the overlay.

    The finished plots are copied to fs_default alongside the rest of the chain's
    products; see DhiPlotMixin for how a dhi plot task is run and its output collected.
    """

    # Not a workflow itself. The parameter exists so that --workflow reaches the tasks
    # below that are -- law's req() only forwards parameters both tasks declare -- and
    # NO_STR leaves each of those at its own default.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    def requires(self):
        # ResonantLimitsTask is what mirrors the cards to datacards_dir(), which is what
        # the globs below resolve against -- the plots cannot be built before it has run.
        if not self.is_combination():
            return [ResonantLimitsTask.req(self)]
        # A combination draws from its members' cards, so it needs their chains -- and it
        # also draws each member's own limit_plots, under <version>/<member>/LimitPlots,
        # so one invocation yields the per-member plots and the combined ones together.
        return [CombinedResonantLimitsTask.req(self)] + [
            self.member_req(name, PlotResonantLimitsTask) for name in self.members()
        ]

    def output(self):
        # fs_default, like CreateDatacardsTask. Holds one
        # <era>/ sub-directory of plots plus plots.json naming them.
        return self.output_dir_target(*self.version_parts(), "LimitPlots")

    def _resolve_config_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.ana_path(), path)

    def common_plot_params(self, entry, era):
        """The dhi parameters an entry's plots share, whatever kind they are.

        Validated against PlotMultipleResonantLimits, which is the subclass and so carries
        every parameter the single-curve PlotResonantLimits has plus its own -- one check
        covers both plot kinds, and the band builder drops what does not apply.
        """
        name = entry.get("name") or "limits"

        params = {
            # Distinguishes the plot files of entries that differ only in labelling;
            # not significant, so it does not move the output directory.
            "plot_postfix": name,
            # Per era, not per entry: the same entry is built for every top-level era and
            # each one carries a different luminosity. plot_params may still override it.
            "campaign": self.get_campaign(era),
        }
        plot_params = entry.get("plot_params") or {}
        self.validate_plot_params(
            PlotMultipleResonantLimits, plot_params, f"limit_plots entry '{name}'"
        )
        params.update(plot_params)

        # dhi turns an unset campaign into None and simply draws no luminosity label, so
        # a missing entry would silently produce an unlabelled plot. Refuse instead.
        if not params.get("campaign"):
            raise RuntimeError(
                f"limit_plots entry '{name}': no campaign for era '{era}'. Add it to the "
                f"'campaigns' map in {self.datacard_config_path()} (a key of "
                "dhi.config.campaign_labels), or set it in this entry's plot_params."
            )
        if params["campaign"] not in campaign_labels:
            # dhi falls back to drawing the key itself, which is a usable escape hatch for
            # a one-off label but is also exactly what a typo looks like.
            print(
                f"WARNING: limit_plots entry '{name}': campaign '{params['campaign']}' is "
                "not in dhi.config.campaign_labels; it will be drawn verbatim as the "
                "luminosity label."
            )

        return params

    def build_plot_spec(self, entry, era, extra_external=()):
        """PlotMultipleResonantLimits parameters for a ``limit_plots`` entry and era.

        Kept as a plain dict so the same values can both construct the task (to learn
        where it will write) and be rendered into a command line (to make it write).

        ``extra_external`` holds external-limit files generated for this run (the
        luminosity projections). They come first, so a projection of our own curve is
        drawn before any fixed reference such as the Run 2 result.
        """
        sequences, names = self.datacard_sequences(entry, era)
        external = tuple(extra_external) + tuple(
            self._resolve_config_path(p) for p in entry.get("external_limits") or []
        )

        return dict(
            version=self.version,
            multi_datacards=tuple(sequences),
            datacard_names=tuple(names),
            external_limits=external,
            **self.common_plot_params(entry, era),
        )

    def build_band_specs(self, entry, era):
        """PlotResonantLimits parameters, one per curve, for an entry with ``bands: true``.

        The band plot is drawn from a single datacard set, so the multi-curve parameters
        (multi_datacards, datacard_names, external_limits, colors, markers) have no meaning
        here and are dropped by filtering against the parameters PlotResonantLimits
        actually declares. Everything else -- axes, xsec, campaign -- is shared with the
        overlay by construction, so the two plots of an entry cannot drift apart.
        """
        if not entry.get("bands"):
            return []

        sequences, names = self.datacard_sequences(entry, era)
        shared = self.common_plot_params(entry, era)
        known = {param_name for param_name, _ in PlotResonantLimits.get_params()}

        specs = []
        for sequence, label in zip(sequences, names):
            spec = {key: value for key, value in shared.items() if key in known}
            spec.update(
                version=self.version,
                datacards=tuple(sequence),
                # Entry name and curve label both. All three entries of the HH->bbWW
                # config draw "*.txt", so dhi would hash them into one output directory;
                # the postfix is what keeps their band plots from overwriting each other.
                # dhi's join_postfix strips whatever is not [a-zA-Z0-9._+-], so a label
                # like "e#mu" needs no sanitising here.
                plot_postfix=f"{shared['plot_postfix']}_{label}",
            )
            specs.append(spec)
        return specs

    def datacard_sequences(self, entry, era):
        """(datacard glob per curve, legend label per curve) for an entry, validated.

        Shared by the plot itself and by the luminosity projection, which has to scale the
        same cards the leading curve is drawn from.
        """
        name = entry.get("name") or "limits"

        sequences, names = [], []
        for card in entry["datacards"]:
            # A curve may draw on several globs: dhi groups every card they match by
            # the mass in its name and combines each group, so globs over two members'
            # cards give their combined limit.
            globs = card["glob"] if isinstance(card["glob"], list) else [card["glob"]]
            patterns = []
            for g in globs:
                base_dir, rel = self.glob_base(g, era, name)
                pattern = os.path.join(base_dir, Template(rel).safe_substitute(ERA=era))
                if not glob.glob(pattern):
                    raise RuntimeError(
                        f"limit_plots entry '{name}': no datacard matches '{pattern}'. "
                        f"Check the glob against the cards under {base_dir}."
                    )
                patterns.append(pattern)
            label = card["name"]
            if "{" in label or "}" in label:
                # law brace-expands datacard_names, so "fb^{-1}" would arrive as "fb^-1".
                raise RuntimeError(
                    f"limit_plots entry '{name}': datacard name '{label}' contains braces, "
                    "which law brace-expands. Put the information in the campaign label instead."
                )
            sequences.append(tuple(patterns))
            names.append(label)
        return sequences, names

    def glob_base(self, card_glob, era, entry_name):
        """(directory, relative glob) a limit_plots glob resolves against.

        In a single configuration that is its own datacards directory for the era. In a
        combination every glob names the member it reads, "SL:${ERA}/e/*.txt", and
        resolves against that member's directory; a bare glob there would have no
        datacards to match, so it is refused rather than left to match nothing.
        """
        if not self.is_combination():
            return self.datacards_dir(era), card_glob
        member, sep, rel = card_glob.partition(":")
        if not sep or member not in self.members():
            raise RuntimeError(
                f"limit_plots entry '{entry_name}': glob '{card_glob}' must name a member "
                f"as '<member>:<glob>'; members are {sorted(self.members())}"
            )
        return self.member_req(member, ResonantLimitsTask).datacards_dir(era), rel

    def limits_npz(self, sequence):
        """The MergeResonantLimits .npz for a datacard sequence, produced if absent.

        The path is taken from dhi's own task rather than assembled here: the store
        directory is a hash of the resolved datacard set, which is not something to
        reimplement. Passing the glob is equivalent to passing the resolved paths --
        dhi resolves it in modify_param_values before hashing.
        """
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
                f"MergeResonantLimits produced no limits for {list(sequence)}; expected "
                f"{target.path}."
            )
        return target.path

    def make_lumi_projection(self, entry, era, projection, sequence, out_dir):
        """Write a dhi external-limits JSON scaling this entry's own limit curve to a
        different integrated luminosity, and return its path.

        Regenerated from the current limits on every run rather than read from a
        checked-in file: the projection is a function of the measured curve, so a stored
        copy silently goes stale the moment the binning, the datacards or the fits change.
        """
        name = entry.get("name") or "limits"
        label = projection["name"]
        campaign = self.get_campaign(era)
        # Defaults to the luminosity dhi prints on the plot for this campaign, so the
        # curve cannot be scaled from a different number than the one drawn beside it.
        lumi_now = projection.get("lumi_now", campaign_lumis.get(campaign))
        if not lumi_now:
            raise RuntimeError(
                f"limit_plots entry '{name}': lumi_projection '{label}' has no lumi_now "
                f"and campaign '{campaign}' is not in dhi.config.campaign_lumis."
            )

        out_path = os.path.join(
            out_dir, re.sub(r"[^\w.-]+", "_", f"{name}_{label}").strip("_") + ".json"
        )

        import numpy as np

        # The same purely statistical scaling dhi's own --lumi-scale applies
        # (plots/limits.py): limit(L_target) = limit(L_now) * sqrt(L_now / L_target).
        # It assumes the systematics scale away with the data, which they do not, so
        # this curve is optimistic -- an extrapolation of the statistical reach, not a
        # projected result. --lumi-scale itself is not usable here: it exists only on
        # plot_limit_scan, and it overwrites the measured curve rather than adding a
        # second one, whereas the point is to draw both side by side.
        #
        # Only "factor" is written, not pre-scaled limits: dhi's read_external_limits
        # (tasks/resonant.py) multiplies by factor * scale itself, so applying it here
        # too would double-count it.
        data = np.load(self.limits_npz(sequence), allow_pickle=True)["data"]
        entry_json = [
            {
                "name": label,
                "scan_parameter": "mhh",
                "factor": math.sqrt(lumi_now / projection["lumi_target"]),
                "limits": {repr(float(r["mhh"])): float(r["limit"]) for r in data},
            }
        ]
        with open(out_path, "w") as f:
            json.dump(entry_json, f, indent=2)
            f.write("\n")
        return out_path

    def run(self):
        entries = self.get_config_data().get("limit_plots", [])
        if not entries:
            raise RuntimeError(
                f"{self.datacard_config_path()} declares no 'limit_plots' block, so there "
                "is nothing to plot."
            )

        # Every entry of an era writes into that era's one directory, and the plot file
        # is named after the entry, so two entries sharing a name silently overwrite each
        # other's plots and only the last one survives. Said here rather than left to be
        # noticed in a missing plot.
        names = [entry.get("name") or "limits" for entry in entries]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise RuntimeError(
                f"{self.datacard_config_path()}: limit_plots entries must have distinct "
                f"names, since the plot file is named after the entry; repeated: "
                f"{', '.join(duplicates)}. An entry with no 'name' is called 'limits'."
            )

        with self.output().localize("w") as local_output:
            produced = []
            for era in self.get_eras_to_plot():
                dest_dir = os.path.join(local_output.abspath, era)
                os.makedirs(dest_dir, exist_ok=True)

                for entry in entries:
                    name = entry.get("name") or "limits"

                    # Projections are written into the output directory before the plot
                    # runs: they are --external-limits inputs, and they are worth keeping
                    # next to the plot as the record of what was actually drawn.
                    sequence = self.datacard_sequences(entry, era)[0][0]
                    projections = [
                        self.make_lumi_projection(entry, era, proj, sequence, dest_dir)
                        for proj in entry.get("lumi_projections") or []
                    ]

                    specs = [
                        (
                            PlotMultipleResonantLimits,
                            self.build_plot_spec(
                                entry, era, extra_external=projections
                            ),
                        )
                    ]
                    specs += [
                        (PlotResonantLimits, s)
                        for s in self.build_band_specs(entry, era)
                    ]

                    for task_cls, spec in specs:
                        produced.extend(
                            os.path.join(era, b)
                            for b in self.draw_plot(task_cls, spec, name, era, dest_dir)
                        )
                    produced.extend(
                        os.path.join(era, os.path.basename(p)) for p in projections
                    )

            with open(os.path.join(local_output.abspath, "plots.json"), "w") as f:
                json.dump(sorted(produced), f, indent=2)

    def get_eras_to_plot(self):
        """Every era the configuration lists -- each has its own limit, so each gets its
        own plots."""
        return self.get_all_eras()
