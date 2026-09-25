import contextlib
import law
import luigi
import os

from string import Template

from FLAF.RunKit.run_tools import ps_call
from FLAF.run_tools.law_customizations import HTCondorWorkflow, copy_param

from StatInference.common.tools import CategoryNaming, importROOT

from .PreprocessShapesTask import PreprocessShapesTask
from .StatInferenceTask import StatInferenceTask


class CreateDatacardsTask(StatInferenceTask, HTCondorWorkflow, law.LocalWorkflow):
    """The datacards for one era, from the shapes the chain was pointed at.

    Runs dc_make/create_datacards.py on whatever `<era>/<variable>/<variable>.root` tree
    it is given -- the configuration's `preprocess:` step if it declares one, otherwise
    the merged histograms directly -- and does no binning of its own. For a meta-era
    (`meta_era` set, `period` one of its real sub-eras) it holds every sub-era's
    histograms at once, which is what the summed shape behind each datacard bin is built
    from, and why the stacked shape plots are drawn here.
    """

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    # If set, build datacards for this meta-era instead of self.period. self.period must
    # then be one of its real sub-eras -- it is only used to construct a valid FLAF Setup
    # (Setup.getGlobal requires a real, known period; a meta-era name isn't one).
    meta_era = luigi.Parameter(default="")

    # Stacked plots of the shapes going into the cards. This task is where they belong:
    # it is the only one holding every sub-era's histograms at once, which is what the
    # merged shape (and hence each datacard bin) is built from.
    make_plots = luigi.BoolParameter(default=True)

    def get_sub_periods(self):
        """Real periods whose histograms feed into datacard_era: its constituent
        sub-eras for a meta-era, otherwise just [self.period]."""
        return self.get_era_groups().get(self.datacard_era, [self.period])

    def input_hist_reqs(self):
        """{key: task} for the shapes the cards are built from.

        The configuration's `preprocess:` step if it declares one, otherwise the merged
        histograms directly. Either way what arrives is "<era>/<variable>/<variable>.root",
        so nothing below here knows which it got.
        """
        if self.preprocess_config():
            return {
                "PreprocessShapes": PreprocessShapesTask.req(
                    self, meta_era=self.meta_era, branches=()
                )
            }
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
        # fs_default. Note that combine cannot read these directly:
        # ResonantLimitsTask mirrors them back to datacards_dir() before handing them to
        # dhi -- see ResonantLimitsTask.stage_datacards.
        return self.output_dir_target(self.version, "Datacards", self.datacard_era)

    def run(self):
        statInf_entry = self.global_params["StatInference"]
        config = self.datacard_config_path()
        # ${ERA} in hist_bins names the era the cards are for, the same way the model's
        # input_file_pattern does. Each era has its own binning -- derived from its own
        # statistics, or its members' summed -- so one configuration serves them all.
        hist_bins_rel = statInf_entry.get("hist_bins")
        hist_bins = (
            os.path.join(
                self.ana_path(),
                Template(hist_bins_rel).safe_substitute(ERA=self.datacard_era),
            )
            if hist_bins_rel
            else None
        )
        param_values = statInf_entry.get("param_values", [])
        create_datacards_py = os.path.join(
            self.ana_path(), "StatInference", "dc_make", "create_datacards.py"
        )
        with contextlib.ExitStack() as stack:
            if self.preprocess_config():
                # PreprocessShapesTask already wrote the "<era>/<variable>/<variable>.root"
                # layout, so its output directory is the base input_file_pattern resolves
                # against -- nothing to stage.
                reqs = self.input_hist_reqs()["PreprocessShapes"]
                base_dir_local = stack.enter_context(reqs.output().localize("r"))
            else:
                targets = {
                    key: req.output()
                    for key, req in self.merged_hist_reqs(
                        self.get_sub_periods()
                    ).items()
                }
                self.check_inputs(targets)
                base_dir_local = stack.enter_context(self.stage_inputs(targets))
            local_output = stack.enter_context(self.output().localize("w"))
            cmd = [
                "python3",
                create_datacards_py,
                "--input",
                base_dir_local.abspath,
                "--output",
                local_output.abspath,
                "--config",
                config,
                "--eras",
                self.datacard_era,
            ]
            if hist_bins:
                cmd += ["--hist-bins", hist_bins]
            if len(param_values) > 0:
                param_values_str = ",".join(str(v) for v in param_values)
                cmd += ["--param_values", param_values_str]
            ps_call(cmd, env=self.cmssw_env, verbose=1)

            if self.make_plots:
                self.plot_rebinned_shapes(
                    base_dir_local.abspath, config, local_output.abspath
                )

    @staticmethod
    def _surviving_axis(base_dir):
        """Which axis of the 2D input the rebinned shapes are binned along: 0 = x, 1 = y.

        Read off the binning record the preprocess step wrote beside the shapes. A DNN
        slice selects on x and bins y; an HME box (a slice with a "y_range") selects on y
        and bins x. With no record the shapes are the DNN-sliced kind, whose surviving
        axis is y -- which is also what this assumed before boxes existed.
        """
        import json

        path = os.path.join(base_dir, "binning.json")
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, ValueError):
            return 1
        node = record.get("binning", {})
        # era -> mass -> channel -> category -> {"slices": [...]}
        while isinstance(node, dict) and "slices" not in node:
            if not node:
                return 1
            node = next(iter(node.values()))
        slices = node.get("slices") if isinstance(node, dict) else None
        return 0 if slices and "y_range" in slices[0] else 1

    def plot_variable(self, variable, axis=1):
        """The variable whose axis the shapes are binned along, for histograms.yaml.

        For input a 2D->1D rebinning produced, that is one of the two variables of the 2D
        entry -- histograms.yaml records both as ``var_list: [x, y]``, so the axis
        metadata the plotter needs is already described and does not have to be restated
        here. `axis` picks which, from _surviving_axis(): the DNN-sliced shapes keep y
        (HME), an HME box keeps x (the DNN). Taking y unconditionally labelled every box
        panel "Deep HME Mass" under DNN-score bins. For input that was always 1D, the
        variable is its own answer.
        """
        try:
            import FLAF.Common.Setup as Setup

            hists = Setup.Setup(self.ana_path(), self.period, self.version).hists
            var_list = hists[variable].get("var_list")
            if var_list and len(var_list) > 1:
                return var_list[axis]
        except Exception as e:
            print(f"Warning: no var_list for {variable} ({e}); plotting it as itself")
        return variable

    @staticmethod
    def _stitch_grid(panels, out_path):
        """Lay rendered panels out on one page: a row per channel, a column per slice.

        The slices of a base category are one physical selection cut into pieces, and the
        fit sees them together -- a slice that looks reasonable in eMu and pathological in
        eE is obvious side by side and invisible in separate files. Only page geometry
        happens here; every panel is drawn by HistPlotter.py, so there is still one
        plotting implementation.

        `panels` is [[path or None, ...] per column] per row; a None leaves its cell blank,
        which is what keeps column N under column N when a slice was skipped.
        """
        from pypdf import PdfWriter

        sheet = CreateDatacardsTask._grid_page(panels)
        if sheet is None:
            return False
        writer = PdfWriter()
        writer.add_page(sheet)
        with open(out_path, "wb") as f:
            writer.write(f)
        return True

    @staticmethod
    def _grid_page(panels):
        """The page _stitch_grid() writes, returned rather than written. None if empty."""
        from pypdf import PageObject, PdfReader, Transformation

        pages = {
            p: PdfReader(p).pages[0] for row in panels for p in row if p is not None
        }
        if not pages:
            return None
        w = max(float(p.mediabox.width) for p in pages.values())
        h = max(float(p.mediabox.height) for p in pages.values())
        n_rows, n_cols = len(panels), max(len(r) for r in panels)

        sheet = PageObject.create_blank_page(width=w * n_cols, height=h * n_rows)
        for r, row in enumerate(panels):
            for c, path in enumerate(row):
                if path is None:
                    continue
                # PDF origin is bottom-left, so the first row goes at the top.
                sheet.merge_transformed_page(
                    pages[path],
                    Transformation().translate(c * w, (n_rows - 1 - r) * h),
                )
        return sheet

    def _write_book(self, cfg, plots_dir, per_variable):
        """Every datacard bin of every mass in one PDF, one page per mass.

        A page is the channels down and every category across -- all the distributions the
        fit sees at that mass, side by side. The columns are the union over all masses, in
        configuration order, so a category absent at one mass (boosted at low mass, say)
        leaves a blank column rather than shifting the others: column N is the same
        category on every page. Each page is bookmarked with its variable.

        `per_variable` is [(variable, {key: panel path}, present)], one per mass.
        """
        from pypdf import PdfWriter

        naming = CategoryNaming.fromConfig(cfg)
        base_order = {}
        for category in cfg["categories"]:
            region, _, cat = category.rpartition("/")
            base_order.setdefault((region, naming.base(cat)), len(base_order))

        def column_key(region_cat):
            region, cat = region_cat
            base, idx = naming.split(cat)
            return (
                base_order.get((region, base), len(base_order)),
                -1 if idx is None else idx,
                cat,
            )

        columns = sorted(
            {(region, cat) for _, _, present in per_variable for _, region, cat in present},
            key=column_key,
        )
        if not columns:
            return None

        def mass_of(variable):
            digits = "".join(ch if ch.isdigit() else " " for ch in variable).split()
            return int(digits[0]) if digits else 0

        writer = PdfWriter()
        for variable, panel_of_key, _ in sorted(
            per_variable, key=lambda v: (mass_of(v[0]), v[0])
        ):
            panels = [
                [
                    (lambda p: p if p and os.path.exists(p) else None)(
                        panel_of_key.get(f"{channel}:{cat}:{region}")
                    )
                    for region, cat in columns
                ]
                for channel in cfg["channels"]
            ]
            sheet = self._grid_page(panels)
            if sheet is None:
                continue
            writer.add_page(sheet)
            writer.add_outline_item(variable, len(writer.pages) - 1)
        if not writer.pages:
            return None
        out = os.path.join(plots_dir, "all_shapes.pdf")
        with open(out, "wb") as f:
            writer.write(f)
        print(f"Wrote {out} ({len(writer.pages)} pages)")
        return out

    @staticmethod
    def _present_keys(shape_file, cfg):
        """The (channel, region, category) triples the summed shape file actually holds.

        The names are read out of the file rather than taken from `categories:`, because
        a configuration may list either the sliced datacard bins ("SR/res2b_dnn0") or the
        base categories they are cut from ("SR/res2b") -- the datacard maker expands the
        latter from the binning record, and plotting has to agree with it. Taking the
        names from the shapes needs no such agreement: whatever the maker wrote is what
        gets plotted, and a configuration in either style works unchanged.

        A configured entry matches a directory when it *is* that directory or is the base
        it was sliced from, so listing "SR/res2b" never drags in "SR/res2b2".
        """
        ROOT = importROOT()
        naming = CategoryNaming.fromConfig(cfg)
        wanted = {
            (category.rpartition("/")[0], naming.base(category.rpartition("/")[2]))
            for category in cfg["categories"]
        }
        present = set()
        f = ROOT.TFile.Open(shape_file)
        if not f or f.IsZombie():
            return present
        try:
            for channel in cfg["channels"]:
                ch_dir = f.Get(channel)
                if not ch_dir:
                    continue
                for region_key in ch_dir.GetListOfKeys():
                    region_dir = region_key.ReadObj()
                    if not region_dir.InheritsFrom("TDirectory"):
                        continue
                    region = region_key.GetName()
                    for cat_key in region_dir.GetListOfKeys():
                        if not cat_key.ReadObj().InheritsFrom("TDirectory"):
                            continue
                        cat = cat_key.GetName()
                        if (region, naming.base(cat)) in wanted:
                            present.add((channel, region, cat))
        finally:
            f.Close()
        return present

    @staticmethod
    def _selection_labels(shape_file, present):
        """{"channel:cat:region": selection} from the category directories' own titles.

        The rebinning step titles each slice directory with the cut it stands for
        ("710.00 < HME < 940.00" for an HME box, the DNN range for a DNN slice), so the
        label travels with the shapes and nothing here has to re-derive it. Empty for a
        category that was never sliced.
        """
        ROOT = importROOT()
        labels = {}
        f = ROOT.TFile.Open(shape_file)
        if not f or f.IsZombie():
            return labels
        try:
            for channel, region, cat in present:
                d = f.Get(f"{channel}/{region}/{cat}")
                title = d.GetTitle() if d else ""
                if title and title != cat:
                    labels[f"{channel}:{cat}:{region}"] = title
        finally:
            f.Close()
        return labels

    @staticmethod
    def _stamp(pdf_path, text):
        """Write `text` onto a HistPlotter panel, as one more line of its label column.

        Drawn as a transparent overlay rather than passed to HistPlotter, which has no
        hook for per-plot free text and belongs to FLAF. The position is the line below
        the region label in HistPlotter's layout.
        """
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(pdf_path)
        page = reader.pages[0]
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        fig = plt.figure(figsize=(w / 72.0, h / 72.0))
        fig.patch.set_alpha(0.0)
        fig.text(0.164, 0.665, text, fontsize=9, ha="left", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", transparent=True)
        plt.close(fig)
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer = PdfWriter()
        writer.add_page(page)
        with open(pdf_path, "wb") as f:
            writer.write(f)

    def _stitch_grids(self, cfg, plots_dir, variable, panel_of_key, present):
        """One grid per base category: its slices across, the channels down."""
        naming = CategoryNaming.fromConfig(cfg)
        bases = {}
        for _, region, cat in present:
            base, slice_idx = naming.split(cat)
            bases.setdefault((region, base), {})[slice_idx] = cat
        for (region, base), slices in bases.items():
            # None for an unsliced category, which is a single-column grid of channels.
            order = sorted(slices, key=lambda i: (i is None, i))
            panels = [
                [
                    (lambda p: p if p and os.path.exists(p) else None)(
                        panel_of_key.get(f"{channel}:{slices[i]}:{region}")
                    )
                    for i in order
                ]
                for channel in cfg["channels"]
            ]
            out = os.path.join(
                plots_dir, f"{variable}_{region.replace('/', '_')}_{base}_grid.pdf"
            )
            if self._stitch_grid(panels, out):
                print(f"Wrote grid {out}")

    def plot_rebinned_shapes(self, base_dir, config, output_dir):
        """Stacked plots of the shapes the cards are built from, one per datacard bin.

        Runs FLAF's own HistPlotter.py -- the script HistPlotTask uses -- rather than a
        plotter of our own, so these come out in the same style as every other plot in the
        analysis and there is one implementation to maintain. It is agnostic about which
        categories exist: it plots whatever `channel:category:region` keys it is handed,
        so the sliced names need nothing on the FLAF side.

        Runs in the default env rather than cmssw_env: the plotter needs PlotKit
        (matplotlib/mplhep), same as HistPlotTask.
        """
        import glob

        cfg = self.get_config_data()
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plotter = os.path.join(self.ana_path(), "FLAF", "Analysis", "HistPlotter.py")
        per_variable = []

        # The datacard bin is the sum over the sub-eras (getCombinedShape does the same at
        # card-build time), so the sub-era files are hadd'ed into the one file the plotter
        # reads. Plotting them separately would show something the fit never sees.
        for variable in self.get_required_variables():
            inputs = [
                p
                for era in self.get_sub_periods()
                for p in glob.glob(
                    os.path.join(base_dir, era, variable, f"{variable}.root")
                )
            ]
            if not inputs:
                continue
            summed = os.path.join(plots_dir, f"_summed_{variable}.root")
            try:
                ps_call(["hadd", "-f", summed] + inputs, verbose=1)
            except Exception as e:
                print(
                    f"WARNING: could not sum the shapes for {variable}, "
                    f"datacards are unaffected: {e}"
                )
                continue

            # Only the categories this mass point actually has. The configuration lists
            # every datacard bin, but the preprocessing gates drop a category where the
            # signal is too small -- boosted at a low mass, say -- and it is then absent
            # from the shapes. HistPlotter exits non-zero on the first key it cannot find,
            # so passing the full list costs every panel and grid that would have come
            # after the gap, not just the missing one.
            present = self._present_keys(summed, cfg)
            keys, outputs = [], []
            # HistPlotter navigates channel -> region -> category, and `present` already
            # holds exactly the triples the file has, in the maker's own naming.
            for channel, region, cat in sorted(present):
                keys.append(f"{channel}:{cat}:{region}")
                outputs.append(
                    os.path.join(plots_dir, f"{variable}_{channel}_{region}_{cat}.pdf")
                )
            if not keys:
                print(f"WARNING: no shapes to plot for {variable}")
                continue
            cmd = [
                "python3",
                plotter,
                "--inFile",
                summed,
                "--all_outFiles",
                ",".join(outputs),
                "--all_keys",
                ",".join(keys),
                "--globalConfig",
                os.path.join(
                    self.ana_path(),
                    self.global_params["analysis_config_area"],
                    "global.yaml",
                ),
                # The surviving axis of the rebinned shapes, and already at its final
                # binning -- so no --rebin, which would coarsen it back to the histograms.yaml
                # grid and undo the whole point of the rebinning.
                "--var",
                self.plot_variable(variable, self._surviving_axis(base_dir)),
                # The plotted shapes are the sum over every sub-era, so the label has to
                # name the combination: HistPlotter reads config/plot/<year>.yaml for the
                # luminosity, and a sub-era's file states that sub-era's luminosity alone.
                # --period stays a real era -- it builds the Setup, which does not know
                # meta-era names.
                "--year",
                self.datacard_era,
                "--ana_path",
                self.ana_path(),
                "--period",
                self.period,
                "--LAWrunVersion",
                self.version,
                "--wantSignals",
                # The slices span ~5 decades in yield (the low-significance slice holds
                # most of the background), so a linear axis hides everything but slice 0.
                "--wantLogScale",
                "y",
            ]
            try:
                ps_call(cmd, verbose=1)
                # the selection each panel stands for -- for an HME box, its window --
                # stamped on before the grids and the book are assembled from the panels
                for key, label in self._selection_labels(summed, present).items():
                    path = dict(zip(keys, outputs)).get(key)
                    if path and os.path.exists(path):
                        self._stamp(path, label)
                self._stitch_grids(
                    cfg, plots_dir, variable, dict(zip(keys, outputs)), present
                )
                per_variable.append((variable, dict(zip(keys, outputs)), present))
            except Exception as e:
                # The datacards are the product that matters; a plotting failure should be
                # loud but must not leave the task looking broken with valid cards on disk.
                print(
                    f"WARNING: shape plotting failed for {variable}, "
                    f"datacards are unaffected: {e}"
                )
            finally:
                if os.path.exists(summed):
                    os.remove(summed)

        try:
            self._write_book(cfg, plots_dir, per_variable)
        except Exception as e:
            print(f"WARNING: could not assemble the all-masses shape book: {e}")
