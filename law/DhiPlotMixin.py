import law
import luigi
import os
import shutil

from FLAF.RunKit.run_tools import ps_call

from .dhi_workflow import dhi_workflow_args


class DhiPlotMixin:
    """Running a dhi plot task from a StatInference task, and keeping what it drew.

    dhi writes its plots under inference/data/store, which is what makes an
    already-drawn plot cheap to re-request but leaves the products somewhere other than
    the rest of the chain's. Every task here follows the same three steps: work out where
    dhi will write, run it as a subprocess, copy the result into the task's own output.

    The dhi tasks are invoked as subprocesses rather than yielded as dynamic dependencies.
    luigi round-trips a dynamic dependency through to_str_params() / from_str_params(),
    and dhi declares parameters whose unset defaults do not survive that -- lumi_scale is
    a FloatParameter(default=None) that serialises to "None" and then raises on
    float("None"). Going through the command line keeps dhi unmodified.
    """

    # Forwarded as "--remove-output 0,a,y": depth 0 drops the plot itself and nothing
    # below it, so no fit is recomputed. Needed because much of the plot styling is
    # significant=False and so does not move the output path -- changing it would
    # otherwise leave the old plot in place and look like the change had no effect.
    redraw = luigi.BoolParameter(
        default=False,
        significant=False,
        description="discard existing plots and draw them again",
    )

    @staticmethod
    def known_plot_params(task_cls):
        """The parameter names a dhi plot task accepts.

        get_params() rather than get_param_names(), which drops the insignificant ones --
        that is most of the plot styling (x_min, campaign, parameters_per_page, ...).
        """
        return {param_name for param_name, _ in task_cls.get_params()}

    def validate_plot_params(self, task_cls, params, context):
        """Reject a plot_params key the dhi task would silently ignore."""
        known = self.known_plot_params(task_cls)
        for key in params:
            if key not in known:
                raise RuntimeError(
                    f"{context}: '{key}' is not a {task_cls.__name__} parameter."
                )

    def plot_targets(self, task_cls, spec):
        """Where the dhi task will write, without scheduling it.

        Constructing the task is safe and is the only honest way to learn its output
        paths -- they are built from a hash of the datacards plus several parameters,
        which is not something to reimplement here. flatten() because a dhi task's
        output() may be a single target, a list, or a dict of both (PlotPullsAndImpacts
        returns {"plots": [...], "plot_data": ...}).
        """
        return law.util.flatten(task_cls(**spec).output())

    def plot_command(self, task_cls, spec, extra_args=()):
        """`law run <dhi plot task> ...` for a spec.

        The task's --workflow is passed on to the dhi fits upstream of the plot
        (--ResonantLimits-workflow, --PullsAndImpacts-workflow; see dhi_workflow): the
        subprocess does not see this process's command line or configuration.

        ``extra_args`` is appended verbatim, after those, so it can override them. It
        exists for the arguments that belong to a task upstream of the one being run rather
        than to it -- law namespaces those as --<TaskName>-<param>, so they are not
        parameters of task_cls and cannot go in the spec.
        """
        cmd = ["law", "run", task_cls.__name__]
        for key, value in spec.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                # Passed as "--flag True", never as a bare "--flag". law sets
                # luigi.BoolParameter.parsing to EXPLICIT_PARSING globally on import
                # (law/parameter.py), which drops argparse's store_true and leaves
                # nargs="?" -- so a bare flag parses to None, luigi falls back to the
                # parameter's default, and the setting is silently lost. "y_log: true"
                # in a plot_params block drew a linear axis for exactly that reason,
                # while plot_targets built the task with y_log=True, so the path this
                # looked the plot up under was not the path it was drawn to.
                cmd += [flag, str(value)]
            elif key == "multi_datacards":
                # colon between datacard sequences, comma within one
                cmd += [flag, ":".join(",".join(seq) for seq in value)]
            elif isinstance(value, (tuple, list)) and all(
                isinstance(v, (tuple, list)) and len(v) == 2 for v in value
            ):
                # dhi's MultiCSVParameter pairs, e.g. parameter_values ("r", 6.8) -> r=6.8
                cmd += [flag, ",".join(f"{a}={b}" for a, b in value)]
            elif isinstance(value, (tuple, list)):
                cmd += [flag, ",".join(str(v) for v in value)]
            else:
                cmd += [flag, str(value)]
        cmd += dhi_workflow_args(getattr(self, "workflow", law.NO_STR))
        cmd += [str(a) for a in extra_args]
        if self.redraw:
            cmd += ["--remove-output", "0,a,y"]
        return cmd

    def draw_plot(self, task_cls, spec, name, era, dest_dir, extra_args=()):
        """Run a dhi plot task, check it actually drew, copy the result into ``dest_dir``.

        Returns the basenames copied, for the manifest.
        """
        # cwd is pinned to the analysis root: dhi's resolve_datacards() takes a
        # different branch when the process happens to sit inside a configured
        # datacards_run2 directory.
        ps_call(
            self.plot_command(task_cls, spec, extra_args),
            cwd=self.ana_path(),
            verbose=1,
        )

        basenames = []
        for target in self.plot_targets(task_cls, spec):
            # luigi's retcode defaults return 0 even when a task fails, and no
            # [retcode] section overrides them here -- so a clean exit is not
            # evidence that anything was drawn. The file is.
            if not os.path.exists(target.path):
                raise RuntimeError(
                    f"'{name}' ({era}): law exited cleanly but {target.path} was not "
                    "written. See the output above for the task that actually failed."
                )
            shutil.copy2(target.path, dest_dir)
            basenames.append(os.path.basename(target.path))
        return basenames
