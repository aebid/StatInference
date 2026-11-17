"""--workflow for the dhi tasks the StatInference tasks run.

The fits are dhi workflows (ResonantLimits: one branch per mass; PullsAndImpacts: one per
parameter), but they are not reached through req(), so a task's --workflow does not reach
them by itself: ResonantLimitsTask builds MergeResonantLimits directly, and the plot tasks
run dhi in a `law run` subprocess. These two functions carry it there, so that
`--workflow local` (the integration CI) fits locally and `--workflow htcondor` sends the fits
to HTCondor like the FLAF tasks. Without --workflow nothing is set and dhi keeps its own
default. A `--ResonantLimits-workflow` / `--PullsAndImpacts-workflow` given explicitly on the
command line still wins: luigi ranks the command line above the configuration.
"""

import law
import luigi

# the dhi workflows behind the limits and the impacts
DHI_WORKFLOW_TASKS = ("ResonantLimits", "PullsAndImpacts")


def _given(workflow):
    return workflow not in (None, "", law.NO_STR)


def dhi_workflow_args(workflow):
    """The `law run` arguments that give the dhi workflows `workflow` (none if not given)."""
    if not _given(workflow):
        return []
    args = []
    for task in DHI_WORKFLOW_TASKS:
        args += [f"--{task}-workflow", str(workflow)]
    return args


def set_dhi_workflow(workflow):
    """Give the dhi workflows built in this process `workflow` (nothing if not given), through
    luigi's configuration -- what `[luigi_ResonantLimits] workflow: ...` in law.cfg does.
    """
    if not _given(workflow):
        return
    config = luigi.configuration.get_config()
    for task in DHI_WORKFLOW_TASKS:
        if not config.has_section(task):
            config.add_section(task)
        config.set(task, "workflow", str(workflow))
