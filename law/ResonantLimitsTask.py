import glob
import law
import luigi
import os
import re
import shutil
import subprocess
import tempfile
import yaml

from dhi.tasks.resonant import MergeResonantLimits

from .StatInferenceTask import CardTask
from .CreateDatacardsTask import CreateDatacardsTask

# Key for the cross-era merge in run()'s dynamic-dependency dict. Not an era name,
# and deliberately not one a configuration could declare.
COMBINED_KEY = "__combined__"


def _same_config(path_a, path_b):
    """Whether two YAML files hold the same configuration, whatever their comments."""
    with open(path_a) as a, open(path_b) as b:
        return yaml.safe_load(a) == yaml.safe_load(b)


class ResonantLimitsTask(CardTask):
    """The limits of one card (`--card`): each of its eras', and the combination of its
    top-level eras. Without `--card`, the limits of every card (see CardTask).
    """

    # Not a workflow itself. The parameter exists so that --workflow reaches the tasks
    # below that are -- law's req() only forwards parameters both tasks declare -- and
    # NO_STR leaves each of those at its own default.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    check_card_eras = True

    def store_parts(self):
        return super().store_parts() + ("combined",)

    def requires(self):
        if self.all_cards:
            return self.card_tasks()
        return CreateDatacardsTask.req(self, branches=self.card_branches())

    def card_branches(self):
        """The CreateDatacardsTask branches of this card, as a `branches` selection: only
        they are built (and, through them, only this card's preprocessing), so one card's
        limits neither wait for nor fail with another card. () -- the whole workflow --
        when this card is every branch, as for a single card."""
        workflow = CreateDatacardsTask.req(self, branches=())
        branch_map = workflow.get_branch_map()
        indices = tuple(
            index
            for index, data in branch_map.items()
            if data["card"] == self.this_card
        )
        if not indices:
            # () would select every branch
            raise ValueError(
                f"{self.get_task_family()}: card '{self.this_card}' has no mass point to "
                "build: none of its masses is among StatInference `param_values`"
            )
        return () if len(indices) == len(branch_map) else indices

    def era_card_outputs(self):
        """{era: [the outputs of the branches that write this card's masses of that
        era]}, each era's in increasing mass, so staging them is deterministic."""
        outputs = {}
        tasks = sorted(
            (
                task
                for task in self.requires().get_branch_tasks().values()
                if task.branch_data["card"] == self.this_card
            ),
            key=lambda task: float(task.branch_data["mass"]),
        )
        for task in tasks:
            outputs.setdefault(task.branch_data["era"], []).append(task.output())
        return outputs

    def output(self):
        if self.all_cards:
            return {card: task.output() for card, task in self.card_tasks().items()}
        return {
            # The limit of the combination this configuration is about: the single
            # top-level era's own, or the cross-era combined cards' when it lists
            # several. See run() on why this is not one merge over every listed era.
            "limits": self.local_target("limits.npz"),
            # One limits_<era>.npz per era in `eras:`, which is what get_all_eras()
            # promises -- each era its own datacards and its own limit.
            "era_limits": law.LocalDirectoryTarget(self.local_path("era_limits")),
            "datacards": law.LocalDirectoryTarget(self.datacards_dir("combined")),
        }

    def stage_datacards(self, era, remote_targets):
        """Mirror an era's datacards from fs_default to a stable local path, returned.

        CreateDatacardsTask writes each mass to its own directory; they are merged here
        into the one directory per era that the limits and the plots glob, laid out as when
        one task wrote every mass. The datacard and shape file names carry the mass, so two
        masses writing the same path is a bug, and it raises rather than one card silently
        replacing the other. The shape plots are named by variable, which several masses
        may share, so each mass's plots/ is mirrored to plots/<its directory>/.

        CreateDatacardsTask writes to fs_default, but the cards must be real local files
        by the time combine sees them: MergeResonantLimits shells out to combine, and
        PlotMultipleResonantLimits resolves --multi-datacards by globbing the filesystem
        outside law entirely. A localize() scratch dir will not do -- run() yields, which
        suspends and re-enters it, so the scratch dir is gone before dhi reads anything.

        The mirror path is the one CreateDatacardsTask used to write to directly, so
        existing --multi-datacards invocations keep working unchanged.
        """
        local_dir = self.datacards_dir(era)
        local_target = law.LocalDirectoryTarget(local_dir)
        # Re-staged on every run, including the re-entry after the yield below. The
        # copy is small (~30 MB) and unconditional refresh is what keeps the mirror
        # from going stale when CreateDatacardsTask reruns within the same version.
        if local_target.exists():
            local_target.remove()
        local_target.touch()
        for remote_target in remote_targets:
            # each mass mirrored on its own, as the whole card was before, then merged in
            part = law.LocalDirectoryTarget(is_tmp=True)
            part.touch()
            try:
                remote_target.copy_to_local(part)
                self._merge_mass_dir(
                    part.abspath,
                    local_dir,
                    os.path.basename(remote_target.path.rstrip("/")),
                )
            finally:
                part.remove(silent=True)
        return local_dir

    @staticmethod
    def _merge_mass_dir(src, dest, mass_dir):
        """Copy one mass's output `src` into the era's `dest`: plots/ to plots/<mass_dir>/,
        everything else to the same relative path, raising if that path already exists.

        The one exception is the merged card every mass keeps (MERGED_CARD_FILE): all
        masses of a card are built from the same one, so it is kept once. The copies are
        compared as configuration, not as bytes -- the header comment names the source
        files, so it differs between analysis areas -- and it raises if their content
        differs.
        """
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            parts = [] if rel == "." else rel.split(os.sep)
            if parts[:1] == ["plots"]:
                parts = ["plots", mass_dir] + parts[1:]
            target_dir = os.path.join(dest, *parts)
            os.makedirs(target_dir, exist_ok=True)
            for name in files:
                target = os.path.join(target_dir, name)
                if os.path.lexists(target):
                    if not parts and name == CardTask.MERGED_CARD_FILE:
                        if _same_config(os.path.join(root, name), target):
                            continue
                        raise RuntimeError(
                            f"staging {mass_dir} into {dest}: its {name} differs from the "
                            "one of the mass points staged before it -- the mass points of "
                            "this card were built from different versions of its "
                            "configuration; rebuild them under one version"
                        )
                    raise RuntimeError(
                        f"staging {mass_dir} into {dest}: {os.path.join(*parts, name)} "
                        "was already written by another mass point"
                    )
                shutil.copy2(os.path.join(root, name), target)

    def run(self):
        if self.all_cards:
            # every card's limits are its requirements; nothing is left to do here
            return
        era_cards = {}

        for e, card_output in self.era_card_outputs().items():
            output_dir = self.stage_datacards(e, card_output)
            era_cards[e] = glob.glob(os.path.join(output_dir, "*.txt"))

        masses = set()
        for e, cards in era_cards.items():
            for c in cards:
                m = re.search(r"_(\d+)\.txt$", c)
                if m:
                    masses.add(m.group(1))

        # Built into a staging directory and published only once every card is there.
        # Creating the outputs first and filling them as we go left a failed
        # combineCards halfway through looking like a complete task -- law's default
        # completeness is output existence -- so the next run skipped it and everything
        # downstream read a partial set. Replacing the directory rather than writing into
        # it also clears cards for masses no longer in the configuration.
        with tempfile.TemporaryDirectory() as staging:
            # The cross-era combination is the one place a group era and its members
            # cannot both appear: the group already *is* their combination, so a card
            # built from both would count those events twice.
            for mass in sorted(masses):
                combine_args = []
                for e in self.get_top_level_eras():
                    for c in era_cards.get(e, []):
                        if c.endswith(f"_{mass}.txt"):
                            combine_args.append(f"{e}={c}")
                            break

                if combine_args:
                    cmd = ["combineCards.py"] + combine_args
                    out_file = os.path.join(staging, f"combined_{mass}.txt")
                    with open(out_file, "w") as f:
                        subprocess.run(cmd, env=self.cmssw_env, stdout=f, check=True)

            out_dc_dir = self.output()["datacards"]
            if out_dc_dir.exists():
                out_dc_dir.remove()
            out_dc_dir.touch()
            combined_cards = []
            for name in sorted(os.listdir(staging)):
                dest = os.path.join(out_dc_dir.path, name)
                shutil.copy2(os.path.join(staging, name), dest)
                combined_cards.append(dest)

        # One merge per era, never one merge over every era's cards together. dhi groups
        # the datacards it is given by the mass in their file name alone and runs
        # combineCards over each group, so a single merge handed five eras' cards fits
        # all five into one workspace per mass. With a configuration listing a group era
        # *and* its members -- x_hh_bbww_DL_run3_1D.yaml does -- that counted every
        # 2022-2023BPix event twice, and the result was one over-strong limit per mass
        # rather than the per-era limits get_all_eras() describes.
        merges = {
            e: MergeResonantLimits(version=self.version, datacards=tuple(cards))
            for e, cards in era_cards.items()
            if cards
        }
        top_level = self.get_top_level_eras()
        # The cross-era combination is only a separate fit when there is more than one
        # era to combine; with a single top-level era that era's own merge already is it.
        combining = len(top_level) > 1 and combined_cards
        if combining:
            merges[COMBINED_KEY] = MergeResonantLimits(
                version=self.version, datacards=tuple(sorted(combined_cards))
            )
        merged = yield merges

        era_dir = self.output()["era_limits"]
        if era_dir.exists():
            era_dir.remove()
        era_dir.touch()
        for e in era_cards:
            if e in merged:
                shutil.copy2(
                    merged[e].path, os.path.join(era_dir.path, f"limits_{e}.npz")
                )

        headline = merged[COMBINED_KEY] if combining else merged.get(top_level[0])
        if headline is None:
            raise RuntimeError(
                "No datacards were found for any top-level era "
                f"({', '.join(top_level)}), so there is no combined limit to write."
            )
        self.output()["limits"].parent.touch()
        shutil.copy2(headline.path, self.output()["limits"].path)
