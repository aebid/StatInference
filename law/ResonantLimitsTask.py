import glob
import law
import luigi
import os
import re
import shutil
import subprocess
import tempfile

from dhi.tasks.resonant import MergeResonantLimits

from .StatInferenceTask import StatInferenceTask
from .CreateDatacardsTask import CreateDatacardsTask

# Key for the cross-era merge in run()'s dynamic-dependency dict. Not an era name,
# and deliberately not one a configuration could declare.
COMBINED_KEY = "__combined__"


class ResonantLimitsTask(StatInferenceTask):
    # Not a workflow itself. The parameter exists so that --workflow reaches the tasks
    # below that are -- law's req() only forwards parameters both tasks declare -- and
    # NO_STR leaves each of those at its own default.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    check_card_eras = True

    def store_parts(self):
        return (self.version, self.__class__.__name__, "combined")

    def requires(self):
        return CreateDatacardsTask.req(self, branches=())

    def era_card_outputs(self):
        """{card era: [the outputs of the branches that write that card's masses]}."""
        outputs = {}
        for task in self.requires().get_branch_tasks().values():
            outputs.setdefault(task.branch_data["era"], []).append(task.output())
        return outputs

    def output(self):
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
        one task wrote every mass (the file names carry the mass, so nothing collides).

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
                shutil.copytree(part.abspath, local_dir, dirs_exist_ok=True)
            finally:
                part.remove(silent=True)
        return local_dir

    def run(self):
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
