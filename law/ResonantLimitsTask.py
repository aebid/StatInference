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


def cards_by_mass(cards):
    """{mass: card} from datacard paths named "..._<mass>.txt"."""
    out = {}
    for c in cards:
        m = re.search(r"_(\d+)\.txt$", c)
        if m:
            out[m.group(1)] = c
    return out


def combine_cards_per_mass(labelled_cards, staging, env):
    """Write staging/combined_<mass>.txt for every mass, combining one card per label.

    `labelled_cards` is [(label, [card, ...])]; the label prefixes the bins combineCards
    writes (era names across eras, member names across a combination), so labels must be
    unique. A mass a label has no card for is combined from the others.
    """
    by_label = [(label, cards_by_mass(cards)) for label, cards in labelled_cards]
    masses = sorted({m for _, cards in by_label for m in cards}, key=int)
    written = []
    for mass in masses:
        args = [f"{label}={cards[mass]}" for label, cards in by_label if mass in cards]
        out_file = os.path.join(staging, f"combined_{mass}.txt")
        with open(out_file, "w") as f:
            subprocess.run(["combineCards.py"] + args, env=env, stdout=f, check=True)
        written.append(out_file)
    return written


def publish_dir(staging, target):
    """Replace directory `target` with the files in `staging`, returning their paths.

    Replaced rather than written into, and only once everything is staged: a failure
    halfway otherwise leaves a directory law takes for a complete output.
    """
    if target.exists():
        target.remove()
    target.touch()
    published = []
    for name in sorted(os.listdir(staging)):
        dest = os.path.join(target.path, name)
        shutil.copy2(os.path.join(staging, name), dest)
        published.append(dest)
    return published


class ResonantLimitsTask(StatInferenceTask):
    # Not a workflow itself. The parameter exists so that --workflow reaches the tasks
    # below that are -- law's req() only forwards parameters both tasks declare -- and
    # NO_STR leaves each of those at its own default.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    def store_parts(self):
        return (*self.version_parts(), self.__class__.__name__, "combined")

    def get_eras(self):
        """Every era the configuration lists: each gets its own cards and its own limit."""
        return self.get_all_eras()

    def _create_datacards_req(self, era, **kwargs):
        era_groups = self.get_era_groups()
        if era in era_groups:
            return CreateDatacardsTask.req(
                self, period=era_groups[era][0], meta_era=era, **kwargs
            )
        return CreateDatacardsTask.req(self, period=era, **kwargs)

    def requires(self):
        return [self._create_datacards_req(e, branches=()) for e in self.get_eras()]

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

    def headline_cards(self):
        """The per-mass cards this configuration's limit is set on, once run() has run:
        the cross-era combined cards when several top-level eras are combined, else the
        single top-level era's own. What a combination configuration combines."""
        top_level = self.get_top_level_eras()
        if len(top_level) > 1:
            return sorted(
                glob.glob(os.path.join(self.datacards_dir("combined"), "*.txt"))
            )
        return sorted(
            glob.glob(os.path.join(self.datacards_dir(top_level[0]), "*.txt"))
        )

    def stage_datacards(self, era, remote_target):
        """Mirror an era's datacards from fs_default to a stable local path, returned.

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
        remote_target.copy_to_local(local_target)
        return local_dir

    def run(self):
        eras = self.get_eras()
        era_cards = {}

        for e in eras:
            create_dc_br0 = self._create_datacards_req(e, branch=0, branches=())
            output_dir = self.stage_datacards(e, create_dc_br0.output())
            era_cards[e] = glob.glob(os.path.join(output_dir, "*.txt"))

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
            combine_cards_per_mass(
                [(e, era_cards.get(e, [])) for e in self.get_top_level_eras()],
                staging,
                self.cmssw_env,
            )
            combined_cards = publish_dir(staging, self.output()["datacards"])

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
