import law
import luigi
import shutil
import tempfile

from dhi.tasks.resonant import MergeResonantLimits

from .StatInferenceTask import StatInferenceTask
from .ResonantLimitsTask import (
    ResonantLimitsTask,
    cards_by_mass,
    combine_cards_per_mass,
    publish_dir,
)


class CombinedResonantLimitsTask(StatInferenceTask):
    """The limit of a combination configuration: its members' datacards, combined per mass.

    A combination configuration (one with `members:`) builds no datacards of its own. Its
    members -- the single- and double-lepton configurations, say -- differ in inputs,
    binning, processes and nuisances, so they cannot be summed histogram by histogram the
    way a group era sums its sub-eras. They are combined at the card level instead: each
    member's own chain runs under <version>/<member>/, and here each member's headline card
    for a mass (ResonantLimitsTask.headline_cards) is combined with the others', labelled by
    member name. Nuisances of the same name are correlated by that combination, which is
    the only coupling between members.
    """

    # Not a workflow itself; see ResonantLimitsTask.
    workflow = luigi.Parameter(default=law.parameter.NO_STR)

    def requires(self):
        if not self.is_combination():
            raise RuntimeError(
                f"{self.datacard_config_path()} declares no 'members', so there is "
                "nothing to combine; run ResonantLimitsTask for a single configuration."
            )
        return {
            name: self.member_req(name, ResonantLimitsTask) for name in self.members()
        }

    def store_parts(self):
        return (*self.version_parts(), self.__class__.__name__)

    def output(self):
        return {
            "limits": self.local_target("limits.npz"),
            "datacards": law.LocalDirectoryTarget(self.datacards_dir("combined")),
        }

    @staticmethod
    def check_members(member_cards):
        """The members must share their mass points, and no bin may belong to two of them
        -- a repeated bin name would be two members claiming the same events."""
        masses = {
            name: sorted(cards_by_mass(cards), key=int)
            for name, cards in member_cards.items()
        }
        reference = next(iter(masses.values()))
        if any(m != reference for m in masses.values()):
            raise RuntimeError(f"the members disagree on the mass points: {masses}")

        owner = {}
        for name, cards in member_cards.items():
            with open(cards[0]) as f:
                # the first "bin" line lists each bin once (the second, per process)
                bins = next(line.split()[1:] for line in f if line.startswith("bin "))
            for b in bins:
                if owner.setdefault(b, name) != name:
                    raise RuntimeError(
                        f"bin '{b}' is in both '{owner[b]}' and '{name}'; the members' "
                        "channels and categories must not overlap"
                    )

    def run(self):
        member_cards = {}
        for name, req in self.requires().items():
            cards = req.headline_cards()
            if not cards:
                raise RuntimeError(f"member '{name}' has no datacards to combine")
            member_cards[name] = cards
        self.check_members(member_cards)

        with tempfile.TemporaryDirectory() as staging:
            combine_cards_per_mass(list(member_cards.items()), staging, self.cmssw_env)
            combined = publish_dir(staging, self.output()["datacards"])

        merged = yield MergeResonantLimits(
            version=self.version, datacards=tuple(sorted(combined))
        )
        self.output()["limits"].parent.touch()
        shutil.copy2(merged.path, self.output()["limits"].path)
