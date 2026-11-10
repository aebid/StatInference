"""Shape uncertainties given as a set of member histograms, reduced to one Up/Down pair.

Some uncertainties are not stored as an Up and a Down shape but as a set of alternative
shapes, one per member: the QCD-scale points (muR, muF) or the members of a Hessian PDF set.
The reduction to the Up/Down pair a datacard needs is not linear, so it has to be taken on
the final shapes -- after the datasets of a process are summed, after the sub-eras of a
meta-era are summed, and in the bins of the fit -- never on parts that are added afterwards.
The datacard maker is where those shapes come together, so it reduces them there.

A card declares such an uncertainty as a shape uncertainty with a `members` block:

    - name: QCDscale_shape_TT
      type: shape
      processes: [ TT ]
      members:
        reduction: envelope
        names: [ muR0p5_muF0p5, muR0p5_muF1, muR1_muF0p5, muR1_muF2, muR2_muF1, muR2_muF2 ]

    - name: pdf_shape_TT
      type: shape
      processes: [ TT ]
      members:
        reduction: hessian
        reference: 0              # optional: deviations from this member, not from the nominal
        names: { first: 1, last: 100 }
        alpha_s: [ 101, 102 ]     # optional: [down, up]

Member m of uncertainty `name` is read as the histogram <process>_<histogram>, where
`histogram` defaults to "{name}_{member}". "{process}" in it is the card process's histogram
name, so a per-mass signal can carry its mass in the input name but not in the nuisance's:

        histogram: QCDscale_shape_{process}_{member}

Reductions, bin by bin, around the nominal n:
  envelope: Up = max(n, members), Down = min(n, members).
  hessian:  d = sqrt(sum_k (m_k - r)^2 + ((a_up - a_down) / 2)^2), r = the reference member
            (the nominal if there is none); Up = n + d, Down = max(n - d, 0).

A member with a non-finite bin is an error: a process without such a member (e.g. a sample
without the alpha_s variations, whose weights are NaN) must not list it.

No ROOT here: the reduction works on bin contents, so an analysis's preprocessing script can
import this module to know which histograms are members.
"""

import math

REDUCTIONS = ("envelope", "hessian")
KEYS = {"reduction", "names", "reference", "alpha_s", "histogram"}


def _names(value, what):
    """A member list: a list of names (numbers are names too), or {first, last} integers."""
    if isinstance(value, dict):
        if set(value) != {"first", "last"}:
            raise ValueError(
                f"{what}: a range is {{first: <int>, last: <int>}}, got {value}"
            )
        first, last = int(value["first"]), int(value["last"])
        if last < first:
            raise ValueError(f"{what}: empty range {value}")
        return [str(k) for k in range(first, last + 1)]
    if isinstance(value, (list, tuple)) and value:
        return [str(v) for v in value]
    raise ValueError(
        f"{what}: expected a non-empty list or {{first, last}}, got {value!r}"
    )


class MemberReduction:
    def __init__(self, reduction, names, reference=None, alpha_s=None, histogram=None):
        if reduction not in REDUCTIONS:
            raise ValueError(
                f"members: reduction must be one of {REDUCTIONS}, got {reduction!r}"
            )
        self.reduction = reduction
        self.names = [str(n) for n in names]
        self.reference = None if reference is None else str(reference)
        self.alpha_s = None if alpha_s is None else [str(a) for a in alpha_s]
        self.histogram = histogram or "{name}_{member}"
        if reduction == "envelope" and (
            self.reference is not None or self.alpha_s is not None
        ):
            raise ValueError("members: reference and alpha_s belong to a hessian set")
        if self.alpha_s is not None and len(self.alpha_s) != 2:
            raise ValueError("members: alpha_s is [down, up]")
        if "{member}" not in self.histogram:
            raise ValueError("members: histogram must contain {member}")
        members = self.all_members()
        if len(set(members)) != len(members):
            raise ValueError("members: a member is listed twice")

    @classmethod
    def fromConfig(cls, entry, unc_name="?"):
        what = f"uncertainty {unc_name}: members"
        if not isinstance(entry, dict):
            raise ValueError(f"{what} must be a mapping")
        unknown = set(entry) - KEYS
        if unknown:
            raise ValueError(f"{what}: unknown keys {sorted(unknown)}")
        if "names" not in entry or "reduction" not in entry:
            raise ValueError(f"{what}: 'reduction' and 'names' are required")
        alpha_s = entry.get("alpha_s")
        try:
            return cls(
                entry["reduction"],
                _names(entry["names"], what),
                reference=entry.get("reference"),
                alpha_s=(
                    _names(alpha_s, what + " alpha_s") if alpha_s is not None else None
                ),
                histogram=entry.get("histogram"),
            )
        except ValueError as e:
            raise ValueError(f"uncertainty {unc_name}: {e}") from None

    def all_members(self):
        """Every member the reduction reads."""
        out = list(self.names)
        if self.reference is not None:
            out.append(self.reference)
        out += self.alpha_s or []
        return out

    def histogram_name(self, unc_name, process, member):
        """The variation part of a member's histogram name, read as <process>_<this>."""
        return self.histogram.format(name=unc_name, process=process, member=member)

    def reduce(self, nominal, members, label=""):
        """(up, down) bin contents from the nominal's and {member: contents}."""
        n_bins = len(nominal)
        for m in self.all_members():
            values = members[m]
            if len(values) != n_bins:
                raise ValueError(
                    f"{label}: member {m} has {len(values)} bins, not {n_bins}"
                )
            bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
            if bad:
                raise ValueError(
                    f"{label}: member {m} is not finite in bins {bad}; a member the "
                    "process does not have must not be listed"
                )
        up, down = [], []
        for i, n in enumerate(nominal):
            if self.reduction == "envelope":
                values = [members[m][i] for m in self.names]
                up.append(max([n] + values))
                down.append(min([n] + values))
                continue
            ref = members[self.reference][i] if self.reference is not None else n
            var = sum((members[m][i] - ref) ** 2 for m in self.names)
            if self.alpha_s:
                a_down, a_up = (members[m][i] for m in self.alpha_s)
                var += ((a_up - a_down) / 2) ** 2
            d = math.sqrt(var)
            up.append(n + d)
            down.append(max(n - d, 0.0))
        return up, down
