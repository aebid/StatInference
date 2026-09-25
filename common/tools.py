import math
import re

from string import Formatter

import numpy as np


class CategoryNaming:
    """Names the sub-categories a 2D -> categorized-1D transformation cuts a base
    category into, and reads those names back.

    The pattern is configuration rather than code: nothing here assumes the sliced axis
    is a DNN score, or that the analysis producing the 2D shapes has one at all. Both
    directions are built from that one pattern, so the code that writes a category name
    and the code that parses it back cannot drift apart -- which is the failure this
    replaces, where a format string wrote the names and hardcoded regexes read them, and
    changing the format silently stopped the regexes matching, leaving every slice as its
    own base category instead of raising.
    """

    # Deliberately neutral: an analysis that wants its discriminant in the name says so in
    # its configuration (HH->bbWW pins "{base_category}_dnn{slice_idx}"), rather than every
    # analysis inheriting one analysis's choice from here. "_slice" rather than "_cat"
    # because hand-written category names do use "_cat" -- x_hh_bbtautau_run2.yaml has
    # "res1b_cat3_masswindow" -- and a default that can collide with an unsliced name would
    # have split() claiming it as a slice of something.
    default_pattern = "{base_category}_slice{slice_idx}"

    # Sub-expression per placeholder. base_category is greedy so a base name that itself
    # ends in something the pattern could match still resolves to the last slice index,
    # which is the one the pattern wrote.
    placeholders = {
        "base_category": r"(?P<base_category>.+)",
        "slice_idx": r"(?P<slice_idx>\d+)",
    }

    def __init__(self, pattern=None):
        self.pattern = pattern or self.default_pattern
        regex = ""
        seen = set()
        for literal, field, _, conversion in Formatter().parse(self.pattern):
            regex += re.escape(literal)
            if field is None:
                continue
            if field not in self.placeholders:
                raise RuntimeError(
                    f"category pattern '{self.pattern}': unknown placeholder "
                    f"'{{{field}}}'; known are "
                    + ", ".join("'{%s}'" % p for p in sorted(self.placeholders))
                )
            if field in seen:
                raise RuntimeError(
                    f"category pattern '{self.pattern}': '{{{field}}}' appears more than "
                    "once, so a name built from it cannot be read back unambiguously"
                )
            if conversion is not None:
                raise RuntimeError(
                    f"category pattern '{self.pattern}': conversion '!{conversion}' on "
                    f"'{{{field}}}' would not survive the round trip back to a category"
                )
            seen.add(field)
            regex += self.placeholders[field]
        missing = sorted(set(self.placeholders) - seen)
        if missing:
            raise RuntimeError(
                f"category pattern '{self.pattern}' must use every placeholder; missing "
                + ", ".join("'{%s}'" % m for m in missing)
            )
        self._regex = re.compile("^" + regex + "$")

    @classmethod
    def fromConfig(cls, cfg):
        """From a datacard configuration's top-level ``category_pattern``, or the default
        when it declares none -- a configuration whose categories were never sliced has
        nothing to parse back, and any name is then its own base.

        A configuration reading shapes from bin_opt_2d/rebin_2d.py must repeat the pattern
        that run used, since that is the only way its sliced category names can be taken
        apart again.
        """
        return cls(cfg.get("category_pattern"))

    def name(self, base_category, slice_idx):
        return self.pattern.format(base_category=base_category, slice_idx=slice_idx)

    def split(self, category):
        """Inverse of name(): "SR/res2b_dnn2" -> ("SR/res2b", 2).

        An unsliced name returns (category, None) rather than raising -- a configuration
        with no ``binning:`` block has categories that were never sliced, and they are
        legitimately their own base.
        """
        match = self._regex.match(category)
        if not match:
            return category, None
        return match.group("base_category"), int(match.group("slice_idx"))

    def base(self, category):
        """The base category a slice belongs to, or the name itself if unsliced."""
        return self.split(category)[0]


class InputCategories:
    """Where a category's shapes live in the input file, when that is not its own name.

    A datacard category is one name for every model point: it names the bin, the
    per-category datacard directory, and what Process/Uncertainty scoping matches
    against. Some inputs do not store it under one name. The HH->bbWW single-lepton
    shapes hold, in the file for mass MX, a region for every mass (SR_SL_M300 ...
    SR_SL_M1000), and only SR_SL_M${MX} is the one the MX hypothesis selected. Listing
    those as categories would put all ten into every datacard as separate bins -- every
    file has all of them, so nothing would ever report one missing.

    So the configuration maps a category to its input directory, with the model's
    parameters substituted at read time:

        input_categories:
          SR_SL/res2b: SR_SL_M${MX}/res2b

    Only exact names are mapped. A sliced name (SR_SL/res2b_hmebox0) is read from shapes
    a preprocessing step has already written under the category's own name, so it
    passes through, as does every category of a configuration that declares no map.
    """

    def __init__(self, mapping=None):
        mapping = mapping or {}
        if not isinstance(mapping, dict):
            raise RuntimeError(
                f"input_categories must map category -> input directory, got {mapping!r}"
            )
        self.mapping = {str(k): str(v) for k, v in mapping.items()}

    @classmethod
    def fromConfig(cls, cfg):
        return cls(cfg.get("input_categories"))

    def unknown(self, categories):
        """Mapped names that are not among the configuration's categories -- a typo
        there would otherwise leave the category read from its own name, silently."""
        return sorted(set(self.mapping) - set(categories))

    def path(self, category, params):
        """The directory holding `category` for model point `params` ({"MX": 500})."""
        pattern = self.mapping.get(category, category)
        for name, value in (params or {}).items():
            pattern = pattern.replace("${" + name + "}", str(value))
        if "${" in pattern:
            raise RuntimeError(
                f"input category '{pattern}' for '{category}' still has an unresolved "
                f"parameter after substituting {dict(params or {})}"
            )
        return pattern


class PackageWrapper:
    def __init__(self, import_fn):
        self._package = None
        self._import_fn = import_fn

    def __getattr__(self, name):
        if self._package is None:
            self._package = self._import_fn()
        return getattr(self._package, name)


def lazyImport(package_name):
    def import_fn():
        return __import__(package_name)

    return PackageWrapper(import_fn)


def importROOT():
    def import_fn():
        import ROOT

        ROOT.gROOT.SetBatch(True)
        ROOT.gROOT.SetMustClean(False)
        ROOT.TH1.AddDirectory(False)
        ROOT.TH2.AddDirectory(False)
        ROOT.EnableThreadSafety()
        return ROOT

    return PackageWrapper(import_fn)


ROOT = importROOT()


def listToVector(x, value_type):
    v = ROOT.std.vector(value_type)(len(x))
    for n in range(len(x)):
        v[n] = x[n]
    return v


def rebinAndFill(new_hist, old_hist, epsilon=1e-7):
    def check_range(old_axis, new_axis):
        old_min = old_axis.GetBinLowEdge(1)
        old_max = old_axis.GetBinUpEdge(old_axis.GetNbins())
        new_min = new_axis.GetBinLowEdge(1)
        new_max = new_axis.GetBinUpEdge(new_axis.GetNbins())
        return old_min <= new_min and old_max >= new_max

    def get_new_bin(old_axis, new_axis, bin_id_old):
        old_low_edge = round(old_axis.GetBinLowEdge(bin_id_old), 4)
        old_up_edge = round(old_axis.GetBinUpEdge(bin_id_old), 4)
        bin_low_new = new_axis.FindFixBin(old_low_edge)
        bin_up_new = new_axis.FindFixBin(old_up_edge)

        new_up_edge = new_axis.GetBinUpEdge(bin_low_new)
        if not (
            bin_low_new == bin_up_new
            or abs(old_up_edge - new_up_edge)
            <= epsilon * abs(old_up_edge + new_up_edge) * 2
        ):
            old_bins = [
                str(old_axis.GetBinLowEdge(n))
                for n in range(1, old_axis.GetNbins() + 2)
            ]
            new_bins = [
                str(new_axis.GetBinLowEdge(n))
                for n in range(1, new_axis.GetNbins() + 2)
            ]
            print("old_bins: [{}]".format(", ".join(old_bins)))
            print("new_bins: [{}]".format(", ".join(new_bins)))

            raise RuntimeError("Incompatible bin edges")
        return bin_low_new

    def add_bin_content(bin_old, bin_new):
        cnt_old = old_hist.GetBinContent(bin_old)
        err_old = old_hist.GetBinError(bin_old)
        cnt_new = new_hist.GetBinContent(bin_new)
        err_new = new_hist.GetBinError(bin_new)
        cnt_upd = cnt_new + cnt_old
        err_upd = math.sqrt(err_new**2 + err_old**2)
        new_hist.SetBinContent(bin_new, cnt_upd)
        new_hist.SetBinError(bin_new, err_upd)

    n_dim = old_hist.GetDimension()
    if new_hist.GetDimension() != n_dim:
        raise RuntimeError("Incompatible number of dimensions")
    if n_dim < 1 or n_dim > 2:
        raise RuntimeError("Unsupported number of dimensions")

    if not check_range(old_hist.GetXaxis(), new_hist.GetXaxis()):
        raise RuntimeError("x ranges are not compatible")

    if n_dim > 1 and not check_range(old_hist.GetYaxis(), new_hist.GetYaxis()):
        raise RuntimeError("y ranges are not compatible")

    for x_bin_old in range(old_hist.GetNbinsX() + 2):
        x_bin_new = get_new_bin(old_hist.GetXaxis(), new_hist.GetXaxis(), x_bin_old)
        if n_dim == 1:
            add_bin_content(x_bin_old, x_bin_new)
        else:
            for y_bin_old in range(old_hist.GetNbinsY() + 2):
                y_bin_new = get_new_bin(
                    old_hist.GetYaxis(), new_hist.GetYaxis(), y_bin_old
                )
                bin_old = old_hist.GetBin(x_bin_old, y_bin_old)
                bin_new = new_hist.GetBin(x_bin_new, y_bin_new)
                add_bin_content(bin_old, bin_new)


def getRelevantBins(
    era,
    channel,
    category,
    signal_processes_histograms,
    signalFractionForRelevantBins,
    unc_name=None,
    unc_scale=None,
    model_params=None,
):
    relevant_bins = set()
    for hist in signal_processes_histograms:
        axis = hist.GetXaxis()
        hist_integral = hist.Integral(1, axis.GetNbins() + 1)
        for nbin in range(1, axis.GetNbins() + 1):
            if (
                hist_integral != 0
                and hist.GetBinContent(nbin) / hist_integral
                >= signalFractionForRelevantBins
            ):
                relevant_bins.add(nbin)
    return list(relevant_bins)


def th1ToNumpy(histogram, include_overflow=False):
    bin_range = (
        (0, histogram.GetNbinsX() + 1)
        if include_overflow
        else (1, histogram.GetNbinsX())
    )
    n_bins = bin_range[1] - bin_range[0] + 1
    bin_contents = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    bin_numbers = np.zeros(n_bins, dtype=int)
    bin_indices = {}
    for bin_idx, bin_number in enumerate(range(bin_range[0], bin_range[1] + 1)):
        bin_contents[bin_idx] = histogram.GetBinContent(bin_number)
        bin_errors[bin_idx] = histogram.GetBinError(bin_number)
        bin_numbers[bin_idx] = bin_number
        bin_indices[bin_number] = bin_idx
    return bin_indices, bin_numbers, bin_contents, bin_errors


class NegativeBinSolution:
    def __init__(self):
        self.accepted = True
        self.integral = None
        self.has_zero_integral = False
        self.has_negative_integral = False
        self.negative_bins = set()
        self.negative_bins_within_error = set()
        self.relevant_negative_bins = set()
        self.changed_bins = set()


def resolveNegativeBins(
    histogram,
    allow_zero_integral=False,
    allow_negative_integral=False,
    allow_negative_bins_within_error=False,
    max_n_sigma_for_negative_bins=1,
    relevant_bins=None,
    include_overflow=False,
):
    relevant_bins = relevant_bins or []
    bin_indices, bin_numbers, bin_contents, bin_errors = th1ToNumpy(
        histogram, include_overflow
    )
    n_bins = len(bin_numbers)

    solution = NegativeBinSolution()
    solution.integral = np.sum(bin_contents)
    if solution.integral == 0:
        solution.has_zero_integral = True
        solution.accepted = allow_zero_integral
        # print("integral = 0")
        return solution
    if solution.integral < 0:
        solution.has_negative_integral = True
        solution.accepted = allow_negative_integral
        # print("integral < 0")
        return solution

    donor_integral = 0.0
    negative_integral = 0.0

    for bin_idx in range(n_bins):
        if bin_contents[bin_idx] >= 0:
            if bin_numbers[bin_idx] not in relevant_bins:
                donor_integral += bin_contents[bin_idx]
            continue
        negative_integral += bin_contents[bin_idx]
        solution.negative_bins.add(bin_numbers[bin_idx])
        # print(bin_idx, bin_numbers[bin_idx])
        zero_within_error = (
            bin_contents[bin_idx] + max_n_sigma_for_negative_bins * bin_errors[bin_idx]
            >= 0
        )
        if zero_within_error:
            solution.negative_bins_within_error.add(bin_numbers[bin_idx])
        if not allow_negative_bins_within_error or not zero_within_error:
            solution.accepted = False
        if bin_numbers[bin_idx] in relevant_bins:
            solution.relevant_negative_bins.add(bin_numbers[bin_idx])
            solution.accepted = False

    if abs(negative_integral) > donor_integral:
        solution.accepted = False

    if not solution.accepted or len(solution.negative_bins) == 0:
        # print("solution not accepted or len solution(negative bins) = 0")
        return solution

    for bin_number in solution.negative_bins:
        bin_idx = bin_indices[bin_number]
        donor_bin_indices = []
        for i in range(n_bins):
            if (
                i != bin_idx
                and bin_numbers[i] not in relevant_bins
                and bin_contents[i] > 0
            ):
                donor_bin_indices.append(i)
        donor_bin_indices = sorted(donor_bin_indices, key=lambda i: abs(i - bin_idx))
        to_balance = -bin_contents[bin_idx]
        bin_errors[bin_idx] = math.sqrt(
            bin_contents[bin_idx] ** 2 + bin_errors[bin_idx] ** 2
        )
        bin_contents[bin_idx] = 0
        solution.changed_bins.add(bin_number)
        for donor_bin_idx in donor_bin_indices:
            donor_contribution = min(to_balance, bin_contents[donor_bin_idx])
            if donor_contribution <= 0:
                continue
            bin_errors[donor_bin_idx] = math.sqrt(
                donor_contribution**2 + bin_errors[donor_bin_idx] ** 2
            )
            bin_contents[donor_bin_idx] -= donor_contribution
            solution.changed_bins.add(bin_numbers[donor_bin_idx])
            to_balance -= donor_contribution
            if to_balance <= 0:
                break
        if (
            to_balance > 0
        ):  # this should not happen, because we checked that negative_integral <= donor_integral
            raise RuntimeError(
                f"resolveNegativeBins: failed to balance bin {bin_number}"
            )
    for bin_number in solution.changed_bins:
        bin_idx = bin_indices[bin_number]
        histogram.SetBinContent(int(bin_number), bin_contents[bin_idx])
        histogram.SetBinError(int(bin_number), bin_errors[bin_idx])

    return solution
