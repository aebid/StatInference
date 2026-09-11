"""The parts of the shape rebinning that do not care how many axes a histogram has.

Split out of bin_opt_2d/rebin_2d.py, which is where all of this was written and where
the comments' point of view comes from. It moved here unchanged when a 1D binner was
added, because the two want the *same* answers to the same questions -- what a slice has
to contain to be worth keeping, which processes are minor enough to exempt, which figure
of merit decides a boundary -- and a second copy of those would be a second place for
them to drift.

`_integral()` branches on GetDimension(), so everything downstream of it reads a TH1 and
a TH2 alike; that is what makes the split possible at all. What stays behind in each
binner is the part that knows what a "bin" is: the 2D driver treats a slice as a
directory of mass bins, the 1D driver treats it as one bin of one histogram.

Nothing here is configuration. BINNING_DEFAULTS and load_binning_config() stay with each
binner because the knob *set* differs -- the mass-axis floors mean nothing in 1D -- and
their "unknown key raises" contract only works if each file states its own valid keys.
"""

import array
import math
import os

import yaml

from StatInference.common.tools import importROOT
from StatInference.common.param_parse import extractParameters
from StatInference.dc_make.model import Model

ROOT = importROOT()


# Written inside each era's own output directory, beside the shapes it describes, and
# read back by --binning to replay them.
BINNING_JSON = "binning.json"

# The figures of merit significance() implements. Named here so the yaml path can be
# checked against the same list the command line's choices= uses.
SIGNIFICANCE_MODES = ("sb", "asimov")


def lookup_frozen(frozen_binning, era, mass, channel, category):
    """The recorded binning for one channel/category, or None if it was skipped then.

    A category absent from the record was skipped by the run that wrote it (too little
    signal, or a background missing from a discovery era), so it is skipped again rather
    than quietly re-optimised -- a replay that rebinned some categories and froze others
    would be neither the old binning nor a new one.
    """
    return (
        frozen_binning.get("binning", {})
        .get(era, {})
        .get(str(mass), {})
        .get(channel, {})
        .get(category)
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = Model.fromConfig(cfg["model"])
    channels = cfg["channels"]
    # Taken verbatim here; run() reduces them to the base categories the 2D input is
    # actually keyed by, once it has the pattern to do it with.
    categories = list(cfg["categories"])

    # Several processes may carry is_signal (e.g. the bbWW(2l) and bbtautau decay
    # modes of the same resonance, both scaled by the same signal strength). They
    # are summed to form the discovery signal that steers the slice
    # boundaries, so the binning is optimised for the total signal in the fit.
    signal_hist_names = []
    mass_values = None
    background_entries = []
    for entry in cfg["processes"]:
        if type(entry) == str:
            background_entries.append((entry, entry, []))
            continue
        if entry.get("is_data", False):
            continue
        base_name = entry["process"]
        hist_name = entry.get("hist_name", base_name)
        if entry.get("is_signal", False):
            signal_hist_names.append(hist_name)
            if mass_values is None:
                mass_values = entry["param_values"]
            elif list(entry["param_values"]) != list(mass_values):
                raise RuntimeError(
                    f"Signal {hist_name} has param_values {entry['param_values']}, "
                    f"which differ from {mass_values}; every signal must be defined "
                    "at the same mass points"
                )
        else:
            background_entries.append((base_name, hist_name, entry.get("channels", [])))

    if not signal_hist_names:
        raise RuntimeError("No signal process found in config")

    return {
        "model": model,
        "channels": channels,
        "categories": categories,
        "era_groups": cfg.get("era_groups", {}),
        # Declared by a configuration that reads sliced shapes, so that run() can check
        # it against the pattern the shapes are actually written with. None when the
        # configuration names no pattern, and then there is nothing to check.
        "category_pattern": cfg.get("category_pattern"),
        "signal_hist_name_patterns": signal_hist_names,
        "signal_param_name": extractParameters(signal_hist_names[0])[0],
        "mass_values": mass_values,
        "background_entries": background_entries,
    }


def open_input_file(input_dir, model, era, mass, param_name):
    file_name = model.getInputFileName(era, {param_name: mass})
    full_path = os.path.join(input_dir, file_name)
    f = ROOT.TFile.Open(full_path, "READ")
    if f is None or f.IsZombie():
        raise RuntimeError(f"Cannot open file {full_path}")
    return f


def _detach(h):
    """Detach a histogram from its TFile and hand ownership to Python.

    SetDirectory(0) alone makes the histogram survive the file's close, but
    leaves it owned by nobody -- so every 2D histogram read here (one per
    systematic variation, per category, per mass) leaked for the lifetime of the
    process. Across the ten-mass loop that grew without bound and got the job
    SIGKILLed partway through the final mass, leaving a truncated ROOT file.
    SetOwnership makes the object die with its last Python reference.
    """
    h.SetDirectory(0)
    ROOT.SetOwnership(h, True)
    return h


def get_hist(f, path):
    h = f.Get(path)
    # TFile.Get() on a fully-missing nested path can return a PyROOT wrapper
    # around a null C++ pointer, which is not `is None` but is falsy.
    if not h:
        return None
    return _detach(h)


def sum_hists(hists):
    total = None
    for h in hists:
        if h is None:
            continue
        if total is None:
            total = _detach(h.Clone())
        else:
            total.Add(h)
    return total


def _integral(hist, lo, hi):
    return (
        hist.Integral(lo, hi, 0, -1)
        if hist.GetDimension() == 2
        else hist.Integral(lo, hi)
    )


def _integral_and_error(hist, lo, hi):
    """(yield, MC statistical error) over [lo, hi]."""
    err = array.array("d", [0.0])
    if hist.GetDimension() == 2:
        value = hist.IntegralAndError(lo, hi, 0, -1, err)
    else:
        value = hist.IntegralAndError(lo, hi, err)
    return value, err[0]


def _bkg_yields(bkg_hists_by_name, lo, hi):
    """For each background, the summed yield across all discovery eras (combined
    statistics). Individual eras are not required to individually clear the
    threshold -- use allow_negative_bins_within_error (maker.py, per process/
    category) for backgrounds/categories where a specific era can land negative
    in a bin that's fine once combined."""
    return {
        name: sum(_integral(h, lo, hi) for h in hists)
        for name, hists in bkg_hists_by_name.items()
    }


def _bkg_errors(bkg_hists_by_name, lo, hi):
    """Per background, the MC statistical error on the summed yield (eras added in
    quadrature)."""
    return {
        name: math.sqrt(sum(_integral_and_error(h, lo, hi)[1] ** 2 for h in hists))
        for name, hists in bkg_hists_by_name.items()
    }


def _total_bkg_error(bkg_hists_by_name, lo, hi):
    errors = _bkg_errors(bkg_hists_by_name, lo, hi)
    return math.sqrt(sum(e**2 for e in errors.values()))


def effective_entries(value, error):
    """(yield / error)^2 -- the unweighted MC event count a weighted yield is worth.

    A DY slice holding a couple of very-high-weight aMC@NLO events can carry a
    sizeable yield with an effective count far below 1, i.e. a background estimate
    that is statistically compatible with almost anything.
    """
    if error <= 0:
        return float("inf") if value > 0 else 0.0
    return (value / error) ** 2


def asimov_significance(s, b, b_err=0.0):
    """Median discovery significance for a counting experiment (Cowan et al.,
    arXiv:1007.1727), with the background uncertainty folded in when given.

    Reduces to sqrt(2*((s+b)*ln(1+s/b) - s)) for b_err = 0. Unlike S/sqrt(B) this
    stays valid when b is O(1), which is exactly the regime the high-mass boosted
    slices live in -- there S/sqrt(B) reports significances of ~30 on ~1.5
    background events, and drives the slice boundary on that basis.
    """
    if b <= 0 or s <= 0:
        return 0.0
    if b_err <= 0:
        return math.sqrt(max(2.0 * ((s + b) * math.log1p(s / b) - s), 0.0))
    var = b_err**2
    term1 = (s + b) * math.log(((s + b) * (b + var)) / (b * b + (s + b) * var))
    term2 = (b * b / var) * math.log1p(var * s / (b * (b + var)))
    return math.sqrt(max(2.0 * (term1 - term2), 0.0))


def significance(s, b, b_err=0.0, mode="sb"):
    """Figure of merit steering the slice boundaries.

    mode="sb":     S / sqrt(B + sigma_B^2). Folding sigma_B into plain S/sqrt(B)
                   stops a downward fluctuation of a statistics-starved background
                   (DY in the b-tagged muMu slices, effective MC count <1) from
                   inflating the apparent significance. Still assumes the Gaussian
                   regime, which breaks down for B of order a few.
    mode="asimov": the Poisson-correct Asimov significance, valid at low B.
    """
    if b is None or b <= 0:
        return 0.0
    if mode == "asimov":
        return asimov_significance(s, b, b_err)
    denominator = b + b_err**2
    if denominator <= 0:
        return 0.0
    return s / (denominator**0.5)


def _slice_passes(
    yields,
    min_sum,
    total_error=None,
    min_neff=0.0,
    errors=None,
    min_each=0.0,
    min_proc_neff=0.0,
    exempt=(),
):
    """Slice validity: the summed background must clear min_sum, and be known
    to better than min_neff effective MC entries -- a window whose background is
    statistically undetermined must not be selectable at all.

    The per-process arms (min_each, min_proc_neff) are what _bin_passes already
    does on the mass axis, applied here as well. Testing only the sum hides an
    individual background that has fluctuated negative behind a large, well
    measured neighbour: muMu/SR/res2b at m500 selected a slice holding
    DY = -5.87 +- 8.92 (N_eff 0.43) because the summed background there was
    +39.9 +- 9.0 (N_eff 22), clearing both summed tests comfortably. That slice
    cannot be turned into a datacard at all -- a negative DY integral is rejected
    outright by resolveNegativeBins -- so the binning has to not select it in the
    first place. Worse, the significance being maximised is S/sqrt(B + sigma_B^2),
    which a downward fluctuation *increases* by lowering B, so such a window is
    mildly preferred rather than merely tolerated.

    `exempt` comes from minor_backgrounds() judged over the whole sliced-axis range of the
    category, never over the candidate window: a background that is negligible in
    this category must not be able to veto every boundary, and judging it inside
    the window under test is circular (a background that is exactly zero there is
    trivially below any fraction of the total).
    """
    total = sum(yields.values())
    if total <= min_sum:
        return False
    if min_neff > 0 and total_error is not None:
        if effective_entries(total, total_error) < min_neff:
            return False
    for name, value in yields.items():
        # As in _bin_passes: exemption covers the magnitude floor, never the sign.
        if value < 0:
            return False
        if name in exempt:
            continue
        if min_each > 0 and value <= min_each:
            return False
        if min_proc_neff > 0 and errors is not None:
            if effective_entries(value, errors.get(name, 0.0)) < min_proc_neff:
                return False
    return True


def minor_backgrounds(bkg_hists_by_name, lo, hi, min_frac):
    """Backgrounds negligible over the *whole* [lo, hi] range, which may therefore
    be exempted from the per-bin min_each floor.

    This must be judged once over the full slice, never inside the candidate bin
    under test. Evaluating the fraction per bin is circular: a background that is
    exactly zero in that bin is trivially below any fraction of the bin total, so
    it is always exempted -- which is precisely the case the floor exists to
    catch. That let a bin through with TT = 0 in a slice where TT is 94% of the
    background (m300, eMu, res2b_dnn2).
    """
    if min_frac <= 0:
        return set()
    yields = _bkg_yields(bkg_hists_by_name, lo, hi)
    total = sum(yields.values())
    if total <= 0:
        return set()
    return {name for name, value in yields.items() if value < min_frac * total}


def grow_slice(
    sig_hist,
    bkg_hists_by_name,
    right,
    first_bin,
    min_sum,
    min_neff=0.0,
    sig_mode="sb",
    min_each=0.0,
    min_proc_neff=0.0,
    exempt=(),
):
    """Among every candidate [left, right] with summed background > min_sum,
    pick the one maximizing S/sqrt(B + sigma_B^2) -- not just the first one that
    clears it. This is what actually drives where the cut lands; the content
    floor is only a validity gate. Falls back to first_bin (best effort) if no
    candidate clears the floor anywhere."""
    best_left = None
    best_sig = -1.0
    for left in range(right, first_bin - 1, -1):
        bkg_y = _bkg_yields(bkg_hists_by_name, left, right)
        b_err = _total_bkg_error(bkg_hists_by_name, left, right)
        bkg_e = (
            _bkg_errors(bkg_hists_by_name, left, right) if min_proc_neff > 0 else None
        )
        if not _slice_passes(
            bkg_y,
            min_sum,
            b_err,
            min_neff,
            bkg_e,
            min_each,
            min_proc_neff,
            exempt,
        ):
            continue
        s = _integral(sig_hist, left, right)
        b = sum(bkg_y.values())
        sig = significance(s, b, b_err, sig_mode)
        if sig > best_sig:
            best_sig = sig
            best_left = left
    return best_left if best_left is not None else first_bin


def find_slices(
    sig_hist,
    bkg_hists_by_name,
    n_slices,
    first_bin,
    last_bin,
    min_sum,
    min_neff=0.0,
    sig_mode="sb",
    min_each=0.0,
    min_proc_neff=0.0,
    min_frac=0.0,
):
    """Split [first_bin, last_bin] into exactly `n_slices` ranges (fixed count,
    required so every mass point shares the same category list), scanning right
    to left. Each slice (except the final, leftover one) is placed to maximize
    signal significance among boundaries with summed background > min_sum.

    Which backgrounds are minor enough to be exempt from the per-process floors is
    decided once here, over the whole [first_bin, last_bin] range, and held fixed
    for every candidate window -- see _slice_passes on why it cannot be re-judged
    per window.
    """
    exempt = (
        minor_backgrounds(bkg_hists_by_name, first_bin, last_bin, min_frac)
        if (min_each > 0 or min_proc_neff > 0)
        else set()
    )
    slices = []
    right = last_bin
    for slice_idx in range(n_slices):
        if slice_idx == n_slices - 1 or right <= first_bin:
            if right < first_bin:
                # The axis ran out before n_slices boundaries were placed. Reported as
                # None rather than invented: the previous code emitted (0,0), (-1,-1),
                # (-2,-2) here to keep the count, and ROOT reads any range whose upper
                # bound is below its lower one as the *whole axis including overflow* --
                # so those became copies of the full plane, and the same events entered
                # the fit two or three times as independent categories. discover_binning
                # rejects the category instead.
                slices.append(None)
                continue
            slices.append((first_bin, right))
            right = first_bin - 1
            continue
        left = grow_slice(
            sig_hist,
            bkg_hists_by_name,
            right,
            first_bin,
            min_sum,
            min_neff,
            sig_mode,
            min_each,
            min_proc_neff,
            exempt,
        )
        slices.append((left, right))
        right = left - 1
    slices.reverse()
    return slices


def extend_outer_edges(ranges, full_lo, full_hi):
    """Widen the first/last range to swallow under/overflow (bin 0 / nbins+1),
    so no events are silently dropped at the extremes of the axis.

    Only real ranges are widened. Widening a None (an exhausted slice) would invent one,
    and widening past a None would put the underflow on a range that is not the outermost
    -- so a list containing any None is left alone and rejected by the caller.
    """
    ranges = list(ranges)
    if any(r is None for r in ranges):
        return ranges
    ranges[0] = (full_lo, ranges[0][1])
    ranges[-1] = (ranges[-1][0], full_hi)
    return ranges


def bin_budget(bkg_hists_by_name, lo, hi, max_bins_per_slice, bkg_per_bin):
    """How many bins this slice can actually afford.

    A fixed max_bins_per_slice is applied blind to slice content: the high-mass boosted
    slices hold ~1.5 total background events and were still being split into 10
    bins, i.e. ~0.15 events per bin. Capping at B_slice / bkg_per_bin ties the
    binning to the statistics that are really there. Returns max_bins_per_slice
    unchanged when bkg_per_bin <= 0 (feature off).
    """
    if bkg_per_bin <= 0:
        return max_bins_per_slice
    total = sum(_bkg_yields(bkg_hists_by_name, lo, hi).values())
    return bins_for_background(total, max_bins_per_slice, bkg_per_bin)


def bins_for_background(total, max_bins_per_slice, bkg_per_bin):
    """bin_budget()'s arithmetic, separated from how the yield was obtained.

    The cell-table binner has the slice's background already and would otherwise have to
    restate this formula to avoid re-projecting the histograms -- and a second statement
    of it is a second thing to keep in step. This is the only copy.
    """
    if bkg_per_bin <= 0:
        return max_bins_per_slice
    if total <= 0:
        return 1
    return max(1, min(max_bins_per_slice, int(total / bkg_per_bin)))


def bin_edges(y_axis, y_ranges):
    """Physical edges of the discovered y ranges, for booking the output TH1.

    The rebinned shapes used to be booked as n_bins over [0, n_bins], which threw
    the axis scale away and left every plot labelled by bin index. The ranges are
    contiguous and ordered, so the edges are each range's low edge plus the last
    range's upper edge. extend_outer_edges() pushes the outer ranges into the
    underflow/overflow bins, which have no finite edge of their own -- those are
    clamped to the axis limits.
    """
    n = y_axis.GetNbins()
    edges = [y_axis.GetBinLowEdge(max(lo, 1)) for lo, _ in y_ranges]
    edges.append(y_axis.GetBinUpEdge(min(y_ranges[-1][1], n)))
    return edges
