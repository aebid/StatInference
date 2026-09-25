import math
import array
import json
import os
import sys

import numpy as np
import yaml

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(file_dir)
    base_dir = os.path.dirname(pkg_dir)
    pkg_dir_name = os.path.split(pkg_dir)[1]
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    __package__ = pkg_dir_name

from StatInference.common.tools import importROOT, CategoryNaming
from StatInference.common.param_parse import applyParameters

# The dimension-agnostic half of the binning: slice finding, the gates it applies,
# the figures of merit, and the histogram plumbing. Shared with the 1D binner, which
# needs the same answers about what a slice has to contain. See that module's
# docstring for what stays here and why.
from StatInference.common.binning_core import (
    BINNING_JSON,
    SIGNIFICANCE_MODES,
    _bkg_errors,
    _bkg_yields,
    _detach,
    _total_bkg_error,
    bin_budget,
    bin_edges,
    bins_for_background,
    effective_entries,
    extend_outer_edges,
    find_slices,
    get_hist,
    load_config,
    lookup_frozen,
    minor_backgrounds,
    open_input_file,
    sum_hists,
)
from StatInference.common.binning_dp import (
    fitted_integral_error,
    fitted_tail,
    fitted_tail_full,
    allocate_budget,
    binning_objective,
    build_cells,
    check_mask_against_gate,
    box_tables,
    check_box_mask,
    partition_dp,
    partition_dp_staged,
    slice_axis_tables,
    slice_tables,
    trim_by_marginal_gain,
)

ROOT = importROOT()


# The knobs that decide the binning, with the values a configuration gets when it does
# not say otherwise. They belong to an analysis rather than to this file -- what a slice
# needs to be worth keeping depends on the sample sizes and the selection, and a test
# configuration wants them at zero -- so every production should state its own in a
# binning yaml -- see config/Datacards/binning_2d.yaml in HH_bbWW for an annotated
# set. This file ships no configuration of its own, only these fallbacks.
BINNING_DEFAULTS = {
    "n_slices": 4,
    "category_pattern": None,  # None -> CategoryNaming's own neutral default
    "slice_var": "x",
    "max_bins_per_slice": 10,
    "min_slice_bkg_sum": 1.0,
    "min_slice_bkg_neff": 4.0,
    "min_slice_bkg_each": 0.01,
    "min_slice_bkg_each_neff": 0.0,
    "min_bin_bkg_each": 0.01,
    "min_bin_bkg_neff": 4.0,
    "min_bin_bkg_each_neff": 0.0,
    "bkg_per_bin": 5.0,
    "min_bkg_frac": 0.05,
    "min_signal": 0.5,
    "significance_mode": "asimov",
    # How the edges are found. "greedy" is the original: each slice boundary is the best
    # one taken on its own, and the mass bins inside a slice are equal-signal quantiles
    # merged until the gates pass. "dp" replaces the mass axis with an exact partition
    # search against the same figure of merit the slice boundaries already use. Default
    # stays "greedy" so adding this moves nothing until a configuration asks for it.
    "strategy": "greedy",
    # "per_slice" gives each slice the bins bin_budget() licenses from its own
    # background, which is what production does. "shared" pools those into one total and
    # spends it where it buys the most Z^2 -- the same number of bins, distributed by
    # what they are worth rather than by where the background happens to be.
    "dp_budget_mode": "per_slice",
    # A bin must earn its place: the split that creates it has to raise this category's
    # Z^2 by at least this fraction of the category's own achievable total, or it is given
    # back. 0.0 leaves the count to the budget alone, which is the previous behaviour.
    "dp_min_bin_gain": 0.0,
    # How many times to re-place the slice boundaries against the mass binning and back.
    # 0 leaves the boundaries where find_slices() put them and only replaces the mass
    # axis, which is the cheaper half and the one that can be compared to greedy directly.
    "dp_max_iterations": 0,
    # Processes the binning does not look at: the edges are chosen as if these did not
    # exist, and they are then written into the shapes like any other. Not a weaker gate
    # -- they are absent from the gates, from the summed background, from the figure of
    # merit and from the bin budget alike.
    #
    # For a background whose MC statistics cannot support any split this is the difference
    # between a binning and no binning. DY in the same-flavour channels holds eE and muMu
    # at one mass bin in the signal-like slice however good the search is: at MX=800,
    # eE/SR/res2b has 10.7 background events and a budget of 10 bins there, and *no*
    # partition into more than one bin exists, because min_bin_bkg_each_neff requires DY
    # to be measured in every bin and no split leaves it so.
    #
    # What this buys has to be checked against the fit, not assumed: a bin the binning
    # declined to see DY in still has DY in it when combine reads the card, and
    # autoMCStats will price its MC uncertainty there. Excluding a process from the choice
    # of edges does not make its statistics better.
    "binning_exclude_processes": [],
    # Written into a negative bin of a *released* process, with an error of a tenth of it.
    # null (the default) leaves negative bins exactly as they are, which is the previous
    # behaviour and leaves them to resolveNegativeBins in the datacard maker.
    #
    # Scoped to the released processes rather than applied to everything, because that is
    # the only place it is needed and it is not free elsewhere. Releasing a process is
    # what makes negative bins expected rather than exceptional, and combine cannot fit a
    # negative background. Applying it to every process instead measurably costs limit in
    # channels that never needed it: at MX=800 filling everywhere moved eMu from 0.0310 to
    # 0.0312 with a bit-identical binning, purely because bins that had been negative or
    # empty now carry 0.01 of background each.
    "fill_negative_bins": None,
    # hme_box only: the search grid for the box edges. A full scan is ny(ny+1)/2 windows
    # and every one of them carries its own dynamic program over the binned axis, so the
    # edges are scanned on a stride and then refined within +-box_refine of the winner.
    "box_stride": 6,
    "box_refine": 5,
    # hme_box only: the box is searched only where the signal is. A window that starts
    # beyond the resonance adds background and no signal; scanning it is wasted time, and
    # including it in the grid costs resolution where the answer actually lies.
    "box_signal_quantile": 0.001,
    # hme_box only: where min_bkg_frac is judged. "box" asks it once, over the whole HME
    # window, and a process under that fraction of the box is then excused from the
    # per-bin floors in *every* bin. That lets the choice of window switch a gate off: a
    # box that happens to hold DY at 4.8% of its background frees DY from
    # min_bin_bkg_each_neff everywhere inside it, including in the bins where DY is a
    # fifth of the total (measured: muMu boosted MX=400, DY N_eff 0.17 over the box).
    # "bin" asks it of each candidate DNN bin on its own, so a process is excused only
    # where it really is negligible. Default "box" keeps earlier binnings reproducible.
    "exempt_scope": "box",
    # Per-process effective-entries requirement for every fit bin, in place of
    # min_bin_bkg_each_neff for the processes it names, e.g. {TT: 25} asks TT to be known
    # to 20% (1/0.2^2). Unnamed processes keep min_bin_bkg_each_neff. Subject to
    # min_bkg_frac exemption like any other per-process gate. hme_box only for now: the
    # DNN-slice strategies' vectorised tables do not read it.
    "min_bin_bkg_neff_by_process": {},
    # hme_box only: processes whose unreliable DNN tail is taken from a fit of their own
    # measured part rather than counted. For DY the tail is one sample's cancelling
    # weights -- errors larger than the yields -- and counting it both inflates the bins'
    # uncertainty and drives the binning, since the gates see that error. The fit is per
    # box because the DNN shape depends on the HME window.
    "fit_tail_processes": [],
    # base categories it applies in. Boosted is deliberately not one for bbWW: its DY has
    # a median effective count of 1.8 over the whole box, so a fitted tail there would be
    # an extrapolation with nothing behind it.
    "fit_tail_categories": [],
    # a bin is replaced once the profile has passed its last bin with this many effective
    # entries -- and then only where the fit is better determined than the count.
    "fit_tail_min_neff": 10.0,
    # how many measured bins the fit is constrained by, counting back from the highest
    "fit_tail_last_k": 12,
}

# Named here so the yaml path can be checked against the same list the command line's
# choices= uses -- significance_mode already learned this lesson: it is read by a
# function that silently treats anything unrecognised as a default, so a typo in the
# yaml quietly changed the answer instead of failing.
BINNING_STRATEGIES = ("greedy", "dp", "hme_box")
BUDGET_MODES = ("per_slice", "shared")
EXEMPT_SCOPES = ("box", "bin")


def load_binning_config(path, overrides=None):
    """Merge the binning yaml over the defaults, then command-line overrides over that.

    Unknown keys raise rather than being ignored: a misspelled floor that silently does
    nothing is the failure mode worth spending an exception on.
    """
    declared = {}
    if path:
        with open(path, "r") as f:
            declared = yaml.safe_load(f) or {}
    unknown = set(declared) - set(BINNING_DEFAULTS)
    if unknown:
        raise RuntimeError(
            f"{path}: unknown binning key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(BINNING_DEFAULTS)}."
        )
    knobs = dict(BINNING_DEFAULTS)
    knobs.update(declared)
    for key, value in (overrides or {}).items():
        if value is not None:
            knobs[key] = value
    # The command line has choices= for this; the yaml had nothing, and significance()
    # treats anything it does not recognise as "sb" -- so a typo silently changed the
    # figure of merit every boundary is chosen by.
    if knobs["significance_mode"] not in SIGNIFICANCE_MODES:
        raise RuntimeError(
            f"{path or 'binning configuration'}: significance_mode "
            f"'{knobs['significance_mode']}' is not one of {sorted(SIGNIFICANCE_MODES)}."
        )
    fill = knobs["fill_negative_bins"]
    if fill is not None and not (isinstance(fill, (int, float)) and fill > 0):
        raise RuntimeError(
            f"{path or 'binning configuration'}: fill_negative_bins must be a positive "
            f"number or null, not {fill!r}. It is the yield a negative bin is written "
            "with, and a zero or negative one would not fix what it exists to fix."
        )
    by_proc = knobs["min_bin_bkg_neff_by_process"]
    if not isinstance(by_proc, dict) or not all(
        isinstance(v, (int, float)) and v >= 0 for v in by_proc.values()
    ):
        raise RuntimeError(
            f"{path or 'binning configuration'}: min_bin_bkg_neff_by_process must map "
            f"process names to non-negative numbers, not {by_proc!r}."
        )
    if (by_proc or knobs["min_bin_bkg_each"] is None) and knobs["strategy"] != "hme_box":
        raise RuntimeError(
            f"{path or 'binning configuration'}: min_bin_bkg_neff_by_process and "
            "min_bin_bkg_each: null are implemented for strategy hme_box only; the "
            f"'{knobs['strategy']}' strategy would silently ignore them."
        )
    for key in ("fit_tail_processes", "fit_tail_categories"):
        if not isinstance(knobs[key], list):
            raise RuntimeError(
                f"{path or 'binning configuration'}: {key} must be a list, "
                f"not {knobs[key]!r}."
            )
    if knobs["fit_tail_processes"] and knobs["strategy"] != "hme_box":
        raise RuntimeError(
            f"{path or 'binning configuration'}: fit_tail_processes is implemented for "
            f"strategy hme_box only, not '{knobs['strategy']}'."
        )
    if not isinstance(knobs["binning_exclude_processes"], list):
        raise RuntimeError(
            f"{path or 'binning configuration'}: binning_exclude_processes must be "
            f"a list of process names, not {knobs['binning_exclude_processes']!r}."
        )
    for key, allowed in (
        ("strategy", BINNING_STRATEGIES),
        ("dp_budget_mode", BUDGET_MODES),
        ("exempt_scope", EXEMPT_SCOPES),
    ):
        if knobs[key] not in allowed:
            raise RuntimeError(
                f"{path or 'binning configuration'}: {key} '{knobs[key]}' is not one of "
                f"{sorted(allowed)}."
            )
    return knobs


def _bin_passes(
    yields,
    min_each,
    exempt=(),
    total_error=None,
    min_neff=0.0,
    errors=None,
    min_proc_neff=0.0,
    proc_neff=None,
):
    """Mass-bin validity: every *relevant* background must exceed min_each, and
    (when min_neff > 0) the summed background must be known to at least min_neff
    effective MC entries.

    `exempt` is the set of backgrounds judged negligible across the whole slice by
    minor_backgrounds(); they are excused from min_each so that a process worth
    0.4% of the yield -- and known only to a few hundred percent -- cannot veto
    every candidate split, which would make find_bins() back off all the way
    to a single bin and discard the mass shape in exactly the high-significance
    slices that matter most.

    The min_neff arm is the same gate _slice_passes() applies to slice boundaries.
    Applying it only there left the mass bins *inside* each slice ungated, and
    since that is where essentially every fit bin lives, the median per-bin
    effective background count came out at ~2.8 against a slice threshold of 4.
    Note it constrains only the *summed* background, so it is satisfied by any one
    well-measured process; min_each/exempt is what protects the individual ones.

    min_each None drops the magnitude floor entirely, leaving only non-negativity (0 is
    not the same thing: it still demands a strictly positive yield). proc_neff maps a
    process to its own effective-entries requirement, in place of min_proc_neff.
    """
    total = sum(yields.values())
    if min_neff > 0 and total_error is not None:
        if effective_entries(total, total_error) < min_neff:
            return False
    for name, value in yields.items():
        # Positivity is never exemptable. `exempt` exists so a negligible process
        # cannot veto every candidate split -- it is about the *magnitude* floor, not
        # about sign. Letting an exempt process go negative does not merely leave an
        # ugly bin: resolveNegativeBins in dc_make/maker.py rejects a negative bin
        # that holds >= signalFractionForRelevantBins of the signal *unconditionally*,
        # ignoring allow_negative_bins_within_error, so the datacard fails to build at
        # all. Observed at m1000 muMu/SR/recovery_dnn1, where ST was 4.24% of the
        # slice -- just under min_bkg_frac 0.05, hence exempt -- and came out at
        # -0.567 +- 0.789, killing the production.
        if value < 0:
            return False
        if name in exempt:
            continue
        if min_each is not None and value <= min_each:
            return False
        # Every background must be measured, not merely present: a yield known only
        # to a few hundred percent is not a background estimate. _slice_passes has
        # carried this arm for the sliced axis; the mass axis, where essentially all
        # fit bins live, had no per-process test at all -- only the summed one, which
        # any single well-measured process satisfies on its own.
        required = (proc_neff or {}).get(name, min_proc_neff)
        if required > 0 and errors is not None:
            if effective_entries(value, errors.get(name, 0.0)) < required:
                return False
    return True


def signal_quantile_ranges(sig_hist, n_bins, first_bin, last_bin):
    """Split [first_bin, last_bin] into `n_bins` ranges each holding an equal share
    of the signal.

    This is what puts the bins where the resonance is. The axis is the same range at
    every mass point (bbWW's HME runs 0-1500 GeV), so the signal occupies a narrow
    window whose position moves with MX while the axis does not; binning must follow
    the signal rather than the axis. Equal-signal quantiles do that automatically -- bin edges
    cluster wherever dS/dm is large (the peak) and a single wide bin absorbs the
    long empty stretches on either side.

    The CDF clamps negative bin contents to zero so it stays monotonic; a
    statistical undershoot in a signal MC bin must not move an edge backwards.
    """
    n_avail = last_bin - first_bin + 1
    n_bins = max(1, min(n_bins, n_avail))
    if n_bins == 1:
        return [(first_bin, last_bin)]

    cumulative = []
    running = 0.0
    for b in range(first_bin, last_bin + 1):
        running += max(sig_hist.GetBinContent(b), 0.0)
        cumulative.append(running)
    total = running
    if total <= 0:
        return [(first_bin, last_bin)]

    ranges = []
    lo = first_bin
    for k in range(1, n_bins):
        target = k * total / n_bins
        b = lo
        while b < last_bin and cumulative[b - first_bin] < target:
            b += 1
        # every one of the n_bins-k ranges still to come needs at least one bin
        b = max(lo, min(b, last_bin - (n_bins - k)))
        ranges.append((lo, b))
        lo = b + 1
    ranges.append((lo, last_bin))
    return ranges


def merge_until_valid(
    ranges,
    sig_hist,
    bkg_hists_by_name,
    min_each,
    exempt=(),
    min_neff=0.0,
    min_proc_neff=0.0,
):
    """Merge adjacent ranges until every one satisfies the background gates.

    The signal quantiles decide where the edges want to be; this decides how many
    of them the background statistics can actually support. A failing range is
    merged into whichever neighbour holds *less* signal, so the dense bins around
    the peak -- the ones carrying the discrimination -- are the last to be given
    up. Terminates because every step removes one range, ending at the single
    full-range bin, which has nothing left to fail against.
    """
    ranges = list(ranges)
    while len(ranges) > 1:
        bad = None
        for i, (lo, hi) in enumerate(ranges):
            if not _bin_passes(
                _bkg_yields(bkg_hists_by_name, lo, hi),
                min_each,
                exempt,
                _total_bkg_error(bkg_hists_by_name, lo, hi),
                min_neff,
                _bkg_errors(bkg_hists_by_name, lo, hi) if min_proc_neff > 0 else None,
                min_proc_neff,
            ):
                bad = i
                break
        if bad is None:
            break
        if bad == 0:
            other = 1
        elif bad == len(ranges) - 1:
            other = bad - 1
        else:
            left_sig = sig_hist.Integral(*ranges[bad - 1])
            right_sig = sig_hist.Integral(*ranges[bad + 1])
            other = bad - 1 if left_sig <= right_sig else bad + 1
        first, second = min(bad, other), max(bad, other)
        ranges[first : second + 1] = [(ranges[first][0], ranges[second][1])]
    return ranges


def find_bins(
    sig_hist,
    bkg_hists_by_name,
    max_bins,
    first_bin,
    last_bin,
    min_each,
    min_frac=0.0,
    min_neff=0.0,
    min_proc_neff=0.0,
):
    """Bins inside one slice: signal quantiles for the edges, background gates for
    the count.

    The two axes need opposite rules, which is why this is not find_slices(). The
    previous version scanned right-to-left growing each bin leftward until the
    backgrounds cleared their floors, mirroring find_slices(). That is right for the
    sliced axis, where the signal piles up at one end so resolution belongs there, and
    wrong for an axis whose signal sits in the middle with both tails empty -- as
    bbWW's HME does, peaking at HME ~ MX.
    Starting from the top of the axis spent the bin budget on the empty
    upper tail and left the entire resonance peak in the single leftover bin:
    at MX=600 in muMu/res2b_dnn3, one bin covered 0-710 GeV holding 97.5% of the
    signal while five bins shared the 710-1500 GeV region holding 2.5%. The fit
    then had no shape to work with in the only region where signal and background
    differ.
    """
    # which backgrounds count as negligible is decided once, over the whole slice
    exempt = minor_backgrounds(bkg_hists_by_name, first_bin, last_bin, min_frac)
    ranges = signal_quantile_ranges(sig_hist, max_bins, first_bin, last_bin)
    return merge_until_valid(
        ranges, sig_hist, bkg_hists_by_name, min_each, exempt, min_neff, min_proc_neff
    )


def _discover_greedy(sig2d, bkg2d_by_name, knobs):
    """bkg2d_by_name: {background_name: [hist per discovery era, ...]}. The list
    is usually a single era's own histogram (standalone limit) or all of a
    meta-era's sub-eras (combined limit) -- see --discovery-eras.
    Yields are summed across whatever's in the list; see _bkg_yields(). sig2d is
    the same discovery reference's (already-summed) signal histogram: its x
    projection picks the significance-maximizing slice boundaries, and its y
    projection within each slice places the mass bin edges by signal quantile."""
    n_slices = knobs["n_slices"]
    max_bins_per_slice = knobs["max_bins_per_slice"]
    min_slice_sum = knobs["min_slice_bkg_sum"]
    min_bin_each = knobs["min_bin_bkg_each"]
    min_slice_bkg_neff = knobs["min_slice_bkg_neff"]
    min_bkg_frac = knobs["min_bkg_frac"]
    min_bin_bkg_neff = knobs["min_bin_bkg_neff"]
    bkg_per_bin = knobs["bkg_per_bin"]
    sig_mode = knobs["significance_mode"]
    min_slice_bkg_each = knobs["min_slice_bkg_each"]
    min_slice_bkg_each_neff = knobs["min_slice_bkg_each_neff"]
    min_bin_bkg_each_neff = knobs["min_bin_bkg_each_neff"]

    any_hist = next(iter(bkg2d_by_name.values()))[0]
    nx = any_hist.GetNbinsX()
    ny = any_hist.GetNbinsY()
    slices = find_slices(
        sig2d,
        bkg2d_by_name,
        n_slices,
        1,
        nx,
        min_slice_sum,
        min_slice_bkg_neff,
        sig_mode,
        min_slice_bkg_each,
        min_slice_bkg_each_neff,
        min_bkg_frac,
    )
    if any(sl is None for sl in slices):
        # The axis could not carry n_slices boundaries: the significance scan reached
        # first_bin with slices still to place, usually because no window anywhere clears
        # the background floors. There is no honest binning to return -- an exhausted
        # slice is neither a selection nor an empty one, since every mass point has to
        # share the same category list -- so the category is refused here and skipped by
        # the caller, the same way one with too little signal is.
        return None
    slices = extend_outer_edges(slices, 0, nx + 1)

    result = []
    for xlo, xhi in slices:
        # ProjectionY attaches its result to gDirectory (here, the open output
        # file); detach so these die with the loop iteration instead of piling up
        # in the output file's in-memory object list.
        bkg_y_by_name = {
            name: [
                _detach(h.ProjectionY(f"_disc_{name}_{xlo}_{xhi}_{i}_y", xlo, xhi, "e"))
                for i, h in enumerate(hists)
            ]
            for name, hists in bkg2d_by_name.items()
        }
        sig_y = _detach(sig2d.ProjectionY(f"_disc_sig_{xlo}_{xhi}_y", xlo, xhi, "e"))
        n_bins = bin_budget(bkg_y_by_name, 1, ny, max_bins_per_slice, bkg_per_bin)
        bin_ranges = find_bins(
            sig_y,
            bkg_y_by_name,
            n_bins,
            1,
            ny,
            min_bin_each,
            min_bkg_frac,
            min_bin_bkg_neff,
            min_bin_bkg_each_neff,
        )
        bin_ranges = extend_outer_edges(bin_ranges, 0, ny + 1)
        result.append({"x_range": (xlo, xhi), "y_ranges": bin_ranges})
    return result


def _slice_dp_curve(cells, xlo, xhi, knobs, max_bins, check=True):
    """For one slice, the best mass-axis partition at every bin count up to max_bins.

    The edges come from an exact search against the same figure of merit that already
    chooses the slice boundaries, instead of from equal-signal quantiles. The quantile
    rule put the edges where the signal is, which is right as far as it goes and was a
    real improvement over scanning the axis from the top -- but it never looks at the
    background, so it cannot tell the steep side of the resonance from the flat one, and
    it cannot decline a split that gains nothing.

    Validity is a mask on the search rather than a term in the score: a bin whose
    background has gone negative is not a bad bin to be outweighed, it is one the
    datacard cannot contain.

    `exempt` is judged once over this slice's whole mass range, never inside the
    candidate bin -- see minor_backgrounds(); judging it per candidate is circular,
    because a background that is zero in the bin under test is trivially below any
    fraction of that bin's total.

    y = 1..ny, *not* the full axis, because that is the range the greedy path judges this
    on: it works from ProjectionY histograms and calls minor_backgrounds(h, 1, ny), so the
    y under/overflow rows are outside the fraction. They are inside the bins that
    eventually get written -- extend_outer_edges pushes them into the first and last -- so
    the two are not quite the same question. Matching the greedy path is deliberate: the
    point of this strategy is to change how the edges are chosen, and a quietly different
    exempt set would mean the comparison was measuring two things at once.
    """
    ny = cells.ny
    exempt = cells.exempt(xlo, xhi, 1, ny, knobs["min_bkg_frac"])
    mode = knobs["significance_mode"]

    def bin_passes(x0, x1, a, b, exempt_set):
        """The canonical gate, for the cross-check below -- and the definition the
        vectorised tables have to agree with."""
        return _bin_passes(
            cells.yields(x0, x1, a, b),
            knobs["min_bin_bkg_each"],
            exempt_set,
            cells.total_bkg_error(x0, x1, a, b),
            knobs["min_bin_bkg_neff"],
            cells.errors(x0, x1, a, b),
            knobs["min_bin_bkg_each_neff"],
        )

    score, valid = slice_tables(cells, xlo, xhi, exempt, knobs, mode)
    if check:
        check_mask_against_gate(
            cells,
            xlo,
            xhi,
            valid,
            score,
            exempt,
            bin_passes,
            mode,
        )
    values, partitions = partition_dp(score, valid, max_bins)
    # back from working indices to bin numbers
    ranges = [None if p is None else [(a + 1, b + 1) for a, b in p] for p in partitions]
    return values, ranges


def _best_count(values, cap):
    """The bin count at or below `cap` that actually scores best, or None if none is
    feasible.

    Not the largest feasible count, which is the obvious choice and the wrong one. With
    the background uncertainty folded into the figure of merit -- which is what
    significance() does, and the reason it does not reward a background that fluctuated
    low -- splitting a bin is no longer guaranteed to raise Z^2: each half carries a
    larger relative sigma_B, and for a slice that is already statistics-limited the
    split can cost more than the extra shape information gains. Taking the largest
    feasible count would then hand back a binning worse than the greedy one it is
    supposed to improve on.

    Choosing the argmax instead makes "at least as good as greedy" true by construction
    rather than by hope: greedy's own partition uses some count m <= cap and passes the
    same gates, so values[m] is one of the candidates here and the winner is at least
    that. It also tends to spend fewer bins than the budget allows, which is the right
    direction -- an extra bin that buys nothing is pure MC-statistical exposure.
    """
    best, best_n = -np.inf, None
    for n in range(1, min(cap, len(values) - 1) + 1):
        if values[n] > best:
            best, best_n = values[n], n
    return best_n


def _y_step(cells, slices, knobs):
    """The mass binning of every slice, and the bin count each one keeps.

    Budgets first, then whichever of them the value of a bin actually justifies. The
    order matters: dp_min_bin_gain only ever gives bins back, so it is applied to the
    counts the budget allowed rather than being a second thing competing with it.
    """
    max_bins = knobs["max_bins_per_slice"]
    curves, budgets = [], []
    for xlo, xhi in slices:
        values, ranges = _slice_dp_curve(cells, xlo, xhi, knobs, max_bins)
        curves.append((values, ranges))
        # y = 1..ny because bin_budget() is handed a ProjectionY and asked for (1, ny),
        # so this is the same yield it would divide
        budgets.append(
            bins_for_background(
                cells.total_bkg(xlo, xhi, 1, cells.ny),
                max_bins,
                knobs["bkg_per_bin"],
            )
        )

    if knobs["dp_budget_mode"] == "shared":
        counts = allocate_budget(
            [values for values, _ in curves], sum(budgets), n_min=1, n_max=max_bins
        )
        if counts is None:
            counts = [None] * len(slices)
    else:
        counts = [None] * len(slices)

    chosen = [
        _best_count(values, counts[i] if counts[i] is not None else budgets[i])
        for i, (values, _) in enumerate(curves)
    ]
    reference = sum(curves[i][0][n] for i, n in enumerate(chosen) if n is not None)
    if knobs["dp_min_bin_gain"] > 0 and reference > 0:
        chosen = [
            None if n is None else m
            for n, m in zip(
                chosen,
                trim_by_marginal_gain(
                    [values for values, _ in curves],
                    [n if n is not None else 1 for n in chosen],
                    knobs["dp_min_bin_gain"] * reference,
                ),
            )
        ]

    partitions = []
    for idx, (values, ranges) in enumerate(curves):
        n = chosen[idx]
        if n is None:
            # Not even one bin covering the whole mass axis clears the gates. The greedy
            # path keeps such a slice anyway, and matching that is deliberate:
            # merge_until_valid() loops `while len(ranges) > 1`, so once it has merged
            # down to a single range it returns it *without* testing it. Refusing here
            # instead would drop categories that production currently builds, and since
            # every mass point has to share one category list, dropping one at one mass
            # is not a stricter binning, it is a broken datacard.
            partitions.append([(1, cells.ny)])
        else:
            partitions.append(list(ranges[n]))
    return partitions


def _x_step(cells, partitions, knobs, exempt_full):
    """Re-place the slice boundaries with the mass binnings held fixed.

    Exact over the whole partition, which is the difference from grow_slice(): that picks
    each boundary to maximise the significance of the one slice it is closing off, taking
    no view on what the remaining axis will be worth. Here every boundary is chosen
    against the total.

    Returns ranges already widened into under/overflow, because slice_axis_tables() scored
    them that way -- the outermost slices are worth what they are worth *with* the events
    that will be written into them.
    """
    score, valid = slice_axis_tables(
        cells,
        partitions,
        knobs,
        knobs["significance_mode"],
        exempt_full,
    )
    parts = partition_dp_staged(score, valid, len(partitions))
    if parts is None:
        return None
    slices = [(i + 1, j + 1) for i, j in parts]
    return extend_outer_edges(slices, 0, cells.nx + 1)


def _objective(cells, slices, partitions, mode):
    """Total Z^2 of a binning, scored on the ranges that would actually be written."""
    return sum(
        cells.score(xlo, xhi, a, b, mode)
        for (xlo, xhi), partition in zip(slices, partitions)
        for a, b in partition
    )


def _discover_dp(sig2d, bkg2d_by_name, knobs):
    """Both axes, alternated until neither can improve the other.

    Seeded from the greedy slice boundaries plus one mass-axis pass, which is what makes
    the result at least as good as the greedy binning rather than merely different: the
    seed is a feasible point, and nothing below is ever accepted unless it scores higher.

    Each step is an exact maximisation of the same total given the other axis, so this is
    block coordinate ascent with globally optimal coordinate steps -- not a heuristic
    relaxation. It converges to a point where neither axis can improve on the other, which
    is not necessarily the global optimum of the pair.

    The accept test is on the objective as finally scored, not on whatever each step
    maximised internally. The two are almost the same thing but not exactly: the mass-axis
    step deliberately judges its gates over y = 1..ny to match the greedy path, while the
    written bins carry the y under/overflow. Comparing the realised objective makes the
    ascent monotone whatever those conventions are, instead of relying on them agreeing.
    """
    any_hist = next(iter(bkg2d_by_name.values()))[0]
    nx = any_hist.GetNbinsX()
    slices = find_slices(
        sig2d,
        bkg2d_by_name,
        knobs["n_slices"],
        1,
        nx,
        knobs["min_slice_bkg_sum"],
        knobs["min_slice_bkg_neff"],
        knobs["significance_mode"],
        knobs["min_slice_bkg_each"],
        knobs["min_slice_bkg_each_neff"],
        knobs["min_bkg_frac"],
    )
    if any(sl is None for sl in slices):
        return None
    slices = extend_outer_edges(slices, 0, nx + 1)

    cells = build_cells(sig2d, bkg2d_by_name)
    mode = knobs["significance_mode"]
    # the slice gates use the exempt set judged once over the whole axis, as find_slices
    # does -- a background negligible in this category must not veto every boundary
    exempt_full = cells.exempt(1, nx, 0, cells.ny + 1, knobs["min_bkg_frac"])

    partitions = _y_step(cells, slices, knobs)
    best = _objective(cells, slices, partitions, mode)
    iterations = 0

    for _ in range(max(0, knobs["dp_max_iterations"])):
        moved = _x_step(cells, partitions, knobs, exempt_full)
        if moved is None or moved == slices:
            break
        candidate = _y_step(cells, moved, knobs)
        value = _objective(cells, moved, candidate, mode)
        # strict improvement, not "the partition changed": the objective is what is
        # guaranteed to be monotone, the partition is not, and ties can cycle forever
        if value <= best * (1.0 + 1e-12):
            break
        slices, partitions, best = moved, candidate, value
        iterations += 1

    if iterations:
        print(f"      [dp] slice boundaries re-placed {iterations}x, Z^2 {best:.4f}")
    return [
        {
            "x_range": (xlo, xhi),
            "y_ranges": extend_outer_edges(list(partition), 0, cells.ny + 1),
        }
        for (xlo, xhi), partition in zip(slices, partitions)
    ]


def _fit_override(cells, y0, y1, knobs):
    """{process: (values, variances)} for this window, or {} when nothing is fitted.

    Built per candidate window: the window is the selection the fit describes, so it has
    to be re-derived as the box moves. Cheap -- one profile and a 3x3 solve.
    """
    names = [n for n in knobs["fit_tail_processes"] if n in cells.names]
    if not names or not knobs.get("_fit_tail_active"):
        return {}
    x = cells.x_centres
    out = {}
    for name in names:
        values, variances = cells.profile(name, y0, y1)
        v, var, n = fitted_tail(x, values, variances, knobs)
        if n:
            out[name] = (v, var)
    return out


def _box_value(cells, y0, y1, knobs, check=False):
    """What one HME box is worth: the best 1D binning of the DNN inside it.

    A box is scored by the binning it admits, not by its own integrated significance.
    Those are different questions and they pick different windows -- scored on its own, a
    box is rewarded for swallowing as much signal as it can and the DNN shape inside it
    plays no part, which is the opposite of what the fit will do with it.
    """
    mode = knobs["significance_mode"]
    # None asks box_tables() and the cross-check to judge exemption bin by bin.
    exempt = (
        None
        if knobs["exempt_scope"] == "bin"
        else cells.exempt(1, cells.nx, y0, y1, knobs["min_bkg_frac"])
    )
    override = _fit_override(cells, y0, y1, knobs)
    score, valid = box_tables(cells, y0, y1, exempt, knobs, mode, override)
    if check and not override:
        # the canonical scalar gate reads the histograms, so it cannot be asked about a
        # fitted profile; the cross-check still runs on every unfitted box.

        def bin_passes(a, b, ylo, yhi, exempt_set):
            return _bin_passes(
                cells.yields(a, b, ylo, yhi),
                knobs["min_bin_bkg_each"],
                exempt_set,
                cells.total_bkg_error(a, b, ylo, yhi),
                knobs["min_bin_bkg_neff"],
                cells.errors(a, b, ylo, yhi),
                knobs["min_bin_bkg_each_neff"],
                knobs["min_bin_bkg_neff_by_process"],
            )

        check_box_mask(
            cells, y0, y1, valid, score, exempt, bin_passes, mode,
            min_frac=knobs["min_bkg_frac"],
        )
    values, partitions = partition_dp(score, valid, knobs["max_bins_per_slice"])
    n = _best_count(values, knobs["max_bins_per_slice"])
    if n is None:
        return float("-inf"), None
    if knobs["dp_min_bin_gain"] > 0 and values[n] > 0:
        n = trim_by_marginal_gain([values], [n], knobs["dp_min_bin_gain"] * values[n])[
            0
        ]
    return values[n], [(a + 1, b + 1) for a, b in partitions[n]]


def _discover_hme_box(sig2d, bkg2d_by_name, knobs):
    """One HME window per category, with the DNN binned inside it.

    The other strategies cut the DNN into slices that each become a datacard category and
    give every slice a shape along HME. This one does the reverse and keeps only one
    category: a single contiguous HME box around the resonance, and a 1D DNN distribution
    inside it. Everything outside the box is dropped -- that is what makes it a cut rather
    than a slice, and it is why the box has to earn its acceptance loss in purity.

    Box and binning are chosen together; see _box_value.
    """
    cells = build_cells(sig2d, bkg2d_by_name)
    ny = cells.ny
    # where the signal actually is -- see box_signal_quantile
    cumulative = np.array(
        [cells.signal(0, cells.nx + 1, 1, j) for j in range(1, ny + 1)]
    )
    total = cumulative[-1] if len(cumulative) else 0.0
    if total <= 0:
        return None
    q = knobs["box_signal_quantile"]
    lo_edge = max(1, int(np.searchsorted(cumulative, q * total)))
    hi_edge = min(ny, int(np.searchsorted(cumulative, (1.0 - q) * total)) + 1)

    best_value, best_box = float("-inf"), None
    stride = max(1, int(knobs["box_stride"]))
    for a in range(lo_edge, hi_edge + 1, stride):
        for b in range(a, hi_edge + 1, stride):
            value, _ = _box_value(cells, a, b, knobs)
            if value > best_value:
                best_value, best_box = value, (a, b)
    if best_box is None:
        return None
    refine = max(0, int(knobs["box_refine"]))
    a0, b0 = best_box
    for a in range(max(lo_edge, a0 - refine), min(hi_edge, a0 + refine) + 1):
        for b in range(max(a, b0 - refine), min(hi_edge, b0 + refine) + 1):
            value, _ = _box_value(cells, a, b, knobs)
            if value > best_value:
                best_value, best_box = value, (a, b)

    y0, y1 = best_box
    value, bins = _box_value(cells, y0, y1, knobs, check=True)
    if bins is None:
        return None
    # A box touching an end of the axis takes that end's overflow with it: the events are
    # inside the selection the box stands for, and dropping them would make the written
    # yield disagree with the cut the category claims to be.
    y0 = 0 if y0 <= 1 else y0
    y1 = ny + 1 if y1 >= ny else y1
    return [{"y_range": (y0, y1), "x_ranges": bins}]


def discover_binning(sig2d, bkg2d_by_name, knobs):
    """Find this category's binning, by whichever strategy the configuration names.

    bkg2d_by_name: {background_name: [hist per discovery era, ...]}. The list is usually
    a single era's own histogram (standalone limit) or all of a meta-era's sub-eras
    (combined limit). Yields are summed across whatever is in the list; see _bkg_yields().
    sig2d is the same discovery reference's already-summed signal histogram.

    Returns None when the category cannot be binned, which the caller reports as a skip.
    """
    excluded = set(knobs["binning_exclude_processes"])
    if excluded:
        # Dropped here, once, rather than special-cased in every gate and every score
        # below. "Choose the edges as if this process were not there" is exactly the
        # statement "do not put it in the dictionary the chooser reads", and expressing it
        # that way means no downstream code has to know the feature exists -- including
        # the greedy strategy, which then gets it for free and stays comparable.
        kept = {n: h for n, h in bkg2d_by_name.items() if n not in excluded}
        if not kept:
            # Every background excluded: there is nothing left to place an edge against,
            # and a binning derived from signal alone would be a fit to the signal MC.
            raise RuntimeError(
                "binning_exclude_processes excludes every background this category has "
                f"({sorted(bkg2d_by_name)}), leaving nothing to bin against."
            )
        bkg2d_by_name = kept
    if knobs["strategy"] == "hme_box":
        return _discover_hme_box(sig2d, bkg2d_by_name, knobs)
    if knobs["strategy"] == "dp":
        return _discover_dp(sig2d, bkg2d_by_name, knobs)
    return _discover_greedy(sig2d, bkg2d_by_name, knobs)


def mkdir_titled(directory, path, title):
    """mkdir a nested path, putting `title` on the leaf directory only.

    TDirectory.mkdir() given a slashed path applies the title to the *first* level and
    hands the rest of the path down as the sub-levels' titles, so every parent would end
    up labelled with a stale fragment of whichever slice was created first ("muMu" titled
    "muMu/SR/res2b_dnn0"). Walking the components keeps the parents clean and puts the
    label where it belongs.
    """
    parts = path.split("/")
    for part in parts[:-1]:
        directory = directory.GetDirectory(part) or directory.mkdir(part)
    return directory.mkdir(parts[-1], title)


def format_var_range(lo, hi, var):
    """One slice's edges on the sliced axis as a selection label, e.g. "1.20 < DNN < 4.50".

    `var` names that axis and comes from the configuration -- this script does not assume
    the analysis slices on a DNN score.

    Formatted here, by the code that discovered the edges, so the label travels with the
    histograms and nothing downstream has to re-derive it. Both edges open means the slice
    covers the whole axis, i.e. there is no selection to state -- that returns an empty
    string rather than a vacuous label.
    """
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"{var} < {hi:.2f}"
    if hi is None:
        return f"{var} > {lo:.2f}"
    return f"{lo:.2f} < {var} < {hi:.2f}"


def slice_ranges(x_axis, y_axis, slices):
    """Physical edges of the discovered slices on the sliced axis, as [[lo, hi], ...].

    The slice x_ranges are bin indices, and extend_outer_edges() has already pushed the
    outermost ones into underflow/overflow -- those have no finite edge, so they are
    recorded as null and read back as an open-ended selection. Written out by run() so
    the plots can say which selection each slice actually is; nothing downstream of
    the datacards needs it.
    """
    ranges = []
    for sl in slices:
        (lo, hi), _, slice_axis = slice_parts(sl)
        axis = x_axis if slice_axis == "x" else y_axis
        n = axis.GetNbins()
        ranges.append(
            [
                None if lo < 1 else axis.GetBinLowEdge(lo),
                None if hi > n else axis.GetBinUpEdge(hi),
            ]
        )
    return ranges


def _slice_record(sl, sel_edges, x_axis, y_axis):
    """One category as plain data, in whichever layout produced it.

    The key names say which axis is the selection and which carries the bins, so a reader
    -- human or replay -- never has to infer it from the strategy that happened to run.
    """
    (lo, hi), bins, slice_axis = slice_parts(sl)
    binned_axis = y_axis if slice_axis == "x" else x_axis
    sel_key, bin_key = (
        ("x_range", "y_ranges")
        if slice_axis == "x"
        else (
            "y_range",
            "x_ranges",
        )
    )
    return {
        sel_key: [lo, hi],
        f"{slice_axis}_edges": sel_edges,
        bin_key: [list(r) for r in bins],
        ("y_edges" if slice_axis == "x" else "x_edges"): bin_edges(binned_axis, bins),
    }


def slices_to_record(slices, x_axis, y_axis, objective=None):
    """The discovered structure as plain data, for binning.json.

    Both forms are written. The bin index ranges are what the code actually cuts on and
    are what a replay restores; the physical edges alongside them are what a human reads,
    and what makes the record meaningful next to a plot.

    `objective` is what the binning scored, recomputed from these ranges rather than
    carried out of the optimiser -- so it is filled in for the greedy strategy too, which
    never computes it, and two runs can be compared from their records alone. It is
    written as a sibling of the slices and never read back: record_to_slices() takes only
    the ranges, so a record written before this existed still replays.
    """
    record = {
        "n_x_bins": x_axis.GetNbins(),
        "n_y_bins": y_axis.GetNbins(),
        "slices": [
            _slice_record(sl, sel_edges, x_axis, y_axis)
            for sl, sel_edges in zip(slices, slice_ranges(x_axis, y_axis, slices))
        ],
    }
    if objective is not None:
        total, per_slice = objective
        record["objective"] = {
            "z": math.sqrt(max(total, 0.0)),
            "z2": total,
            "z2_per_slice": per_slice,
            "n_bins": sum(len(slice_parts(sl)[1]) for sl in slices),
        }
    return record


def record_to_slices(record, x_axis, y_axis, where):
    """Inverse of slices_to_record(): what discover_binning() would have returned.

    The axis sizes are checked rather than trusted. Bin indices only mean anything against
    the axes they were found on, so a binning.json replayed over input with a different
    binning would otherwise cut the shapes in silently wrong places -- which is exactly
    what freezing a binning is supposed to rule out.
    """
    for name, axis, recorded in (
        ("x", x_axis, record["n_x_bins"]),
        ("y", y_axis, record["n_y_bins"]),
    ):
        if axis.GetNbins() != recorded:
            raise RuntimeError(
                f"{where}: recorded binning was found on a {recorded}-bin {name} axis, "
                f"but the input has {axis.GetNbins()}. The binning.json does not belong "
                "to this input."
            )
    return [
        (
            {
                "y_range": tuple(sl["y_range"]),
                "x_ranges": [tuple(r) for r in sl["x_ranges"]],
            }
            if "y_range" in sl
            else {
                "x_range": tuple(sl["x_range"]),
                "y_ranges": [tuple(r) for r in sl["y_ranges"]],
            }
        )
        for sl in record["slices"]
    ]


def _is_released(key, released):
    """Whether a histogram key belongs to one of the released processes.

    Keys are "DY" for the nominal and "DY_<nuisance>_<Up|Down>" for a variation, so a
    process owns its own name and anything under it with the separator. The separator is
    required so that releasing "DY" cannot also claim a process called "DYJets".
    """
    return any(key == name or key.startswith(name + "_") for name in released)


def slice_parts(sl):
    """(selection range, bin ranges, which axis the selection is on) for one category.

    Two layouts reach this point. The DNN-sliced strategies cut x into categories and give
    each a shape along y: {"x_range", "y_ranges"}. The HME-box strategy cuts y and gives
    the one category a shape along x: {"y_range", "x_ranges"}. Everything downstream --
    the writer, the record, the objective -- needs the same three things from either, and
    asking here is what keeps that code from having to know which strategy ran.
    """
    if "y_range" in sl:
        return sl["y_range"], sl["x_ranges"], "y"
    return sl["x_range"], sl["y_ranges"], "x"


def fitted_bins(cells, slices, knobs, name):
    """([(content, error, first_fitted_fine_bin)] per bin, last measured fine bin).

    Only the *fitted part* of each bin: the fine bins past the last measured one. The
    rest of the bin stays with the era's own histogram, so a well-measured region keeps
    its per-era shape and its systematic variations keep theirs -- replacing the whole
    template with a scaled copy of the summed one would erase both.

    The fit is derived from the discovery statistics -- the eras summed, which is what it
    is constrained by -- and handed to each source era scaled by that era's own share of
    the counted yield in the box. The share is a well-measured number even where the tail
    is not, so each era keeps its own normalisation while the shape comes from the fit.
    """
    (y0, y1), bin_ranges, slice_axis = slice_parts(slices[0])
    if slice_axis != "y":
        return None
    values, variances = cells.profile(name, max(y0, 0), min(y1, cells.ny + 1))
    fitted, fitted_var, n, info = fitted_tail_full(
        cells.x_centres, values, variances, knobs
    )
    if not n:
        return None
    replaced = set(info["replaced"])
    out = []
    for lo, hi in bin_ranges:
        fine = range(max(lo, 0), min(hi, cells.nx + 1) + 1)
        from_fit = [i for i in fine if i in replaced]
        if not from_fit:
            out.append((0.0, 0.0, None))
            continue
        # the fitted fine bins share the fit's three parameters, so their contributions
        # to this bin's integral are propagated together rather than in quadrature
        error = fitted_integral_error(
            cells.x_centres, info["p"], info["cov"], from_fit
        )
        out.append((float(sum(fitted[i] for i in from_fit)), error, min(from_fit)))
    return out


def rebin_hist_2d(hist2d, slices, name, naming, fill_negative=None, override=None):
    """Given the discovered slice structure, produce one final TH1 per slice
    for this specific histogram (nominal or a systematic variation).

    With fill_negative set, a bin whose content comes out negative is written as that
    value with a tenth of it as the error, rather than as the negative it is. combine
    cannot fit a negative background, and resolveNegativeBins rejects a negative bin
    outright once it holds enough of the signal to count as relevant -- so for a process
    released from the binning gates, which is where negative bins are expected rather than
    exceptional, this is what keeps the datacard buildable.

    Applied to every histogram of that process including its systematic variations,
    because a negative variation is no more usable than a negative nominal. Which
    processes get here at all is the caller's decision -- see _is_released().

    What it costs is worth stating: a filled bin claims a background of fill +- fill/10,
    i.e. known to 10%, when what was actually measured is a yield consistent with zero and
    of unknown sign. That is an assertion of knowledge the MC does not support, and
    autoMCStats will not widen it back out. It buys a fit that runs; it does not buy a
    background estimate."""
    outputs = []
    for slice_idx, sl in enumerate(slices):
        (sel_lo, sel_hi), bin_ranges, slice_axis = slice_parts(sl)
        xlo, xhi = (sel_lo, sel_hi) if slice_axis == "x" else (None, None)
        # ROOT reinterprets an inverted or negative range as the full axis including
        # under/overflow, silently and with a plausible-looking positive yield, so an
        # invalid range must never reach IntegralAndError below. Checked here rather
        # than trusted because this is the last point where it is still cheap to say so.
        if not 0 <= sel_lo <= sel_hi:
            raise RuntimeError(
                f"invalid {slice_axis} bin range ({sel_lo}, {sel_hi}) for slice "
                f"{slice_idx} of {name}; ROOT would read this as the whole plane."
            )
        binned_axis = hist2d.GetYaxis() if slice_axis == "x" else hist2d.GetXaxis()
        edges = array.array("d", bin_edges(binned_axis, bin_ranges))
        # Detached for the same reason as the projections above: Write() targets
        # gDirectory regardless, so nothing needs these to stay attached.
        h = _detach(
            ROOT.TH1D(
                naming.name(name, slice_idx),
                name,
                len(edges) - 1,
                edges,
            )
        )
        for bin_idx, (blo, bhi) in enumerate(bin_ranges, start=1):
            if override is not None:
                fit_content, fit_error, first_fitted = override[bin_idx - 1]
                if first_fitted is None:
                    pass  # nothing fitted in this bin: counted, as any other histogram
                else:
                    content, error = fit_content, fit_error
                    if first_fitted > blo:
                        # the bin straddles the boundary: the measured part comes from
                        # this era's own histogram, the rest from the fit
                        err = array.array("d", [0.0])
                        counted = hist2d.IntegralAndError(
                            blo, first_fitted - 1, sel_lo, sel_hi, err
                        )
                        content += counted
                        error = math.sqrt(error**2 + err[0] ** 2)
                    h.SetBinContent(bin_idx, content)
                    h.SetBinError(bin_idx, error)
                    continue
            err = array.array("d", [0.0])
            # the selection is on one axis and the bin on the other; IntegralAndError
            # always wants (x_lo, x_hi, y_lo, y_hi)
            if slice_axis == "x":
                content = hist2d.IntegralAndError(sel_lo, sel_hi, blo, bhi, err)
            else:
                content = hist2d.IntegralAndError(blo, bhi, sel_lo, sel_hi, err)
            error = err[0]
            if fill_negative is not None and content < 0:
                content = fill_negative
                error = fill_negative / 10.0
            h.SetBinContent(bin_idx, content)
            h.SetBinError(bin_idx, error)
        outputs.append(h)
    return outputs


def as_2d(hist):
    """A 1D input as the 2D shape the binning reads: x is the input's own axis, and y a
    single bin standing for the whole of the axis the input does not have.

    This is how a 1D discriminant takes the same binning as the 2D DNN-vs-HME shapes
    with nothing written twice. Every strategy then sees one y bin. hme_box's box can
    only be that bin, which it widens to take the overflow, so what is left is its DNN
    binning inside the box -- the same partition, floors and marginal-gain trim -- on
    the full input. The written shapes are 1D along x, as they are from a 2D input.

    2D histograms pass through; anything else (None, 3D) comes back as None, which the
    callers already treat as absent.
    """
    if hist is None:
        return None
    if hist.GetDimension() == 2:
        return hist
    if hist.GetDimension() != 1:
        return None
    x_axis = hist.GetXaxis()
    nx = x_axis.GetNbins()
    edges = array.array("d", [x_axis.GetBinLowEdge(i) for i in range(1, nx + 2)])
    h2 = ROOT.TH2D(
        f"{hist.GetName()}_as2d",
        hist.GetTitle(),
        nx,
        edges,
        1,
        array.array("d", [0.0, 1.0]),
    )
    h2.SetDirectory(0)
    h2.Sumw2()
    h2.GetXaxis().SetTitle(x_axis.GetTitle())
    for i in range(0, nx + 2):
        h2.SetBinContent(i, 1, hist.GetBinContent(i))
        h2.SetBinError(i, 1, hist.GetBinError(i))
    h2.SetEntries(hist.GetEntries())
    return h2


def process_category(
    sources,
    channel,
    category,
    cfg,
    mass,
    era,
    discovery_files,
    knobs,
    frozen=None,
    replaying=False,
):
    """Rebin one channel/category, returning the binning it used (None if skipped).

    The edges are discovered once from `discovery_files` -- every source era summed -- and
    then applied to each source era separately, so a group era's members are cut on the
    combination's statistics but stay in their own files. The datacard step still sums
    them, which is what keeps a per-era lnN expressible: scaling one sub-era's own shape
    needs that sub-era to still exist as a shape.

    `sources` is [(source_era, in_file, out_file)]; for a plain era there is one.

    With `replaying` set the edges come from a previous run's binning.json and no
    optimisation happens at all: the discovery files are then only read for the axes,
    the content gates are not applied, and `frozen` being None means the record does not
    hold this category -- which is a skip, not a licence to derive one.
    """
    min_signal = knobs["min_signal"]
    # The resonance parameter is named by the configuration, not by this script -- it is
    # "MX" for bbWW but the slicing knows nothing about which parameter it is scanning.
    param_name = cfg["signal_param_name"]
    # Read from where the input keeps this category for this mass, written under the
    # category's own name -- so the output, binning.json and the datacard bins are the
    # same for every mass even when the input's directory is not (see InputCategories).
    in_dir = f"{channel}/{cfg['input_categories'].path(category, {param_name: mass})}"
    prefix = f"{in_dir}/"
    # The key list comes from the first source era; a systematic that only some eras carry
    # is filled in from their nominal when the slices are written below.
    cat_dir = sources[0][1].Get(in_dir)
    if not cat_dir:
        print(f"  [skip] {in_dir}: not found in {sources[0][1].GetName()}")
        return

    signal_keys = [
        applyParameters(pattern, {cfg["signal_param_name"]: mass})
        for pattern in cfg["signal_hist_name_patterns"]
    ]
    background_names = [
        hist_name
        for (base_name, hist_name, allowed_channels) in cfg["background_entries"]
        if not allowed_channels or channel in allowed_channels
    ]

    def load2d(f, key):
        return as_2d(get_hist(f, prefix + key))

    disc_sig = sum_hists(
        [load2d(f, key) for f in discovery_files for key in signal_keys]
    )
    disc_bkg_by_name = {}
    for bkg_key in background_names:
        per_era = [load2d(f, bkg_key) for f in discovery_files]
        per_era = [h for h in per_era if h is not None]
        if len(per_era) == len(discovery_files):
            disc_bkg_by_name[bkg_key] = per_era

    sig_integral = disc_sig.Integral() if disc_sig is not None else 0
    if replaying:
        # A replay reproduces a recorded category set exactly, or it is not a replay.
        # The gates below decide which categories get binned; re-running them against a
        # new input can only disagree with the record -- a category whose signal has
        # since dipped under min_signal would be dropped even though its edges are
        # written down, which is precisely the drift freezing exists to prevent.
        if frozen is None:
            print(
                f"    [skip] {channel}/{category} {param_name}={mass}: not in the "
                "binning record, so it was skipped by the run that wrote it; skipped "
                "again rather than optimised afresh."
            )
            return
        if disc_sig is None:
            raise RuntimeError(
                f"{era}/{param_name}={mass}/{channel}/{category} is in the binning "
                "record, but its signal histogram is missing from the input, so the "
                "axes the recorded bin indices refer to cannot be read."
            )
    elif disc_sig is None or sig_integral < min_signal or not disc_bkg_by_name:
        if not disc_bkg_by_name:
            reason = "no background histograms found in all discovery eras"
        else:
            reason = f"signal too small for discovery ({sig_integral} < {min_signal})"
        print(
            f"    [skip] {channel}/{category} {param_name}={mass}: {reason}, skipping"
        )
        return

    where = f"{era}/{param_name}={mass}/{channel}/{category}"
    if frozen is not None:
        slices = record_to_slices(
            frozen, disc_sig.GetXaxis(), disc_sig.GetYaxis(), where
        )
    else:
        slices = discover_binning(disc_sig, disc_bkg_by_name, knobs)
        if slices is None:
            print(
                f"    [skip] {channel}/{category} {param_name}={mass}: the "
                f"{knobs['n_slices']} slices could not be placed -- the axis was "
                "exhausted before the last boundary, so no window clears the background "
                "floors. Lower n_slices or the floors for this configuration."
            )
            return

    # The selection each slice stands for is otherwise nowhere in the output: the sliced
    # categories are named by index and the surviving axis is the rebinned one. It is the
    # slice directory's own title, so it travels with the histograms and there is no
    # side-car path to hand a reader correctly and no key name for the two ends to agree
    # on -- anything that can open the shapes can already read it.
    naming = cfg["naming"]
    ranges = slice_ranges(disc_sig.GetXaxis(), disc_sig.GetYaxis(), slices)

    # Processes whose tail is written from the fit rather than counted, and the summed
    # counted yield each one has in the box, which is what the per-era scaling is against.
    fit_bins, summed_nominal, fit_err_norm = {}, {}, {}
    if knobs.get("_fit_tail_active") and len(slices) == 1:
        cells_for_fit = build_cells(disc_sig, disc_bkg_by_name)
        (fy0, fy1), _, faxis = slice_parts(slices[0])
        for name in knobs["fit_tail_processes"]:
            if name not in disc_bkg_by_name or faxis != "y":
                continue
            template = fitted_bins(cells_for_fit, slices, knobs, name)
            if template is None:
                continue
            values, _ = cells_for_fit.profile(
                name, max(fy0, 0), min(fy1, cells_for_fit.ny + 1)
            )
            fit_bins[name] = template
            summed_nominal[name] = float(values.sum())
            # Each source era is written as the same fitted shape scaled by its own share
            # of the counted yield, so the four templates' errors are fully correlated --
            # but the datacard sums the eras and adds their errors in quadrature. Scaling
            # each era's error by f/sqrt(sum f^2) rather than by f makes that quadrature
            # sum come out at the fit's actual uncertainty instead of ~half of it.
            fractions = []
            for _, era_file, _ in sources:
                era_hist = as_2d(get_hist(era_file, prefix + name))
                fractions.append(
                    era_hist.Integral(0, era_hist.GetNbinsX() + 1, fy0, fy1)
                    / summed_nominal[name]
                    if era_hist is not None and summed_nominal[name]
                    else 0.0
                )
            norm = math.sqrt(sum(f * f for f in fractions))
            fit_err_norm[name] = norm if norm > 0 else 1.0
            fitted_total = sum(c for c, _, first in template if first is not None)
            print(
                f"    [fit] {channel}/{category} {param_name}={mass}: {name} tail fitted "
                f"in {sum(1 for _, _, first in template if first is not None)} of "
                f"{len(template)} bins, {fitted_total:.2f} events of a box yield "
                f"{summed_nominal[name]:.2f}"
            )

    # One shared set of edges, applied to every source era in its own file.
    for source_era, in_file, out_file in sources:
        for slice_idx in range(len(slices)):
            mkdir_titled(
                out_file,
                f"{channel}/{naming.name(category, slice_idx)}",
                format_var_range(*ranges[slice_idx], var=cfg["slice_var"]),
            )
        cat_dir = in_file.Get(in_dir)
        if not cat_dir:
            print(f"  [skip] {source_era} {in_dir}: not in the input")
            continue
        for key in [k.GetName() for k in cat_dir.GetListOfKeys()]:
            hist2d = as_2d(get_hist(in_file, prefix + key))
            if hist2d is None:
                continue
            override = None
            base_proc = key.split("_")[0] if "_" in key else key
            if fit_bins and base_proc in fit_bins:
                template = fit_bins[base_proc]
                # This era's share of the counted yield in the box -- and for a
                # systematic variation, its size relative to the nominal, so the
                # variation keeps the effect it describes where the fit supplies the
                # shape. Errors scale by f/sqrt(sum f^2) rather than f: every era carries
                # the same fitted shape, so their errors are fully correlated, while the
                # datacard adds them in quadrature when it sums the eras.
                (ylo, yhi), _, _ = slice_parts(slices[0])
                era_total = hist2d.Integral(0, hist2d.GetNbinsX() + 1, ylo, yhi)
                scale = (
                    era_total / summed_nominal[base_proc]
                    if summed_nominal.get(base_proc)
                    else 0.0
                )
                err_scale = scale / fit_err_norm[base_proc]
                override = [
                    (c * scale, e * err_scale, first) for c, e, first in template
                ]
            for slice_idx, h in enumerate(
                rebin_hist_2d(
                    hist2d,
                    slices,
                    key,
                    naming,
                    (
                        knobs.get("fill_negative_bins")
                        if _is_released(
                            key, knobs.get("binning_exclude_processes") or []
                        )
                        else None
                    ),
                    override,
                )
            ):
                out_file.cd(f"{channel}/{naming.name(category, slice_idx)}")
                h.Write(key)

    # Scored from the ranges that were just written, for both strategies and for a replay
    # alike, so the number in the record always describes the shapes beside it. Cheap
    # next to the search, and the greedy path has no other way to report what it achieved.
    # Only when there is something to score. A replay is allowed to proceed with a
    # background missing from the input -- it applies recorded edges and asks no
    # questions -- and a figure of merit computed against no background would be a
    # number, but not a true one.
    objective = (
        binning_objective(
            build_cells(disc_sig, disc_bkg_by_name), slices, knobs["significance_mode"]
        )
        if disc_bkg_by_name
        else None
    )
    return slices_to_record(slices, disc_sig.GetXaxis(), disc_sig.GetYaxis(), objective)


def run(
    input_dir,
    output_dir,
    config_path,
    era,
    knobs,
    frozen_binning=None,
):
    """Produce one era of the datacard configuration's `eras:` list.

    Which era it is decides everything. A plain era is binned on its own statistics, for
    a standalone limit. An era that is a key of `era_groups:` is binned on its members'
    summed statistics -- a combination supports finer bins than any single era can -- and
    those edges are then applied to each member separately.

    Output layout is "<output_dir>/<source_era>/<variable>/<variable>.root", plus the
    binning.json that produced it. One era per --output directory, since the same source
    era carries different edges under different target eras and they must not overwrite
    each other. The members are kept in their own files rather than summed here: the
    datacard step sums them, and a per-era lnN can only be built by scaling a sub-era's
    own shape, which a pre-summed shape no longer has.
    """
    cfg = load_config(config_path)
    # A replay is bound by the record, not by the current knobs. The pattern the shapes
    # were written with and the axis they were sliced on are properties of that binning,
    # and binning.json stores both precisely so a replay does not depend on being handed
    # the same --binning-config again. Taking them from the knobs meant a standalone
    # replay without it silently wrote its categories under different names.
    pattern = knobs["category_pattern"]
    slice_var = knobs["slice_var"]
    if frozen_binning is not None:
        pattern = frozen_binning.get("category_pattern", pattern)
        slice_var = frozen_binning.get("slice_var", slice_var)
    cfg["naming"] = CategoryNaming(pattern)
    cfg["slice_var"] = slice_var

    # Both configurations name the pattern, and both carry a comment saying they must
    # agree -- so check it rather than trust it. A mismatch does not raise on its own:
    # base() simply fails to take the datacard config's names apart, every slice becomes
    # its own base category, and the shapes come out under names the datacard step will
    # not find.
    config_pattern = cfg.get("category_pattern")
    if config_pattern is not None and config_pattern != cfg["naming"].pattern:
        raise RuntimeError(
            f"category_pattern mismatch: the datacard configuration {config_path} says "
            f"'{config_pattern}', this binning says '{cfg['naming'].pattern}'. They name "
            "the same categories from the two ends and have to be identical."
        )
    # The datacard configuration lists the sliced names this script *writes*
    # ("SR/res2b_dnn0"); the 2D input it *reads* is keyed by the base categories those
    # slices are cut from ("SR/res2b"). Recover them with the same pattern that names
    # them, so the two ends cannot disagree about which is which. A configuration that
    # lists base names already is unchanged by this -- base() returns an unsliced name
    # as-is.
    cfg["categories"] = list(
        dict.fromkeys(cfg["naming"].base(c) for c in cfg["categories"])
    )
    model = cfg["model"]

    # A group era is built from its members; a plain era is built from itself.
    source_eras = cfg["era_groups"].get(era, [era])
    if era in cfg["era_groups"]:
        print(f"{era} is a group of {source_eras}: binning on their summed statistics")
    else:
        print(f"{era} is a standalone era: binning on its own statistics")

    record = {
        "slice_var": knobs["slice_var"],
        "category_pattern": cfg["naming"].pattern,
        "knobs": {k: v for k, v in sorted(knobs.items())},
        "binning": {},
    }
    by_mass = record["binning"].setdefault(era, {})

    for mass in cfg["mass_values"]:
        # One handle per source era, used both to discover the edges and to write the
        # shapes -- so the binning is derived from exactly the statistics it is applied to.
        sources = []
        for src in source_eras:
            in_file = open_input_file(
                input_dir, model, src, mass, cfg["signal_param_name"]
            )
            # getInputFileName() yields "<src>/<variable>/<variable>.root", which is
            # the layout input_file_pattern resolves against. --output is one era's own
            # directory, so the era being produced is not repeated inside it.
            out_path = os.path.join(
                output_dir,
                model.getInputFileName(src, {cfg["signal_param_name"]: mass}),
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sources.append((src, in_file, ROOT.TFile.Open(out_path, "RECREATE")))
        discovery_files = [f for _, f, _ in sources]

        print(
            f"Rebinning {era} {cfg['signal_param_name']}={mass} from "
            f"{len(sources)} source era(s) -> "
            f"{os.path.join(output_dir, era)}"
        )
        by_channel = by_mass.setdefault(str(mass), {})
        for channel in cfg["channels"]:
            for category in cfg["categories"]:
                frozen = None
                if frozen_binning is not None:
                    frozen = lookup_frozen(frozen_binning, era, mass, channel, category)
                # Whether this category's fit-tail processes are fitted here. Decided
                # by the category, so a configuration can fit where DY is measurable
                # (res2b) and count it where it is not (boosted).
                cat_knobs = dict(
                    knobs,
                    _fit_tail_active=bool(knobs["fit_tail_processes"])
                    and category in knobs["fit_tail_categories"],
                )
                used = process_category(
                    sources,
                    channel,
                    category,
                    cfg,
                    mass,
                    era,
                    discovery_files,
                    cat_knobs,
                    frozen=frozen,
                    replaying=frozen_binning is not None,
                )
                if used is not None:
                    by_channel.setdefault(channel, {})[category] = used

        for _, in_file, out_file in sources:
            out_file.Close()
            in_file.Close()

    # Written beside the shapes it describes, in the same output as them: the binning is
    # a product of this run, not configuration, so it travels with what it produced and a
    # reader never has to work out which binning a set of shapes came from.
    if not any(
        by_ch for by_mass_ in record["binning"].values() for by_ch in by_mass_.values()
    ):
        raise RuntimeError(
            f"{era}: every channel/category was skipped, so no shapes were written and "
            "the binning record is empty. The [skip] lines above say why -- usually the "
            "signal is below min_signal, or a background is missing from a discovery "
            "era. Exiting 0 here would hand the datacard step an empty input."
        )

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, BINNING_JSON)
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebin 2D histograms into significance-sliced 1D shapes.\n\n"
        "A standalone pre-step, not part of the datacard chain: it writes a "
        "'<era>/<variable>/<variable>.root' tree in the same layout HistMergerTask "
        "produces, so a production of these is consumed by pointing the chain's "
        "--hists-version at it. It also writes " + BINNING_JSON + ", which --binning "
        "replays to reproduce a binning instead of re-deriving one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="base directory containing <era>/<var>/<var>.root",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="output base directory, mirrors --input layout",
    )
    parser.add_argument(
        "--config", required=True, type=str, help="datacard configuration yaml"
    )
    parser.add_argument(
        "--binning-config",
        required=False,
        type=str,
        default=None,
        help="binning yaml holding the knobs below; anything it does not set takes the "
        "built-in default, and an explicit flag overrides both",
    )
    parser.add_argument(
        "--binning",
        required=False,
        type=str,
        default=None,
        help=f"a previous run's {BINNING_JSON}. Given one, the recorded edges are applied "
        "as-is and nothing is optimised -- this is how a binning is frozen and a "
        "production reproduced. The knobs are then unused",
    )
    parser.add_argument(
        "--era",
        required=True,
        type=str,
        help="the era of the datacard configuration's `eras:` list to produce. A plain "
        "era is binned on its own statistics; an era that is a key of `era_groups:` is "
        "binned on its members' summed statistics and its shapes are that sum. Each is "
        "written under its own era name, since a combination supports finer bins than "
        "any single era and the two must not overwrite each other",
    )

    # Knob overrides. All default to None so that "not given" is distinguishable from
    # "given the same value as the default", which is what lets --binning-config win.
    knob_args = {
        "n_slices": (int, "fixed number of slices per category (same for every mass)"),
        "category_pattern": (
            str,
            "pattern naming the sliced categories, e.g. "
            "'{base_category}_dnn{slice_idx}'; must use both {base_category} and "
            "{slice_idx}, and the datacard configuration reading these shapes must "
            "declare the same one",
        ),
        "slice_var": (
            str,
            "name of the sliced axis, used to label each slice directory with the "
            "selection it stands for (e.g. 'DNN' -> '0.80 < DNN < 1.00')",
        ),
        "max_bins_per_slice": (int, "bins to aim for inside each slice"),
        "min_slice_bkg_sum": (
            float,
            "minimum summed-background yield required in a slice",
        ),
        "min_slice_bkg_neff": (
            float,
            "minimum effective MC entries of the summed background for a slice boundary "
            "to be selectable",
        ),
        "min_slice_bkg_each": (
            float,
            "minimum yield required of every non-negligible background in a slice",
        ),
        "min_slice_bkg_each_neff": (
            float,
            "minimum effective MC entries of every non-negligible background in a slice; "
            "much stronger than the yield floor and correspondingly expensive",
        ),
        "min_bin_bkg_each": (
            float,
            "minimum yield required of every non-negligible background in a bin",
        ),
        "min_bin_bkg_neff": (
            float,
            "minimum effective MC entries of the summed background in a bin",
        ),
        "min_bin_bkg_each_neff": (
            float,
            "minimum effective MC entries of every non-negligible background in a bin. "
            "The summed test above is satisfied by any one well-measured process, so "
            "this is what makes each background individually measured; bins are merged "
            "into their lower-signal neighbour until it holds",
        ),
        "bkg_per_bin": (
            float,
            "summed background to aim for per bin; overrides max_bins_per_slice when it "
            "binds",
        ),
        "min_bkg_frac": (
            float,
            "backgrounds below this fraction of the category total are exempt from the "
            "per-background floors, so a negligible process cannot veto every boundary",
        ),
        "min_signal": (float, "minimum signal integral for a category to be rebinned"),
    }
    for name, (typ, help_text) in knob_args.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            required=False,
            type=typ,
            default=None,
            help=help_text,
        )
    parser.add_argument(
        "--fill-negative-bins",
        required=False,
        type=float,
        default=None,
        help="write any negative bin as this yield, with a tenth of it as the error; "
        "omit to leave negative bins alone",
    )
    parser.add_argument(
        "--binning-exclude-processes",
        required=False,
        type=str,
        default=None,
        help="comma-separated processes the binning ignores entirely, e.g. 'DY'; the "
        "edges are chosen as if they did not exist and they are still written out",
    )
    parser.add_argument(
        "--strategy",
        required=False,
        type=str,
        default=None,
        choices=list(BINNING_STRATEGIES),
        help="how the edges are found: 'greedy' = each slice boundary taken on its own "
        "and equal-signal quantiles inside it, 'dp' = an exact partition search on the "
        "mass axis against the same figure of merit",
    )
    parser.add_argument(
        "--dp-budget-mode",
        required=False,
        type=str,
        default=None,
        choices=list(BUDGET_MODES),
        help="'per_slice' gives each slice the bins its own background licenses; "
        "'shared' pools the same total and spends it where it buys the most (dp only)",
    )
    parser.add_argument(
        "--significance-mode",
        required=False,
        type=str,
        default=None,
        choices=list(SIGNIFICANCE_MODES),
        help="figure of merit for slice boundaries: 'sb' = S/sqrt(B+sigmaB^2), "
        "'asimov' = Poisson-correct Asimov significance (valid at low B)",
    )
    args = parser.parse_args()

    overrides = {name: getattr(args, name) for name in knob_args}
    overrides["significance_mode"] = args.significance_mode
    overrides["strategy"] = args.strategy
    overrides["fill_negative_bins"] = args.fill_negative_bins
    overrides["binning_exclude_processes"] = (
        args.binning_exclude_processes.split(",")
        if args.binning_exclude_processes
        else None
    )
    overrides["dp_budget_mode"] = args.dp_budget_mode
    knobs = load_binning_config(args.binning_config, overrides)

    frozen_binning = None
    if args.binning:
        with open(args.binning, "r") as f:
            frozen_binning = json.load(f)
        print(f"Replaying the binning recorded in {args.binning}; not optimising")

    run(
        args.input,
        args.output,
        args.config,
        args.era,
        knobs,
        frozen_binning=frozen_binning,
    )
