"""Rebin a 1D discriminant into significance-optimised bins.

The 1D counterpart of bin_opt_2d/rebin_2d.py, and deliberately its DNN axis and nothing
else: the boundaries are placed by the same find_slices() scan, under the same gates, with
the same figure of merit. Everything that decides *where* a boundary goes lives in
StatInference/common/binning_core.py and is shared between the two, so the two binners
cannot answer the same question differently.

What differs is only what a range becomes. In 2D a range on the DNN axis is a *category*
-- a directory holding the mass shape cut from it -- and the mass axis is then binned
inside it. Here a range is one *bin* of one histogram, so there is no second axis, no
sliced category names, no category_pattern, and the output keeps the input's
<channel>/<category> layout untouched. That also means the knobs are named for bins
(min_bin_bkg_*): they gate the very objects that end up in the fit, with no second level
below them.

Two properties of find_slices() this inherits and does not paper over:

  * the leftmost range is the leftover -- it takes whatever the scan did not consume and
    is never gated. In 2D that is the low-DNN slice, and here it is the low-DNN *bin*.
    It is the background-dominated end, so its yield is never the problem, but it does
    mean exactly one bin escapes the floors and it should not be read as though it had
    passed them.
  * grow_slice() does not reserve axis for the ranges still to be placed, so a scan can
    run out and return None. That is never a binning with a hole in it. Here it means the
    category cannot carry that many bins, so process_category() retries with one fewer,
    down to min_bins, and rejects the category outright only if even that fails -- which
    is what discover_binning() does immediately in 2D, where the count cannot vary because
    a range there is a whole category.

Usage is the same as rebin_2d.py -- a standalone pre-step writing the
"<era>/<variable>/<variable>.root" layout HistMergerTask produces, consumed by pointing
the chain's --hists-version at it.
"""

import array
import json
import os
import sys

import yaml

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(file_dir)
    base_dir = os.path.dirname(pkg_dir)
    pkg_dir_name = os.path.split(pkg_dir)[1]
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    __package__ = pkg_dir_name

from StatInference.common.tools import importROOT
from StatInference.common.param_parse import applyParameters

from StatInference.common.binning_core import (
    BINNING_JSON,
    SIGNIFICANCE_MODES,
    _detach,
    bin_edges,
    extend_outer_edges,
    find_slices,
    get_hist,
    load_config,
    lookup_frozen,
    open_input_file,
    sum_hists,
)

ROOT = importROOT()


# The knobs that decide the binning, with the values a configuration gets when it does
# not say otherwise. They belong to an analysis rather than to this file, so every
# production should state its own in a binning yaml -- see config/Datacards/binning_1d.yaml
# in HH_bbWW for an annotated set. This file ships no configuration of its own, only these
# fallbacks.
#
# This is a strict subset of rebin_2d.py's set, minus everything about a second axis
# (max_bins_per_slice, bkg_per_bin and the 2D min_bin_* family, which there gate the mass
# bins inside a slice). The names are the bin-level ones because that is what a range is
# here; they are handed to find_slices() in the argument positions rebin_2d fills from its
# min_slice_* knobs, which is the same scan gated the same way.
BINNING_DEFAULTS = {
    # n_bins is the most bins a category may take; min_bins the fewest it may be cut to
    # before it is rejected instead. Equal values reproduce a fixed count exactly, which is
    # what the defaults do, so this stays a no-op unless a configuration asks for a range.
    "n_bins": 5,
    "min_bins": 5,
    "min_bin_bkg_sum": 1.0,
    "min_bin_bkg_neff": 4.0,
    "min_bin_bkg_each": 0.01,
    "min_bin_bkg_each_neff": 0.0,
    "min_bkg_frac": 0.05,
    "min_signal": 0.5,
    "significance_mode": "asimov",
}


def load_binning_config(path, overrides=None):
    """Merge the binning yaml over the defaults, then command-line overrides over that.

    Unknown keys raise rather than being ignored: a misspelled floor that silently does
    nothing is the worst outcome here, because the run still succeeds and the shapes look
    plausible. Note that this is why the knob set is stated per binner rather than shared
    -- the 2D mass-axis knobs are not merely unused here, they are errors, and saying so
    is the point.
    """
    declared = {}
    if path:
        with open(path, "r") as f:
            declared = yaml.safe_load(f) or {}
    unknown = sorted(set(declared) - set(BINNING_DEFAULTS))
    if unknown:
        raise RuntimeError(
            f"{path}: unknown binning knob(s) {unknown}. Known knobs are "
            f"{sorted(BINNING_DEFAULTS)}. (The 2D mass-axis knobs do not apply to a 1D "
            "fit -- there is no second axis to budget.)"
        )
    knobs = dict(BINNING_DEFAULTS)
    knobs.update(declared)
    for key, value in (overrides or {}).items():
        if value is not None:
            knobs[key] = value
    if knobs["significance_mode"] not in SIGNIFICANCE_MODES:
        raise RuntimeError(
            f"{path or 'binning configuration'}: significance_mode "
            f"'{knobs['significance_mode']}' is not one of {sorted(SIGNIFICANCE_MODES)}."
        )
    return knobs


def mkdir_path(directory, path):
    """mkdir a nested path, returning the leaf. Existing levels are reused.

    Unlike the 2D binner's mkdir_titled() there is no title to place: a range here is a
    bin of a histogram whose axis carries the real edges, so the selection it stands for
    is already readable off the shape itself.
    """
    for part in path.split("/"):
        directory = directory.GetDirectory(part) or directory.mkdir(part)
    return directory


def ranges_to_record(ranges, axis):
    """The discovered binning as plain data, for binning.json.

    Both forms are written, for the same reason as in 2D: the bin index ranges are what
    the code cuts on and what a replay restores, and the physical edges beside them are
    what a human reads. The axis size is recorded as `n_x_bins`, the name the 2D record
    uses, rather than as `n_bins`: it is the *input* axis the indices were found
    against, and is not the `n_bins` knob -- that is how many ranges came out of it,
    which is len(ranges).
    """
    return {
        "n_x_bins": axis.GetNbins(),
        "ranges": [list(r) for r in ranges],
        "edges": bin_edges(axis, ranges),
    }


def record_to_ranges(record, axis, where):
    """Inverse of ranges_to_record().

    The axis size is checked rather than trusted. Bin indices only mean anything against
    the axis they were found on, so a binning.json replayed over input binned differently
    would cut the shapes in silently wrong places -- exactly what freezing a binning is
    supposed to rule out.
    """
    if axis.GetNbins() != record["n_x_bins"]:
        raise RuntimeError(
            f"{where}: recorded binning was found on a {record['n_x_bins']}-bin axis, but "
            f"the input has {axis.GetNbins()}. The binning.json does not belong to this "
            "input."
        )
    return [tuple(r) for r in record["ranges"]]


def rebin_hist_1d(hist, ranges, name):
    """One rebinned TH1 for this histogram (nominal or a systematic variation).

    Booked on the discovered physical edges rather than on bin indices, so the shape keeps
    the discriminant's own scale and a plot of it is labelled by score, not by bin number.
    """
    edges = array.array("d", bin_edges(hist.GetXaxis(), ranges))
    # Detached because Write() targets gDirectory regardless; nothing needs this attached
    # to the input file it was read from.
    out = _detach(ROOT.TH1D(name, name, len(edges) - 1, edges))
    for bin_idx, (lo, hi) in enumerate(ranges, start=1):
        # ROOT reinterprets an inverted or negative range as the whole axis including
        # under/overflow, silently and with a plausible-looking positive yield. Checked
        # here because this is the last point at which it is still cheap to say so.
        if not 0 <= lo <= hi:
            raise RuntimeError(
                f"invalid bin range ({lo}, {hi}) for bin {bin_idx} of {name}; ROOT would "
                "read this as the whole axis."
            )
        err = array.array("d", [0.0])
        content = hist.IntegralAndError(lo, hi, err)
        out.SetBinContent(bin_idx, content)
        out.SetBinError(bin_idx, err[0])
    return out


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
    optimisation happens at all: the discovery files are then only read for the axis, the
    content gates are not applied, and `frozen` being None means the record does not hold
    this category -- which is a skip, not a licence to derive one.
    """
    min_signal = knobs["min_signal"]
    # The resonance parameter is named by the configuration, not by this script -- it is
    # "MX" for bbWW but the binning knows nothing about which parameter it is scanning.
    param_name = cfg["signal_param_name"]
    prefix = f"{channel}/{category}/"
    # The key list comes from the first source era; a systematic that only some eras carry
    # is filled in from their nominal when the bins are written below.
    cat_dir = sources[0][1].Get(f"{channel}/{category}")
    if not cat_dir:
        print(f"  [skip] {channel}/{category}: not found in {sources[0][1].GetName()}")
        return

    signal_keys = [
        applyParameters(pattern, {param_name: mass})
        for pattern in cfg["signal_hist_name_patterns"]
    ]
    background_names = [
        hist_name
        for (base_name, hist_name, allowed_channels) in cfg["background_entries"]
        if not allowed_channels or channel in allowed_channels
    ]

    def load(f, key):
        return get_hist(f, prefix + key)

    disc_sig = sum_hists([load(f, key) for f in discovery_files for key in signal_keys])
    disc_bkg_by_name = {}
    for bkg_key in background_names:
        per_era = [load(f, bkg_key) for f in discovery_files]
        per_era = [h for h in per_era if h is not None]
        if len(per_era) == len(discovery_files):
            disc_bkg_by_name[bkg_key] = per_era

    sig_integral = disc_sig.Integral() if disc_sig is not None else 0
    if replaying:
        # A replay reproduces a recorded category set exactly, or it is not a replay. The
        # gates below decide which categories get binned; re-running them against a new
        # input can only disagree with the record -- a category whose signal has since
        # dipped under min_signal would be dropped even though its edges are written down,
        # which is precisely the drift freezing exists to prevent.
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
                "record, but its signal histogram is missing from the input, so the axis "
                "the recorded bin indices refer to cannot be read."
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

    axis = disc_sig.GetXaxis()
    nx = axis.GetNbins()
    where = f"{era}/{param_name}={mass}/{channel}/{category}"
    if frozen is not None:
        ranges = record_to_ranges(frozen, axis, where)
    else:
        # n_bins is a maximum, not a quota: take the most bins this category can actually
        # support under the floors, and step down when it cannot carry them.
        #
        # A single global count does not work here. The categories differ in background by
        # three orders of magnitude -- at m600 eE/res2b holds ~53000 background events and
        # muMu/boosted ~85 -- so one number is set by the poorest category and leaves the
        # rich ones far coarser than their statistics allow. Measured: a fixed n_bins of 8
        # rejected the same-flavour boosted categories at 18 of 20 mass points, while
        # res2b and recovery took 8 comfortably.
        #
        # The step-down uses the gates themselves as the criterion rather than a separate
        # heuristic, which matters because the binding constraint is not yield. bin_budget()
        # in the 2D binner divides by background *yield*, and by that measure muMu/boosted's
        # 85 events would license many bins; what actually stops it is DY's per-process
        # effective entries. The gates already encode what a bin has to contain, so asking
        # them is both cheaper and more honest than modelling it twice.
        #
        # Unlike 2D this is safe to vary per category and per mass. There a range is a
        # *category*, so the count has to be fixed or the datacards would not share a
        # category list; here a range is a bin inside one shape, each mass builds its own
        # datacard, and the 2D mass axis already varies its bin count per slice.
        ranges = None
        n_used = 0
        for n in range(knobs["n_bins"], knobs["min_bins"] - 1, -1):
            candidate = find_slices(
                disc_sig,
                disc_bkg_by_name,
                n,
                1,
                nx,
                knobs["min_bin_bkg_sum"],
                knobs["min_bin_bkg_neff"],
                knobs["significance_mode"],
                knobs["min_bin_bkg_each"],
                knobs["min_bin_bkg_each_neff"],
                knobs["min_bkg_frac"],
            )
            if not any(r is None for r in candidate):
                ranges, n_used = candidate, n
                break
        if ranges is None:
            print(
                f"    [skip] {channel}/{category} {param_name}={mass}: even "
                f"{knobs['min_bins']} bins could not be placed -- the axis was exhausted "
                "before the last boundary, so no window clears the background floors. "
                "Lower min_bins or the floors for this configuration."
            )
            return
        if n_used != knobs["n_bins"]:
            # Deliberately not worded as a failure, and deliberately not containing the
            # phrase the rejection message uses: this is the mechanism working, and a log
            # grep for rejections must not match it.
            print(
                f"    [bins] {channel}/{category} {param_name}={mass}: took {n_used} of "
                f"up to {knobs['n_bins']} bins, the most its background supports"
            )
        ranges = extend_outer_edges(ranges, 0, nx + 1)

    # One shared set of edges, applied to every source era in its own file.
    for source_era, in_file, out_file in sources:
        mkdir_path(out_file, f"{channel}/{category}")
        cat_dir = in_file.Get(f"{channel}/{category}")
        if not cat_dir:
            print(f"  [skip] {source_era} {channel}/{category}: not in the input")
            continue
        for key in [k.GetName() for k in cat_dir.GetListOfKeys()]:
            hist = get_hist(in_file, prefix + key)
            if hist is None or hist.GetDimension() != 1:
                continue
            out_file.cd(f"{channel}/{category}")
            rebin_hist_1d(hist, ranges, key).Write(key)

    return ranges_to_record(ranges, axis)


def run(
    input_dir,
    output_dir,
    config_path,
    era,
    knobs,
    frozen_binning=None,
):
    """Produce one era of the datacard configuration's `eras:` list.

    Which era it is decides everything. A plain era is binned on its own statistics, for a
    standalone limit. An era that is a key of `era_groups:` is binned on its members'
    summed statistics -- a combination supports finer bins than any single era can -- and
    those edges are then applied to each member separately.

    Output layout is "<output_dir>/<source_era>/<variable>/<variable>.root", plus the
    binning.json that produced it. One era per --output directory, since the same source
    era carries different edges under different target eras and they must not overwrite
    each other. The members are kept in their own files rather than summed here: the
    datacard step sums them, and a per-era lnN can only be built by scaling a sub-era's own
    shape, which a pre-summed shape no longer has.
    """
    cfg = load_config(config_path)
    # Nothing to reconcile with a category_pattern here: this binner writes its shapes
    # under the input's own <channel>/<category>, so the names the datacard configuration
    # lists are the names on disk. A configuration that declares a category_pattern is
    # describing sliced 2D shapes and is not this one.
    if cfg.get("category_pattern") is not None:
        raise RuntimeError(
            f"{config_path} declares category_pattern "
            f"'{cfg['category_pattern']}', which names the sliced categories rebin_2d.py "
            "writes. This binner does not slice into categories -- its ranges are bins of "
            "one histogram -- so a configuration reading its output must not declare one."
        )
    model = cfg["model"]

    # A group era is built from its members; a plain era is built from itself.
    source_eras = cfg["era_groups"].get(era, [era])
    if era in cfg["era_groups"]:
        print(f"{era} is a group of {source_eras}: binning on their summed statistics")
    else:
        print(f"{era} is a standalone era: binning on its own statistics")

    record = {
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
            # getInputFileName() yields "<src>/<variable>/<variable>.root", which is the
            # layout input_file_pattern resolves against. --output is one era's own
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
            f"{len(sources)} source era(s) -> {os.path.join(output_dir, era)}"
        )
        by_channel = by_mass.setdefault(str(mass), {})
        for channel in cfg["channels"]:
            for category in cfg["categories"]:
                frozen = None
                if frozen_binning is not None:
                    frozen = lookup_frozen(frozen_binning, era, mass, channel, category)
                used = process_category(
                    sources,
                    channel,
                    category,
                    cfg,
                    mass,
                    era,
                    discovery_files,
                    knobs,
                    frozen=frozen,
                    replaying=frozen_binning is not None,
                )
                if used is not None:
                    by_channel.setdefault(channel, {})[category] = used

        for _, in_file, out_file in sources:
            out_file.Close()
            in_file.Close()

    # Written beside the shapes it describes, in the same output as them: the binning is a
    # product of this run, not configuration, so it travels with what it produced and a
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
        description="Rebin 1D discriminant histograms into significance-optimised bins."
        "\n\nA standalone pre-step, not part of the datacard chain: it writes a "
        "'<era>/<variable>/<variable>.root' tree in the same layout HistMergerTask "
        "produces, so a production of these is consumed by pointing the chain's "
        "--hists-version at it. It also writes " + BINNING_JSON + ", which --binning "
        "replays to reproduce a binning instead of re-deriving one.\n\n"
        "The boundaries are placed by the same scan rebin_2d.py uses on its DNN axis; "
        "the difference is that a range here is a bin rather than a category.",
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
        "binned on its members' summed statistics. Each is written under its own era "
        "name, since a combination supports finer bins than any single era and the two "
        "must not overwrite each other",
    )

    # Knob overrides. All default to None so that "not given" is distinguishable from
    # "given the same value as the default", which is what lets --binning-config win.
    knob_args = {
        "n_bins": (int, "most bins a category may take"),
        "min_bins": (
            int,
            "fewest bins a category may be cut down to before it is rejected instead. "
            "Equal to n_bins means a fixed count",
        ),
        "min_bin_bkg_sum": (
            float,
            "minimum summed-background yield required in a bin",
        ),
        "min_bin_bkg_neff": (
            float,
            "minimum effective MC entries of the summed background for a bin boundary to "
            "be selectable",
        ),
        "min_bin_bkg_each": (
            float,
            "minimum yield required of every non-negligible background in a bin",
        ),
        "min_bin_bkg_each_neff": (
            float,
            "minimum effective MC entries of every non-negligible background in a bin. "
            "The summed test above is satisfied by any one well-measured process, so this "
            "is what makes each background individually measured",
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
        "--significance-mode",
        required=False,
        type=str,
        default=None,
        choices=list(SIGNIFICANCE_MODES),
        help="figure of merit for bin boundaries: 'sb' = S/sqrt(B+sigmaB^2), "
        "'asimov' = Poisson-correct Asimov significance (valid at low B)",
    )
    args = parser.parse_args()

    overrides = {name: getattr(args, name) for name in knob_args}
    overrides["significance_mode"] = args.significance_mode
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
