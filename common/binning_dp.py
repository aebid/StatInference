"""Prefix-sum cells over a 2D shape, so any candidate binning can be scored in O(1).

The greedy binner asks its questions one candidate at a time and re-integrates the
histograms for each, which is affordable because it only ever looks at ~nx windows. An
exact partition search looks at ~nx^2 windows on the sliced axis and ~ny^2 inside every
slice, so the integrals have to become table lookups or the search is not worth running.
That is all this module does: it turns the TH2s into cumulative arrays and hands back the
same yields, errors and figure of merit the greedy path would have computed.

It deliberately knows nothing about what a valid bin is. The gates live in
binning_core (and _bin_passes in the 2D binner), and are passed in as callables, so
there is exactly one definition of what a bin has to contain and this module cannot
drift from it. What it provides is the *arguments* those gates take -- the yields and
errors dicts -- built from the tables instead of from ROOT.

Under- and overflow are carried, and that is load-bearing rather than tidy. _integral()
in binning_core reads a TH2 with Integral(lo, hi, 0, -1), and ROOT reads biny2 < biny1 as
"the whole y axis including under- and overflow", so every slice-level quantity in the
greedy path already includes the y under/overflow rows. A table that stopped at bin ny
would disagree with the greedy path by a fraction of an event, which is easily enough to
move a boundary and produce a difference nobody can explain. So the arrays are indexed
0..n+1 throughout, and a slice-level query spans y = 0..ny+1.

Differencing prefix sums is not exact, and one of the ways it is inexact has teeth.
A window whose content is ~1e8 times smaller than the whole plane's -- routine for DY in
a narrow high-HME window -- loses about eight digits to cancellation, so a quantity that
is exactly 0.0 in ROOT comes back as -2e-16 here. That is numerically nothing and
physically nothing, but _bin_passes() tests `value < 0` to reject a background that has
gone negative, and it would reject 137 of the windows sampled below on the strength of a
rounding sign. So every yield is snapped to zero below a per-process epsilon scaled by
that process's own L1 norm, which is the size of the cancellation that can occur. The
epsilon lands around 1e-8 events for the largest process here, against a magnitude floor
(min_bin_bkg_each) of 0.01 -- six orders of margin, so the snap cannot mask a real
negative.

The check that all of this is true is parity_check(), not inspection. It is the only
reason to trust anything built on top of this file.
"""

import math

import numpy as np

from StatInference.common.binning_core import (
    _bkg_errors,
    _bkg_yields,
    _integral,
    _total_bkg_error,
    significance,
)


def _hist_arrays(hist):
    """(values, variances) as (nx+2, ny+2) arrays indexed [binx, biny], bins 0..n+1.

    Read through GetBinContent/GetBinError rather than the internal buffer: the buffer
    is faster but its layout is a ROOT implementation detail, and getting it wrong is
    silent. This runs once per histogram per category, which is nothing next to the
    search it feeds.
    """
    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()
    values = np.empty((nx + 2, ny + 2), dtype=np.float64)
    variances = np.empty((nx + 2, ny + 2), dtype=np.float64)
    for bx in range(nx + 2):
        for by in range(ny + 2):
            values[bx, by] = hist.GetBinContent(bx, by)
            err = hist.GetBinError(bx, by)
            variances[bx, by] = err * err
    return values, variances


def _prefix(a):
    """Inclusive 2D prefix sums, shape (nx+3, ny+3), with P[i, j] = sum(a[:i, :j])."""
    p = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    p[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return p


def _cancellation_eps(a, rel=1e-12):
    """How large a value this array's prefix sums can invent out of rounding.

    Scaled by the L1 norm rather than the sum, because the cancellation that matters
    comes from negative-weight events: a process whose bins nearly cancel has a small
    sum and a large L1, and it is the L1 that sets how much precision the difference of
    two prefix sums can lose. rel is a few hundred times the double-precision epsilon,
    which covers accumulating over the ~1e4 cells of one of these planes.
    """
    return rel * float(np.abs(a).sum())


def _snap(value, eps):
    """Zero out a value that is indistinguishable from zero at this array's precision."""
    return 0.0 if abs(value) < eps else value


def _rect(p, x0, x1, y0, y1):
    """Sum over the inclusive bin-index rectangle [x0..x1] x [y0..y1]."""
    return float(p[x1 + 1, y1 + 1] - p[x0, y1 + 1] - p[x1 + 1, y0] + p[x0, y0])


class Cells:
    """Every quantity the binning gates and the figure of merit need, as O(1) lookups.

    Background yields are summed across discovery eras and their MC errors added in
    quadrature, which is what _bkg_yields()/_bkg_errors() do -- the eras are summed
    because the binning is derived from the combination it will be applied to.
    """

    def __init__(self, nx, ny, sig, bkg, var, x_centres=None):
        self.nx = nx
        self.ny = ny
        # bin centres of the x axis including under/overflow, for a profile fit; the
        # outer two are placeholders one bin width outside, never fitted.
        self.x_centres = (
            x_centres if x_centres is not None else np.arange(nx + 2, dtype=np.float64)
        )
        self.names = list(bkg)
        self._sig = _prefix(sig)
        self._bkg = {name: _prefix(bkg[name]) for name in self.names}
        self._var = {name: _prefix(var[name]) for name in self.names}
        # An empty background dict is reachable -- a replay whose input is missing a
        # process it was derived with -- and sum() of nothing is the integer 0, which is
        # not an array. Zeros keep every query defined and answering zero, which is the
        # truthful answer when there is no background to integrate.
        zero = np.zeros_like(sig)
        bkg_tot = sum(bkg.values()) if bkg else zero
        var_tot = sum(var.values()) if var else zero
        self._bkg_tot = _prefix(bkg_tot)
        self._var_tot = _prefix(var_tot)
        self._eps_sig = _cancellation_eps(sig)
        self._eps_bkg = {name: _cancellation_eps(bkg[name]) for name in self.names}
        self._eps_tot = _cancellation_eps(bkg_tot)
        self._eps_var = {name: _cancellation_eps(var[name]) for name in self.names}
        self._eps_var_tot = _cancellation_eps(var_tot)

    def profile(self, name, y0, y1):
        """(values, variances) per x bin for one process inside the y window [y0, y1].

        Length nx+2, under- and overflow included, which is the array a fit and
        _ranges_from_column() both expect.
        """
        col = self._bkg[name][:, y1 + 1] - self._bkg[name][:, y0]
        var = self._var[name][:, y1 + 1] - self._var[name][:, y0]
        return np.diff(col), np.diff(var)

    # -- the whole y axis, under- and overflow included: what a slice-level query means
    def full_y(self):
        return 0, self.ny + 1

    def signal(self, x0, x1, y0, y1):
        return _snap(_rect(self._sig, x0, x1, y0, y1), self._eps_sig)

    def total_bkg(self, x0, x1, y0, y1):
        return _snap(_rect(self._bkg_tot, x0, x1, y0, y1), self._eps_tot)

    def total_bkg_error(self, x0, x1, y0, y1):
        return math.sqrt(max(_rect(self._var_tot, x0, x1, y0, y1), 0.0))

    def variances(self, x0, x1, y0, y1):
        """{background: squared MC statistical error}. Exposed because this, not the
        error, is the quantity the tables actually add -- so it is the one a parity
        check can bound the rounding of."""
        return {n: _rect(self._var[n], x0, x1, y0, y1) for n in self.names}

    def total_bkg_variance(self, x0, x1, y0, y1):
        return _rect(self._var_tot, x0, x1, y0, y1)

    def yields(self, x0, x1, y0, y1):
        """{background: yield}, the dict the gates take."""
        return {
            n: _snap(_rect(self._bkg[n], x0, x1, y0, y1), self._eps_bkg[n])
            for n in self.names
        }

    def errors(self, x0, x1, y0, y1):
        """{background: MC statistical error}, the dict the gates take."""
        return {
            n: math.sqrt(max(_rect(self._var[n], x0, x1, y0, y1), 0.0))
            for n in self.names
        }

    def score(self, x0, x1, y0, y1, mode):
        """The figure of merit for one cell, squared.

        Squared because Asimov Z^2 is what adds across independent counting bins, so a
        partition's value is the sum of its cells' scores and the search can be a
        dynamic program. significance() is binning_core's, so the cells are ranked by
        exactly the quantity the greedy path maximises.
        """
        s = self.signal(x0, x1, y0, y1)
        b = self.total_bkg(x0, x1, y0, y1)
        b_err = self.total_bkg_error(x0, x1, y0, y1)
        return significance(s, b, b_err, mode) ** 2

    def exempt(self, x0, x1, y0, y1, min_frac):
        """minor_backgrounds() over a rectangle, for the range the caller is about to
        subdivide -- never for the candidate sub-range itself, which is circular.
        """
        if min_frac <= 0:
            return set()
        y = self.yields(x0, x1, y0, y1)
        total = sum(y.values())
        if total <= 0:
            return set()
        return {n for n, v in y.items() if v < min_frac * total}


def build_cells(sig2d, bkg2d_by_name):
    """Cells for one channel/category, from the same histograms discover_binning() reads.

    sig2d is the already-summed discovery signal; bkg2d_by_name is
    {background: [hist per discovery era]}, summed here the way _bkg_yields() sums it.
    """
    nx = sig2d.GetNbinsX()
    ny = sig2d.GetNbinsY()
    sig, _ = _hist_arrays(sig2d)
    bkg = {}
    var = {}
    for name, hists in bkg2d_by_name.items():
        values = np.zeros((nx + 2, ny + 2), dtype=np.float64)
        variances = np.zeros((nx + 2, ny + 2), dtype=np.float64)
        for h in hists:
            v, e2 = _hist_arrays(h)
            values += v
            variances += e2
        bkg[name] = values
        var[name] = variances
    axis = sig2d.GetXaxis()
    width = axis.GetBinWidth(1)
    centres = np.array(
        [axis.GetBinLowEdge(1) - width]
        + [axis.GetBinCenter(i) for i in range(1, nx + 1)]
        + [axis.GetBinUpEdge(nx) + width]
    )
    return Cells(nx, ny, sig, bkg, var, centres)


def parity_check(cells, sig2d, bkg2d_by_name, n_samples=200, seed=0):
    """Assert the tables agree with the ROOT integrals they replace.

    Everything downstream is only as trustworthy as this. Over random x windows it
    compares the quantities the gates and the figure of merit are built from --
    per-process yields and variances, the summed variance, and the signal -- against
    binning_core's own _bkg_yields/_bkg_errors/_total_bkg_error/_integral, which is what
    the greedy path calls. Slice level (full y, under- and overflow included), because
    that is the query whose ROOT spelling is easy to get wrong.

    **The tolerance is absolute and scaled by the cancellation, not relative.** A
    relative tolerance is the wrong test for a difference of prefix sums: a window
    holding 1e-8 of the plane's content has already spent eight of its sixteen digits
    before the comparison starts, so demanding a fixed relative agreement on it is
    demanding that floating point not be floating point. What *can* be bounded is the
    absolute rounding, and it is bounded by the same per-process epsilon Cells uses to
    snap -- the L1 norm times a few hundred machine epsilons. Errors are compared as
    variances for the same reason: the variance is what the tables add, and the square
    root only inflates the relative error of an already-cancelled quantity.

    It also asserts the two agree on the *sign* of every yield, which is a separate
    property from agreeing on its value and the one the binning actually turns on:
    _bin_passes rejects a bin outright when a background is negative, so a yield that is
    0.0 in ROOT and -2e-16 here is not a rounding difference, it is a different binning.

    Raises on the first disagreement rather than returning a verdict: a table that is
    subtly wrong must not be usable.
    """
    rng = np.random.default_rng(seed)
    y0, y1 = cells.full_y()
    checked = 0
    for _ in range(n_samples):
        lo = int(rng.integers(1, cells.nx + 1))
        hi = int(rng.integers(lo, cells.nx + 1))

        got_y = cells.yields(lo, hi, y0, y1)
        want_y = _bkg_yields(bkg2d_by_name, lo, hi)
        got_v = cells.variances(lo, hi, y0, y1)
        want_e = _bkg_errors(bkg2d_by_name, lo, hi)
        for name in want_y:
            _assert_close(
                got_y[name],
                want_y[name],
                cells._eps_bkg[name],
                f"yield[{name}] {lo}..{hi}",
            )
            _assert_close(
                got_v[name],
                want_e[name] ** 2,
                cells._eps_var[name],
                f"variance[{name}] {lo}..{hi}",
            )
            _assert_same_sign(got_y[name], want_y[name], f"yield[{name}] {lo}..{hi}")

        _assert_close(
            cells.total_bkg_variance(lo, hi, y0, y1),
            _total_bkg_error(bkg2d_by_name, lo, hi) ** 2,
            cells._eps_var_tot,
            f"total bkg variance {lo}..{hi}",
        )
        _assert_close(
            cells.signal(lo, hi, y0, y1),
            _integral(sig2d, lo, hi),
            cells._eps_sig,
            f"signal {lo}..{hi}",
        )
        checked += 1
    return checked


def _assert_same_sign(got, want, what):
    """The gates branch on sign, so agreeing to 1e-9 is not the same as agreeing."""
    if (got < 0) != (want < 0):
        raise AssertionError(
            f"binning_dp parity failure on the sign of {what}: table says {got!r}, "
            f"ROOT says {want!r}. _bin_passes rejects a bin whose background is "
            "negative, so these two would not produce the same binning even though the "
            "magnitudes agree. The zero-snapping in Cells is what should have prevented "
            "this."
        )


def _assert_close(got, want, atol, what):
    """Agreement to within the rounding the prefix sums can actually produce."""
    if abs(got - want) > atol:
        raise AssertionError(
            f"binning_dp parity failure on {what}: table says {got!r}, ROOT says "
            f"{want!r}; they differ by {abs(got - want):.3e}, which exceeds the "
            f"{atol:.3e} this quantity's cancellation can account for. The cell tables "
            "do not reproduce the integrals the gates are defined on, so any binning "
            "derived from them would be cut in the wrong places."
        )


NEG = -np.inf


def partition_dp(score, valid, max_parts):
    """Best partition of [0..n-1] into exactly k contiguous valid ranges, for every k.

    Returns (values, partitions): values[k] is the total score of the best k-part
    partition, or -inf if there is none, and partitions[k] is that partition as a list of
    (lo, hi) index pairs. Index 0 is unused in both so that k reads as the part count.

    Every k is returned rather than just the requested one, because the caller needs the
    whole curve and it costs nothing to keep: the budget allocator picks each slice's bin
    count by comparing what the slices would do with one more bin, which is exactly
    values[k] against values[k+1]. Computing them one at a time would re-run the same DP.

    Exact, not greedy. The score is additive over the parts -- Asimov Z^2 is what adds
    across independent counting bins -- so the optimal k-part partition of a prefix is
    built from an optimal (k-1)-part partition of a shorter prefix, and the usual
    interval DP applies. O(max_parts * n^2).

    `valid` is a mask, not a penalty: an invalid range is unreachable rather than merely
    expensive, so a partition containing one cannot be returned at any score. That is what
    keeps the background gates hard constraints instead of preferences.
    """
    n = score.shape[0]
    max_parts = max(1, min(max_parts, n))
    masked = np.where(valid, score, NEG)

    dp = np.full((max_parts + 1, n + 1), NEG)
    arg = np.full((max_parts + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, max_parts + 1):
        for e in range(k, n + 1):
            # candidate: the last part is [s .. e-1], the rest is a (k-1)-part prefix
            cand = dp[k - 1, :e] + masked[:e, e - 1]
            s = int(np.argmax(cand))
            if cand[s] > NEG:
                dp[k, e] = cand[s]
                arg[k, e] = s

    values = [NEG] * (max_parts + 1)
    partitions = [None] * (max_parts + 1)
    for k in range(1, max_parts + 1):
        if dp[k, n] == NEG:
            continue
        values[k] = float(dp[k, n])
        parts = []
        e = n
        for kk in range(k, 0, -1):
            s = int(arg[kk, e])
            parts.append((s, e - 1))
            e = s
        partitions[k] = list(reversed(parts))
    return values, partitions


def allocate_budget(values_per_slice, total, n_min=1, n_max=None):
    """Spend a fixed total number of bins across the slices, where it buys the most.

    values_per_slice[k][n] is what slice k scores with n bins, as partition_dp returns it.
    Returns the list of counts maximizing the summed score subject to sum(n) <= total and
    n >= n_min everywhere, or None if even n_min is infeasible for some slice.

    This is the part that fixes the allocation, and it is worth being explicit about why
    a per-slice rule cannot. bin_budget() sizes a slice from its own background *yield*,
    and yield is largest exactly where signal purity is lowest -- so on the bbWW DL shapes
    the lowest DNN slice, carrying 0.1% of the total Asimov Z^2, was given a median of ten
    HME bins, while the top slice carrying 87.5% of it was given a median of one and was a
    single bin in 53% of categories. No local rule can see that, because the comparison
    that matters is between slices. Here it is the only comparison being made.

    n_min >= 1 is not negotiable: a slice with no bins is not a category, and every mass
    point has to produce the same category list.

    A knapsack rather than a sort, because the value of the n-th bin in a slice depends on
    how many that slice already has -- Z^2 has diminishing returns in bin count -- so
    "give the next bin to whoever gains most" is only correct if taken to convergence,
    which is what the DP does in one pass. O(n_slices * total * n_max).
    """
    n_slices = len(values_per_slice)
    if n_max is None:
        n_max = max(len(v) - 1 for v in values_per_slice)
    if total < n_min * n_slices:
        total = n_min * n_slices

    dp = np.full((n_slices + 1, total + 1), NEG)
    arg = np.full((n_slices + 1, total + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, n_slices + 1):
        vals = values_per_slice[k - 1]
        for spent in range(total + 1):
            best, best_n = NEG, -1
            for n in range(n_min, min(n_max, len(vals) - 1, spent) + 1):
                if vals[n] == NEG or dp[k - 1, spent - n] == NEG:
                    continue
                v = dp[k - 1, spent - n] + vals[n]
                if v > best:
                    best, best_n = v, n
            dp[k, spent] = best
            arg[k, spent] = best_n

    spent = int(np.argmax(dp[n_slices]))
    if dp[n_slices, spent] == NEG:
        return None
    counts = []
    for k in range(n_slices, 0, -1):
        n = int(arg[k, spent])
        counts.append(n)
        spent -= n
    return list(reversed(counts))


def _self_test(trials=400, seed=0):
    """Check partition_dp and allocate_budget against exhaustive enumeration.

    Both are short dynamic programs whose failure mode is silence: a subtly wrong
    recurrence still returns a partition, and the binning it produces still looks like a
    binning. On problems small enough to enumerate completely there is no reason to
    settle for a plausible answer, so this compares against every partition there is.

    Pure numpy, no ROOT and no input files, so it runs anywhere:
        python3 -m StatInference.common.binning_dp
    The half of this module that does need ROOT is checked by parity_check() instead.
    """
    import itertools

    def brute_partitions(n, k):
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = (0,) + cuts + (n,)
            parts = [(bounds[i], bounds[i + 1] - 1) for i in range(k)]
            yield parts

    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(trials):
        n = int(rng.integers(2, 9))
        score = rng.normal(size=(n, n))
        valid = rng.random((n, n)) > rng.uniform(0.0, 0.6)
        for a in range(n):
            for b in range(n):
                if b < a:
                    valid[a, b] = False
                    score[a, b] = 0.0
        values, partitions = partition_dp(score, valid, n)
        for k in range(1, n + 1):
            best = NEG
            for part in brute_partitions(n, k):
                if all(valid[a, b] for a, b in part):
                    total = sum(score[a, b] for a, b in part)
                    best = max(best, total)
            got = values[k]
            if best == NEG:
                assert got == NEG, f"found a {k}-part partition where none exists"
            else:
                assert (
                    abs(got - best) == 0 or abs(got - best) < 1e-9
                ), f"partition_dp n={n} k={k}: {got} != optimum {best}"
                part = partitions[k]
                assert len(part) == k
                assert (
                    part[0][0] == 0 and part[-1][1] == n - 1
                ), "does not cover the axis"
                assert all(
                    part[i][1] + 1 == part[i + 1][0] for i in range(k - 1)
                ), "parts are not contiguous"
                assert all(valid[a, b] for a, b in part), "returned an invalid range"
            checked += 1

    alloc_checked = 0
    for _ in range(trials // 2):
        n_slices = int(rng.integers(2, 5))
        n_max = int(rng.integers(2, 7))
        total = int(rng.integers(n_slices, n_slices * n_max + 1))
        values_per_slice = []
        for _ in range(n_slices):
            # concave by construction, as Z^2 in bin count is
            curve = [NEG] + list(np.sort(rng.random(n_max))[::-1].cumsum())
            for i in range(1, n_max + 1):
                if rng.random() < 0.15:
                    curve[i] = NEG
            values_per_slice.append(curve)
        got = allocate_budget(values_per_slice, total, n_min=1, n_max=n_max)
        best, best_combo = NEG, None
        for combo in itertools.product(range(1, n_max + 1), repeat=n_slices):
            if sum(combo) > total:
                continue
            if any(values_per_slice[k][combo[k]] == NEG for k in range(n_slices)):
                continue
            value = sum(values_per_slice[k][combo[k]] for k in range(n_slices))
            if value > best:
                best, best_combo = value, combo
        if best_combo is None:
            assert got is None, "allocated a budget where no feasible split exists"
        else:
            assert got is not None, "found no allocation though one exists"
            assert sum(got) <= total, "allocation overspends the budget"
            value = sum(values_per_slice[k][got[k]] for k in range(n_slices))
            assert (
                abs(value - best) < 1e-9
            ), f"allocate_budget: {value} != optimum {best}"
        alloc_checked += 1

    return checked, alloc_checked


if __name__ == "__main__":
    n_part, n_alloc = _self_test()
    print(f"partition_dp:    {n_part} (n, k) cases vs exhaustive enumeration -- OK")
    print(f"allocate_budget: {n_alloc} cases vs exhaustive enumeration -- OK")


def binning_objective(cells, slices, mode):
    """(total Z^2, per-slice Z^2) for a finished binning.

    The quantity the search maximises, recomputed from the binning that was actually
    written rather than carried out of the optimiser. That makes it meaningful for the
    greedy strategy too -- which never computes it -- so the two can be compared from
    their binning.json alone, and it means a discrepancy between what the optimiser
    thought it achieved and what the shapes contain shows up as a discrepancy rather
    than going unnoticed.

    Note this is evaluated on the ranges as recorded, i.e. after extend_outer_edges has
    pushed the outermost ones into under/overflow, so it describes the shapes on disk.
    """
    per_slice = []
    for sl in slices:
        if "y_range" in sl:  # HME box: selection on y, bins along x
            ylo, yhi = sl["y_range"]
            per_slice.append(
                sum(cells.score(a, b, ylo, yhi, mode) for a, b in sl["x_ranges"])
            )
        else:  # DNN slice: selection on x, bins along y
            xlo, xhi = sl["x_range"]
            per_slice.append(
                sum(cells.score(xlo, xhi, a, b, mode) for a, b in sl["y_ranges"])
            )
    return sum(per_slice), per_slice


def _ranges_from_column(column, n, include_outer=False):
    """M[i, j] = the rectangle sum for bins (i+1)..(j+1), from one column of a prefix sum.

    For a fixed x window, P[x1+1, :] - P[x0, :] is the running sum along y, and every
    y range is a difference of two of its entries -- so the whole (n, n) table of
    candidate ranges is one outer subtraction rather than n^2 lookups.

    With include_outer the first and last bins swallow the under- and overflow, which is
    what extend_outer_edges() does to the outermost ranges before they are written. A
    search that scores the ranges without them is optimising something slightly different
    from what ends up in the datacard -- 0.11% of the background on the DNN axis of these
    shapes, small but not nothing, and concentrated entirely in the two outermost slices.
    """
    lo = column[1 : n + 1].copy()
    hi = column[2 : n + 2].copy()
    if include_outer:
        lo[0] = column[0]
        hi[-1] = column[n + 2]
    return hi[None, :] - lo[:, None]


def _neff_matrix(value, error):
    """effective_entries() over arrays, with its two degenerate cases kept.

    error <= 0 means the yield carries no MC uncertainty at all: infinite effective
    entries if there is something there, none if there is not. Taking (v/e)^2 blindly
    would make the first case a division by zero and the second a nan, and a nan
    compares false against every threshold -- which would silently reject exactly the
    empty bins the scalar gate accepts.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(error > 0, value / np.where(error > 0, error, 1.0), 0.0) ** 2
    degenerate = np.where(value > 0, np.inf, 0.0)
    return np.where(error > 0, ratio, degenerate)


def _asimov_z2_matrix(s, b, b_err):
    """asimov_significance(...)**2 over arrays, matching the scalar branch for branch."""
    out = np.zeros_like(s)
    live = (b > 0) & (s > 0)
    if not live.any():
        return out
    var = b_err**2
    no_err = live & (b_err <= 0)
    if no_err.any():
        ss, bb = s[no_err], b[no_err]
        out[no_err] = np.maximum(2.0 * ((ss + bb) * np.log1p(ss / bb) - ss), 0.0)
    with_err = live & (b_err > 0)
    if with_err.any():
        ss, bb, vv = s[with_err], b[with_err], var[with_err]
        term1 = (ss + bb) * np.log(((ss + bb) * (bb + vv)) / (bb * bb + (ss + bb) * vv))
        term2 = (bb * bb / vv) * np.log1p(vv * ss / (bb * (bb + vv)))
        out[with_err] = np.maximum(2.0 * (term1 - term2), 0.0)
    return out


def _sb_z2_matrix(s, b, b_err):
    """significance(..., mode="sb")**2 over arrays: S^2 / (B + sigma_B^2)."""
    denom = b + b_err**2
    ok = (b > 0) & (denom > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, s**2 / np.where(ok, denom, 1.0), 0.0)


def slice_tables(cells, x0, x1, exempt, knobs, mode):
    """(score, valid) over every candidate mass bin of one slice, as (ny, ny) arrays.

    The scalar path -- building a yields dict and an errors dict per candidate and
    calling _bin_passes -- costs about 80 microseconds a candidate, and there are
    ny(ny+1)/2 of them per slice. That is minutes per era spent in dict construction, so
    the arithmetic is done as whole-array operations here instead.

    This is the one place that restates the gate rather than calling it, which is a real
    risk: two spellings of the same rule are two rules as soon as one of them is edited.
    It is contained by _check_mask_against_gate(), which re-evaluates the canonical
    _bin_passes on a random sample of candidates every time these tables are built and
    raises on the first disagreement -- so the fast path is checked against the slow one
    on real inputs, on every run, rather than at review time.
    """
    ny = cells.ny
    sig = _ranges_from_column(cells._sig[x1 + 1] - cells._sig[x0], ny)
    b_tot = _ranges_from_column(cells._bkg_tot[x1 + 1] - cells._bkg_tot[x0], ny)
    v_tot = _ranges_from_column(cells._var_tot[x1 + 1] - cells._var_tot[x0], ny)
    b_tot = np.where(np.abs(b_tot) < cells._eps_tot, 0.0, b_tot)
    err_tot = np.sqrt(np.maximum(v_tot, 0.0))

    valid = np.triu(np.ones((ny, ny), dtype=bool))
    if knobs["min_bin_bkg_neff"] > 0:
        valid &= _neff_matrix(b_tot, err_tot) >= knobs["min_bin_bkg_neff"]

    for name in cells.names:
        b_p = _ranges_from_column(cells._bkg[name][x1 + 1] - cells._bkg[name][x0], ny)
        b_p = np.where(np.abs(b_p) < cells._eps_bkg[name], 0.0, b_p)
        # positivity is never exemptable -- see _bin_passes
        valid &= b_p >= 0
        if name in exempt:
            continue
        valid &= b_p > knobs["min_bin_bkg_each"]
        if knobs["min_bin_bkg_each_neff"] > 0:
            v_p = _ranges_from_column(
                cells._var[name][x1 + 1] - cells._var[name][x0], ny
            )
            err_p = np.sqrt(np.maximum(v_p, 0.0))
            valid &= _neff_matrix(b_p, err_p) >= knobs["min_bin_bkg_each_neff"]

    if mode == "asimov":
        score = _asimov_z2_matrix(sig, b_tot, err_tot)
    else:
        score = _sb_z2_matrix(sig, b_tot, err_tot)
    score = np.where(valid, score, 0.0)
    return score, valid


def check_mask_against_gate(
    cells,
    x0,
    x1,
    valid,
    score,
    exempt,
    bin_passes,
    mode,
    n_samples=150,
    seed=0,
):
    """Re-derive a sample of the vectorised tables with the canonical scalar gate.

    Cheap -- a few hundred calls against the ~11000 the tables replace -- and it is what
    makes the fast path safe to trust: if slice_tables() and _bin_passes ever disagree
    about a candidate bin, the run stops here instead of quietly producing a different
    binning.
    """
    ny = cells.ny
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        a = int(rng.integers(1, ny + 1))
        b = int(rng.integers(a, ny + 1))
        want = bool(bin_passes(x0, x1, a, b, exempt))
        got = bool(valid[a - 1, b - 1])
        if got != want:
            raise AssertionError(
                f"binning_dp vectorised gate disagrees with _bin_passes on bins "
                f"{a}..{b} of slice x={x0}..{x1}: fast path says "
                f"{'valid' if got else 'invalid'}, _bin_passes says "
                f"{'valid' if want else 'invalid'}. The two spellings of the bin gate "
                "have drifted; slice_tables() must be brought back in step with it."
            )
        if want:
            expected = cells.score(x0, x1, a, b, mode)
            if abs(expected - score[a - 1, b - 1]) > 1e-9 * max(abs(expected), 1.0):
                raise AssertionError(
                    f"binning_dp vectorised score disagrees with significance() on bins "
                    f"{a}..{b} of slice x={x0}..{x1}: {score[a - 1, b - 1]!r} vs "
                    f"{expected!r}."
                )


def trim_by_marginal_gain(values_per_slice, counts, threshold):
    """Give back every bin whose last split bought less than `threshold`.

    Pooling the budget was supposed to move resolution from the slices that cannot use it
    to the one that can. Measured over the 90 bbWW DL category-mass problems, it did not:
    it raised the total bin count 23% and left the top slice's share of Z^2 where it was,
    because the pool is what bin_budget() *licenses* and that has enough slack for every
    slice to take more. With Z^2 very nearly monotone in bin count, a search told only to
    maximise it will always spend whatever it is given.

    So the binding constraint has to be the value of a bin, not the size of a pot. A bin
    is kept only if the split that created it raised this category's Z^2 by at least
    `threshold` -- expressed as a fraction of the category's own achievable total, so it
    means "worth something on the scale that matters here" rather than a yield in events.
    The lowest DNN slice carries 0.1% of the total Z^2 across hundreds of bins, and every
    one of those bins is MC-statistical exposure bought for nothing; this is what declines
    them.

    Stepping down one bin at a time and stopping at the first marginal gain that clears
    the threshold is exact for a concave curve, which Z^2 in bin count is: each extra
    split has less left to separate. It is not assumed -- a curve that is not concave
    simply stops at its first qualifying step, which is still a count whose last bin paid
    for itself.
    """
    if threshold <= 0:
        return list(counts)
    trimmed = []
    for values, n in zip(values_per_slice, counts):
        while n > 1 and values[n] - values[n - 1] < threshold:
            n -= 1
        trimmed.append(n)
    return trimmed


def _exempt_masks(cells, min_frac, y0, y1, n, axis):
    """Per-process boolean masks saying, for every candidate range, whether that process
    is negligible inside it.

    minor_backgrounds() judged over every candidate window at once. It stays the same
    question it answers scalar-side -- is this process below min_frac of the total over
    the range about to be subdivided -- but on the sliced axis there are nx(nx+1)/2 such
    ranges and each one has its own answer, so it has to be a mask rather than a set.
    """
    masks = {}
    if min_frac <= 0:
        return masks
    total = _axis_ranges(cells._bkg_tot, y0, y1, n, axis, True)
    for name in cells.names:
        value = _axis_ranges(cells._bkg[name], y0, y1, n, axis, True)
        with np.errstate(invalid="ignore"):
            masks[name] = (total > 0) & (value < min_frac * total)
    return masks


def _axis_ranges(prefix, a, b, n, axis, include_outer=False):
    """All ranges along `axis`, at a fixed window [a, b] on the other one."""
    if axis == "y":
        column = prefix[b + 1] - prefix[a]
    else:
        column = prefix[:, b + 1] - prefix[:, a]
    return _ranges_from_column(column, n, include_outer)


def slice_axis_tables(cells, y_partitions, knobs, mode, exempt_full):
    """(score, valid) for every candidate slice on the sliced axis, per slice index.

    Returns arrays of shape (n_slices, nx, nx): entry [k, i, j] is what the x range
    (i+1)..(j+1) would score if it were slice k, carrying slice k's current mass binning,
    and whether it may be used that way at all.

    Three conditions beyond "the slice itself passes its gates", each of which the obvious
    version omits and each of which breaks something different:

    - the candidate must be able to *afford* the mass bins it would inherit,
      len(P_k) <= bin_budget(candidate). Without it the search shrinks a slice while
      keeping the bin count the wider slice licensed, which silently defeats bkg_per_bin
      -- the one knob holding the bin count down.
    - every inherited mass bin must still pass _bin_passes inside the candidate, judged
      against the *candidate's own* exempt set. The exempt set depends on the x range, so
      the mass-bin validity mask is not invariant under a move of the slice boundaries.
      This is also what makes the alternation monotone: with it, the incumbent binning is
      a member of every step's feasible set, so a step can never return something worse.
    - the slice-level gates use the exempt set computed once over the whole axis, which is
      what find_slices() does -- a background negligible in this category must not be able
      to veto every boundary.
    """
    nx = cells.nx
    ny = cells.ny
    n_slices = len(y_partitions)
    score = np.zeros((n_slices, nx, nx))
    valid = np.zeros((n_slices, nx, nx), dtype=bool)

    # slice-level quantities span the whole y axis including under/overflow, because that
    # is what _integral() on a TH2 does -- see this module's docstring
    b_slice = _axis_ranges(cells._bkg_tot, 0, ny + 1, nx, "x", True)
    b_slice = np.where(np.abs(b_slice) < cells._eps_tot, 0.0, b_slice)
    v_slice = _axis_ranges(cells._var_tot, 0, ny + 1, nx, "x", True)
    err_slice = np.sqrt(np.maximum(v_slice, 0.0))

    slice_ok = np.triu(np.ones((nx, nx), dtype=bool))
    slice_ok &= b_slice > knobs["min_slice_bkg_sum"]
    if knobs["min_slice_bkg_neff"] > 0:
        slice_ok &= _neff_matrix(b_slice, err_slice) >= knobs["min_slice_bkg_neff"]
    for name in cells.names:
        b_p = _axis_ranges(cells._bkg[name], 0, ny + 1, nx, "x", True)
        b_p = np.where(np.abs(b_p) < cells._eps_bkg[name], 0.0, b_p)
        slice_ok &= b_p >= 0
        if name in exempt_full:
            continue
        if knobs["min_slice_bkg_each"] > 0:
            slice_ok &= b_p > knobs["min_slice_bkg_each"]
        if knobs["min_slice_bkg_each_neff"] > 0:
            v_p = _axis_ranges(cells._var[name], 0, ny + 1, nx, "x", True)
            slice_ok &= (
                _neff_matrix(b_p, np.sqrt(np.maximum(v_p, 0.0)))
                >= knobs["min_slice_bkg_each_neff"]
            )

    # how many mass bins each candidate slice could afford, from the same yield and the
    # same arithmetic bin_budget() uses (y = 1..ny, as the greedy path's ProjectionY)
    b_for_budget = _axis_ranges(cells._bkg_tot, 1, ny, nx, "x", True)
    if knobs["bkg_per_bin"] > 0:
        with np.errstate(invalid="ignore"):
            afford = np.where(
                b_for_budget <= 0,
                1,
                np.minimum(
                    knobs["max_bins_per_slice"],
                    (b_for_budget / knobs["bkg_per_bin"]).astype(int),
                ),
            )
        afford = np.maximum(afford, 1)
    else:
        afford = np.full((nx, nx), knobs["max_bins_per_slice"])

    exempt_masks = _exempt_masks(cells, knobs["min_bkg_frac"], 1, ny, nx, "x")

    for k, partition in enumerate(y_partitions):
        ok = slice_ok & (afford >= len(partition))
        total = np.zeros((nx, nx))
        for a, b in partition:
            sig = _axis_ranges(cells._sig, a, b, nx, "x", True)
            b_tot = _axis_ranges(cells._bkg_tot, a, b, nx, "x", True)
            b_tot = np.where(np.abs(b_tot) < cells._eps_tot, 0.0, b_tot)
            v_tot = _axis_ranges(cells._var_tot, a, b, nx, "x", True)
            err = np.sqrt(np.maximum(v_tot, 0.0))
            if knobs["min_bin_bkg_neff"] > 0:
                ok &= _neff_matrix(b_tot, err) >= knobs["min_bin_bkg_neff"]
            for name in cells.names:
                b_p = _axis_ranges(cells._bkg[name], a, b, nx, "x", True)
                b_p = np.where(np.abs(b_p) < cells._eps_bkg[name], 0.0, b_p)
                ok &= b_p >= 0
                magnitude = b_p > knobs["min_bin_bkg_each"]
                if knobs["min_bin_bkg_each_neff"] > 0:
                    v_p = _axis_ranges(cells._var[name], a, b, nx, "x", True)
                    magnitude &= (
                        _neff_matrix(b_p, np.sqrt(np.maximum(v_p, 0.0)))
                        >= knobs["min_bin_bkg_each_neff"]
                    )
                mask = exempt_masks.get(name)
                ok &= magnitude if mask is None else (magnitude | mask)
            if mode == "asimov":
                total += _asimov_z2_matrix(sig, b_tot, err)
            else:
                total += _sb_z2_matrix(sig, b_tot, err)
        valid[k] = ok
        score[k] = np.where(ok, total, 0.0)
    return score, valid


def partition_dp_staged(score, valid, n_parts):
    """partition_dp where each part has its own score matrix, indexed by part number.

    The sliced axis needs this because a candidate range is not worth the same amount in
    every position: it carries the mass binning of whichever slice it becomes, so
    score[k] is the table for the k-th slice and the DP has to keep track of which one it
    is placing. Otherwise identical -- exactly n_parts contiguous valid ranges covering
    the axis, maximising the total.
    """
    n = score.shape[1]
    dp = np.full((n_parts + 1, n + 1), NEG)
    arg = np.full((n_parts + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, n_parts + 1):
        masked = np.where(valid[k - 1], score[k - 1], NEG)
        for e in range(k, n + 1):
            cand = dp[k - 1, :e] + masked[:e, e - 1]
            s = int(np.argmax(cand))
            if cand[s] > NEG:
                dp[k, e] = cand[s]
                arg[k, e] = s
    if dp[n_parts, n] == NEG:
        return None
    parts = []
    e = n
    for k in range(n_parts, 0, -1):
        s = int(arg[k, e])
        parts.append((s, e - 1))
        e = s
    return list(reversed(parts))


def fit_log_quadratic(x, values, variances, min_neff=1.0, last_k=12, monotone=True):
    """Weighted fit of log y = a + b x + c x^2 to the falling side of a profile.

    Returns (params, covariance, indices used) or None. The weights are the bins' own
    effective entries, so neither the well-populated bulk nor the near-empty tail can
    dominate; a linear-space least squares over a distribution spanning four decades is
    dominated by the tail bins, whose absolute errors are tiny, and under-predicts by 40%.

    The turn-over is found on a smoothed profile because these distributions are flat
    through the bulk and fall above it -- fitted across both, the quadratic comes out
    rising. With `monotone` the curve is constrained to fall over the whole range it will
    be used on (c <= 0 and b + 2 c x_start <= 0); unconstrained, ~10% of boxes produced a
    tail that turns back up, which is the quadratic fitting noise at the sparse end.
    """
    errors = np.sqrt(np.maximum(variances, 0.0))
    neff = _neff_matrix(values, errors)
    smooth = np.convolve(np.where(neff >= 1, values, 0.0), np.ones(5) / 5.0, "same")
    # errors > 0 as well as the effective-entry cut: _neff_matrix reports infinite
    # effective entries for a bin with content and no error (a sample with a single
    # unweighted event there), and such a bin would carry infinite weight below.
    good = np.where((neff >= min_neff) & (values > 0) & (errors > 0))[0]
    good = good[good > int(np.argmax(smooth))]
    if len(good) < 4:
        return None
    used = good[-last_k:]
    X = np.vstack([x[used] ** k for k in range(3)]).T
    w = (values[used] / errors[used]) ** 2
    if not np.all(np.isfinite(w)):
        return None
    try:
        cov = np.linalg.pinv(X.T @ (w[:, None] * X))
    except np.linalg.LinAlgError:
        # a degenerate window: no fit rather than a fit nobody can trust
        return None
    p = cov @ (X.T @ (w * np.log(values[used])))
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(cov)):
        return None
    if monotone and (p[2] > 0 or p[1] + 2 * p[2] * x[used].min() > 0):
        from scipy.optimize import minimize

        logy = np.log(values[used])
        x0 = float(x[used].min())
        res = minimize(
            lambda q: float(np.sum(w * (logy - X @ q) ** 2)),
            p,
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda q: -q[2]},
                {"type": "ineq", "fun": lambda q: -(q[1] + 2 * q[2] * x0)},
            ],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success:
            p = res.x
    return p, cov, used


def fitted_tail_full(x, values, variances, knobs):
    """A profile with its unreliable tail replaced by the fit of its own measured part.

    Hybrid on purpose: bins the MC measures are kept as measured, and only the bins past
    the last one with `fit_tail_min_neff` effective entries are taken from the fit, and
    only where the fit is better determined than the count it replaces. The fit carries
    the normalisation of the measured bins it was constrained by, so nothing is rescaled.

    Returns (values, variances, n_replaced, info), where info carries the fit parameters,
    their covariance and which bins were replaced -- needed to propagate the uncertainty
    of an *integral* over several fitted bins, whose errors are not independent. The
    inputs are returned unchanged when the fit cannot be constrained.
    """
    out_v, out_var = values.copy(), variances.copy()
    res = fit_log_quadratic(
        x, values, variances, knobs.get("fit_tail_fit_min_neff", 1.0),
        int(knobs.get("fit_tail_last_k", 12)),
    )
    if res is None:
        return out_v, out_var, 0, None
    p, cov, used = res
    errors = np.sqrt(np.maximum(variances, 0.0))
    neff = _neff_matrix(values, errors)
    threshold = knobs["fit_tail_min_neff"]
    measured = np.where(neff >= threshold)[0]
    last_measured = measured.max() if len(measured) else -1
    n = 0
    replaced = []
    started = False
    for i in range(int(last_measured) + 1, len(x)):
        X = np.array([x[i] ** k for k in range(3)])
        mu = float(np.exp(X @ p))
        sigma = mu * math.sqrt(max(float(X @ cov @ X), 0.0))
        if not started:
            # where the tail starts: the first bin past the measured region where the fit
            # is better determined than the count and is itself pinned down. Once it
            # starts, it runs to the end of the axis -- the tail is one object, and
            # leaving counted bins interspersed inside it would mean a template that is
            # part fit and part noise over the same falling edge.
            if sigma >= errors[i] > 0 or (mu > 0 and sigma / mu > 1.0):
                continue
            started = True
        out_v[i], out_var[i] = mu, sigma**2
        replaced.append(i)
        n += 1
    return out_v, out_var, n, {"p": p, "cov": cov, "replaced": replaced}


def fitted_tail(x, values, variances, knobs):
    """fitted_tail_full() without the fit record, for callers that only need the profile."""
    v, var, n, _ = fitted_tail_full(x, values, variances, knobs)
    return v, var, n


def fitted_integral_error(x, p, cov, indices):
    """Uncertainty of the summed fitted yield over `indices`, correlations included.

    The bins of one box share three fitted parameters, so their uncertainties are nearly
    fully correlated: summing their variances in quadrature understates the error on an
    integral over them by a median factor 1.4 on these shapes, and by up to 2.2. The
    integral's gradient is sum_i mu_i * X_i, and its variance g^T C g.
    """
    if not len(indices):
        return 0.0
    g = np.zeros(len(p))
    for i in indices:
        X = np.array([x[i] ** k for k in range(len(p))])
        g = g + math.exp(float(X @ p)) * X
    return math.sqrt(max(float(g @ cov @ g), 0.0))


def profile_column(values):
    """The prefix-sum column _ranges_from_column() expects, from a per-bin profile."""
    return np.concatenate([[0.0], np.cumsum(values)])


def box_tables(cells, y0, y1, exempt, knobs, mode, override=None):
    """(score, valid) over every candidate bin of the *sliced-on-y* layout.

    slice_tables() with the axes exchanged: there the window is on x and the bins run
    along y, here the window is a cut on y -- an HME box around the resonance -- and the
    bins run along x, the DNN. Same gates, same figure of merit; only which axis is the
    selection and which carries the shape has changed.

    The bins tile the whole x axis including its under/overflow, because everything inside
    the box is kept and binned. The box itself is a selection: what falls outside it is
    discarded, not swept into an outer bin, which is the difference between a cut and a
    slice.

    `exempt` is the set of processes excused from the per-bin floors over the whole box,
    or None to judge that per candidate bin: a process is then excused in exactly the
    bins where it is under min_bkg_frac of that bin's own background. Positivity is never
    excused either way -- see _bin_passes.
    """
    nx = cells.nx
    ranges = lambda prefix: _axis_ranges(prefix, y0, y1, nx, "x", True)
    override = override or {}

    def proc(name):
        """This process's range table inside the window, fitted profile or counted."""
        if name in override:
            v, _ = override[name]
            return _ranges_from_column(profile_column(v), nx, True)
        return ranges(cells._bkg[name])

    def proc_var(name):
        if name in override:
            _, var = override[name]
            return _ranges_from_column(profile_column(var), nx, True)
        return ranges(cells._var[name])

    sig = ranges(cells._sig)
    b_tot = ranges(cells._bkg_tot)
    for name, (v, var) in override.items():
        # the total follows whatever this process was replaced by
        b_tot = b_tot - ranges(cells._bkg[name]) + _ranges_from_column(
            profile_column(v), nx, True
        )
    b_tot = np.where(np.abs(b_tot) < cells._eps_tot, 0.0, b_tot)
    v_tot = ranges(cells._var_tot)
    for name, (v, var) in override.items():
        v_tot = v_tot - ranges(cells._var[name]) + _ranges_from_column(
            profile_column(var), nx, True
        )
    err = np.sqrt(np.maximum(v_tot, 0.0))

    valid = np.triu(np.ones((nx, nx), dtype=bool))
    if knobs["min_bin_bkg_neff"] > 0:
        valid &= _neff_matrix(b_tot, err) >= knobs["min_bin_bkg_neff"]
    per_bin = exempt is None
    min_frac = knobs["min_bkg_frac"]
    for name in cells.names:
        b_p = proc(name)
        b_p = np.where(np.abs(b_p) < cells._eps_bkg[name], 0.0, b_p)
        valid &= b_p >= 0
        if not per_bin and name in exempt:
            continue
        # None: no magnitude floor, only the non-negativity already applied above
        min_each = knobs["min_bin_bkg_each"]
        magnitude = (
            b_p > min_each if min_each is not None else np.ones_like(valid, dtype=bool)
        )
        required = knobs.get("min_bin_bkg_neff_by_process", {}).get(
            name, knobs["min_bin_bkg_each_neff"]
        )
        if required > 0:
            v_p = proc_var(name)
            magnitude &= _neff_matrix(b_p, np.sqrt(np.maximum(v_p, 0.0))) >= required
        if per_bin and min_frac > 0:
            # the same test Cells.exempt() makes, asked of each candidate bin
            with np.errstate(invalid="ignore"):
                magnitude |= (b_tot > 0) & (b_p < min_frac * b_tot)
        valid &= magnitude
    score = (
        _asimov_z2_matrix(sig, b_tot, err)
        if mode == "asimov"
        else _sb_z2_matrix(sig, b_tot, err)
    )
    return np.where(valid, score, 0.0), valid


def check_box_mask(
    cells,
    y0,
    y1,
    valid,
    score,
    exempt,
    bin_passes,
    mode,
    n_samples=100,
    seed=0,
    min_frac=0.0,
):
    """The cross-check of box_tables() against the canonical scalar gate.

    Same contract as check_mask_against_gate(): the fast path is re-derived on a random
    sample with _bin_passes and significance() on every build, so the two spellings of the
    rule cannot drift unnoticed.
    """
    nx = cells.nx
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        a = int(rng.integers(1, nx + 1))
        b = int(rng.integers(a, nx + 1))
        # the outermost bins carry the x under/overflow, because box_tables builds them
        # with include_outer -- the reference has to ask the same question
        lo = 0 if a == 1 else a
        hi = nx + 1 if b == nx else b
        exempt_here = (
            cells.exempt(lo, hi, y0, y1, min_frac) if exempt is None else exempt
        )
        want = bool(bin_passes(lo, hi, y0, y1, exempt_here))
        got = bool(valid[a - 1, b - 1])
        if got != want:
            raise AssertionError(
                f"binning_dp box gate disagrees with _bin_passes on DNN bins {a}..{b} of "
                f"the HME box {y0}..{y1}: fast path says {'valid' if got else 'invalid'}, "
                f"_bin_passes says {'valid' if want else 'invalid'}."
            )
        if want:
            expected = cells.score(lo, hi, y0, y1, mode)
            if abs(expected - score[a - 1, b - 1]) > 1e-9 * max(abs(expected), 1.0):
                raise AssertionError(
                    f"binning_dp box score disagrees with significance() on DNN bins "
                    f"{a}..{b} of box {y0}..{y1}: {score[a - 1, b - 1]!r} vs {expected!r}."
                )
