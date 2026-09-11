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

    def __init__(self, nx, ny, sig, bkg, var):
        self.nx = nx
        self.ny = ny
        self.names = list(bkg)
        self._sig = _prefix(sig)
        self._bkg = {name: _prefix(bkg[name]) for name in self.names}
        self._var = {name: _prefix(var[name]) for name in self.names}
        self._bkg_tot = _prefix(sum(bkg.values()))
        self._var_tot = _prefix(sum(var.values()))
        self._eps_sig = _cancellation_eps(sig)
        self._eps_bkg = {name: _cancellation_eps(bkg[name]) for name in self.names}
        self._eps_tot = _cancellation_eps(sum(bkg.values()))
        self._eps_var = {name: _cancellation_eps(var[name]) for name in self.names}
        self._eps_var_tot = _cancellation_eps(sum(var.values()))

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
    return Cells(nx, ny, sig, bkg, var)


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
