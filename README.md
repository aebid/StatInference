# StatInference

## Two ways bins get decided

Two independent things in this repository choose the binning of the shapes that go into
the datacards. They do not share code, and an analysis uses one or the other. **Neither is
part of the datacard chain**: both are offline pre-steps that produce input the chain then
consumes, so `CreateDatacardsTask` reads whatever shapes it is pointed at and does no
binning of its own.

**1. Shape-driven 2D→1D binning — `bin_opt_2d/rebin_2d.py`.** Derives bin edges from the
shapes themselves (no fits, no limits): it cuts the x axis of a 2D input into slices that
become datacard categories, and rebins y into the bins of each slice. It writes the
rebinned shapes as `<output>/<source era>/<variable>/<variable>.root`, plus the
`binning.json` recording the edges it chose, beside them.

It runs as the datacard configuration's **preprocessing step**, not as part of this
repository's logic. `PreprocessShapesTask` runs whatever `preprocess:` names, supplying
`--input`, `--output`, `--era` and `--config` (the datacard configuration); nothing here
knows what the step does, and a
configuration that declares no `preprocess:` block skips the task entirely and reads the
merged histograms as they are:

```yaml
preprocess:
  script: StatInference/bin_opt_2d/rebin_2d.py
  args: [--binning-config, config/Datacards/binning_2d.yaml]
```

The knobs live with the analysis (`config/Datacards/binning_2d.yaml`) because they are
analysis configuration; the derived `binning.json` lives with the shapes it produced,
because it is a product of the run rather than an input to it.

Which era is being produced decides how it is derived. A plain era is binned on its own
statistics, for a standalone limit; an era that is a key of `era_groups:` is binned on its
members' summed statistics, and those edges are then applied to each member separately --
kept in their own files, because the datacard step sums them and a per-era lnN can only be
built by scaling a sub-era's own shape.

Each base category (`SR/res2b`) becomes per-slice categories named by the `category_pattern`
knob, e.g. `{base_category}_dnn{slice_idx}`. The pattern is the analysis's choice — nothing
here assumes the sliced axis is a DNN score. `common/tools.py:CategoryNaming` both writes
those names and parses them back from that one pattern.

**2. Limit-driven binning optimisation — `bin_opt/`.** A search harness, documented below:
it builds candidate binnings, runs limits with combine for each, and ranks them. Its product
is a `hist_bins` JSON, applied at datacard time by `dc_make/binner.py`. Every module is a
script driven by `bin_opt/bin_optimization.yaml`, and it exposes no importable API.

Which one an analysis is on is visible in its configuration: the `bin_opt` path sets
`hist_bins` (as `config/x_hh_bbtautau_run2.yaml` does), while an analysis reading rebinned
shapes leaves `hist_bins` unset and instead lists the sliced `categories:` and the
`category_pattern` that names them (as HH→bbWW's `config/Datacards/x_hh_bbww_DL_run3.yaml`
does). An analysis using neither simply lists the categories its input already has.

`categories:` is always taken verbatim — it is the set of directories that exist in the
input shapes, and nothing in `dc_make` derives or expands it.

## Correlating uncertainties across eras

A datacard bin belonging to a meta-era (an entry of `era_groups:`) holds the summed
histograms of its sub-eras, so "which era is this nuisance for" has to be decided while
that sum is being built rather than by which bin it lands in.

The nuisance name and the input histogram name are the same string — `dc_make/maker.py`
reads `<process>_<name>_<Up|Down>` — so **an era-specific nuisance is named by the
producer**, not by `dc_make`. To split a source per era, give it an era-suffixed `name:`
in each era's `weights.yaml` on the analysis side and declare one scoped entry per era:

```yaml
  - name: CMS_pileup_2022
    type: shape
    eras: [Run3_2022]
  - name: CMS_pileup_2022EE
    type: shape
    eras: [Run3_2022EE]
```

A source that stays correlated keeps one unsuffixed `name:` in every era's `weights.yaml`
and one unscoped datacard entry. There is no `decorrelate:` key: whether a source is split
is visible from what the producer writes.

`DatacardMaker` makes this work in two places, both of which treat a real era as a
one-element meta-era so configurations without `era_groups:` are unaffected:

- `uncAppliesTo` registers the nuisance on the meta-era bin when **any** sub-era matches,
  since `Uncertainty.appliesTo` compares against the meta-era name and would otherwise
  drop the entry silently -- 64 of 76 shape sources for HH->bbWW, the jet energy scale
  among them. `getMVLnNValue` covers `hasMultipleValues` lookups, merging the sub-eras it
  finds a value for into one number weighted by their yields.
- `getCombinedShape` varies **only** the matching sub-eras and takes the others at their
  nominal, so the sum is the meta-era shape with one era shifted. This is what the lnN
  path has always done in `_getSubEraLnNVariedShapes`.

A sub-era the uncertainty does apply to is still required to have its variation
histogram; a missing one raises rather than being quietly treated as nominal. Suffixing
the producer without the datacard, or the reverse, therefore fails loudly at build time.

## How to run binning optimisation on lxplus
Open two separate LXPLUS terminals: one for **server side** and the other for **worker side**.
Set up the environment and proxy in analysis area as usual on both terminals.
```sh
source env.sh
voms-proxy-init -voms cms -rfc -valid 192:00
cd StatInference
```
Configure the `bin_opt/bin_optimization.yaml` file for the intended analysis (resonant or nonresonant), channels, era, etc.

Get the server side script running first (it creates worker output directories) and takes about a few seconds. Then move on to the worker side terminal and run the worker side script there.
### Server side
Better to run on screen. If screen does not work, you will need to have this server script below running on an active lxplus terminal.

Example:
For the current version of code, you will need to run the following script separately for each channel.
```sh
python3 bin_opt/optimize_channel.py --channel tauTau
```

### Worker side
Running on screen is not required, jobs are submitted on HTCondor.

Example:
for HTCondor jobs, run the following:
```sh
python3 bin_opt/submitLimitWorkers.py
```
The worker script can also be run locally as following:
```sh
python3 bin_opt/rebinAndRunLimitsWorker.py --channel tauTau
```

If runtime is longer than one day, both server and workers should be restarted.
Workers directory should be removed before restarting server/workers.

E.g.
```sh
rm -r output/binning_v2/tauTau_2018/workers
```
