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
  args: [--binning-config, config/Datacards/DL/binning_2d.yaml]
```

The knobs live with the analysis (`config/Datacards/DL/binning_2d.yaml`) because they are
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

## Datacard cards

The chain builds the datacard *cards* the analysis's `global.yaml` names in its
`StatInference` block: either `cards:`, each card's name and file (paths relative to the
analysis), or `config:`, a single card, which then goes by its file name. Setting both is an
error.

```yaml
StatInference:
  cards:
    DL_spin0: config/Datacards/DL/dl_spin0.yaml
    DL_spin2: config/Datacards/DL/dl_spin2.yaml
```

The name is used for the law branches (`{card, era[, mass]}`), the output paths
(`Datacards/<card>/<era>/...`, `ResonantLimitsTask/<card>/...`) and `--card` of the limit and
plot tasks (without it they run every card); it does not appear in the datacards.

A card can `include:` shared files, paths relative to the including file. The merge is strict
and never overrides: `processes`, `uncertainties`, `limit_plots` and `impact_plots` are joined
and an entry name defined twice raises; every other top-level key is set in one file only.
Includes do not nest. A card without `include:` reads as before. To see a card merged, with the
file each setting comes from:

```bash
python3 StatInference/common/card_config.py <card.yaml>
```

The chain hands its scripts (the preprocess step, `create_datacards.py`) the merged card as one
file, `card_config.yaml`, kept next to their output. See `common/card_config.py` for the rules.

## Unit tests

`test/test_law_chain.py` checks the law tasks on the real FLAF and law classes over a scratch
analysis (`test/statinference_fixture.py`): for each card, what it writes, which merged
histograms it reads and the commands it runs. `test/test_card_eras.py` checks the card
workflows over several eras, and `test/test_shape_plots.py` which histograms the shape plots
of a card show, for categories read through `input_categories`.
`test/test_card_config.py` checks the card loader's merge rules and a chain over several
cards, one of them with an include (not yet in the `unit-tests` workflow's file list).
They need no ROOT, Combine or dhi. The `unit-tests` workflow runs them against a pinned FLAF
commit; to run them locally, put a FLAF checkout at that commit (`ref:` in
`.github/workflows/unit-tests.yaml`) next to this one and run from the directory holding both:

```bash
pip install pyyaml numpy luigi==3.8.1 law==0.1.21 pytest
PYTHONPATH=$PWD python -m pytest -q StatInference/test/test_law_chain.py \
    StatInference/test/test_card_eras.py StatInference/test/test_shape_plots.py \
    StatInference/test/test_card_config.py
```
