"""A scratch analysis for the StatInference law tasks, on the real FLAF and law classes.

The analysis has three real eras behind the cards (Run3_2022, Run3_2022EE, Run3_2023), one
more configured era (Run3_2023BPix) and the datacard configurations below. Run3_2022EE reads
its merged histograms from a store of its own (`fs_HistTuple` in
config/Run3_2022EE/global.yaml), so an input built with another era's Setup points to the
wrong store and shows up in the paths.

The checkouts under test are found next to each other: `<parent>/StatInference` (this
repository) and `<parent>/FLAF`, which is how the unit-test workflow lays them out.
"""

import importlib.util
import os
import sys
from unittest import mock

statinference_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkout_parent = os.path.dirname(statinference_repo)
flaf_repo = os.path.join(checkout_parent, "FLAF")
if checkout_parent not in sys.path:
    sys.path.insert(0, checkout_parent)

import luigi.task_register  # noqa: E402

# FLAF's task modules import ROOT, awkward, uproot and Corrections at module level, and the
# limit and plot tasks import dhi. The unit-test runner has none of them, and nothing tested
# here uses them: a stand-in is registered, for these imports alone, for each one that cannot
# be imported for real, and the previous entries are put back afterwards.
_HEAVY_MODULES = [
    "ROOT",
    "awkward",
    "uproot",
    "Corrections",
    "Corrections.Corrections",
    "Corrections.CorrectionsCore",
    "Corrections.lumi",
    "dhi",
    "dhi.config",
    "dhi.tasks",
    "dhi.tasks.resonant",
    "dhi.tasks.pulls_impacts",
]


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_replaced = {
    name: sys.modules.get(name) for name in _HEAVY_MODULES if not _importable(name)
}
sys.modules.update({name: mock.MagicMock() for name in _replaced})
try:
    import FLAF.Common.Setup as Setup  # noqa: E402
    from FLAF.Analysis import tasks as analysis  # noqa: E402
    from FLAF.run_tools import law_customizations as lc  # noqa: E402
    from StatInference.law import tasks as si  # noqa: E402
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

ERA_A = "Run3_2022"
ERA_B = "Run3_2022EE"
ERA_C = "Run3_2023"
GROUP = "Run3_Early"
ERAS = (ERA_A, ERA_B, ERA_C)
#: a configured era behind no card
EXTRA_ERA = "Run3_2023BPix"

#: the eras of every card: a group, one of its members and a standalone era
CARD_ERAS = [GROUP, ERA_A, ERA_C]
SUB_ERAS = {GROUP: [ERA_A, ERA_B], ERA_A: [ERA_A], ERA_C: [ERA_C]}

#: the mass points of every card, each a branch of its own: global.yaml restricts none
MASSES = [300, 500]
VARIABLES = [f"DNN_m{m}_vs_HME" for m in MASSES]

CARD = """
analysis: test
type: resonant
model:
  parameters: [ MX ]
  input_file_pattern: ${{ERA}}/DNN_m${{MX}}_vs_HME/DNN_m${{MX}}_vs_HME.root
eras: [ {group}, {era_a}, {era_c} ]
era_groups:
  {group}: [ {era_a}, {era_b} ]
channels: [ eE, muMu ]
categories: [ OS_Iso/res2b, OS_Iso/recovery ]
{preprocess}
processes:
  - process: data_obs
    is_data: true
  - process: Signal_${{MX}}
    param_values: [ {masses} ]
    is_signal: true
  - TT
"""

PREPROCESS = """preprocess:
  script: StatInference/bin_opt_2d/rebin_2d.py
  args:
    - --binning-config
    - config/Datacards/binning_${ERA}.yaml
    - --tag
    - ${ERA}
"""


def card(preprocess):
    return CARD.format(
        group=GROUP,
        era_a=ERA_A,
        era_b=ERA_B,
        era_c=ERA_C,
        masses=", ".join(str(m) for m in MASSES),
        preprocess=PREPROCESS if preprocess else "",
    )


#: the analysis configuration; {store} is the scratch directory the file systems point to,
#: {card} the datacard configuration StatInference reads
FILES = {
    "config/global.yaml": """
phys_model: M
channelSelection: [ eE, muMu ]
categories: [ res2b ]
regions: [ OS_Iso ]
analysis_config_area: config
fs_default: {store}/default
fs_HistTuple: {store}/hists
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: [ DNN_m300_vs_HME, DNN_m500_vs_HME ]
    fullResolution_variables: []
StatInference:
  config: {card}
  hist_bins: config/Datacards/bins_${ERA}.json
""",
    f"config/{ERA_B}/global.yaml": "fs_HistTuple: {store}/hists_2022EE\n",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n  data: [ data ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
    "config/plot/histograms.yaml": "".join(
        f"{v}:\n  var_list: [ DNN_m{m}, HME ]\n" for v, m in zip(VARIABLES, MASSES)
    ),
    "config/Datacards/card.yaml": card(preprocess=False),
    "config/Datacards/card_preprocess.yaml": card(preprocess=True),
}
for _era in ERAS + (EXTRA_ERA,):
    FILES[f"config/{_era}/processes.yaml"] = "data:\n  datasets: [ Data ]\n"
    FILES[f"config/{_era}/datasets.yaml"] = "Data: {}\nTTto2L2Nu: {}\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def make_analysis(tmp, card_file):
    """The scratch analysis under `tmp`, reading the datacard configuration `card_file`
    (relative to the analysis); returns its path."""
    ana = os.path.join(tmp, "ana")
    os.makedirs(ana)
    os.symlink(flaf_repo, os.path.join(ana, "FLAF"))
    os.symlink(statinference_repo, os.path.join(ana, "StatInference"))
    store = os.path.join(tmp, "store")
    for path, text in FILES.items():
        text = text.replace("{store}", store).replace("{card}", card_file)
        write(os.path.join(ana, path), text)
    return ana


def hist_store(era):
    """The fs_HistTuple directory of `era` in the scratch analysis, relative to the scratch
    directory."""
    return os.path.join("store", "hists_2022EE" if era == ERA_B else "hists")


def clear_process_caches():
    """Forget what earlier tasks of this process cached: instances, maps, Setups."""
    luigi.task_register.Register.clear_instance_cache()
    lc.Task._branch_map_cache.clear()
    Setup.Setup._global_instances.clear()
