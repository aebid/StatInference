#!/usr/bin/env python3
"""The histograms dc_make copies with Clone() must be freed.

From ROOT 6.40 PyROOT does not own what TH1::Clone() returns, and with importROOT's
TH1.AddDirectory(False) no directory does either, so a copy not handed to Python is never freed.
The meta-era shapes are summed into such a copy on every getShape() call: the loss grows with
the processes, bins and uncertainties of a card. ROOT 6.36, the one of the CMSSW release the
cards are built in, still owns the copies, so without the fix this test fails only under
ROOT >= 6.40.

Uncertainty.fitFlat's TF1 is counted as well: PyROOT gives a TF1 made in Python to C++, where
nothing deletes it, under every ROOT version.

The live objects are counted with ROOT's object table, which counts only the objects made
after it is switched on, so the calls run in a child process that switches it on first.

Needs ROOT: run inside an analysis environment (source env.sh).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.needs_root

statinference_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkout_parent = os.path.dirname(statinference_repo)

CARD = """
analysis: test
type: resonant
model:
  parameters: [ MX ]
  input_file_pattern: ${ERA}.root
signalFractionForRelevantBins: 0.2
eras: [ Run3_Early ]
era_groups:
  Run3_Early: [ Run3_2022, Run3_2023 ]
channels: [ eE ]
categories: [ SR ]
processes:
  - process: data_obs
    is_data: true
    is_asimov_data: true
  - process: Signal_${MX}
    param_values: [ 300 ]
    is_signal: true
  - TT
uncertainties:
  - name: lumi_2022
    type: lnN
    eras: [ Run3_2022 ]
    value: 0.014
  - name: jes
    type: auto
"""

CHILD = """
import gc, importlib.util, json, os, sys
from unittest import mock
sys.path.insert(0, sys.argv[1])
import ROOT
ROOT.TObject.SetObjectStat(True)
# dc_make imports CombineHarvester at module level, and nothing counted here calls it: outside the
# CMSSW environment a stand-in is registered in its place, as statinference_fixture does.
if importlib.util.find_spec("CombineHarvester") is None:
    for name in ("CombineHarvester", "CombineHarvester.CombineTools", "CombineHarvester.CombineTools.ch"):
        sys.modules[name] = mock.MagicMock()
from StatInference.dc_make.maker import DatacardMaker
from StatInference.dc_make.uncertainty import UncertaintyScale

def live():
    gc.collect()
    ROOT.gObjectTable.UpdateInstCount()
    return sum(
        int(getattr(ROOT, c).Class().GetInstanceCount()) for c in ("TH1F", "TH1D", "TF1")
    )

# In a function, so that no name keeps the proxy of a histogram that Close() deleted: cppyy
# would hand that proxy out again for a histogram made at the same address.
def write_input(path, scale):
    shapes = {
        "Signal_300": [1, 4, 9, 4],
        "TT": [40, 30, 20, 10],
        "TT_jes_Up": [44, 32, 20, 9],
        "TT_jes_Down": [36, 28, 20, 11],
    }
    f = ROOT.TFile(path, "RECREATE")
    f.mkdir("eE").mkdir("SR").cd()
    for name, contents in shapes.items():
        h = ROOT.TH1D(name, name, len(contents), 0.0, 1.0)
        for i, content in enumerate(contents, 1):
            h.SetBinContent(i, scale * content)
            h.SetBinError(i, 0.1 * scale * content)
        h.Write()
    f.Close()

tmp = sys.argv[2]
write_input(os.path.join(tmp, "Run3_2022.root"), 1.0)
write_input(os.path.join(tmp, "Run3_2023.root"), 2.0)
maker = DatacardMaker(os.path.join(tmp, "card.yaml"), tmp)
tt, data, signal = (maker.processes[n] for n in ("TT", "data_obs", "Signal_300"))
lumi, jes = maker.uncertainties["lumi_2022"], maker.uncertainties["jes"]
meta, era, ch, cat = "Run3_Early", "Run3_2022", "eE", "SR"

def resolve_jes():
    nominal = maker.getShape(tt, era, ch, cat, None)
    shapes = {s: maker.getShape(tt, era, ch, cat, None, "jes", s.name) for s in UncertaintyScale}
    return jes.resolveType(nominal, shapes, maker.autolnNThr, maker.asymlnNThr)

calls = {
    "getShape meta-era": lambda: maker.getShape(tt, meta, ch, cat, None),
    "getShape meta-era asimov": lambda: maker.getShape(data, meta, ch, cat, None),
    "getShape asimov": lambda: maker.getShape(data, era, ch, cat, None),
    "getMetaEraLnNShapeUnc": lambda: maker.getMetaEraLnNShapeUnc(lumi, tt, meta, ch, cat, None),
    "resolveType": resolve_jes,
}
# The signal shapes the negative-bin check consults, read first as createDatacards does.
maker.getShape(signal, meta, ch, cat, signal.params)
counts = {}
for name, call in calls.items():
    call()  # fills the shape cache
    base = live()
    counts[name] = []
    for _ in range(3):
        call()
        counts[name].append(live() - base)
for f in maker.input_files.values():
    f.Close()
del maker, calls
counts["left after the maker"] = live()
print(json.dumps(counts))
"""


class TestHistOwnership(unittest.TestCase):
    def test_copied_histograms_are_freed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "card.yaml"), "w") as f:
                f.write(CARD)
            result = subprocess.run(
                [sys.executable, "-c", CHILD, checkout_parent, tmp],
                capture_output=True,
                text=True,
                # FLAF's HistHelper, which the binner imports, includes its header from here
                env=dict(os.environ, ANALYSIS_PATH=checkout_parent),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            counts = json.loads(result.stdout.strip().splitlines()[-1])
            left = counts.pop("left after the maker")
            for name, growth in counts.items():
                self.assertEqual(
                    growth, [0, 0, 0], f"{name} left copies owned by nobody"
                )
            self.assertEqual(left, 0, "the cached copies outlived the maker")


if __name__ == "__main__":
    unittest.main()
