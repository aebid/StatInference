"""Every task in one namespace.

Kept as a module of its own so an analysis can go on naming
``StatInference.law.tasks`` in its law.cfg ``[modules]`` list; each task lives in the
file beside this one that carries its name. law discovers tasks by walking
``Task.__subclasses__()``, so importing them here is what puts them in the index.
"""

from .StatInferenceTask import StatInferenceTask
from .MergedHists import MergedHists
from .PreprocessShapesTask import PreprocessShapesTask
from .CreateDatacardsTask import CreateDatacardsTask
from .ResonantLimitsTask import ResonantLimitsTask
from .CombinedResonantLimitsTask import CombinedResonantLimitsTask
from .PlotResonantLimitsTask import PlotResonantLimitsTask
from .PlotPullsAndImpactsTask import PlotPullsAndImpactsTask
from .ResonantLimitsAndHistPlotTask import ResonantLimitsAndHistPlotTask

__all__ = [
    "StatInferenceTask",
    "MergedHists",
    "PreprocessShapesTask",
    "CreateDatacardsTask",
    "ResonantLimitsTask",
    "CombinedResonantLimitsTask",
    "PlotResonantLimitsTask",
    "PlotPullsAndImpactsTask",
    "ResonantLimitsAndHistPlotTask",
]
