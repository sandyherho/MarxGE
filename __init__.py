"""
MarxGE: Marxian General Equilibrium Model
-----------------------------------------

A computational implementation of John E. Roemer's "A General Equilibrium Approach to Marxian Economics"
(Econometrica, 1980).

This package provides tools to model, simulate, and analyze Marxian economic concepts in a general
equilibrium framework, including:
1. Production with general convex technologies
2. Labor values and exploitation
3. The Fundamental Marxian Theorem
4. Reproducible solutions
5. Social determination of workers' consumption
"""

# Import core models for convenience
from marxge.core.linear_model import MarxianLinearModel
from marxge.core.convex_model import ConvexProductionModel
from marxge.core.social_model import SocialDeterminationModel

# Import analysis tools
from marxge.analysis.fmt import FundamentalMarxianTheorem
from marxge.analysis.general_equilibrium import GeneralEquilibriumAnalysis
from marxge.analysis.visualization import Visualizer

# Import utility functions
from marxge.utils.consumption import (
    technological_needs_consumption,
    historical_moral_consumption
)
from marxge.utils.examples import ExampleModels

# Import experiment runners
from marxge.experiments.runner import MarxianExperiments

__version__ = '0.1.0'
