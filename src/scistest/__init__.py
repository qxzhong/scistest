"""scisTest: conditional independence testing for censored survival data."""

from .aggregation import (
    AggregatedPValueResult,
    aggregate_p_value,
    aggregated_scis_test,
    calibrate_aggregated_p_value,
    make_subsamples,
    normal_rank_transform,
)
from .config import (
    DirectionConfig,
    GeneratorConfig,
    HazardConfig,
    RandomForestDirectionConfig,
    RandomForestHazardConfig,
    SciTestConfig,
)
from .core import ScisTestResult, scis_test, scisTest

__all__ = [
    "AggregatedPValueResult",
    "DirectionConfig",
    "GeneratorConfig",
    "HazardConfig",
    "RandomForestDirectionConfig",
    "RandomForestHazardConfig",
    "SciTestConfig",
    "ScisTestResult",
    "SciTestResult",
    "sci_test",
    "sciTest",
    "aggregate_p_value",
    "aggregated_scis_test",
    "calibrate_aggregated_p_value",
    "make_subsamples",
    "normal_rank_transform",
    "framingham_configs",
    "forest_configs",
    "forest_simulation_configs",
    "simulation_configs",
    "scis_test",
    "scisTest",
]

__version__ = "0.2.0"

from .core import SciTestResult, sci_test, sciTest
from .manuscript import (
    forest_configs,
    forest_simulation_configs,
    framingham_configs,
    simulation_configs,
)
