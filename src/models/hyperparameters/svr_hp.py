from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class SVRHyperparametersNames(StrEnum):
    """Names of the hyperparameters for SVR, used as keys in parameter grids."""
    KERNEL = "kernel"
    SHRINKING = "shrinking"
    CACHE_SIZE = "cache_size"
    C = "C"
    EPSILON = "epsilon"
    GAMMA = "gamma"


SVR_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant hyperparameters for SVR
    HyperparameterMetadata(
        name=SVRHyperparametersNames.KERNEL,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=None,
        const_value="rbf",
        description="Radial Basis Function kernel - best for non-linear relationships."
    ),
    HyperparameterMetadata(
        name=SVRHyperparametersNames.SHRINKING,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=None,
        const_value=True,
        description="Whether to use the shrinking heuristic to speed up computation."
    ),
    HyperparameterMetadata(
        name=SVRHyperparametersNames.CACHE_SIZE,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=None,
        const_value=1000,
        description="Increase cache size (MB) for faster training."
    ),
    # Tunable hyperparameters for SVR
    HyperparameterMetadata(
        name=SVRHyperparametersNames.C,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=lambda: loguniform(1e-2, 1e3),
        const_value=None,
        description="Regularization parameter. Vital to tune on log scale."
    ),
    HyperparameterMetadata(
        name=SVRHyperparametersNames.EPSILON,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=lambda: uniform(0.01, 0.5),
        const_value=None,
        description="Distance within which no penalty is associated in the loss function (margin of tolerance)."
    ),
    HyperparameterMetadata(
        name=SVRHyperparametersNames.GAMMA,
        algorithm=ModelAlgorithm.SVR,
        search_space_factory=lambda: loguniform(1e-4, 1e-1),
        const_value=None,
        description="Kernel coefficient for RBF. Controls the reach of local influence."
    )
]
    