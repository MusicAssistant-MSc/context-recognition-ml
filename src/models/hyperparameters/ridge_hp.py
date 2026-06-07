from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class RidgeHyperparametersNames(StrEnum):
    """Names of the hyperparameters for Ridge Regression, used as keys in parameter grids."""
    FIT_INTERCEPT = "fit_intercept"
    COPY_X = "copy_X"
    SOLVER = "solver"
    MAX_ITER = "max_iter"
    TOL = "tol"
    RANDOM_STATE = "random_state"
    ALPHA = "alpha"
    

RIDGE_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant parameters for LightGBM
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.FIT_INTERCEPT,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value=True,
        description="Whether to fit intercept."
    ),
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.COPY_X,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value=True,
        description="If True, X will be copied; else, it may be overwritten."
    ),
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.SOLVER,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value="auto",
        description="Automatically selected solver."
    ),
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.MAX_ITER,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value=10000,
        description="Maximum number of iterations."
    ),
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.TOL,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value=1e-4,
        description="Tolerance for optimization stopping criterion."
    ),
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.RANDOM_STATE,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=None,
        const_value=RANDOM_STATE,
        description="Random seed."
    ),
    # Tunable parameters for LightGBM
    HyperparameterMetadata(
        name=RidgeHyperparametersNames.ALPHA,
        algorithm=ModelAlgorithm.RIDGE,
        search_space_factory=lambda: loguniform(1e-4, 1e3),
        const_value=None,
        description="Regularization strength."
    ),
]