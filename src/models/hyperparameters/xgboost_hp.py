from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class XGBoostHyperparametersNames(StrEnum):
    """Names of the hyperparameters for XGBoost, used as keys in parameter grids."""
    OBJECTIVE = "objective"
    N_JOBS = "n_jobs"
    TREE_METHOD = "tree_method"
    RANDOM_STATE = "random_state"
    N_ESTIMATORS = "n_estimators"
    LEARNING_RATE = "learning_rate"
    MAX_DEPTH = "max_depth"
    MIN_CHILD_WEIGHT = "min_child_weight"
    SUBSAMPLE = "subsample"
    COLSAMPLE_BYTREE = "colsample_bytree"
    GAMMA = "gamma"
    REG_ALPHA = "reg_alpha"
    REG_LAMBDA = "reg_lambda"


XGBOOST_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant parameters for XGBoost
    HyperparameterMetadata(
        name=XGBoostHyperparametersNames.OBJECTIVE,
        algorithm=ModelAlgorithm.XGBOOST,
        search_space_factory=None,
        const_value="reg:squarederror",
        description="Standard regression objective loss function (MSE)."
    ),
    HyperparameterMetadata(
        name=XGBoostHyperparametersNames.N_JOBS,
        algorithm=ModelAlgorithm.XGBOOST,
        search_space_factory=None,
        const_value=1,
        description="Use a single CPU core for training."
    ),
    HyperparameterMetadata(
        name=XGBoostHyperparametersNames.TREE_METHOD,
        algorithm=ModelAlgorithm.XGBOOST,
        search_space_factory=None,
        const_value="hist",
        description="Histogram-based algorithm for faster training."
    ),
    HyperparameterMetadata(
        name=XGBoostHyperparametersNames.RANDOM_STATE,
        algorithm=ModelAlgorithm.XGBOOST,
        search_space_factory=None,
        const_value=RANDOM_STATE,
        description="Random seed."
    ),
    # Tunable hyperparameters for XGBoost
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.N_ESTIMATORS,
        search_space_factory=lambda: randint(100, 600),
        const_value=None,
        description="Number of boosting trees."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.LEARNING_RATE,
        search_space_factory=lambda: loguniform(0.005, 0.2),
        const_value=None,
        description="Boosting learning rate."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.MAX_DEPTH,
        search_space_factory=lambda: randint(2, 6),
        const_value=None,
        description="Maximum tree depth."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.MIN_CHILD_WEIGHT,
        search_space_factory=lambda: randint(1, 12),
        const_value=None,
        description="Minimum child weight."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.SUBSAMPLE,
        search_space_factory=lambda: uniform(0.6, 0.4),
        const_value=None,
        description="Row subsampling ratio."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.COLSAMPLE_BYTREE,
        search_space_factory=lambda: uniform(0.5, 0.5),
        const_value=None,
        description="Feature subsampling ratio."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.GAMMA,
        search_space_factory=lambda: uniform(0, 5),
        const_value=None,
        description="Minimum split gain."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.REG_ALPHA,
        search_space_factory=lambda: loguniform(1e-4, 10),
        const_value=None,
        description="L1 regularization."
    ),
    HyperparameterMetadata(
        algorithm=ModelAlgorithm.XGBOOST,
        name=XGBoostHyperparametersNames.REG_LAMBDA,
        search_space_factory=lambda: loguniform(1e-3, 10),
        const_value=None,
        description="L2 regularization."
    ),
]