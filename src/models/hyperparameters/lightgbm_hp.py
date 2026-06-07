from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class LightGBMHyperparametersNames(StrEnum):
    """Names of the hyperparameters for LightGBM, used as keys in parameter grids."""
    OBJECTIVE = "objective"
    N_JOBS = "n_jobs"
    VERBOSITY = "verbosity"
    IMPORTANCE_TYPE = "importance_type"
    BOOSTING_TYPE = "boosting_type"
    RANDOM_STATE = "random_state"
    N_ESTIMATORS = "n_estimators"
    LEARNING_RATE = "learning_rate"
    NUM_LEAVES = "num_leaves"
    MAX_DEPTH = "max_depth"
    MIN_CHILD_SAMPLES = "min_child_samples"
    COLSAMPLE_BYTREE = "colsample_bytree"
    BAGGING_FRACTION = "bagging_fraction"
    BAGGING_FREQ = "bagging_freq"
    MIN_SPLIT_GAIN = "min_split_gain"
    REG_ALPHA = "reg_alpha"
    REG_LAMBDA = "reg_lambda"


LIGHTGBM_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant parameters for LightGBM
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.OBJECTIVE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value="regression",
        description="Regression objective for continuous target prediction."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.N_JOBS,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value=-1,
        description="Use all available CPU cores."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.VERBOSITY,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value=-1,
        description="Disable LightGBM training logs."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.IMPORTANCE_TYPE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value="gain",
        description="Calculate feature importance based on information gain (better for SHAP)."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.BOOSTING_TYPE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value="gbdt",
        description="Standard Gradient Boosting Decision Trees."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.RANDOM_STATE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=None,
        const_value=RANDOM_STATE,
        description="Random seed."
    ),
    # Tunable parameters for LightGBM
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.N_ESTIMATORS,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: randint(100, 1000),
        const_value=None,
        description="Number of boosting iterations."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.LEARNING_RATE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: loguniform(0.005, 0.2),
        const_value=None,
        description="Boosting learning rate."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.NUM_LEAVES,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: randint(8, 64),
        const_value=None,
        description="Maximum number of leaves in one tree."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.MAX_DEPTH,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: randint(3, 10),
        const_value=None,
        description="Maximum tree depth."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.MIN_CHILD_SAMPLES,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: randint(5, 50),
        const_value=None,
        description="Minimum samples required in a leaf."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.BAGGING_FRACTION,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: uniform(0.6, 0.4),
        const_value=None,
        description="Fraction of samples randomly selected per iteration."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.BAGGING_FREQ,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: randint(1, 10),
        const_value=None,
        description="Bagging frequency."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.COLSAMPLE_BYTREE,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: uniform(0.6, 0.4),
        const_value=None,
        description="Fraction of features to be sampled for each iteration."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.MIN_SPLIT_GAIN,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: loguniform(1e-4, 1.0),
        const_value=None,
        description="Minimum gain required to perform split."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.REG_ALPHA,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: loguniform(1e-4, 10),
        const_value=None,
        description="L1 regularization."
    ),
    HyperparameterMetadata(
        name=LightGBMHyperparametersNames.REG_LAMBDA,
        algorithm=ModelAlgorithm.LIGHTGBM,
        search_space_factory=lambda: loguniform(1e-4, 10),
        const_value=None,
        description="L2 regularization."
    )
]
