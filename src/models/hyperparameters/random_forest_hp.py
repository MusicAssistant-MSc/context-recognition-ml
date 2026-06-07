from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class RandomForestHyperparametersNames(StrEnum):
    """Names of the hyperparameters for Random Forest, used as keys in parameter grids."""
    CRITERION = "criterion"
    BOOTSTRAP = "bootstrap"
    N_JOBS = "n_jobs"
    RANDOM_STATE = "random_state"
    N_ESTIMATORS = "n_estimators"
    MAX_DEPTH = "max_depth"
    MAX_FEATURES = "max_features"
    MIN_SAMPLES_SPLIT = "min_samples_split"
    MIN_SAMPLES_LEAF = "min_samples_leaf"
    MAX_SAMPLES = "max_samples"


RANDOM_FOREST_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant hyperparameters for Random Forest
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.CRITERION,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=None,
        const_value="squared_error",
        description="The function to measure the quality of a split (Mean Squared Error)."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.BOOTSTRAP,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=None,
        const_value=True,
        description="Whether bootstrap samples are used when building trees (essential for RF)."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.N_JOBS,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=None,
        const_value=-1,
        description="Utilize all available CPU cores for parallel forest construction."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.RANDOM_STATE,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=None,
        const_value=RANDOM_STATE,
        description="Random seed."
    ),
    # Tunable hyperparameters for Random Forest
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.N_ESTIMATORS,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: randint(100, 1000),
        const_value=None,
        description="Number of trees in the forest."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.MAX_DEPTH,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: randint(5, 30),
        const_value=None,
        description="Maximum tree depth."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.MAX_FEATURES,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        const_value=None,
        description="Number of features considered for best split."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.MIN_SAMPLES_SPLIT,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: randint(2, 20),
        const_value=None,
        description="Minimum samples required to split an internal node."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.MIN_SAMPLES_LEAF,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: randint(1, 10),
        const_value=None,
        description="Minimum samples required in leaf nodes."
    ),
    HyperparameterMetadata(
        name=RandomForestHyperparametersNames.MAX_SAMPLES,
        algorithm=ModelAlgorithm.RANDOM_FOREST,
        search_space_factory=lambda: uniform(0.5, 0.5),
        const_value=None,
        description="Fraction of samples drawn for each tree."
    )
]