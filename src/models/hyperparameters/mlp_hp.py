from enum import StrEnum
from scipy.stats import uniform, loguniform, randint

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm, RANDOM_STATE

class MLPHyperparametersNames(StrEnum):
    """Names of the hyperparameters for MLP, used as keys in parameter grids."""
    MAX_ITER = "max_iter"
    EARLY_STOPPING = "early_stopping"
    VALIDATION_FRACTION = "validation_fraction"
    SHUFFLE = "shuffle"
    N_ITER_NO_CHANGE = "n_iter_no_change"
    RANDOM_STATE = "random_state"
    HIDDEN_LAYER_SIZES = "hidden_layer_sizes"
    ACTIVATION = "activation"
    SOLVER = "solver"
    ALPHA = "alpha"
    LEARNING_RATE_INIT = "learning_rate_init"
    BATCH_SIZE = "batch_size"


MLP_HYPERPARAMETERS: list[HyperparameterMetadata] = [
    # Constant parameters for MLP
    HyperparameterMetadata(
        name=MLPHyperparametersNames.MAX_ITER,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=2000,
        description="Maximum number of training iterations."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.EARLY_STOPPING,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=True,
        description="Enable early stopping."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.VALIDATION_FRACTION,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=0.1,
        description="Validation split fraction for early stopping."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.SHUFFLE,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=True,
        description="Whether to shuffle samples in each iteration (for Adam/SGD)."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.N_ITER_NO_CHANGE,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=20,
        description="Early stopping patience."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.RANDOM_STATE,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=None,
        const_value=RANDOM_STATE,
        description="Random seed."
    ),
    # Tunable parameters for MLP
    HyperparameterMetadata(
        name=MLPHyperparametersNames.HIDDEN_LAYER_SIZES,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=lambda: [
            (8,), (12,), (16,), (24,), (32,), (48,), (64,)
        ],
        const_value=None,
        description="The ith element represents the number of neurons in the ith hidden layer."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.ACTIVATION,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=lambda: ["relu", "tanh"],
        const_value=None,
        description="Activation function."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.SOLVER,
        algorithm=ModelAlgorithm.MLP,
        # search_space_factory=lambda: ["adam", "lbfgs"],
        search_space_factory=lambda: ["adam"],
        const_value=None,
        description="Optimization solver."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.ALPHA,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=lambda: loguniform(1e-4, 1e1),
        const_value=None,
        description="L2 regularization strength."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.LEARNING_RATE_INIT,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=lambda: loguniform(1e-4, 5e-2),
        const_value=None,
        description="Initial learning rate."
    ),
    HyperparameterMetadata(
        name=MLPHyperparametersNames.BATCH_SIZE,
        algorithm=ModelAlgorithm.MLP,
        search_space_factory=lambda: [16, 32, 64],
        const_value=None,
        description="Mini-batch size."
    ),
]
    
