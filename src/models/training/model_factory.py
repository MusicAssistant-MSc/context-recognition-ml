from typing import Any

from src.models.const import ModelAlgorithm
from src.models.training import tree_models
from src.models.hyperparameters.registry import HYPERPARAMETERS_REGISTRY

def get_unfitted_model(
    algorithm: ModelAlgorithm,
    **hyperparams: Any
) -> Any:
    """Factory function to create an unfitted model instance based on the specified algorithm.
    
    Args:
        algorithm (ModelAlgorithm): The type of ML algorithm to create.
        **hyperparams: Additional hyperparameters specific to the model type.
    """
    # Retrieve constant hyperparameters for the specified algorithm and merge with provided hyperparameters
    params = HYPERPARAMETERS_REGISTRY.get_const_params_grid(algorithm)
    params.update(hyperparams)

    match algorithm:
        case ModelAlgorithm.XGBOOST:
            return tree_models.create_xgboost_regressor(**params)
        case ModelAlgorithm.LIGHTGBM:
            return tree_models.create_lightgbm_regressor(**params)
        case ModelAlgorithm.RANDOM_FOREST:
            return tree_models.create_random_forest_regressor(**params)
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")