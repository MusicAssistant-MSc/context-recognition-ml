from typing import Any, Final

from src.models.hyperparameters.metadata import HyperparameterMetadata
from src.models.const import ModelAlgorithm
from src.models.hyperparameters.xgboost_hp import XGBOOST_HYPERPARAMETERS
from src.models.hyperparameters.lightgbm_hp import LIGHTGBM_HYPERPARAMETERS
from src.models.hyperparameters.random_forest_hp import RANDOM_FOREST_HYPERPARAMETERS
from src.models.hyperparameters.svr_hp import SVR_HYPERPARAMETERS
from src.models.hyperparameters.ridge_hp import RIDGE_HYPERPARAMETERS
from src.models.hyperparameters.mlp_hp import MLP_HYPERPARAMETERS


class HyperparameterRegistry:
    def __init__(self, hyperparameters: list[HyperparameterMetadata]):
        self._hyperparameters: dict[str, HyperparameterMetadata] = {hp.name: hp for hp in hyperparameters}

    def __getitem__(self, name: str) -> HyperparameterMetadata:
        try:
            return self._hyperparameters[name]
        except KeyError:
            raise KeyError(f"Hyperparameter '{name}' not found in registry.")

    def get_by_algorithm(self, algorithm: ModelAlgorithm) -> list[HyperparameterMetadata]:
        """Retrieve all hyperparameters associated with a specific algorithm."""
        return [hp for hp in self._hyperparameters.values() if hp.algorithm == algorithm]
    
    def get_tuning_params_grid(self, algorithm: ModelAlgorithm) -> dict[str, Any]:
        """Construct a parameter grid for hyperparameter tuning for a specified algorithm."""
        return {
            hp.name: hp.search_space_factory()
            for hp in self.get_by_algorithm(algorithm)
            if hp.search_space_factory is not None
        }
    
    def get_const_params_grid(self, algorithm: ModelAlgorithm) -> dict[str, Any]:
        """Construct a parameter grid for constant hyperparameter values for a specific algorithm."""
        return {
            hp.name: hp.const_value
            for hp in self.get_by_algorithm(algorithm)
            if hp.const_value is not None
        }
    

class HyperparameterNamesRegistry:
    """Registry for standardized hyperparameter names used across the codebase."""
    pass
    

_hp_metadata_lists: list[list[HyperparameterMetadata]] = [
    XGBOOST_HYPERPARAMETERS, 
    LIGHTGBM_HYPERPARAMETERS,
    RANDOM_FOREST_HYPERPARAMETERS,
    SVR_HYPERPARAMETERS,
    RIDGE_HYPERPARAMETERS,
    MLP_HYPERPARAMETERS,
    []
]

HYPERPARAMETERS_REGISTRY: Final[HyperparameterRegistry] = HyperparameterRegistry([
    hp for hp_list in _hp_metadata_lists for hp in hp_list
])