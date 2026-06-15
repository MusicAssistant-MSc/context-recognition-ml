from dataclasses import dataclass
from typing import Any, Callable

from src.models.const import ModelAlgorithm


@dataclass(frozen=True)
class HyperparameterMetadata:
    """Metadata for a single hyperparameter, including its name, search space or const value and description.
    
    Attributes:
        name (str): The name of the hyperparameter, used as the key in parameter grids.
        algorithm (ModelAlgorithm): The machine learning algorithm a hyperparameter is associated with.
        search_space_factory (Callable[[], Any] | None): A function that returns a search space distribution or list for tuning. 
            Set to None if the hyperparameter is not being tuned.
        const_value (Any | None): A fixed value for the hyperparameter if it is not being tuned. 
            This is used when the hyperparameter should be set to a specific value rather than searched over.
        is_log_scale (bool): Indicates if the hyperparameter should be sampled on a logarithmic scale during tuning.
        description (str | None): An optional description of the hyperparameter for documentation purposes.
    """
    name: str
    algorithm: ModelAlgorithm
    search_space_factory: Callable[[], Any] | None 
    const_value: Any | None = None 
    is_log_scale: bool = False
    description: str | None = None