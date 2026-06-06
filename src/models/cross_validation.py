from typing import Any
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from src.models.const import CrossValidationStrategy


def get_cv_splitter(
    strategy: CrossValidationStrategy,
    n_splits: int
) -> Any:
    """Create a cross-validation splitter based on the specified strategy.
    
    Args:
        strategy (CrossValidationStrategy): The type of cross-validation strategy to use.
        n_splits (int): Number of splits for GroupKFold. Ignored for LOSO.
    
    Returns:
        Any: An instance of a cross-validation splitter.
    """
    match strategy:
        case CrossValidationStrategy.GROUP_K_FOLD:
            return GroupKFold(n_splits=n_splits)
        case CrossValidationStrategy.LOSO:
            return LeaveOneGroupOut()
        case _:
            raise ValueError(f"Unsupported cross-validation strategy: {strategy}")
        

def get_nested_cv_setup(
    outer_strategy: CrossValidationStrategy,
    inner_strategy: CrossValidationStrategy,
    n_splits_outer: int,
    n_splits_inner: int
) -> tuple[Any, Any]:
    """
    Create nested cross-validation splitters for outer and inner loops.

    The outer splitter is used for final model performance estimation.
    The inner splitter is used for hyperparameter tuning and model selection.
    
    Args:
        outer_strategy (CrossValidationStrategy): Strategy for the outer loop.
        inner_strategy (CrossValidationStrategy): Strategy for the inner loop.
        n_splits_outer (int): Number of splits for the outer loop (if applicable).
        n_splits_inner (int): Number of splits for the inner loop (if applicable).
    
    Returns:
        tuple[Any, Any]: A tuple containing the outer and inner cross-validation splitters.
    """
    outer_cv = get_cv_splitter(outer_strategy, n_splits_outer)
    inner_cv = get_cv_splitter(inner_strategy, n_splits_inner)
    return outer_cv, inner_cv