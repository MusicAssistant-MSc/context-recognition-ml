from src.models.const import ModelAlgorithm, TREE_MODELS_WITH_NATIVE_NAN_SUPPORT

def use_imputer_for_tree_model(algorithm: ModelAlgorithm) -> bool:
    """Determine whether to a tree-based model requires explicit imputation."""
    return algorithm not in TREE_MODELS_WITH_NATIVE_NAN_SUPPORT