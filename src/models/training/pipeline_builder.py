from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Any

from src.models.const import PREPROCESSOR_STEP_NAME, MODEL_STEP_NAME

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
    preprocessor_step_name: str = PREPROCESSOR_STEP_NAME,
    model_step_name: str = MODEL_STEP_NAME
) -> Pipeline:
    """Build a machine learning pipeline with the given preprocessor and model.
    
    Args:
        preprocessor (ColumnTransformer): The preprocessing steps to apply to the data.
        model (Any): The machine learning model to use in the pipeline.
        preprocessor_step_name (str): The name of the preprocessor step in the pipeline.
        model_step_name (str): The name of the model step in the pipeline.

    Returns:
        Pipeline: A scikit-learn Pipeline object that combines the preprocessor and model.
    """
    return Pipeline(steps=[
        (preprocessor_step_name, preprocessor),
        (model_step_name, model)
    ])