import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.const import RegressionMetrics

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """Calculate standard regression metrics (MAE, MSE, RMSE) for model evaluation.
    
    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values from the model.

    Returns:
        dict[str, float]: A dictionary containing the calculated metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        RegressionMetrics.MAE: mae,
        RegressionMetrics.MSE: mse,
        RegressionMetrics.RMSE: rmse
    }