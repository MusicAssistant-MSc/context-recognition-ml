from typing import Any
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def create_svr_regressor(**kwargs: Any) -> SVR:
    """Initialize a Support Vector Regression model."""
    return SVR(**kwargs)


def create_ridge_regressor(**kwargs: Any) -> Ridge:
    """Initialize a Ridge Regression model."""
    return Ridge(**kwargs)


def create_mlp_regressor(**kwargs: Any) -> MLPRegressor:
    """Initialize an MLP regressor."""
    return MLPRegressor(**kwargs)