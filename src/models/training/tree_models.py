from typing import Any
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


def create_xgboost_regressor(**kwargs: Any) -> XGBRegressor:
    """Initialize an XGBoost regressor with specified parameters."""
    return XGBRegressor(**kwargs)


def create_lightgbm_regressor(**kwargs: Any) -> LGBMRegressor:
    """Initialize a LightGBM regressor with specified parameters."""
    return LGBMRegressor(**kwargs)


def create_random_forest_regressor(**kwargs: Any) -> RandomForestRegressor:
    """Initialize a Random Forest regressor with specified parameters."""
    return RandomForestRegressor(**kwargs)