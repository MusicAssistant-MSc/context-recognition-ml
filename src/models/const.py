from enum import StrEnum, auto
from typing import Final
from pathlib import Path

from src.utils.const import SiTunesInterimInterEnvCols, SiTunesInterimMusicCols, SAVED_MODELS_DIR, REPORTS_DIR


class ModelAlgorithm(StrEnum):
    """Supported ML algorithm types."""
    XGBOOST = auto()
    LIGHTGBM = auto()
    RANDOM_FOREST = auto()
    SVR = auto()
    MLP = auto()
    RIDGE = auto()


class ModelVariant(StrEnum):
    """Ablation study model variants based on feature sets."""
    BIOMETRIC = auto()
    MOTION = auto()
    CONTEXT = auto()
    ALL = auto()


class CrossValidationStrategy(StrEnum):
    """Cross-validation strategies for model evaluation."""
    GROUP_K_FOLD = auto()
    LOSO = auto()


class RegressionMetrics(StrEnum):
    """Names of the regression metrics used for model evaluation."""
    MAE = auto()
    MSE = auto()
    RMSE = auto()


# Random state constant
RANDOM_STATE: Final[int] = 42

# Default file extensions
MODEL_EXT: Final[str] = ".joblib"
METRICS_EXT: Final[str] = ".json"
SUMMARY_CSV_NAME: Final[str] = "experiment_summary.csv"

# Directories for saving models and reports
SAVED_MODELS_DIR: Final[Path] = SAVED_MODELS_DIR
REPORTS_DIR: Final[Path] = REPORTS_DIR
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"

# Default hyperparameter name prefix
HYPERPARAMETER_MODEL_PREFIX: Final[str] = "model"

# Default pipeline step names
PREPROCESSOR_STEP_NAME: Final[str] = "preprocessor"
MODEL_STEP_NAME: Final[str] = HYPERPARAMETER_MODEL_PREFIX

# Default names for ID and target columns in the datasets
USER_ID_COL: Final[str] = SiTunesInterimInterEnvCols.USER_ID
INTER_ID_COL: Final[str] = SiTunesInterimInterEnvCols.INTER_ID
MUSIC_ID_COL: Final[str] = SiTunesInterimMusicCols.ITEM_ID
TARGET_COL: Final[str] = SiTunesInterimInterEnvCols.RATING

# --- Default cross-validation settings ---
# Default cross-validation strategies
DEFAULT_OUTER_CV_STRATEGY: Final[CrossValidationStrategy] = CrossValidationStrategy.GROUP_K_FOLD
DEFAULT_OUTER_CV_SPLITS: Final[int] = 5
DEFAULT_INNER_CV_STRATEGY: Final[CrossValidationStrategy] = CrossValidationStrategy.GROUP_K_FOLD
DEFAULT_INNER_CV_SPLITS: Final[int] = 4

# --- Default hyperparameter tuning settings ---
RANDOMIZED_SEARCH_ITER: Final[int] = 50
RANDOMIZED_SEARCH_SCORING: Final[str] = "neg_mean_squared_error"
RANDOMIZED_SEARCH_N_JOBS: Final[int] = -1 # Use all available CPU cores for tuning

# Tree-based models that support native handling of NaN values
TREE_MODELS_WITH_NATIVE_NAN_SUPPORT: Final[set[ModelAlgorithm]] = {
    ModelAlgorithm.XGBOOST,
    ModelAlgorithm.LIGHTGBM
}