import json
from pathlib import Path
from typing import Any, Literal
import re
import logging
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import pingouin as pg

from src.models.const import RegressionMetrics, MIN_MODELS_FOR_FRIEDMAN_TEST, METRICS_DIR, FILE_TIMESTAMP_SUFFIX_PATTERN
from src.utils.const import ALPHA_SIGNIFICANCE_LEVEL, POSTHOC_CONOVER_P_ADJUST

logger = logging.getLogger(__name__)


def check_global_normality(
    results_matrix: pd.DataFrame,
    alpha_significance_level: float = ALPHA_SIGNIFICANCE_LEVEL
)-> bool:
    """
    Check the normality of each column (model) in a matrix of results.
    As per PQStat documentation (p. 140), this is the first step.

    Args:
        results_matrix (pd.DataFrame): A DataFrame where each column represents a model
            and each row represents an outer fold result.
        alpha_significance_level (float): The significance level for the Shapiro-Wilk test.

    Returns:
        bool: True if all columns are normally distributed, False otherwise.
    """
    for col in results_matrix.columns:
        _, p_value = shapiro(results_matrix[col])
        if p_value <= alpha_significance_level:
            logger.info(f"Normality violated by model: {col} (p={p_value:.4f})")
            return False

    return True


def perform_friedman_analysis(
    aggregated_df: pd.DataFrame,
    min_models: int = MIN_MODELS_FOR_FRIEDMAN_TEST
) -> tuple[float, float]:
    """
    Perform the Friedman test on a matrix of results.

    The Friedman test is a non-parametric statistical test used to detect 
    differences in treatments across multiple test attempts.

    Args:
        metrics_matrix (pd.DataFrame): DataFrame where each column represents a group of models
            and each row represents an outer fold aggregated result.
        min_models (int): The minimum number of models required for the Friedman test.

    Returns:
        tuple[float, float]: A tuple containing the test statistic and the p-value.

    Raises:
        ValueError: If the number of columns (models) is less than 3.
    """
    # Pingouin expects long format for its friedman implementation
    df_long = aggregated_df.copy()
    df_long = aggregated_df.reset_index().melt(
        id_vars='index', var_name='group', value_name='metric_value'
    )
    df_long.rename(columns={'index': 'fold'}, inplace=True)

    friedman_results = pg.friedman(
        data=df_long, dv='metric_value', within='group', subject='fold'
    )
    
    return friedman_results


def perform_conover_posthoc(
    aggregated_df: pd.DataFrame,
    p_adjust: str = POSTHOC_CONOVER_P_ADJUST
) -> pd.DataFrame:
    """Performs Conover-Inman post-hoc test (PQStat p. 261).

    Args:
        aggregated_df: Matrix where rows are folds and columns are groups.
        p_adjust: Method for multiple comparisons correction.
    """
    df_copy = aggregated_df.copy()
    # Use posthoc_conover_friedman version for Friedman post-hoc tests
    posthoc_matrix = sp.posthoc_conover_friedman(
        df_copy, 
        melted=False,
        p_adjust=p_adjust
    )
    
    return posthoc_matrix