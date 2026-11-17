import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from src.utils.const import (
    PROJECT_ROOT, 
    PIPELINE_RUN_ID_ENV_VAR, 
    DATA_CLI_ARGS, 
    NOTEBOOK_CLI_ARGS,
    NOTEBOOKS_DIR,
    NOTEBOOKS_OUTPUT_DIR
)
from src.utils.run_utils import generate_run_id, parse_args, build_cli_str_args
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_stage(stage_name: str, module_path: str, env: dict[str, str], args: list[str] | None = None) -> None:
    """Execute a specific stage of the pipeline as a separate isolated subprocess.

    Args:
        stage_name (str): The name of the stage to run.
        module_path (str): The path to the module to execute.
        env (dict[str, str]): The environment variables for the subprocess.
        args (list[str] | None): Additional command-line arguments for the module.

    Returns:
        None
    """
    command = [sys.executable, "-m", module_path]
    if args:
        command.extend(args)

    logger.info(f"Running stage '{stage_name}' with command: {' '.join(command)}")
    
    start_time = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    end_time = time.time()

    if result.returncode != 0:
        logger.error(f"Stage '{stage_name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

    logger.info(f"Stage '{stage_name}' completed successfully in {end_time - start_time:.2f} seconds.\n")


def run_notebook(input_nb: Path, output_nb: Path, env: dict[str, str]) -> None:
    """Execute a Jupyter notebook via Papermill.

    Args:
        input_nb (Path): The path to the input notebook file.
        output_nb (Path): The path to the output notebook file.
        env (dict[str, str]): The environment variables for the subprocess.

    Returns:
        None
    """
    command = [sys.executable, "-m", "papermill", str(input_nb), str(output_nb), "--log-output"]

    logger.info(f"Running notebook '{input_nb}' with command: {' '.join(command)}")
    
    start_time = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    end_time = time.time()

    if result.returncode != 0:
        logger.error(f"Notebook '{input_nb}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

    logger.info(f"Notebook executed and saved to '{output_nb}' in {end_time - start_time:.2f} seconds.\n")


def main() -> None:
    setup_logging()
    run_id = generate_run_id()

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",  # Ensure unbuffered output for real-time logging
        PIPELINE_RUN_ID_ENV_VAR: run_id,
    }
    # Build the command-line arguments for each stage
    args = parse_args("Orchestrate the global ML pipeline.", DATA_CLI_ARGS, NOTEBOOK_CLI_ARGS)

    total_start_time = time.time()

    # Stage 1: Data Download and Cleaning
    data_args = build_cli_str_args(args, DATA_CLI_ARGS)
    run_stage("Data Download and Cleaning", "src.data.make_dataset", base_env, args=data_args)

    # Stage 2: Feature Analysis (Notebook)
    if not args.skip_notebook:
        notebook_input_path = NOTEBOOKS_DIR / "01_eda_feature_distribution.ipynb"
        NOTEBOOKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        notebook_output_path = NOTEBOOKS_OUTPUT_DIR / "executed_01_eda_feature_distribution.ipynb"
        run_notebook(notebook_input_path, notebook_output_path, base_env)
    else:
        logger.info("Skipping feature distribution analysis notebook execution (--skip-notebook).\n")

    # Stage 3: Feature Extraction and Engineering
    run_stage("Feature Extraction andEngineering", "src.features.build_features", base_env)

    # Stage 4: Model Training and Evaluation
    run_stage("Model Training and Evaluation", "src.models.run_experiment", base_env)

    # Stage 5: Statistical Evaluation (Notebook)
    if not args.skip_notebook:
        notebook_input_path = NOTEBOOKS_DIR / "02_experiment_results_analysis.ipynb"
        NOTEBOOKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        notebook_output_path = NOTEBOOKS_OUTPUT_DIR / "executed_02_experiment_results_analysis.ipynb"
        run_notebook(notebook_input_path, notebook_output_path, base_env)
    else:
        logger.info("Skipping statistical evaluation notebook execution (--skip-notebook).\n")

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Pipeline completed successfully in {total_elapsed_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
