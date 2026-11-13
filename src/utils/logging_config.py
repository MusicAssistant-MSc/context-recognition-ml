import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils.const import LOG_FORMAT
from src.utils.time_utils import format_utc_log_timestamp_to_rfc


class UTCFormatter(logging.Formatter):
    """Custom logging formatter for unified UTC timestamps in logs."""
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return timestamp.strftime(format=datefmt)
        return format_utc_log_timestamp_to_rfc(timestamp)  


def setup_logging(
    log_file: str | Path | None = None,
    log_format: str = LOG_FORMAT,
    log_level: int = logging.INFO,
    log_time_format: str | None = None
) -> None:
    """Set up logging configuration.

    Args:
        log_file (str | Path | None): Path to the log file. If None, logs will be printed to the console only.
        log_format (str): The format for the log messages.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_time_format (str | None): The override format for the timestamp in the log messages.
    Returns:
        None
    """
    formatter = UTCFormatter(fmt=log_format, datefmt=log_time_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)  # Ensure that existing handlers are overridden
    