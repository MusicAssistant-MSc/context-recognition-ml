from datetime import datetime, timezone

from src.utils.const import FILE_TIMESTAMP_FORMAT


def get_current_timestamp_for_filename() -> str:
    """Get the current UTC timestamp formatted for filenames.

    Returns:
        str: The current UTC timestamp in the format 'YYYY-MM-DD_HHMMSS'.
    """
    utc_now = datetime.now(timezone.utc)
    return utc_now.strftime(FILE_TIMESTAMP_FORMAT)


def get_current_timestamp_for_log() -> str:
    """Get the current UTC timestamp formatted for logging.

    Returns:
        str: The current UTC timestamp in the format 'YYYY-MM-DDTHH:MM:SS.mmm+00:00'.
    """
    utc_now = datetime.now(timezone.utc)
    return utc_now.isoformat(timespec='milliseconds')