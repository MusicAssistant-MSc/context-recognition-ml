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


def format_utc_log_timestamp_to_rfc(timestamp: datetime) -> str:
    """Format a UTC timestamp for logging in RFC 3339 format.

    Args:
        timestamp (datetime): The UTC timestamp to format.

    Returns:
        str: The current UTC timestamp in the format 'YYYY-MM-DD HH:MM:SS.mmmZ'.
    """
    return timestamp.astimezone(timezone.utc).isoformat(sep=' ', timespec='milliseconds').replace('+00:00', 'Z')
