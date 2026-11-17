from uuid import uuid4
import argparse

from src.utils.time_utils import get_current_timestamp_for_filename
from src.utils.const import CliArgsConfigKeys


def generate_run_id() -> str:
    """Generate a unique run ID based on the current UTC timestamp and a random UUID.

    Returns:
        str: A unique run ID.
    """
    timestamp = get_current_timestamp_for_filename()
    unique_id = uuid4().hex[:8]  # Use the first 8 characters of the UUID
    return f"{timestamp}_{unique_id}"


def parse_args(description: str, *flag_groups: list[dict[str, str]]) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline.

    Args:
        description (str): Description of the command-line interface.
        *flag_groups (list[dict[str, str]]): Variable number of argument flag groups.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    exclusive_groups: dict[str, argparse._MutuallyExclusiveGroup] = {}
    
    for group in flag_groups:
        if isinstance(group, list):
            for flag in group:
                # Check if the flag belongs to an exclusive group
                exclusive_group_name = flag.get(CliArgsConfigKeys.EXCLUSIVE_GROUP)
                
                if exclusive_group_name:
                    # Create the exclusive group if it doesn't exist yet
                    if exclusive_group_name not in exclusive_groups:
                        exclusive_groups[exclusive_group_name] = parser.add_mutually_exclusive_group()
                    add_to_target = exclusive_groups[exclusive_group_name]
                else:
                    add_to_target = parser
                
                # Add the argument to the appropriate target (parser or exclusive group)
                add_to_target.add_argument(
                    flag.get(CliArgsConfigKeys.NAME),
                    dest=flag.get(CliArgsConfigKeys.DEST),
                    action=flag.get(CliArgsConfigKeys.ACTION),
                    help=flag.get(CliArgsConfigKeys.HELP)
                )
        else:
            parser.add_argument(
                group.get(CliArgsConfigKeys.NAME),
                dest=group.get(CliArgsConfigKeys.DEST),
                action=group.get(CliArgsConfigKeys.ACTION),
                help=group.get(CliArgsConfigKeys.HELP)
            )

    return parser.parse_args()


def build_cli_str_args(args: argparse.Namespace, flag_group: list[dict[str, str]]) -> list[str]:
    """Build a list of command-line string arguments based on the provided argparse.Namespace and flag group.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        flag_group (list[dict[str, str]]): A list of argument flag dictionaries.

    Returns:
        list[str]: A list of command-line string arguments.
    """
    return [
        flag.get(CliArgsConfigKeys.NAME)
        for flag in flag_group
        if getattr(args, flag.get(CliArgsConfigKeys.DEST), False)
    ]
