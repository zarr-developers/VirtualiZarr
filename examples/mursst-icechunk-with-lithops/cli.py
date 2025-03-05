"""
Command-line interface.

This module provides a command-line interface for the package.
"""

import argparse

from lithops_functions import (
    lithops_calc_icechunk_store_mean,
    lithops_calc_original_files_mean,
    lithops_check_data_store_access,
    lithops_list_installed_packages,
    write_to_icechunk,
)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run lithops functions.")
    parser.add_argument(
        "function",
        choices=[
            "write_to_icechunk",
            "check_data_store_access",
            "calc_icechunk_store_mean",
            "calc_original_files_mean",
            "list_installed_packages",
        ],
        help="The function to run.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for data processing (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for data processing (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--append_dim",
        type=str,
        help="Append dimension for writing to icechunk.",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the command-line interface.
    """
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    append_dim = args.append_dim

    if args.function == "write_to_icechunk":
        write_to_icechunk(
            start_date=start_date, end_date=end_date, append_dim=append_dim
        )
    elif args.function == "check_data_store_access":
        lithops_check_data_store_access()
    elif args.function == "calc_icechunk_store_mean":
        lithops_calc_icechunk_store_mean(start_date=start_date, end_date=end_date)
    elif args.function == "calc_original_files_mean":
        lithops_calc_original_files_mean(start_date=start_date, end_date=end_date)
    elif args.function == "list_installed_packages":
        lithops_list_installed_packages()


if __name__ == "__main__":
    main()
