"""
Data models for MUR SST data processing.

This module contains data structures used throughout the package.
"""

from dataclasses import dataclass


@dataclass
class Task:
    """
    Represents a data processing task.

    Attributes:
        var: The variable name to process
        dt: The datetime string
        time_idx: The time index in the array
    """

    var: str
    dt: str
    time_idx: int
