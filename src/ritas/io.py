"""RITAS I/O module."""
from pathlib import Path

import pandas as pd


def read_input(infile: Path) -> pd.DataFrame:
    """Read the input file.

    Args:
        infile (Path): The input file to read.

    Returns:
        pd.DataFrame: The input data.
    """
    return pd.read_csv(infile)
