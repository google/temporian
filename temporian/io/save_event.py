from typing import List, Optional
import pandas as pd
from pathlib import Path

from temporian.implementation.numpy.data.event import NumpyEvent


def save_event(
    event: NumpyEvent,
    path: str,
    sep: str = ",",
    na_rep: Optional[str] = None,
    columns: Optional[List[str]] = None,
):
    """Save a NumpyEvent to a file.

    Args:
        event (NumpyEvent): NumpyEvent to be saved.
        path (str): Path to the file.
        sep (str, optional): Separator to use. Defaults to ",".
        na_rep (Optional[str], optional): Representation to use for missing values. Defaults to None.
        columns (Optional[List[str]], optional): Columns to save. Defaults to None.

    """
    df = event.to_dataframe()
    df.to_csv(path, index=False, sep=sep, na_rep=na_rep, columns=columns)
