from typing import List, Optional
import pandas as pd
from pathlib import Path
from temporian.implementation.numpy.data.event import NumpyEvent


def read_event(
    path: str,
    timestamp_column: str,
    index_names: List[str] = [],
    sep: str = ",",
) -> NumpyEvent:
    """Read a NumpyEvent from a file.

    Args:
        path (str): Path to the file.
        timestamp_column (str): Name of the column to be used as timestamp for the event.
        index_names (List[str], optional): Names of the DataFrame columns to be used as index for the event. Defaults to [].
        sep (str, optional): Separator to use. Defaults to ",".


    Returns:
        NumpyEvent: NumpyEvent read from file.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a csv file.

    """
    # check file exists and is a csv
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    if path.suffix != ".csv":
        raise ValueError(f"File {path} is not a csv file.")

    df = pd.read_csv(path, sep=sep)
    return NumpyEvent.from_dataframe(
        df, index_names=index_names, timestamp_column=timestamp_column
    )
