from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple, Optional
from enum import Enum
from temporian.core.data.duration_utils import Timestamp


class Style(Enum):
    """Plotting style."""

    # Determine the style according to the data.
    auto = "auto"
    # Connect numerical values with a line.
    line = "line"
    # A discreet marker showing a feature value.
    marker = "marker"
    # A discreet marker not showing a feature value.
    vline = "vline"


class Options(NamedTuple):
    """Options for plotting."""

    backend: Optional[str]
    height_per_plot_px: int
    width_px: int
    max_points: Optional[int]
    min_time: Optional[Timestamp]
    max_time: Optional[Timestamp]
    max_num_plots: int
    style: Style
    interactive: bool


class PlotterBackend(ABC):
    """Create a plot.

    For instance, this class can be used as follow:

    ```
    __init__

    # Plot 1
    new_subplot
    plot_feature
    plot_feature
    finalize_subplot

    # Plot 2
    new_subplot
    plot_sampling
    plot_feature
    finalize_subplot

    finalize
    ```
    """

    @abstractmethod
    def __init__(self, num_plots: int, options: Options):
        """Starts the plot."""

        pass

    @abstractmethod
    def new_subplot(
        self,
        title: Optional[str],
        num_items: int,
        is_unix_timestamp: bool,
    ):
        """Adds a new sub plot."""

        raise NotImplementedError

    @abstractmethod
    def finalize_subplot(
        self,
    ):
        """Finalizes a previously added sub plot."""

        raise NotImplementedError

    @abstractmethod
    def plot_feature(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        name: Optional[str],
        style: Style,
        color_idx: int,
    ):
        """Plots a feature in the last added sub plot."""

        raise NotImplementedError

    @abstractmethod
    def plot_sampling(
        self,
        xs: np.ndarray,
        color_idx: int,
    ):
        """Plots samplings in the last added sub plot."""

        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalizes the plot."""

        raise NotImplementedError
