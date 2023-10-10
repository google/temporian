"""Matplotlib plotting backend."""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
import numpy as np


from temporian.core.data.duration_utils import (
    convert_timestamp_to_datetime,
)
from temporian.implementation.numpy.data.plotter_base import (
    Options,
    Style,
    PlotterBackend,
)


class Plotter(PlotterBackend):
    def __init__(self, num_plots: int, options: Options):
        super().__init__(num_plots, options)

        self.colors = get_cmap("tab10").colors
        px = 1 / plt.rcParams["figure.dpi"]

        self.fig, self.axs = plt.subplots(
            num_plots,
            figsize=(
                options.width_px * px,
                options.height_per_plot_px * num_plots * px,
            ),
            squeeze=False,
            sharex=True,
        )

        self.fig_idx = 0
        self.options = options

    def ax(self):
        return self.axs[self.fig_idx, 0]

    def new_subplot(
        self,
        title: Optional[str],
        num_items: int,
        is_unix_timestamp: bool,
    ):
        self.cur_num_items = num_items
        self.cur_is_unix_timestamp = is_unix_timestamp

        if title is not None:
            self.ax().set_title(title, fontsize=8)

    def finalize_subplot(
        self,
    ):
        if self.cur_num_items > 1:
            self.ax().legend(fontsize=8)
        self.fig_idx += 1

    def plot_feature(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        name: Optional[str],
        style: Style,
        color_idx: int,
    ):
        _matplotlib_sub_plot(
            ax=self.ax(),
            xs=xs,
            ys=ys,
            options=self.options,
            color=self.colors[color_idx % len(self.colors)],
            name=name if self.cur_num_items == 1 else None,
            legend=name if self.cur_num_items != 1 else None,
            is_unix_timestamp=self.cur_is_unix_timestamp,
            style=style,
        )

    def plot_sampling(
        self,
        xs: np.ndarray,
        color_idx: int,
        name: str,
    ):
        _matplotlib_sub_plot(
            ax=self.ax(),
            xs=xs,
            ys=np.zeros(len(xs)),
            options=self.options,
            color=self.colors[color_idx % len(self.colors)],
            name=name if self.cur_num_items == 1 else None,
            legend=name if self.cur_num_items != 1 else None,
            is_unix_timestamp=self.cur_is_unix_timestamp,
            style=Style.vline,
        )

    def finalize(self):
        self.fig.tight_layout()
        return self.fig


def _matplotlib_sub_plot(
    ax,
    xs,
    ys,
    options: Options,
    color,
    name: Optional[str],
    is_unix_timestamp: bool,
    style: Style,
    legend: Optional[str] = None,
    **wargs,
):
    """Plots "(xs, ys)" on the axis "ax"."""

    if style == Style.line:
        mat_style = {}  # Default
    elif style == Style.marker:
        mat_style = {"marker": "2", "linestyle": "None"}
    elif style == Style.vline:
        mat_style = {"marker": "|", "linestyle": "None"}
    else:
        raise ValueError("Non implemented style")

    if legend is not None:
        wargs["label"] = legend

    ax.plot(xs, ys, lw=0.5, color=color, **mat_style, **wargs)
    if options.min_time is not None or options.max_time is not None:
        args = {}
        if options.min_time is not None:
            args["left"] = (
                convert_timestamp_to_datetime(options.min_time)
                if is_unix_timestamp
                else options.min_time
            )
        if options.max_time is not None:
            args["right"] = (
                convert_timestamp_to_datetime(options.max_time)
                if is_unix_timestamp
                else options.max_time
            )
        ax.set_xlim(**args)

    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    if name is not None:
        ax.set_ylabel(name, size=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.grid(lw=0.4, ls="--", axis="x")
