"""Matplotlib plotting backend."""

import datetime
from typing import Optional, List, Set

import numpy as np

from temporian.core.data.duration_utils import (
    convert_timestamp_to_datetime,
    convert_timestamps_to_datetimes,
)
from temporian.implementation.numpy.data.event_set import EventSet, IndexType
from temporian.implementation.numpy.data.plotter import (
    Options,
    Style,
    is_uniform,
    get_num_plots,
    auto_style,
    Groups,
)


def plot_matplotlib(
    groups: Groups,
    indexes: List[IndexType],
    options: Options,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    colors = get_cmap("tab10").colors

    px = 1 / plt.rcParams["figure.dpi"]

    num_plots = get_num_plots(groups, indexes, options)

    fig, axs = plt.subplots(
        num_plots,
        figsize=(
            options.width_px * px,
            options.height_per_plot_px * num_plots * px,
        ),
        squeeze=False,
        sharex=True,
    )

    index_names = groups[0].items[0].evtset.schema.index_names()

    # Actual plotting
    plot_idx = 0
    for index in indexes:
        assert len(index_names) == len(index)
        if plot_idx >= num_plots:
            # Too many plots are displayed already.
            break

        title = " ".join([f"{k}={v}" for k, v in zip(index_names, index)])

        # Index of the next color to use in the plot.
        color_idx = 0

        for group in groups:
            if plot_idx >= num_plots:
                break
            group_has_one_item = len(group.items) == 1

            for group_item in group.items:
                xs = group_item.evtset.data[index].timestamps
                uniform = is_uniform(xs)

                plot_mask = np.full(len(xs), True)
                if options.min_time is not None:
                    plot_mask = plot_mask & (xs >= options.min_time)
                if options.max_time is not None:
                    plot_mask = plot_mask & (xs <= options.max_time)
                if (
                    options.max_points is not None
                    and len(xs) > options.max_points
                ):
                    # Too many timestamps. Only keep the fist ones.
                    plot_mask = plot_mask & (
                        np.cumsum(plot_mask) <= options.max_points
                    )

                xs = xs[plot_mask]

                if group_item.evtset.schema.is_unix_timestamp:
                    # Matplotlib understands datetimes.
                    xs = convert_timestamps_to_datetimes(xs)

                if group_item.feature_idx == -1:
                    # Plot the timestamps.
                    _matplotlib_sub_plot(
                        ax=axs[plot_idx, 0],
                        xs=xs,
                        ys=np.zeros(len(xs)),
                        options=options,
                        color=colors[color_idx % len(colors)],
                        name="[sampling]",
                        is_unix_timestamp=group_item.evtset.schema.is_unix_timestamp,
                        title=title,
                        style=Style.vline,
                    )
                else:
                    feature_name = group_item.evtset.schema.features[
                        group_item.feature_idx
                    ].name

                    ys = group_item.evtset.data[index].features[
                        group_item.feature_idx
                    ]
                    ys = ys[plot_mask]
                    if options.style == Style.auto:
                        effective_stype = auto_style(uniform, xs, ys)
                    else:
                        effective_stype = options.style

                    _matplotlib_sub_plot(
                        ax=axs[plot_idx, 0],
                        xs=xs,
                        ys=ys,
                        options=options,
                        color=colors[color_idx % len(colors)],
                        name=feature_name if group_has_one_item else None,
                        legend=feature_name if not group_has_one_item else None,
                        is_unix_timestamp=group_item.evtset.schema.is_unix_timestamp,
                        title=title,
                        style=effective_stype,
                    )

                # Only print the index / title once
                title = None

                color_idx += 1

            if not group_has_one_item:
                axs[plot_idx, 0].legend(fontsize=8)

            plot_idx += 1

    fig.tight_layout()
    return fig


def _matplotlib_sub_plot(
    ax,
    xs,
    ys,
    options: Options,
    color,
    name: Optional[str],
    is_unix_timestamp: bool,
    title: Optional[str],
    style: Style,
    legend: Optional[str] = None,
    **wargs,
):
    """Plots "(xs, ys)" on the axis "ax"."""

    import matplotlib.ticker as ticker

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
    if title:
        ax.set_title(title, fontsize=8)
