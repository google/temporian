from typing import NamedTuple, Optional, Union, List, Any
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.core.data import duration
import datetime

import numpy as np

DEFAULT_BACKEND = "matplotlib"


class Options(NamedTuple):
    """Options for plotting."""

    backend: str
    height_per_plot_px: int
    width_px: int
    max_points: Optional[int]
    min_time: Optional[duration.Timestamp]
    max_time: Optional[duration.Timestamp]


def plot(
    event: Union[List[NumpyEvent], NumpyEvent],
    index: Optional[Union[tuple, Any]] = (),
    backend: str = DEFAULT_BACKEND,
    width_px: int = 1024,
    height_per_plot_px: int = 150,
    max_points: Optional[int] = None,
    min_time: Optional[duration.Timestamp] = None,
    max_time: Optional[duration.Timestamp] = None,
):
    """Plots an event.

    Args:
        index: The index of the event to plot. Use 'event.index()' for the
            list of available indices. If index=None, select arbitrarily
            (non deterministically) an index to plot.
        backend: Plotting library to use.
        width_px: Width of the figure in pixel.
        height_per_plot_px: Height of each sub-plot (one per feature) in pixel.
        max_points: Maximum number of points to plot.
        min_time: If set, only plot events after min_time.
        max_time: If set, only plot events before min_time.
    """

    if isinstance(event, list):
        events = event
    else:
        events = [event]

    if index is None and len(events) > 0:
        index = events[0]._first_index_value

    if not isinstance(index, tuple):
        index = (index,)

    options = Options(
        backend=backend,
        width_px=width_px,
        height_per_plot_px=height_per_plot_px,
        max_points=max_points,
        min_time=min_time,
        max_time=max_time,
    )

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown plotting backend {backend}. Available "
            f"backends: {BACKENDS}"
        )

    return BACKENDS[backend](events=events, index=index, options=options)


def _plot_matplotlib(events: List[NumpyEvent], index: tuple, options: Options):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    colors = get_cmap("tab10").colors

    px = 1 / plt.rcParams["figure.dpi"]

    num_plots = 0
    for event in events:
        if index not in event.data:
            raise ValueError(
                f"Index '{index}' does not exist in event. Check the available"
                " indexes with 'event.index()' and provide one of those index"
                " to the 'index' argument of 'plot'. Alternatively, set "
                '"index=None" to select a random index value (e.g., '
                f"{event._first_index_value}."
            )
        num_features = len(event.feature_names)
        if num_features == 0:
            # We plot the sampling
            num_features = 1
        num_plots += num_features

    fig, axs = plt.subplots(
        num_plots,
        figsize=(
            options.width_px * px,
            options.height_per_plot_px * num_plots * px,
        ),
        squeeze=False,
    )

    plot_idx = 0
    for event in events:
        feature_names = event.feature_names

        xs = event.sampling.data[index]
        if options.max_points is not None and len(xs) > options.max_points:
            xs = xs[: options.max_points]

        if event.sampling.is_unix_timestamp:
            xs = [
                datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
                for x in xs
            ]

        if len(feature_names) == 0:
            # Plot the ticks

            ax = axs[plot_idx, 0]

            _matplotlib_sub_plot(
                ax=ax,
                xs=xs,
                ys=np.zeros(len(xs)),
                options=options,
                color=colors[feature_idx % len(colors)],
                name="[sampling]",
                marker="+",
                is_unix_timestamp=event.sampling.is_unix_timestamp,
            )

            plot_idx += 1

        for feature_idx, feature_name in enumerate(feature_names):
            ax = axs[plot_idx, 0]

            ys = event.data[index][feature_idx].data
            if options.max_points is not None and len(ys) > options.max_points:
                ys = ys[: options.max_points]

            _matplotlib_sub_plot(
                ax=ax,
                xs=xs,
                ys=ys,
                options=options,
                color=colors[feature_idx % len(colors)],
                name=feature_name,
                is_unix_timestamp=event.sampling.is_unix_timestamp,
            )

            plot_idx += 1

    fig.tight_layout()
    return fig


def _matplotlib_sub_plot(
    ax, xs, ys, options, color, name, is_unix_timestamp, **wargs
):
    import matplotlib.ticker as ticker

    ax.plot(xs, ys, lw=0.5, color=color, **wargs)

    if options.min_time is not None or options.max_time is not None:
        args = {}
        if options.min_time is not None:
            args["left"] = (
                duration.convert_date_to_duration(options.min_time)
                if not is_unix_timestamp
                else options.min_time
            )
        if options.max_time is not None:
            args["right"] = (
                duration.convert_date_to_duration(options.max_time)
                if not is_unix_timestamp
                else options.max_time
            )
        ax.set_xlim(**args)

    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_ylabel(name, size=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.grid(lw=0.4, ls="--", axis="x")


BACKENDS = {"matplotlib": _plot_matplotlib}
