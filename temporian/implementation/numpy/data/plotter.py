from typing import NamedTuple, Optional, Union, List, Any
from temporian.implementation.numpy.data.event import NumpyEvent

import numpy as np

DEFAULT_BACKEND = "matplotlib"


class Options(NamedTuple):
    """Options for plotting."""

    backend: str
    height_per_plot_px: int
    width_px: int
    max_points: Optional[int]


def plot(
    event: Union[List[NumpyEvent], NumpyEvent],
    index: Union[tuple, Any] = (),
    backend: str = DEFAULT_BACKEND,
    width_px: int = 1024,
    height_per_plot_px: int = 150,
    max_points: Optional[int] = None,
):
    """Plots an event.

    Args:
        index: The index of the event to plot. Use 'event.index()' for the
            list of available indices.
        backend: Plotting library to use.
        width_px: Width of the figure in pixel.
        height_per_plot_px: Height of each sub-plot (one per feature) in pixel.
        max_points: Maximum number of points to plot.
    """

    if not isinstance(index, tuple):
        index = (index,)

    options = Options(
        backend=backend,
        width_px=width_px,
        height_per_plot_px=height_per_plot_px,
        max_points=max_points,
    )

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown plotting backend {backend}. Available "
            "backends: {BACKENDS}"
        )

    if isinstance(event, list):
        events = event
    else:
        events = [event]

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
                " to the 'index' argument of 'plot'."
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
            )

            plot_idx += 1

    fig.tight_layout()
    return fig


def _matplotlib_sub_plot(ax, xs, ys, options, color, name, **wargs):
    import matplotlib.ticker as ticker

    ax.plot(xs, ys, lw=0.5, color=color, **wargs)

    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_ylabel(name, size=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.grid(lw=0.4, ls="--", axis="x")


BACKENDS = {"matplotlib": _plot_matplotlib}
