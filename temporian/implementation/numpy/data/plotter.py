# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from typing import NamedTuple, Optional, Union, List, Any

import numpy as np

from temporian.core.data import duration
from temporian.implementation.numpy.data.event_set import EventSet

DEFAULT_BACKEND = "matplotlib"


class Options(NamedTuple):
    """Options for plotting."""

    backend: str
    height_per_plot_px: int
    width_px: int
    max_points: Optional[int]
    min_time: Optional[duration.Timestamp]
    max_time: Optional[duration.Timestamp]
    max_num_plots: int


def plot(
    evsets: Union[List[EventSet], EventSet],
    indexes: Optional[Union[Any, tuple, List[tuple]]] = None,
    backend: str = DEFAULT_BACKEND,
    width_px: int = 1024,
    height_per_plot_px: int = 150,
    max_points: Optional[int] = None,
    min_time: Optional[duration.Timestamp] = None,
    max_time: Optional[duration.Timestamp] = None,
    max_num_plots: int = 20,
):
    """Plots event sets.

    Args:
        evsets: Single or list of event sets to plot.
        indexes: The index or list of indexes to plot. If index=None, plots all
            the available indexes. Indexes should be provided as single value
            (e.g. string) or tuple of values. Example: index="a", index=("a",),
            index=("a", "b",), index=["a", "b"], index=[("a", "b"), ("a", "c")].
        backend: Plotting library to use.
        width_px: Width of the figure in pixel.
        height_per_plot_px: Height of each sub-plot (one per feature) in pixel.
        max_points: Maximum number of points to plot.
        min_time: If set, only plot events after it.
        max_time: If set, only plot events before it.
        max_num_plots: Maximum number of plots to display. If more plots are
            available, only plot the first `max_num_plots` ones and print a
            warning.
    """

    original_indexes = indexes

    if not isinstance(evsets, list):
        evsets = [evsets]

    if len(evsets) == 0:
        raise ValueError("Events is empty")

    if indexes is None:
        # All the indexes
        indexes = list(evsets[0].data.keys())

    elif isinstance(indexes, tuple):
        # e.g. indexes=("a",)
        indexes = [indexes]

    elif isinstance(indexes, list):
        # e.g. indexes=["a", ("b",)]
        indexes = [v if isinstance(v, tuple) else (v,) for v in indexes]

    else:
        # e.g. indexes="a"
        indexes = [(indexes,)]

    for index in indexes:
        if not isinstance(index, tuple):
            raise ValueError(
                "An index should be a tuple or a list of tuples. Instead"
                f' receives "indexes={original_indexes}"'
            )

    options = Options(
        backend=backend,
        width_px=width_px,
        height_per_plot_px=height_per_plot_px,
        max_points=max_points,
        min_time=min_time,
        max_time=max_time,
        max_num_plots=max_num_plots,
    )

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown plotting backend {backend}. Available "
            f"backends: {BACKENDS}"
        )

    return BACKENDS[backend](evsets=evsets, indexes=indexes, options=options)


def _plot_matplotlib(
    evsets: List[EventSet], indexes: List[tuple], options: Options
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    colors = get_cmap("tab10").colors

    px = 1 / plt.rcParams["figure.dpi"]

    # Compute the number of sub-plots + extra checks.
    num_plots = 0
    for index in indexes:
        for evset in evsets:
            if index not in evset.data:
                raise ValueError(
                    f"Index '{index}' does not exist in event set. Check the"
                    " available indexes with 'evset.index' and provide one of"
                    " those index to the 'index' argument of 'plot'."
                    ' Alternatively, set "index=None" to select a random'
                    f" index value (e.g., {evset.first_index_key()}."
                )
            num_features = len(evset.feature_names)
            if num_features == 0:
                # We plot the sampling
                num_features = 1
            num_plots += num_features

    if num_plots == 0:
        raise ValueError("There is nothing to plot.")

    if num_plots > options.max_num_plots:
        print(
            f"The number of plots ({num_plots}) is larger than "
            f'"options.max_num_plots={options.max_num_plots}". Only the first '
            "plots will be printed."
        )
        num_plots = options.max_num_plots

    fig, axs = plt.subplots(
        num_plots,
        figsize=(
            options.width_px * px,
            options.height_per_plot_px * num_plots * px,
        ),
        squeeze=False,
    )

    # Actual plotting
    plot_idx = 0
    for index in indexes:
        if plot_idx >= num_plots:
            # Too much plots are displayed already.
            break

        # Note: Don't display the tuple parenthesis is the index contain a
        # single dimension.
        title = str(index[0] if len(index) == 1 else index)

        for evset in evsets:
            if plot_idx >= num_plots:
                break

            feature_names = evset.feature_names

            xs = evset.data[index].timestamps
            if options.max_points is not None and len(xs) > options.max_points:
                # Too many timestamps. Only keep the fist ones.
                xs = xs[: options.max_points]

            uniform = is_uniform(xs)

            if evset.is_unix_timestamp:
                # Matplotlib understands datetimes.
                xs = [
                    datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
                    for x in xs
                ]

            if len(feature_names) == 0:
                # There is not features to plot. Instead, plot the timestamps.
                _matplotlib_sub_plot(
                    ax=axs[plot_idx, 0],
                    xs=xs,
                    ys=np.zeros(len(xs)),
                    options=options,
                    color=colors[0],
                    name="[sampling]",
                    marker="+",
                    linestyle="None",
                    is_unix_timestamp=evset.is_unix_timestamp,
                    title=title,
                )
                # Only print the index / title once
                title = None

                plot_idx += 1

            for feature_idx, feature_name in enumerate(feature_names):
                if plot_idx >= num_plots:
                    # Too much plots are displayed already.
                    break

                ys = evset.data[index].features[feature_idx]
                if (
                    options.max_points is not None
                    and len(ys) > options.max_points
                ):
                    # Too many timestamps. Only keep the fist ones.
                    ys = ys[: options.max_points]

                _matplotlib_sub_plot(
                    ax=axs[plot_idx, 0],
                    xs=xs,
                    ys=ys,
                    options=options,
                    color=colors[feature_idx % len(colors)],
                    name=feature_name,
                    is_unix_timestamp=evset.is_unix_timestamp,
                    title=title,
                    marker="None" if uniform else "+",
                )

                # Only print the index / title once
                title = None

                plot_idx += 1

    fig.tight_layout()
    return fig


def _matplotlib_sub_plot(
    ax,
    xs,
    ys,
    options: Options,
    color,
    name: str,
    is_unix_timestamp: bool,
    title: Optional[str],
    **wargs,
):
    """Plots "(xs, ys)" on the axis "ax"."""

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
    if title:
        ax.set_title(title)


def is_uniform(xs) -> bool:
    """Checks if timestamps are uniformly sampled."""

    diff = np.diff(xs)
    if len(diff) == 0:
        return True
    return np.allclose(diff, diff[0])


BACKENDS = {"matplotlib": _plot_matplotlib}
