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

"""Plotting utility."""

from typing import Optional, Union, List, Set, Tuple, Dict, Callable, Type

from dataclasses import dataclass
import numpy as np

from temporian.core.data.duration_utils import (
    convert_timestamps_to_datetimes,
)
from temporian.implementation.numpy.data.plotter_base import Options, Style
from temporian.core.data import duration_utils
from temporian.implementation.numpy.data.event_set import (
    EventSet,
    normalize_index_key,
    IndexItemType,
    IndexType,
)
from temporian.implementation.numpy.data.plotter_base import (
    Options,
    Style,
    PlotterBackend,
)

# "evtset" argument in tp.plot.
InputEventSet = Union[
    EventSet,
    List[EventSet],
    Tuple[EventSet, ...],
    List[Tuple[EventSet, ...]],
]

# "indexes" argument in tp.plot.
InputIndex = Optional[
    Union[
        IndexItemType,
        IndexType,
        List[IndexItemType],
        List[IndexType],
    ]
]

# "features" argument in tp.plot.
InputFeatures = Optional[
    Union[
        str,
        List[str],
        Set[str],
    ]
]


@dataclass
class GroupItem:
    evtset: EventSet

    # Index of the feature in "evtset". If -1, plots the timestamp.
    feature_idx: int


@dataclass
class Group:
    """Features / timestaps that get plotted together."""

    items: List[GroupItem]


Groups = List[Group]


def matplotlib_backend():
    from temporian.implementation.numpy.data import plotter_matplotlib

    return plotter_matplotlib.Plotter


def bokeh_backend():
    from temporian.implementation.numpy.data import plotter_bokeh

    return plotter_bokeh.Plotter


BACKENDS: Dict[str, Callable] = {
    "matplotlib": matplotlib_backend,
    "bokeh": bokeh_backend,
}


def error_message_import_backend(backend: str) -> str:
    return (
        f"Cannot plot with selected backend={backend}. Solutions: (1) Install"
        f" {backend} e.g. 'pip install {backend}', or (2) use a different"
        " plotting backen, for example with 'plot(..., backend=\"<other"
        f' backend>"). The supported backends are: {list(BACKENDS.keys())}.'
    )


def build_groups(
    evsets: InputEventSet, features: Optional[Set[str]], allow_list: bool = True
) -> Groups:
    """Sort user inputs into groups of features to plot together."""

    if isinstance(evsets, EventSet):
        # Plot each feature individually
        groups = []
        for feature_idx, feature in enumerate(evsets.schema.features):
            if features is not None and feature.name not in features:
                continue
            groups.append(Group([GroupItem(evsets, feature_idx)]))
        if len(groups) == 0:
            # Plot the timestamps
            groups.append(Group([GroupItem(evsets, -1)]))
        return groups

    if isinstance(evsets, tuple):
        # Plot all the event sets and their features together
        group_items = []
        for evset in evsets:
            if not isinstance(evset, EventSet):
                raise ValueError(
                    f"Expecting tuple of eventsets. Got {type(evset)} instead."
                )
            plot_for_current_evtset = False
            for feature_idx, feature in enumerate(evset.schema.features):
                if features is not None and feature.name not in features:
                    continue
                group_items.append(GroupItem(evset, feature_idx))
                plot_for_current_evtset = True
            if not plot_for_current_evtset:
                group_items.append(GroupItem(evset, -1))

        return [Group(group_items)]

    if allow_list and isinstance(evsets, list):
        groups = []
        for x in evsets:
            groups.extend(build_groups(x, features, allow_list=False))
        return groups
    raise ValueError("Non supported evsets input")


def normalize_features(features: InputFeatures) -> Optional[Set[str]]:
    """Normalizes the "features" argument of plot."""

    if features is None:
        return None
    if isinstance(features, str):
        return {features}
    if isinstance(features, list):
        return set(features)
    if isinstance(features, set):
        return features
    raise ValueError(f"Non supported feature type {features}")


def normalize_indexes(indexes: InputIndex, groups: Groups) -> List[IndexType]:
    """Normalizes the "indexes" argument of plot."""

    if indexes is None:
        # All the available index
        normalized_indexes = list(
            groups[0].items[0].evtset.get_index_keys(sort=True)
        )

    elif isinstance(indexes, list):
        # e.g. indexes=["a", ("b",)]
        normalized_indexes = [
            v if isinstance(v, tuple) else (v,) for v in indexes
        ]

    elif isinstance(indexes, tuple):
        # e.g. indexes=("a",)
        normalized_indexes = [indexes]

    else:
        # e.g. indexes="a"
        normalized_indexes = [(indexes,)]

    normalized_indexes = [normalize_index_key(x) for x in normalized_indexes]
    validate_indexes(normalized_indexes, groups)
    return normalized_indexes


def validate_indexes(indexes: List[IndexType], groups: Groups):
    """Ensures that indexes are valid i.e. available in all event-sets."""

    for g in groups:
        for item in g.items:
            for index in indexes:
                if index not in item.evtset.data:
                    raise ValueError(
                        f"Index {index!r} does not exist in event set:"
                        f" {item.evtset}"
                    )


def plot(
    evsets: InputEventSet,
    indexes: InputIndex = None,
    features: InputFeatures = None,
    width_px: int = 1024,
    height_per_plot_px: int = 150,
    max_points: Optional[int] = None,
    min_time: Optional[duration_utils.Timestamp] = None,
    max_time: Optional[duration_utils.Timestamp] = None,
    max_num_plots: int = 20,
    style: Union[Style, str] = Style.auto,
    return_fig: bool = False,
    interactive: bool = False,
    backend: Optional[str] = None,
):
    """Plots one or several [`EventSets`][temporian.EventSet].


    If multiple eventsets are provided, they should all have the same index.
    The time axis (i.e., horizontal axis) is shared among all the plots.
    Different features can be plotted independently or on the same plots.
    Plotting an eventset without features plots timestamps instead.

    When plotting a single eventset, this function is equivalent to
    [`EventSet.plot()`][temporian.EventSet.plot].

    Examples:
        ```python
        >>> evset = tp.event_set(timestamps=[1, 2, 4],
        ...     features={"f1": [0, 42, 10], "f2": [10, -10, 20]})

        # Plot each feature individually
        >>> tp.plot(evset)

        # Plots multiple features in the same sub-plot.
        >>> tp.plot((evset,), interactive=True)

        # Make the plot interractive
        >>> tp.plot(evset, interactive=True)

        # Save figure to file
        >>> fig = tp.plot(evset, return_fig=True)
        >>> fig.savefig("/tmp/fig.png")

        # Change drawing style
        >>> tp.plot(evset, style="line")

        ```

    Args:
        evsets: Single or list of EventSets to plot.
        indexes: The index keys or list of indexes keys to plot. If
            indexes=None, plots all the available indexes. Indexes should be
            provided as single value (e.g. string) or tuple of values. Example:
            indexes="a", indexes=("a",), indexes=("a", "b",),
            indexes=["a", "b"], indexes=[("a", "b"), ("a", "c")].
        features: Feature names of the event(s) to plot. Use
            'evset.feature_names' for the list of available names.
            If a feature doesn't exist in an event, it's silently skipped.
            If None, plots all features of all events.
        width_px: Width of the figure in pixel.
        height_per_plot_px: Height of each sub-plot (one per feature) in pixel.
        max_points: Maximum number of points to plot.
        min_time: If set, only plot events after it.
        max_time: If set, only plot events before it.
        max_num_plots: Maximum number of plots to display. If more plots are
            available, only plot the first `max_num_plots` ones and print a
            warning.
        style: A `Style` or equivalent string like: `line`, `marker` or `vline`.
        return_fig: If true, returns the figure object. The figure object
            depends on the backend.
        interactive: If true, creates an interactive plotting. interactive=True
            effectively selects a backend that support interactive plotting.
            Ignored if "backend" is set.
        backend: Plotting library to use. Possible values are: matplotlib,
            bokeh. If set, overrides the "interactive" argument.
    """

    normalized_features = normalize_features(features)
    groups = build_groups(evsets, normalized_features)
    normalized_indexes = normalize_indexes(indexes, groups)

    if len(groups) == 0:
        raise ValueError("Not input eventsets")

    if isinstance(style, str):
        style = Style[style]
    assert isinstance(style, Style)

    options = Options(
        interactive=interactive,
        backend=backend,
        width_px=width_px,
        height_per_plot_px=height_per_plot_px,
        max_points=max_points,
        min_time=(
            duration_utils.normalize_timestamp(min_time)
            if min_time is not None
            else None
        ),
        max_time=(
            duration_utils.normalize_timestamp(max_time)
            if max_time is not None
            else None
        ),
        max_num_plots=max_num_plots,
        style=style,
    )

    if backend is None:
        backend = "bokeh" if interactive else "matplotlib"

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown plotting backend {backend}. Available "
            f"backends: {BACKENDS}"
        )

    try:
        plotter_class = BACKENDS[backend]()
        fig = plot_with_plotter(
            plotter_class=plotter_class,
            groups=groups,
            indexes=normalized_indexes,
            options=options,
        )

    except ImportError:
        print(error_message_import_backend(backend))
        raise

    return fig if return_fig else None


def plot_with_plotter(
    plotter_class: Type[PlotterBackend],
    groups: Groups,
    indexes: List[IndexType],
    options: Options,
):
    num_plots = get_num_plots(groups, indexes, options)
    plotter: PlotterBackend = plotter_class(num_plots, options)

    index_names = groups[0].items[0].evtset.schema.index_names()

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

            plotter.new_subplot(
                title=title,
                num_items=len(group.items),
                is_unix_timestamp=group.items[
                    0
                ].evtset.schema.is_unix_timestamp,
            )

            # Only print the index / title once
            title = None

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
                    xs = convert_timestamps_to_datetimes(xs)

                if group_item.feature_idx == -1:
                    # Plot the timestamps.
                    plotter.plot_sampling(xs=xs, color_idx=color_idx)
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

                    plotter.plot_feature(
                        xs=xs,
                        ys=ys,
                        name=feature_name,
                        style=effective_stype,
                        color_idx=color_idx,
                    )
                color_idx += 1

            plotter.finalize_subplot()

            plot_idx += 1

    return plotter.finalize()


def get_num_plots(
    groups: Groups,
    indexes: List[tuple],
    options: Options,
):
    """Computes the number of sub-plots."""

    num_plots = len(indexes) * len(groups)
    if num_plots == 0:
        raise ValueError("There is nothing to plot.")

    if num_plots > options.max_num_plots:
        print(
            f"The number of plots ({num_plots}) is larger than "
            f'"options.max_num_plots={options.max_num_plots}". Only the first '
            "plots will be printed."
        )
        num_plots = options.max_num_plots

    return num_plots


def auto_style(uniform: bool, xs, ys) -> Style:
    """Finds the best plotting style."""

    if len(ys) <= 1:
        return Style.marker

    if len(ys) == 0:
        all_ys_are_equal = True
    else:
        all_ys_are_equal = np.all(ys == ys[0])

    if not uniform and (len(xs) <= 1000 or all_ys_are_equal):
        return Style.marker
    else:
        return Style.line


def is_uniform(xs) -> bool:
    """Checks if timestamps are uniformly sampled."""

    diff = np.diff(xs)
    if len(diff) == 0:
        return True
    return np.allclose(diff, diff[0])
