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

from dataclasses import dataclass

from typing import NamedTuple, Optional, Union, List, Any, Set, Tuple
from enum import Enum

import numpy as np

from temporian.core.data import duration_utils
from temporian.implementation.numpy.data.event_set import (
    EventSet,
    normalize_index_key,
    IndexItemType,
    IndexType,
)

# How to input event sets in the plotter.
InputEventSet = Union[
    EventSet,
    List[EventSet],
    Tuple[EventSet, ...],
    List[Tuple[EventSet, ...]],
]

# How to input indexes in the plotter.
InputIndex = Optional[
    Union[
        IndexItemType,
        IndexType,
        List[IndexItemType],
        List[IndexType],
    ]
]

InputFeatures = Optional[
    Union[
        str,
        List[str],
        Set[str],
    ]
]


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
    min_time: Optional[duration_utils.Timestamp]
    max_time: Optional[duration_utils.Timestamp]
    max_num_plots: int
    style: Style
    interactive: bool


@dataclass
class GroupItem:
    evtset: EventSet
    feature_idx: int  # Index of the feature. If -1, plots the timestamp.


@dataclass
class Group:
    """Features / timestaps that get plotted together."""

    items: List[GroupItem]


Groups = List[Group]


def build_groups(
    evsets: InputEventSet, features: Optional[Set[str]], allow_list: bool = True
) -> Groups:
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
    """Plots [`EventSets`][temporian.EventSet].

    Plots one or several event sets. If multiple eventsets are provided, they
    should all have the same index. If plotting an eventset without features,
    the timestamps are plotted with vertical bars. The time axis (i.e.,
    horizontal axis) is shared among all the plots.

    This method can also be called from
    [`EventSet.plot()`][temporian.EventSet.plot] with the same args (except
    `evsets`).

    Examples:
        ```python
        >>> evset = tp.event_set(timestamps=[1, 2, 4],
        ...     features={"f1": [0, 42, 10], "f2": [10, -10, 20]})

        # Default
        >>> tp.plot(evset)

        # Lines instead of markers, only f2, limit x-axis to t=2
        >>> tp.plot(evset, style="line", features="f2", max_time=2)

        # Access figure and axes
        >>> fig = tp.plot(evset, return_fig=True)
        >>> fig.tight_layout(pad=3.0)
        >>> _ = fig.axes[0].set_ylim([-50, 50])

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
        fig = BACKENDS[backend](
            groups=groups,
            indexes=normalized_indexes,
            options=options,
        )
    except ImportError:
        print(error_message_import_backend(backend))
        raise

    return fig if return_fig else None


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


from temporian.implementation.numpy.data.plotter_bokeh import plot_bokeh
from temporian.implementation.numpy.data.plotter_matplotlib import (
    plot_matplotlib,
)

BACKENDS = {"matplotlib": plot_matplotlib, "bokeh": plot_bokeh}


def error_message_import_backend(backend: str) -> str:
    return (
        f"Cannot plot with selected backend={backend}. Solutions: (1) Install"
        f" {backend} e.g. 'pip install {backend}', or (2) use a different"
        " plotting backen, for example with 'plot(..., backend=\"<other"
        f' backend>"). The supported backends are: {list(BACKENDS.keys())}.'
    )
