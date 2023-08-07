import sys
from typing import Any, List

import numpy as np

from temporian.utils import string
from temporian.utils import config
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import convert_timestamp_to_datetime
from temporian.implementation.numpy.data.event_set import EventSet

ELLIPSIS = "…"


def display_html(evset: EventSet) -> str:
    """HTML representation, mainly for IPython notebooks."""
    from xml.dom import minidom

    def create_table_row(
        columns: List[str], header: bool = False
    ) -> minidom.Element:
        tr = dom.createElement("tr")
        for col in columns:
            if header:
                td = dom.createElement("th")
                text = dom.createElement("b")
                text.appendChild(dom.createTextNode(col))
            else:
                td = dom.createElement("td")
                text = dom.createTextNode(col)
            td.appendChild(text)
            tr.appendChild(td)
        return tr

    # Create DOM
    impl = minidom.getDOMImplementation()
    assert impl is not None
    dom = impl.createDocument(None, "div", None)
    top = dom.documentElement

    # Other configs
    convert_datetime = evset.schema.is_unix_timestamp
    index_schemas = evset.schema.indexes
    feature_schemas = evset.schema.features
    all_index_keys = evset.get_index_keys(sort=True)
    num_indexes = len(all_index_keys)
    num_features = len(evset.schema.features)

    # If limit=0, set limit=len
    max_indexes = config.max_display_indexes or num_indexes
    max_features = config.max_display_features or num_features
    has_hidden_feats = num_features > max_features
    visible_feats = feature_schemas[:max_features]

    # Index and features schemas
    evset_info = (
        f"{num_indexes} indexes × {num_features} features"
        f" (memory usage: {string.pretty_num_bytes(evset.memory_usage())})"
    )
    top.appendChild(dom.createTextNode(evset_info))

    # Create one table and header per index value
    for index_key in all_index_keys[:max_indexes]:
        # Index header text (n events x n features)
        index_text = ", ".join(
            [
                f"{idx.name}={repr_value_html(val, idx.dtype)}"
                for val, idx in zip(index_key, index_schemas)
            ]
        )
        index_data = evset.data[index_key]
        num_timestamps = len(index_data.timestamps)
        max_timestamps = config.max_display_events or num_timestamps
        title = dom.createElement("h3")
        title.appendChild(dom.createTextNode(f"Index: ({index_text})"))
        descript = dom.createTextNode(f"{num_timestamps} events")
        top.appendChild(title)
        top.appendChild(descript)

        # Table with column names
        table = dom.createElement("table")
        col_names = ["timestamp"] + [feature.name for feature in visible_feats]
        if has_hidden_feats:
            col_names += [ELLIPSIS]
        table.appendChild(create_table_row(col_names, header=True))

        # Rows with events
        for timestamp_idx, timestamp in enumerate(
            index_data.timestamps[:max_timestamps]
        ):
            row = []

            # Timestamp column
            timestamp_repr = (
                convert_timestamp_to_datetime(timestamp)
                if convert_datetime
                else repr_float_html(timestamp)
            )
            row.append(f"{timestamp_repr}")

            # Feature values
            for val, feature in zip(
                index_data.features[:max_features], visible_feats
            ):
                row.append(repr_value_html(val[timestamp_idx], feature.dtype))

            # Add ... column on the right
            if has_hidden_feats:
                row.append(ELLIPSIS)

            # Create row and add
            table.appendChild(create_table_row(row))

        # Add ... row at the bottom
        if num_timestamps > max_timestamps:
            # Timestamp + features + <... column if was added>
            row = [ELLIPSIS] * (1 + len(visible_feats) + int(has_hidden_feats))
            table.appendChild(create_table_row(row))

        top.appendChild(table)

    # If there are hidden indexes, show how many
    if num_indexes > max_indexes:
        hidden_indexes = num_indexes - max_indexes
        top.appendChild(
            dom.createTextNode(
                f"{ELLIPSIS} ({hidden_indexes} more indexes not shown)"
            )
        )
    return top.toxml()


def display_text(evset: EventSet) -> str:
    # Configs and defaults
    max_events = config.max_printed_events or sys.maxsize  # see np.printoptions
    max_indexes = config.max_printed_indexes  # 0 will print all

    # Representation of the "data" field
    with np.printoptions(
        precision=config.print_precision,
        threshold=max_events,
    ):
        data_repr = []

        for i, index_key in enumerate(evset.get_index_keys(sort=True)):
            index_data = evset.data[index_key]
            if max_indexes and (i + 1) > max_indexes:
                data_repr.append(
                    f"... (showing {max_indexes} of {len(evset.data)} indexes)"
                )
                break
            index_key_repr = []
            for index_value, index_name in zip(
                index_key, evset.schema.index_names()
            ):
                index_key_repr.append(f"{index_name}={index_value}")
            index_key_repr = " ".join(index_key_repr)
            data_repr.append(
                f"{index_key_repr} ({len(index_data.timestamps)} events):\n"
                f"    timestamps: {index_data.timestamps}\n"
                f"{string.indent(repr_features_text(evset, index_data.features))}"
            )
        data_repr = string.indent("\n".join(data_repr))

    return (
        f"indexes: {evset.schema.indexes}\n"
        f"features: {evset.schema.features}\n"
        "events:\n"
        f"{data_repr}\n"
        f"memory usage: {string.pretty_num_bytes(evset.memory_usage())}\n"
    )


def repr_value_html(value: Any, dtype: DType) -> str:
    if dtype == DType.STRING:
        assert isinstance(value, bytes)
        repr = value.decode()
    elif dtype.is_float:
        repr = repr_float_html(value)
    else:
        repr = str(value)
    max_chars = config.max_display_chars or None
    if max_chars is not None and len(repr) > max_chars:
        repr = repr[:max_chars] + ELLIPSIS
    return repr


def repr_float_html(value: float) -> str:
    # Create string format with precision, e.g "{:.6g}"
    float_template = "{:.%d%s}" % (config.print_precision, "g")
    return float_template.format(value)


def repr_features_text(evset: EventSet, features: List[np.ndarray]) -> str:
    """Repr for a list of features."""

    max_features = config.max_printed_features  # 0 will print all
    feature_repr = []
    for idx, (feature_schema, feature_data) in enumerate(
        zip(evset.schema.features, features)
    ):
        if max_features and (idx + 1) > max_features:
            feature_repr.append("...")
            break

        feature_repr.append(f"'{feature_schema.name}': {feature_data}")
    return "\n".join(feature_repr)
