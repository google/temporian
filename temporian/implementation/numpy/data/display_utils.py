from typing import Any, List

import numpy as np

from temporian.utils import string
from temporian.utils import config
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import convert_timestamp_to_datetime
from temporian.implementation.numpy.data.event_set import EventSet


def display_text(evset: EventSet) -> str:
    def repr_features(features: List[np.ndarray]) -> str:
        """Repr for a list of features."""

        feature_repr = []
        for idx, (feature_schema, feature_data) in enumerate(
            zip(evset.schema.features, features)
        ):
            if idx > config.max_printed_features:
                feature_repr.append("...")
                break

            feature_repr.append(f"'{feature_schema.name}': {feature_data}")
        return "\n".join(feature_repr)

    # Representation of the "data" field
    with np.printoptions(
        precision=config.print_precision,
        threshold=config.max_printed_events,
    ):
        data_repr = []

        for i, index_key in enumerate(evset.get_index_keys(sort=True)):
            index_data = evset.data[index_key]
            if i > config.max_printed_indexes:
                data_repr.append(f"... ({len(evset.data) - i} remaining)")
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
                f"{string.indent(repr_features(index_data.features))}"
            )
        data_repr = string.indent("\n".join(data_repr))

    return (
        f"indexes: {evset.schema.indexes}\n"
        f"features: {evset.schema.features}\n"
        "events:\n"
        f"{data_repr}\n"
        f"memory usage: {string.pretty_num_bytes(evset.memory_usage())}\n"
    )


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

    features = evset.schema.features[: config.max_display_features]
    n_features = len(evset.schema.features)
    cut_features = n_features > config.max_display_features
    indexes = evset.schema.indexes
    convert_datetime = evset.schema.is_unix_timestamp
    all_index_keys = evset.get_index_keys(sort=True)

    impl = minidom.getDOMImplementation()
    assert impl is not None
    dom = impl.createDocument(None, "div", None)
    top = dom.documentElement

    for index_key in all_index_keys[: config.max_display_indexes]:
        # Index and size (n events x n features)
        index_text = ", ".join(
            [
                f"{idx.name}={repr_value(val, idx.dtype)}"
                for val, idx in zip(index_key, indexes)
            ]
        )
        index_data = evset.data[index_key]
        n_events = len(index_data.timestamps)
        title = dom.createElement("h3")
        title.appendChild(dom.createTextNode(f"Index: ({index_text})"))
        descript = dom.createTextNode(
            f"{n_events} events × {n_features} features"
        )
        top.appendChild(title)
        top.appendChild(descript)

        # Table with header column names
        table = dom.createElement("table")
        col_names = ["timestamp"] + [feature.name for feature in features]
        table.appendChild(create_table_row(col_names, header=True))

        # Rows with events
        for i, timestamp in enumerate(
            index_data.timestamps[: config.max_display_events]
        ):
            row = []

            # Timestamp column
            timestamp = (
                convert_timestamp_to_datetime(timestamp)
                if convert_datetime
                else repr_float(timestamp)
            )
            row.append(f"{timestamp}")

            # Feature values
            for val, feature in zip(
                index_data.features[: config.max_display_features], features
            ):
                row.append(repr_value(val[i], feature.dtype))

            # Add ... column on the right
            if cut_features:
                row.append("...")

            # Create row and add
            table.appendChild(create_table_row(row))

        # Add ... row at the bottom
        if n_events > config.max_display_events:
            # Timestamp + features + <... column if was added>
            empty_row = ["..."] * (1 + len(features) + int(cut_features))
            table.appendChild(create_table_row(empty_row))

        top.appendChild(table)

    # If there are hidden indexes, show how many
    if len(all_index_keys) > config.max_display_indexes:
        top.appendChile(
            dom.createTextNode(
                f"... (showing {config.max_display_indexes} of"
                f" {len(all_index_keys)} indexes)"
            )
        )
    return top.toxml()


def repr_value(value: Any, dtype: DType) -> str:
    if dtype == DType.STRING:
        assert isinstance(value, bytes)
        repr = value.decode()
    elif dtype.is_float:
        repr = repr_float(value)
    else:
        repr = str(value)
    if len(repr) > config.max_display_chars:
        repr = repr[: config.max_display_chars] + "..."
    return repr


def repr_float(value: float) -> str:
    # Create string format with precision, e.g "{:.6g}"
    float_template = "{:.%d%s}" % (config.print_precision, "g")
    return float_template.format(value)
