import sys
from typing import Any, List, Dict, Union
from xml.dom import minidom

import numpy as np

from temporian.utils import string
from temporian.utils import config
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import convert_timestamp_to_datetime
from temporian.implementation.numpy.data.event_set import EventSet

ELLIPSIS = "…"

Html = Any
Dom = Any
StyleHtml = Union[Html, str]

_HTML_STYLE_FEATURE_KEY = {"color": "#D81B60"}
_HTML_STYLE_INDEX_KEY = {"color": "#d55e00"}
_HTML_STYLE_DTYPE = {"color": "#4c9041"}
_HTML_STYLE_INDEX_VALUE = {"color": "#1E88E5"}
_HTML_STYLE_HEADER_DIV = {
    "margin-bottom": "11px",
    "padding": "5px",
    "font-size": "small",
    "line-height": "120%",
    "border": "1px solid rgba(127, 127, 127, 0.2)",
}
_HTML_STYLE_TABLE = {
    "margin-left": "20px",
    "border": "1px solid rgba(127, 127, 127, 0.2)",
}


def display_html(evset: EventSet) -> str:
    """HTML representation, mainly for IPython notebooks."""

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
    feature_schemas = evset.schema.features
    all_index_keys = evset.get_index_keys(sort=True)
    num_indexes = len(all_index_keys)
    num_features = len(evset.schema.features)

    # If limit=0 or None, set limit=len
    max_indexes = config.max_display_indexes or num_indexes
    max_features = config.max_display_features or num_features
    has_hidden_feats = num_features > max_features
    visible_feats = feature_schemas[:max_features]

    # Header with features and indexes.
    top.appendChild(display_html_header(dom, evset))

    # Create one table and header per index value
    for index_key in all_index_keys[:max_indexes]:
        index_data = evset.data[index_key]
        num_timestamps = len(index_data.timestamps)
        max_timestamps = config.max_display_events or num_timestamps

        # Display index values
        html_index_value = dom.createElement("div")
        top.appendChild(html_index_value)
        html_index_value.appendChild(html_style_bold(dom, "index"))
        html_index_value.appendChild(dom.createTextNode(" ("))

        for idx, (item_value, item_schema) in enumerate(
            zip(index_key, evset.schema.indexes)
        ):
            if idx != 0:
                html_index_value.appendChild(dom.createTextNode(", "))

            html_index_value.appendChild(
                html_style(dom, item_schema.name, _HTML_STYLE_INDEX_KEY)
            )
            html_index_value.appendChild(dom.createTextNode("="))
            if isinstance(item_value, bytes):
                item_value = item_value.decode()
            html_index_value.appendChild(
                html_style(dom, str(item_value), _HTML_STYLE_INDEX_VALUE)
            )

        html_index_value.appendChild(
            dom.createTextNode(f") with {num_timestamps} events")
        )

        # Table with column names
        table = html_style(dom, dom.createElement("table"), _HTML_STYLE_TABLE)
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

    return top.toprettyxml(indent="  ")


def html_style(
    dom: Dom, item: StyleHtml, style_attributes: Dict[str, Any]
) -> Html:
    if isinstance(item, minidom.Text):
        raw_item = item
        item = dom.createElement("span")
        item.appendChild(raw_item)
    elif isinstance(item, str):
        raw_item = item
        item = dom.createElement("span")
        item.appendChild(dom.createTextNode(raw_item))

    style_key = "style"

    # Get existing style, if any.
    style = (
        item.getAttribute(style_key) + "; "
        if item.hasAttribute(style_key)
        else ""
    )

    # Add new style
    style += "; ".join([f"{k}:{v}" for k, v in style_attributes.items()])

    item.setAttribute("style", style)
    return item


def html_style_bold(dom: Dom, item: StyleHtml) -> Html:
    return html_style(dom, item, {"font-weight": "bold"})


def html_style_italic(dom: Dom, item: StyleHtml) -> Html:
    return html_style(dom, item, {"font-style": "italic"})


def display_html_vertical_space(dom: Dom, px: int) -> Html:
    root = dom.createElement("div")
    root.setAttribute("style", f"padding-bottom: {px}px")
    root.appendChild(dom.createTextNode(""))
    return root


def display_html_memory_usage(dom: Dom, evset: EventSet) -> Html:
    root = dom.createElement("div")
    root.appendChild(html_style_bold(dom, "memory usage: "))
    root.appendChild(
        dom.createTextNode(string.pretty_num_bytes(evset.memory_usage()))
    )
    return root


def display_html_header(dom: Dom, evset: EventSet) -> Html:
    root = html_style(dom, dom.createElement("div"), _HTML_STYLE_HEADER_DIV)

    # Features

    html_features = dom.createElement("div")
    root.appendChild(html_features)

    html_features_header = dom.createElement("span")
    html_features.appendChild(html_features_header)
    html_features_header.appendChild(html_style_bold(dom, "features"))
    html_features_header.appendChild(dom.createTextNode(" ["))
    html_features_header.appendChild(
        dom.createTextNode(str(len(evset.schema.features)))
    )
    html_features_header.appendChild(dom.createTextNode("]: "))

    if len(evset.schema.features) == 0:
        html_features.appendChild(html_style_italic(dom, "none"))

    # If limit=0 or None, set limit=len
    num_features = len(evset.schema.features)
    max_features = config.max_display_feature_dtypes or num_features

    for idx, feature in enumerate(evset.schema.features[:max_features]):
        if idx != 0:
            html_features.appendChild(dom.createTextNode(", "))

        html_feature = dom.createElement("span")
        html_feature.appendChild(dom.createTextNode("("))
        html_feature.appendChild(
            html_style(dom, feature.name, _HTML_STYLE_FEATURE_KEY)
        )
        html_feature.appendChild(dom.createTextNode(", "))
        html_feature.appendChild(
            html_style(dom, str(feature.dtype), _HTML_STYLE_DTYPE)
        )
        html_feature.appendChild(dom.createTextNode(")"))
        html_features.appendChild(html_feature)

    if max_features < num_features:
        html_features.appendChild(dom.createTextNode(f", ..."))

    # Indexes

    html_indexes = dom.createElement("div")
    root.appendChild(html_indexes)

    html_indexes_header = dom.createElement("span")
    html_indexes.appendChild(html_indexes_header)
    html_indexes_header.appendChild(html_style_bold(dom, "indexes"))
    html_indexes_header.appendChild(dom.createTextNode(" ["))
    html_indexes_header.appendChild(
        dom.createTextNode(str(len(evset.schema.indexes)))
    )
    html_indexes_header.appendChild(dom.createTextNode("]: "))

    if len(evset.schema.indexes) == 0:
        html_indexes.appendChild(html_style_italic(dom, "none"))

    # If limit=0 or None, set limit=len
    num_indexes = len(evset.schema.indexes)
    max_indexes = config.max_display_index_dtypes or num_indexes

    for idx, index in enumerate(evset.schema.indexes[:max_indexes]):
        if idx != 0:
            html_indexes.appendChild(dom.createTextNode(", "))

        html_index = dom.createElement("span")
        html_index.appendChild(dom.createTextNode("("))
        html_index.appendChild(
            html_style(dom, index.name, _HTML_STYLE_INDEX_KEY)
        )
        html_index.appendChild(dom.createTextNode(", "))
        html_index.appendChild(
            html_style(dom, str(index.dtype), _HTML_STYLE_DTYPE)
        )
        html_index.appendChild(dom.createTextNode(")"))
        html_indexes.appendChild(html_index)

    if max_indexes < num_indexes:
        html_indexes.appendChild(dom.createTextNode(f", ..."))

    # Number of events
    html_num_examples = dom.createElement("div")
    html_num_examples.appendChild(html_style_bold(dom, "events: "))
    html_num_examples.appendChild(dom.createTextNode(str(evset.num_events())))
    root.appendChild(html_num_examples)

    # Number of indexes
    html_num_examples = dom.createElement("div")
    html_num_examples.appendChild(html_style_bold(dom, "index values: "))
    html_num_examples.appendChild(dom.createTextNode(str(len(evset.data))))
    root.appendChild(html_num_examples)

    # Memory usage
    root.appendChild(display_html_memory_usage(dom, evset))

    return root


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
