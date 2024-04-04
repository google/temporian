"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Dict, Any, Tuple, Iterator, Sequence, List

import numpy as np
import apache_beam as beam

from temporian.core.data.dtype import DType, tp_dtype_to_py_type
from temporian.io.format import DictEventSetFormat, DictEventSetFormatChoices
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)
from temporian.beam.typing import (
    SingleFeatureValue,
    BeamIndexKeyItem,
    StructuredRow,
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    StructuredRowValue,
    POS_FEATURE_IDX,
    POS_TIMESTAMP_VALUES,
    POS_FEATURE_VALUES,
    FeatureItemWithIdxValue,
    FeatureItemWithIdx,
)
from temporian.core.data.schema import Schema, FeatureSchema
from temporian.beam.io import np_array_coder  # pylint: disable=unused-import


def _cast_feature_value(value: Any, dtype: DType) -> SingleFeatureValue:
    """Convert a user feature value to the internal representation."""

    py_type = tp_dtype_to_py_type(dtype)
    if py_type is bytes and isinstance(value, str):
        return bytes(value, encoding="utf-8")
    return py_type(value)


def _cast_index_value(value: Any, dtype: DType) -> BeamIndexKeyItem:
    """Convert a user index value to the internal representation."""

    return _cast_feature_value(value, dtype)


def _parse_and_index(
    row: Dict[str, Any], schema: Schema, timestamp_key: str
) -> StructuredRow:
    """Transforms a dict of key:value to a StructuredRow.

    In essence, this function replaces the string index feature and index values
    with a integer index (based on a schema).

    Example:
        row = {"timestamps": 10, "f1": 11, "f2": 12, "i1": 13}
        schema: features = [f1, f2], indexes = [i1]
        timestamp_key: timestamps

        Output
            (13, ), (10, (11, 12))

    This function is used during the conversion of key:value features feed by
    the user into BeamEventSet, the working format used by Temporian.
    """

    index_values = tuple(
        _cast_index_value(row[index.name], index.dtype)
        for index in schema.indexes
    )
    feature_values = tuple(
        _cast_feature_value(row[feature.name], feature.dtype)
        for feature in schema.features
    )
    timestamp = float(row[timestamp_key])
    return index_values, (timestamp, feature_values)


class _MergeTimestamps(beam.DoFn):
    """Aggregates + split StructuredRows into IndexValue.

    This function aggregates together all the timestamps+values having the same
    feature+index, and splits aggregated values by features.

    Example:
        Input
            (20, ), (100, (11, 12))
            (20, ), (101, (13, 14))
            (21, ), (102, (15, 16))

        Output
            (20, ), (0, (100, 101), (11, 13))
            (20, ), (1, (100, 101), (12, 14))
            (21, ), (0, (102,), (15,))
            (21, ), (1, (102,), (16,))

    This function is used during the conversion of key:value features feed by
    the user into BeamEventSet, the working format used by Temporian.
    """

    def __init__(self, features: List[FeatureSchema]):
        self._feature_np_dtypes = [
            tp_dtype_to_np_dtype(feature.dtype) for feature in features
        ]

    def process(
        self,
        item: Tuple[BeamIndexKey, Sequence[StructuredRowValue]],
    ) -> Iterator[FeatureItemWithIdx]:
        index, feat_and_ts = item
        timestamps = np.fromiter(
            (v[0] for v in feat_and_ts),
            dtype=np.float64,
            count=len(feat_and_ts),
        )
        for feature_idx, feature_np_type in enumerate(self._feature_np_dtypes):
            if feature_np_type is np.bytes_:
                # Let numpy figure the length of the longest string.
                values = np.array([v[1][feature_idx] for v in feat_and_ts])
            else:
                values = np.fromiter(
                    (v[1][feature_idx] for v in feat_and_ts),
                    dtype=feature_np_type,
                    count=len(feat_and_ts),
                )
            yield index, (timestamps, values, feature_idx)


def _merge_timestamps_no_features(
    item: Tuple[BeamIndexKey, Sequence[StructuredRowValue]],
) -> FeatureItem:
    """Same as _MergeTimestamps, but when there are no features."""

    index, feat_and_ts = item
    timestamps = np.fromiter(
        (v[0] for v in feat_and_ts), dtype=np.float64, count=len(feat_and_ts)
    )
    return index, (timestamps, None)


def _event_set_dict_to_event_set(
    input: Dict[str, Any], schema: Schema, timestamp_key: str
) -> Iterator[FeatureItemWithIdx]:
    """Converts a `indexedEvents` event-set into an internal event-set.

    Example:
        Input
            Schema
                features=[("f1", DType.INT64), ("f2", DType.STRING)]
                indexes=[("i1", DType.INT64), ("i2", DType.STRING)]

            input (one item):
                {
                "timestamp": [100.0, 101.0, 102.0],
                "f1": [1, 2, 3],
                "f2": [b"a", b"b", b"c"],
                "i1": 10,
                "i2": b"x",
                }

        Output (two items)
            # Feature "f1"
            ((10, b"x"), 0, ([100.0, 101.0, 102.0], [1, 2, 3])
            # Feature "f2"
            ((10, b"x"), 1, ([100.0, 101.0, 102.0], [b"a", b"b", b"c"])
    """

    timestamps = input[timestamp_key]
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.dtype.type != np.float64
    ):
        raise ValueError(
            f"Timestamp with value {timestamps} is expected to be np.float64"
            f" numpy array, but has dtype {type(timestamps)} instead."
        )

    index = []
    for index_schema in schema.indexes:
        expected_type = tp_dtype_to_py_type(index_schema.dtype)
        src_value = input[index_schema.name]

        if not isinstance(src_value, expected_type):
            raise ValueError(
                f'Index "{index_schema.name}" with value "{src_value}" is'
                f" expected to be of type {expected_type} (since the Temporian "
                f" dtype is {index_schema.dtype}) but type"
                f" {type(src_value)} was found."
            )
        index.append(src_value)
    index_tuple = tuple(index)

    for feature_idx, feature_schema in enumerate(schema.features):
        expected_dtype = tp_dtype_to_np_dtype(feature_schema.dtype)
        src_value = input[feature_schema.name]

        if (
            not isinstance(src_value, np.ndarray)
            or src_value.dtype.type != expected_dtype
        ):
            if isinstance(src_value, np.ndarray):
                effective_type = src_value.dtype.type
            else:
                effective_type = type(src_value)

            raise ValueError(
                f'Feature "{feature_schema.name}" with value "{src_value}" is'
                " expected to be a numpy array with dtype"
                f" {expected_dtype} (since the Temporian dtype is"
                f" {feature_schema.dtype}) but numpy dtype"
                f" {effective_type} was found."
            )

        yield index_tuple, (timestamps, src_value, feature_idx)


def _event_set_dict_to_event_set_no_features(
    input: Dict[str, Any], schema: Schema, timestamp_key: str
) -> FeatureItem:
    """Same as _event_set_dict_to_event_set, but without features"""

    timestamps = input[timestamp_key]
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.dtype.type != np.float64
    ):
        raise ValueError(
            f"Timestamp with value {timestamps} is expected to be np.float64"
            f" numpy array, but has dtype {type(timestamps)} instead."
        )

    index = []
    for index_schema in schema.indexes:
        expected_type = tp_dtype_to_py_type(index_schema.dtype)
        src_value = input[index_schema.name]

        if not isinstance(src_value, expected_type):
            raise ValueError(
                f'Index "{index_schema.name}" with value "{src_value}" is'
                f" expected to be of type {expected_type} (since the Temporian "
                f" dtype is {index_schema.dtype}) but type"
                f" {type(src_value)} was found."
            )
        index.append(src_value)
    index_tuple = tuple(index)

    return index_tuple, (timestamps, None)


def _reshuffle_item_in_tuples(items: tuple) -> tuple:
    return tuple(
        (e | f"Shuffle #{i}" >> beam.Reshuffle()) for i, e in enumerate(items)
    )


@beam.ptransform_fn
def to_event_set(
    pipe: beam.PCollection[Dict[str, Any]],
    schema: Schema,
    timestamp_key: str = "timestamp",
    format: DictEventSetFormatChoices = DictEventSetFormat.GROUPED_BY_INDEX,
) -> BeamEventSet:
    """Converts a PCollection of key:value to a Beam EventSet.

    This method is compatible with the output of `from_csv_raw` and the
    Official Beam IO connectors.

    When importing data from csv files, use `from_csv` to convert csv files
    directly into EventSets.

    Unlike Temporian in-process EventSet import method (
    [tp.event_set][temporian.event_set])), this method (`tpb.to_event_set`)
    requires for timestamps to be numerical values.

    Args:
        pipe: Beam pipe of key values.
        schema: Schema of the data. Note: The schema of a Temporian node is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        format: Format of the events inside the received dictionary. See
            [DictEventSetFormat][temporian.io.format.DictEventSetFormat] for
            more.

    Returns:
        Beam EventSet.
    """

    # TODO: Add support for datetime timestamps.
    num_features = len(schema.features)
    if format == DictEventSetFormat.GROUPED_BY_INDEX:
        if num_features != 0:
            return partition_by_feature_idx(
                pipe
                | "Parse dict"
                >> beam.FlatMap(
                    _event_set_dict_to_event_set, schema, timestamp_key
                ),
                num_features=num_features,
                reshuffle=True,
            )
        else:
            return _reshuffle_item_in_tuples(
                (
                    pipe
                    | "Parse dict"
                    >> beam.Map(
                        _event_set_dict_to_event_set_no_features,
                        schema,
                        timestamp_key,
                    ),
                )
            )
    elif format == DictEventSetFormat.SINGLE_EVENTS:
        indexed = (
            pipe
            | "Parse and index"
            >> beam.Map(_parse_and_index, schema, timestamp_key)
            # Group by index values and feature index
            | "Aggregate" >> beam.GroupByKey()
            # Build feature and timestamps arrays.
        )
        if num_features != 0:
            return partition_by_feature_idx(
                indexed
                # Build feature and timestamps arrays.
                | "Merge by timestamps"
                >> beam.ParDo(_MergeTimestamps(schema.features)),
                num_features=num_features,
                reshuffle=True,
            )
        else:
            return _reshuffle_item_in_tuples(
                (
                    indexed
                    # Build feature and timestamps arrays.
                    | "Merge by timestamps"
                    >> beam.Map(_merge_timestamps_no_features),
                )
            )
    else:
        raise ValueError(f"Unknown format {format}")


def _convert_to_dict_event_key_value(
    item: Tuple[
        BeamIndexKey,
        Sequence[FeatureItemWithIdxValue],
    ],
    schema: Schema,
    timestamp_key: str,
) -> Iterator[Dict[str, Any]]:
    index, feature_blocks = item

    # Sort the feature by feature index.
    feature_blocks = sorted(feature_blocks, key=lambda x: x[POS_FEATURE_IDX])
    assert len(feature_blocks) > 0

    # All the feature blocks have the same timestamps. We use the first one.
    common_item_dict = {}
    for index_schema, index_value in zip(schema.indexes, index):
        common_item_dict[index_schema.name] = index_value

    timestamps = feature_blocks[0][POS_TIMESTAMP_VALUES]
    for event_idx, timestamp in enumerate(timestamps):
        item_dict = common_item_dict.copy()
        item_dict[timestamp_key] = timestamp.item()
        for feature_schema, feature in zip(schema.features, feature_blocks):
            values = feature[POS_FEATURE_VALUES]
            assert values is not None
            item_dict[feature_schema.name] = values[event_idx].item()

        yield item_dict


def _convert_to_dict_event_set_key_value(
    item: Tuple[
        BeamIndexKey,
        Sequence[FeatureItemWithIdxValue],
    ],
    schema: Schema,
    timestamp_key: str,
) -> Dict[str, Any]:
    index, feature_blocks = item

    # Sort the feature by feature index.
    feature_blocks = sorted(feature_blocks, key=lambda x: x[POS_FEATURE_IDX])
    assert len(feature_blocks) > 0

    item_dict = {}
    for index_schema, index_value in zip(schema.indexes, index):
        item_dict[index_schema.name] = index_value

    # All the feature blocks have the same timestamps. We use the first one.
    item_dict[timestamp_key] = feature_blocks[0][POS_TIMESTAMP_VALUES]
    for feature_schema, feature in zip(schema.features, feature_blocks):
        item_dict[feature_schema.name] = feature[POS_FEATURE_VALUES]

    return item_dict


def _add_feature_idx(src: FeatureItem, feature_idx: int) -> FeatureItemWithIdx:
    index, (timestamps, features) = src
    return index, (timestamps, features, feature_idx)


def _extract_feature_idx(item: FeatureItemWithIdx, num_features: int) -> int:
    return item[1][POS_FEATURE_IDX]


def _remove_feature_idx(src: FeatureItemWithIdx) -> FeatureItem:
    index, (timestamps, features, feature_idx) = src
    del feature_idx
    return index, (timestamps, features)


def add_feature_idx_and_flatten(
    pipe: BeamEventSet,
) -> beam.PCollection[FeatureItemWithIdx]:
    """Groups a tuple of PCollection of features into a single PCollection."""

    with_idx = []
    for feature_idx, feature in enumerate(pipe):
        with_idx.append(
            feature
            | f"Add feature idx {feature_idx}"
            >> beam.Map(_add_feature_idx, feature_idx)
        )
    return tuple(with_idx) | "Flatten features" >> beam.Flatten()


def partition_by_feature_idx(
    pipe: beam.PCollection[FeatureItemWithIdx],
    num_features: int,
    reshuffle: bool,
) -> BeamEventSet:
    """Splits a PCollection of features into a tuple of PCollections.

    The inverse of add_feature_idx_and_flatten.
    """

    partitions = pipe | "Partition by features" >> beam.Partition(
        _extract_feature_idx, num_features
    )
    without_idx = []
    for feature_idx, feature in enumerate(partitions):
        item = feature | f"Remove feature idx {feature_idx}" >> beam.Map(
            _remove_feature_idx
        )
        if reshuffle:
            item = item | f"Shuffle feature {feature_idx}" >> beam.Reshuffle()
        without_idx.append(item)
    return tuple(without_idx)


@beam.ptransform_fn
def to_dict(
    pipe: BeamEventSet,
    schema: Schema,
    timestamp_key: str = "timestamp",
    format: DictEventSetFormatChoices = DictEventSetFormat.GROUPED_BY_INDEX,
) -> beam.PCollection[Dict[str, Any]]:
    """Converts a Beam EventSet to PCollection of key->value.

    This method is compatible with the output of `from_csv_raw` and the
    Official Beam IO connectors. This method is the inverse of `to_event_set`.

    Args:
        pipe: PCollection of Beam EventSet.
        schema: Schema of the data.
        timestamp_key: Key containing the timestamps.
        format: Format of the events inside the output dictionary. See
            [DictEventSetFormat][temporian.io.format.DictEventSetFormat] for
            more.

    Returns:
        Beam pipe of key values.
    """

    # TODO: Add support for datetime timestamps.
    grouped_by_features = (
        add_feature_idx_and_flatten(pipe)
        | "Group by index " >> beam.GroupByKey()
    )

    if format == DictEventSetFormat.GROUPED_BY_INDEX:
        return grouped_by_features | "Convert to dict" >> beam.Map(
            _convert_to_dict_event_set_key_value, schema, timestamp_key
        )
    elif format == DictEventSetFormat.SINGLE_EVENTS:
        return grouped_by_features | "Convert to dict" >> beam.FlatMap(
            _convert_to_dict_event_key_value, schema, timestamp_key
        )
    else:
        raise ValueError(f"Unknown format {format}")
