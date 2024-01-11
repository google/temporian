"""Numpy Coder for Beam

This file define a coder to improve the serialization of Numpy arrays in Beam.
This code is copied from the TensorFlow Graph Neural Network project
(https://github.com/tensorflow/gnn).
"""

import numpy as np

import apache_beam as beam
from apache_beam.coders import typecoders
from apache_beam.typehints import typehints


class NDArrayCoder(beam.coders.Coder):
    """Beam coder for Numpy N-dimensional array of TF-compatible data types.

    Supports all numeric data types and bytes (represented as `np.object_`).
    The numpy array is serialized as a tuple of `(dtype, shape, flat values)`.
    For numeric values serialization we rely on `tobytes()` and `frombuffer` from
    the numpy library. It, seems, has the best speed/space tradeoffs. Tensorflow
    represents `tf.string` as `np.object_` (as `np.string_` is for arrays
    containing fixed-width byte strings, which can lead to lots of wasted
    memory). Because `np.object_` is an array of references to arbitrary
    objects, we could not rely on numpy native serialization and using
    `IterableCoder` from the Beam library instead.

    NOTE: for some simple stages the execution time may be dominated by data
    serialization/deserialization, so any imporvement here translates directly to
    the total execution costs.
    """

    def __init__(self):
        encoded_struct = typehints.Tuple[str, typehints.Tuple[int, ...], bytes]
        self._coder = typecoders.registry.get_coder(encoded_struct)
        self._bytes_coder = typecoders.registry.get_coder(
            typehints.Iterable[bytes]
        )

    def encode(self, value: np.ndarray) -> bytes:
        if value.dtype == np.object_:
            flat_values = self._bytes_coder.encode(value.flat)
        else:
            flat_values = value.tobytes()
        return self._coder.encode((value.dtype.str, value.shape, flat_values))

    def decode(self, encoded: bytes) -> np.ndarray:
        dtype_str, shape, serialized_values = self._coder.decode(encoded)
        dtype = np.dtype(dtype_str)
        if dtype == np.object_:
            flat_values = np.array(
                self._bytes_coder.decode(serialized_values), dtype=np.object_
            )
        else:
            flat_values = np.frombuffer(serialized_values, dtype=dtype)
        return np.reshape(flat_values, shape)

    def is_deterministic(self):
        return True

    def to_type_hint(self):
        return np.ndarray


beam.coders.registry.register_coder(np.ndarray, NDArrayCoder)
