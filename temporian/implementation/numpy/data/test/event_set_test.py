import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set, IndexData
from temporian.utils import config
from temporian.utils import golden


class EventSetTest(absltest.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            features={
                "a": [1, 2, 3, 7, 8],
                "b": [4, 5, 6, 9, 10],
                "x": [1, 1, 1, 2, 2],
                "y": ["hello", "hello", "hello", "world", "world"],
            },
            indexes=["x", "y"],
        )

    def test_get_index_value(self):
        index_data = self.evset.get_index_value((2, b"world"), normalize=False)
        self.assertTrue(isinstance(index_data, IndexData))
        self.assertTrue(
            (np.array(index_data.features) == [[7, 8], [9, 10]]).all()
        )
        self.assertTrue((abs(index_data.timestamps - [0.4, 0.5]) < 1e-6).all())

    def test_get_index_value_normalized(self):
        index_data = self.evset.get_index_value((2, "world"))
        self.assertTrue(isinstance(index_data, IndexData))
        self.assertTrue(
            (np.array(index_data.features) == [[7, 8], [9, 10]]).all()
        )
        self.assertTrue((abs(index_data.timestamps - [0.4, 0.5]) < 1e-6).all())

    def test_set_index_value(self):
        value = self.evset.get_index_value((1, b"hello"))
        modified = IndexData(
            timestamps=value.timestamps,
            features=[f + 1 for f in value.features],
        )
        self.evset.set_index_value((2, b"world"), modified)
        self.assertEqual(self.evset.get_index_value((2, b"world")), modified)

    def test_data_access(self):
        self.assertEqual(
            repr(self.evset.schema.features), "[('a', int64), ('b', int64)]"
        )
        self.assertEqual(
            repr(self.evset.schema.indexes), "[('x', int64), ('y', str_)]"
        )

    def test_memory_usage(self):
        memory_usage = self.evset.memory_usage()
        print("memory_usage:", memory_usage)

        self.assertLessEqual(memory_usage, 1200 + 500)
        self.assertGreaterEqual(memory_usage, 1200 - 500)

    def test_repr_nolimits(self):
        config.print_max_events = 0
        config.print_max_features = 0
        config.print_max_indexes = 0
        print(self.evset)
        self.assertEqual(
            repr(self.evset),
            """indexes: [('x', int64), ('y', str_)]
features: [('a', int64), ('b', int64)]
events:
    x=1 y=b'hello' (3 events):
        timestamps: [0.1 0.2 0.3]
        'a': [1 2 3]
        'b': [4 5 6]
    x=2 y=b'world' (2 events):
        timestamps: [0.4 0.5]
        'a': [7 8]
        'b': [ 9 10]
memory usage: 1.2 kB
""",
        )

    def test_repr_limits(self):
        config.print_max_features = 1
        config.print_max_indexes = 1
        # Numpy summarization only makes sense for limit >= 6
        config.print_max_events = 6
        evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            features={
                "a": [1, 2, 3, 7, 8, 9, 10, 11],
                "b": [4, 5, 6, 9, 10, 15, 16, 17],
                "x": [1, 1, 1, 1, 1, 1, 1, 2],
                "y": ["h", "h", "h", "h", "h", "h", "h", "w"],
            },
            indexes=["x", "y"],
        )
        self.assertEqual(
            repr(evset),
            """indexes: [('x', int64), ('y', str_)]
features: [('a', int64), ('b', int64)]
events:
    x=1 y=b'h' (7 events):
        timestamps: [0.1 0.2 0.3 ... 0.5 0.6 0.7]
        'a': [ 1  2  3 ...  8  9 10]
        ...
    ... (showing 1 of 2 indexes)
memory usage: 1.2 kB
""",
        )

    def test_html_repr_no_limits(self):
        config.display_max_indexes = 0
        config.display_max_features = 0
        config.display_max_events = 0

        golden.check_string(
            self,
            self.evset._repr_html_(),
            "temporian/implementation/numpy/data/test/test_data/test_html_repr_no_limits_golden.html",
        )

    def test_html_repr_limits(self):
        config.display_max_indexes = 1
        config.display_max_features = 1
        config.display_max_events = 2

        golden.check_string(
            self,
            self.evset._repr_html_(),
            "temporian/implementation/numpy/data/test/test_data/test_html_repr_limits_golden.html",
        )


if __name__ == "__main__":
    absltest.main()
