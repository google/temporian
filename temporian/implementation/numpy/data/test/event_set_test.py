import numpy as np
from absl.testing import absltest

from temporian.implementation.numpy.data.io import event_set, IndexData
from temporian.utils import config


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
        config.max_printed_events = 0
        config.max_printed_features = 0
        config.max_printed_indexes = 0
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
        config.max_printed_features = 1
        config.max_printed_indexes = 1
        # Numpy summarization only makes sense for limit >= 6
        config.max_printed_events = 6
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
        config.max_display_indexes = 0
        config.max_display_features = 0
        config.max_display_events = 0

        self.assertEqual(
            self.evset._repr_html_(),
            "<div>2 indexes × 2 features (memory usage: 1.2 kB)"
            + "<h3>Index: (x=1, y=hello)</h3>"
            + "3 events"
            + "<table>"
            + "<tr><th><b>timestamp</b></th><th><b>a</b></th><th><b>b</b></th></tr>"
            + "<tr><td>0.1</td><td>1</td><td>4</td></tr>"
            + "<tr><td>0.2</td><td>2</td><td>5</td></tr>"
            + "<tr><td>0.3</td><td>3</td><td>6</td></tr>"
            + "</table>"
            + "<h3>Index: (x=2, y=world)</h3>"
            + "2 events"
            + "<table>"
            + "<tr><th><b>timestamp</b></th><th><b>a</b></th><th><b>b</b></th></tr>"
            + "<tr><td>0.4</td><td>7</td><td>9</td></tr>"
            + "<tr><td>0.5</td><td>8</td><td>10</td></tr>"
            + "</table></div>",
        )

    def test_html_repr_limits(self):
        config.max_display_indexes = 1
        config.max_display_features = 1
        config.max_display_events = 2
        dots = "…"  # (ellipsis)

        self.assertEqual(
            self.evset._repr_html_(),
            "<div>2 indexes × 2 features (memory usage: 1.2 kB)"
            + "<h3>Index: (x=1, y=hello)</h3>"
            + "3 events"
            + "<table>"
            + f"<tr><th><b>timestamp</b></th><th><b>a</b></th><th><b>{dots}</b></th></tr>"
            + f"<tr><td>0.1</td><td>1</td><td>{dots}</td></tr>"
            + f"<tr><td>0.2</td><td>2</td><td>{dots}</td></tr>"
            + f"<tr><td>{dots}</td><td>{dots}</td><td>{dots}</td></tr>"
            + "</table>"
            + f"{dots} (1 more indexes not shown)</div>",
        )


if __name__ == "__main__":
    absltest.main()
