from typing import List, Dict, Set, Union, Tuple, Optional

from absl.testing import absltest
import re
import numpy as np

from temporian.utils.rtcheck import (
    rtcheck,
    _check_annotation,
    _Trace,
    runtime_check_raise_exception,
)
from temporian.core.compilation import compile
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.data.node import EventSetNode, input_node
from temporian.implementation.numpy.data.io import event_set


def check(v, a):
    try:
        _check_annotation(_Trace(), False, v, a)
    except Exception as e:
        print("Check annotation failed: ", e, flush=True)
        return False
    return True


@rtcheck
def f(a, b: int, c: str = "aze") -> List[str]:
    del a
    del b
    del c
    return ["a", "b"]


class SomeClass:
    pass


class RTCheckTest(absltest.TestCase):
    def test_base(self):
        f(1, 2, "a")
        f(1, 2)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Expecting value of type <class 'str'> but received value of"
                " type <class 'int'>. The value is \"3\""
            ),
        ):
            f(1, 2, 3)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Expecting value of type <class 'int'> but received value of"
                " type <class 'str'>. The value is \"aze\"."
            ),
        ):
            f(1, "aze")

    def test_compile(self):
        @rtcheck
        @compile
        def g(a: EventSetNode) -> EventSetNode:
            return a

        g(event_set([1, 2, 3]))
        g(input_node([]))

    def test_wrong_compile_order(self):
        with self.assertRaisesRegex(
            ValueError, "Apply @compile before @rtcheck"
        ):

            @compile
            @rtcheck
            def h(a: EventSetNode) -> EventSetNode:
                return a

    def test_disable_rt_check(self):
        runtime_check_raise_exception(False)
        f(1, 2, 3)
        runtime_check_raise_exception(True)
        with self.assertRaises(ValueError):
            f(1, 2, 3)

    def test_m_int(self):
        self.assertTrue(check(1, int))

        self.assertFalse(check(1.5, int))
        self.assertFalse(check("hello", int))
        self.assertFalse(check([], int))
        self.assertFalse(check({}, int))
        self.assertFalse(check(SomeClass(), int))

    def test_m_float(self):
        self.assertTrue(check(1.5, float))

        self.assertFalse(check(1, float))
        self.assertFalse(check("hello", float))
        self.assertFalse(check([], float))
        self.assertFalse(check({}, float))
        self.assertFalse(check(SomeClass(), float))

    def test_m_list(self):
        self.assertTrue(check([], List))
        self.assertTrue(check(["a", 1], List))
        self.assertTrue(check([1, 2], List))
        self.assertTrue(check([1, 2], List[int]))
        self.assertTrue(check(["a", "b"], List[str]))

        self.assertFalse(check(1.5, List))
        self.assertFalse(check("hello", List))
        self.assertFalse(check({}, List))
        self.assertFalse(check(SomeClass(), List))

        self.assertFalse(check([1, 2], List[str]))
        self.assertFalse(check(["a", 2], List[int]))
        self.assertFalse(check(["a", 1], List[str]))
        self.assertFalse(check([[], [1, 2]], List[str]))

    def test_m_dict(self):
        self.assertTrue(check({}, Dict))
        self.assertTrue(check({1: 2, 3: 4}, Dict))
        self.assertTrue(check({"a": 2, "b": 4}, Dict[str, int]))

        self.assertFalse(check(1.5, Dict))
        self.assertFalse(check("hello", Dict))
        self.assertFalse(check([], Dict))
        self.assertFalse(check(SomeClass(), Dict))

        self.assertFalse(check({"a": 2, "b": 4}, Dict[int, str]))
        self.assertFalse(check({1: 2, 3: 4}, Dict[str, int]))

    def test_m_numpy(self):
        self.assertTrue(check(np.array(1), np.ndarray))

        self.assertFalse(check(1.5, np.ndarray))
        self.assertFalse(check("hello", np.ndarray))
        self.assertFalse(check([], np.ndarray))
        self.assertFalse(check({}, np.ndarray))
        self.assertFalse(check(SomeClass(), np.ndarray))

    def test_m_set(self):
        self.assertTrue(check(set(), Set))
        self.assertTrue(check({"a", 1}, Set))
        self.assertTrue(check({1, 2}, Set))
        self.assertTrue(check({1, 2}, Set[int]))
        self.assertTrue(check({"a", "b"}, Set[str]))

        self.assertFalse(check(1.5, Set))
        self.assertFalse(check("hello", Set))
        self.assertFalse(check([], Set))
        self.assertFalse(check(SomeClass(), Set))

        self.assertFalse(check({1, 2}, Set[str]))
        self.assertFalse(check({"a", 2}, Set[int]))
        self.assertFalse(check({"a", 1}, Set[str]))

    def test_m_union(self):
        self.assertTrue(check(1, Union[int, str]))

        self.assertFalse(check([], Union[int, str]))

    def test_m_optional(self):
        self.assertTrue(check(1, Optional[int]))
        self.assertTrue(check(None, Optional[int]))
        self.assertTrue(check(1, Optional))

        self.assertFalse(check("hello", Optional[int]))

    def test_m_tuple(self):
        self.assertTrue(check((1, "aze"), Tuple[int, str]))
        self.assertTrue(check((1, "aze"), Tuple))

        self.assertFalse(check((1, 2), Tuple[int, str]))

    def test_args(self):
        @rtcheck
        def a(*x: int) -> int:
            return sum(x)

        a()
        a(1)
        a(1, 2)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Expecting value of type <class 'int'> but received value of"
                " type <class 'str'>."
            ),
        ):
            a("a", "b")

    def test_wargs(self):
        @rtcheck
        def a(**x: int) -> int:
            return sum([v for _, v in x.items()])

        a()
        a(x=1)
        a(x=1, y=2)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Expecting value of type <class 'int'> but received value of"
                " type <class 'str'>."
            ),
        ):
            a(x="1", y="2")


if __name__ == "__main__":
    absltest.main()
