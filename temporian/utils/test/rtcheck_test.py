from typing import List, Dict, Set, Union, Tuple, Optional

from absl.testing import absltest
import re
import numpy as np

from temporian.utils.rtcheck import rtcheck, _check_annotation, _Trace
from temporian.core.compilation import compile
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.data.node import EventSetNode, input_node
from temporian.implementation.numpy.data.io import event_set


def _m(v, a):
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
                'When checking function "f". When checking argument "c".'
                " Found value of type <class 'int'> when type <class 'str'> was"
                " expected. The exact value is 3."
            ),
        ):
            f(1, 2, 3)

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                'When checking function "f". When checking argument "b".'
                " Found value of type <class 'str'> when type <class 'int'> was"
                " expected. The exact value is aze."
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
            ValueError, "Apply @rtcheck before @compile"
        ):

            @compile
            @rtcheck
            def h(a: EventSetNode) -> EventSetNode:
                return a

    def test_m_int(self):
        self.assertTrue(_m(1, int))

        self.assertFalse(_m(1.5, int))
        self.assertFalse(_m("hello", int))
        self.assertFalse(_m([], int))
        self.assertFalse(_m({}, int))
        self.assertFalse(_m(SomeClass(), int))

    def test_m_float(self):
        self.assertTrue(_m(1.5, float))

        self.assertFalse(_m(1, float))
        self.assertFalse(_m("hello", float))
        self.assertFalse(_m([], float))
        self.assertFalse(_m({}, float))
        self.assertFalse(_m(SomeClass(), float))

    def test_m_list(self):
        self.assertTrue(_m([], List))
        self.assertTrue(_m(["a", 1], List))
        self.assertTrue(_m([1, 2], List))
        self.assertTrue(_m([1, 2], List[int]))
        self.assertTrue(_m(["a", "b"], List[str]))

        self.assertFalse(_m(1.5, List))
        self.assertFalse(_m("hello", List))
        self.assertFalse(_m({}, List))
        self.assertFalse(_m(SomeClass(), List))

        self.assertFalse(_m([1, 2], List[str]))
        self.assertFalse(_m(["a", 2], List[int]))
        self.assertFalse(_m(["a", 1], List[str]))
        self.assertFalse(_m([[], [1, 2]], List[str]))

    def test_m_dict(self):
        self.assertTrue(_m({}, Dict))
        self.assertTrue(_m({1: 2, 3: 4}, Dict))
        self.assertTrue(_m({"a": 2, "b": 4}, Dict[str, int]))

        self.assertFalse(_m(1.5, Dict))
        self.assertFalse(_m("hello", Dict))
        self.assertFalse(_m([], Dict))
        self.assertFalse(_m(SomeClass(), Dict))

        self.assertFalse(_m({"a": 2, "b": 4}, Dict[int, str]))
        self.assertFalse(_m({1: 2, 3: 4}, Dict[str, int]))

    def test_m_numpy(self):
        self.assertTrue(_m(np.array(1), np.ndarray))

        self.assertFalse(_m(1.5, np.ndarray))
        self.assertFalse(_m("hello", np.ndarray))
        self.assertFalse(_m([], np.ndarray))
        self.assertFalse(_m({}, np.ndarray))
        self.assertFalse(_m(SomeClass(), np.ndarray))

    def test_m_set(self):
        self.assertTrue(_m(set(), Set))
        self.assertTrue(_m({"a", 1}, Set))
        self.assertTrue(_m({1, 2}, Set))
        self.assertTrue(_m({1, 2}, Set[int]))
        self.assertTrue(_m({"a", "b"}, Set[str]))

        self.assertFalse(_m(1.5, Set))
        self.assertFalse(_m("hello", Set))
        self.assertFalse(_m([], Set))
        self.assertFalse(_m(SomeClass(), Set))

        self.assertFalse(_m({1, 2}, Set[str]))
        self.assertFalse(_m({"a", 2}, Set[int]))
        self.assertFalse(_m({"a", 1}, Set[str]))

    def test_m_union(self):
        self.assertTrue(_m(1, Union[int, str]))

        self.assertFalse(_m([], Union[int, str]))

    def test_m_optional(self):
        self.assertTrue(_m(1, Optional[int]))
        self.assertTrue(_m(None, Optional[int]))
        self.assertTrue(_m(1, Optional))

        self.assertFalse(_m("hello", Optional[int]))

    def test_m_tuple(self):
        self.assertTrue(_m((1, "aze"), Tuple[int, str]))
        self.assertTrue(_m((1, "aze"), Tuple))

        self.assertFalse(_m((1, 2), Tuple[int, str]))

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
                "Found value of type <class 'str'> when type <class 'int'> was"
                " expected."
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
                "Found value of type <class 'str'> when type <class 'int'> was"
                " expected."
            ),
        ):
            a(x="1", y="2")


if __name__ == "__main__":
    absltest.main()
