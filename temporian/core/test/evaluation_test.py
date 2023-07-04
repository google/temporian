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

from absl.testing import absltest

from temporian.core import evaluation
from temporian.core.test import utils
from temporian.implementation.numpy.data.event_set import EventSet

import temporian as tp


class EvaluationTest(absltest.TestCase):
    def test_schedule_trivial(self):
        a = utils.create_source_node()
        b = utils.OpI1O1(a)

        schedule = evaluation.build_schedule(
            inputs={a}, outputs={b.outputs["output"]}
        )
        self.assertEqual(schedule.ordered_operators, [b])

    def test_schedule_empty(self):
        a = utils.create_source_node()

        schedule = evaluation.build_schedule(inputs={a}, outputs={a})
        self.assertEqual(schedule.ordered_operators, [])

    def test_schedule_two_delayed_inputs(self):
        i1 = utils.create_source_node()
        i2 = utils.create_source_node()
        o1 = utils.OpI1O1(i1)
        o2 = utils.OpI1O1(i2)
        o3 = utils.OpI2O1(o1.outputs["output"], o2.outputs["output"])
        schedule = evaluation.build_schedule(
            inputs={i1, i2}, outputs={o3.outputs["output"]}
        )
        self.assertTrue(
            (schedule.ordered_operators == [o2, o1, o3])
            or (schedule.ordered_operators == [o1, o2, o3])
        )

    def test_schedule_basic(self):
        i1 = utils.create_source_node()
        o2 = utils.OpI1O1(i1)
        i3 = utils.create_source_node()
        o4 = utils.OpI2O1(o2.outputs["output"], i3)
        o5 = utils.OpI1O2(o4.outputs["output"])

        schedule = evaluation.build_schedule(
            inputs={i1, i3},
            outputs={o5.outputs["output_1"], o4.outputs["output"]},
        )
        self.assertTrue(
            (schedule.ordered_operators == [o2, o4, o5])
            or (schedule.ordered_operators == [o4, o2, o5])
        )

    def test_schedule_mid_chain(self):
        i1 = utils.create_source_node()
        o2 = utils.OpI1O1(i1)
        o3 = utils.OpI1O1(o2.outputs["output"])
        o4 = utils.OpI1O1(o3.outputs["output"])
        o5 = utils.OpI1O1(o4.outputs["output"])

        schedule = evaluation.build_schedule(
            inputs={o3.outputs["output"]}, outputs={o5.outputs["output"]}
        )
        self.assertEqual(schedule.ordered_operators, [o4, o5])

    def test_run_value(self):
        i1 = utils.create_source_node()
        result = evaluation.run(i1, {i1: utils.create_input_event_set()})
        self.assertIsInstance(result, EventSet)

    def test_run_query_list(self):
        i1 = utils.create_source_node()
        i2 = utils.create_source_node()
        result = evaluation.run(
            [i1, i2],
            {
                i1: utils.create_input_event_set(),
                i2: utils.create_input_event_set(),
            },
        )
        self.assertIsInstance(result, list)
        self.assertLen(result, 2)

    def test_run_query_dict(self):
        i1 = utils.create_source_node()
        i2 = utils.create_source_node()
        result = evaluation.run(
            {"i1": i1, "i2": i2},
            {
                i1: utils.create_input_event_set(),
                i2: utils.create_input_event_set(),
            },
        )
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertLen(result, 2)
        self.assertEqual(set(result.keys()), {"i1", "i2"})

    def test_run_input_event_set(self):
        input_evset = utils.create_input_event_set()
        result = evaluation.run(input_evset.node(), input_evset)
        self.assertIsInstance(result, EventSet)

    def test_run_input_list_event_set(self):
        input_1 = utils.create_input_event_set()
        input_2 = utils.create_input_event_set()
        result = evaluation.run(
            [input_1.node(), input_2.node()],
            [input_1, input_2],
        )
        self.assertIsInstance(result, list)

    def test_run_single_unnamed_input_named(self):
        e1 = utils.create_input_event_set("i1")
        i1 = e1.node()

        result = evaluation.run(i1, e1)

        self.assertIsInstance(result, EventSet)
        self.assertTrue(result is e1)

    def test_run_list_unnamed_inputs_named(self):
        e1 = utils.create_input_event_set("i1")
        i1 = e1.node()
        e2 = utils.create_input_event_set("i2")
        i2 = e2.node()
        e3 = utils.create_input_event_set("i3")
        i3 = e3.node()

        result = evaluation.run([i3, i1, i2], [e1, e2, e3])
        isinstance(result, list)

        self.assertIsInstance(result, list)
        self.assertTrue(result[0] is e3)
        self.assertTrue(result[1] is e1)
        self.assertTrue(result[2] is e2)

    def test_run_repeated_unnamed_inputs(self):
        i1 = utils.create_source_node(name="i1")
        evset_1 = utils.create_input_event_set(name="i1")
        evset_2 = utils.create_input_event_set(name="i1")

        with self.assertRaises(ValueError):
            evaluation.run(i1, [evset_1, evset_2])

    def test_has_leak(self):
        a = tp.input_node([("f", float)])
        b = tp.moving_sum(a, 5)
        c = tp.leak(b, 5)
        d = tp.prefix("something_", c)
        e = tp.moving_sum(d, 2)

        self.assertTrue(tp.has_leak(e))
        self.assertTrue(tp.has_leak(e, a))
        self.assertTrue(tp.has_leak([e], [a]))
        self.assertTrue(tp.has_leak({"e": e}, {"a": a}))

        self.assertFalse(tp.has_leak(e, c))
        self.assertFalse(tp.has_leak(e, d))


if __name__ == "__main__":
    absltest.main()
