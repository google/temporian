#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "temporian/implementation/numpy_cc/operators/add_index.h"
#include "temporian/implementation/numpy_cc/operators/calendar.h"
#include "temporian/implementation/numpy_cc/operators/filter_moving_count.h"
#include "temporian/implementation/numpy_cc/operators/join.h"
#include "temporian/implementation/numpy_cc/operators/resample.h"
#include "temporian/implementation/numpy_cc/operators/since_last.h"
#include "temporian/implementation/numpy_cc/operators/tick_calendar.h"
#include "temporian/implementation/numpy_cc/operators/until_next.h"
#include "temporian/implementation/numpy_cc/operators/window.h"

namespace {
namespace py = pybind11;
}  // namespace

PYBIND11_MODULE(operators_cc, m) {
  init_since_last(m);
  init_resample(m);
  init_window(m);
  init_join(m);
  init_add_index(m);
  init_tick_calendar(m);
  init_filter_moving_count(m);
  init_until_next(m);
  init_calendar_ops(m);
}
