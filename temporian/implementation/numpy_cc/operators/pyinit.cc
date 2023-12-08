#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "temporian/implementation/numpy_cc/operators/add_index.h"
#include "temporian/implementation/numpy_cc/operators/calendar/day_of_month.h"
#include "temporian/implementation/numpy_cc/operators/calendar/day_of_week.h"
#include "temporian/implementation/numpy_cc/operators/calendar/day_of_year.h"
#include "temporian/implementation/numpy_cc/operators/calendar/hour.h"
#include "temporian/implementation/numpy_cc/operators/calendar/isoweek.h"
#include "temporian/implementation/numpy_cc/operators/calendar/minute.h"
#include "temporian/implementation/numpy_cc/operators/calendar/month.h"
#include "temporian/implementation/numpy_cc/operators/calendar/second.h"
#include "temporian/implementation/numpy_cc/operators/calendar/year.h"
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
  init_calendar_year(m);
  init_calendar_month(m);
  init_calendar_day_of_month(m);
  init_calendar_hour(m);
  init_calendar_minute(m);
  init_calendar_day_of_year(m);
  init_calendar_day_of_week(m);
  init_calendar_isoweek(m);
}
