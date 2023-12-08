#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "temporian/implementation/numpy_cc/operators/calendar/common.h"
#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

py::array_t<int> calendar_day_of_week(const py::array_t<double> &timestamps,
                                      const py::object tz) {
  auto parsed_tz = handle_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto days = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_days = days.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and get the day_of_week
    auto local_time = absl::ToCivilDay(utc_time, parsed_tz);
    v_days[i] = map_week_day(absl::GetWeekday(local_time));
  }

  return days;
}
}  // namespace

void init_calendar_day_of_week(py::module &m) {
  m.def("calendar_day_of_week", &calendar_day_of_week, "",
        py::arg("timestamps").noconvert(), py::arg("tz"));
}