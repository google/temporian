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

py::array_t<int> calendar_month(const py::array_t<double> &timestamps,
                                const py::object tz) {
  auto parsed_tz = handle_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto months = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_months = months.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and get the month
    auto local_time = absl::ToCivilDay(utc_time, parsed_tz);
    v_months[i] = local_time.month();
  }

  return months;
}
}  // namespace

void init_calendar_month(py::module &m) {
  m.def("calendar_month", &calendar_month, "",
        py::arg("timestamps").noconvert(), py::arg("tz"));
}